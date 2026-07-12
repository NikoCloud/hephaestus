# ===----------------------------------------------------------------------=== #
# Hephaestus -- Qwen3 forward pass (Phase 1a, BF16)
#
# Op order verified against HF transformers 5.13.1 Qwen3Attention.forward:
#   q = q_norm(q_proj(x).view(heads, head_dim))   <- norm AFTER proj+reshape
#   k = k_norm(k_proj(x).view(kv_heads, head_dim))
#   v =        v_proj(x).view(kv_heads, head_dim) <- V is never normed
#   q, k = rope(q, k)                             <- RoPE AFTER qk-norm
#   attn -> [seq, heads*head_dim] -> o_proj
#
# Non-square projections are the norm here, not the exception: q_out (heads *
# head_dim) != hidden. Nothing below assumes otherwise.
# ===----------------------------------------------------------------------=== #

from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import sqrt
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from nn.gather_scatter import gather
from nn.normalization import rms_norm_gpu

from hephaestus.kernels import (
    F32,
    apply_rope_inplace,
    attention,
    cache_write,
    linear,
    residual_add,
    silu_mul,
    BF16,
    MAX_KEYS,
)
from hephaestus.model import Qwen3Weights

comptime EPS = Float32(1e-6)


def _rms_norm(
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    gamma: TileTensor[BF16, ...],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """RMSNorm over the last axis. multiply_before_cast=False and
    weight_offset=0 reproduce HF Qwen3RMSNorm bit-exactly (exp4).

    Takes raw pointers so the QK-norm case (in == out) doesn't trip the
    exclusivity checker. In-place is safe and is what MAX's own rms_norm test
    does (identity_output_fn writes back into the input buffer).
    """
    var in_buf = TileTensor(in_ptr, row_major(Coord(Index(rows, cols))))
    var out_buf = TileTensor(out_ptr, row_major(Coord(Index(rows, cols))))

    @always_inline
    @__copy_capture(in_buf)
    @parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[BF16, width]:
        return in_buf.raw_load[width=width](in_buf.layout(coords))

    @always_inline
    @__copy_capture(out_buf)
    @parameter
    def output_fn[
        width: SIMDSize, alignment: Int
    ](coords: Coord, val: SIMD[BF16, width]) -> None:
        out_buf.raw_store[width=width, alignment=alignment](
            out_buf.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, output_fn, multiply_before_cast=False](
        Coord(rows, cols), gamma, EPS, Scalar[BF16](0), ctx
    )


struct Activations[
    hidden: Int, q_out: Int, kv_out: Int, inter: Int, vocab: Int
](Movable):
    """Scratch buffers, allocated once. Sized for MAX_SEQ tokens at a time."""

    var x: DeviceBuffer[BF16]  # [seq, hidden] residual stream
    var xn: DeviceBuffer[BF16]  # [seq, hidden] normed
    var q: DeviceBuffer[BF16]  # [seq, q_out]
    var k: DeviceBuffer[BF16]  # [seq, kv_out]
    var v: DeviceBuffer[BF16]  # [seq, kv_out]
    var attn_out: DeviceBuffer[BF16]  # [seq, q_out]
    var proj: DeviceBuffer[BF16]  # [seq, hidden] o_proj / down_proj output
    var gate: DeviceBuffer[BF16]  # [seq, inter]
    var up: DeviceBuffer[BF16]  # [seq, inter]
    var act: DeviceBuffer[BF16]  # [seq, inter] silu(gate)*up
    var logits: DeviceBuffer[F32]  # [seq, vocab] -- fp32, never bf16
    var max_seq: Int

    def __init__(out self, ctx: DeviceContext, max_seq: Int) raises:
        self.x = ctx.enqueue_create_buffer[BF16](max_seq * Self.hidden)
        self.xn = ctx.enqueue_create_buffer[BF16](max_seq * Self.hidden)
        self.q = ctx.enqueue_create_buffer[BF16](max_seq * Self.q_out)
        self.k = ctx.enqueue_create_buffer[BF16](max_seq * Self.kv_out)
        self.v = ctx.enqueue_create_buffer[BF16](max_seq * Self.kv_out)
        self.attn_out = ctx.enqueue_create_buffer[BF16](max_seq * Self.q_out)
        self.proj = ctx.enqueue_create_buffer[BF16](max_seq * Self.hidden)
        self.gate = ctx.enqueue_create_buffer[BF16](max_seq * Self.inter)
        self.up = ctx.enqueue_create_buffer[BF16](max_seq * Self.inter)
        self.act = ctx.enqueue_create_buffer[BF16](max_seq * Self.inter)
        self.logits = ctx.enqueue_create_buffer[F32](max_seq * Self.vocab)
        self.max_seq = max_seq


struct KVCache[n_layers: Int, kv_out: Int](Movable):
    """Contiguous per-layer K/V. Layout per layer: [MAX_KEYS, kv_out]."""

    var k: DeviceBuffer[BF16]
    var v: DeviceBuffer[BF16]
    var length: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.k = ctx.enqueue_create_buffer[BF16](
            Self.n_layers * MAX_KEYS * Self.kv_out
        )
        self.v = ctx.enqueue_create_buffer[BF16](
            Self.n_layers * MAX_KEYS * Self.kv_out
        )
        self.length = 0


def forward[
    vocab: Int,
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
    n_layers: Int,
    n_heads: Int,
    n_kv_heads: Int,
    theta: Float64,
](
    weights: Qwen3Weights[_, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers],
    mut acts: Activations[hidden, q_out, kv_out, inter, vocab],
    mut cache: KVCache[n_layers, kv_out],
    token_ids: DeviceBuffer[DType.int32],
    seq: Int,
    ctx: DeviceContext,
) raises:
    """Runs `seq` tokens through the model, appending to the KV cache.

    Logits for all `seq` positions land in acts.logits; the caller reads the
    last row. `cache.length` is the number of tokens already cached (0 for a
    fresh prefill), and is advanced by `seq` on return.
    """
    comptime group = n_heads // n_kv_heads
    var past = cache.length

    # --- embedding lookup: gather rows of embed_tokens ----------------------
    var x = TileTensor(acts.x, row_major(Coord(Index(seq, hidden))))
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(token_ids, row_major(Coord(Index(seq)))),
        context=ctx,
    )

    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, hidden))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, q_out))))
    var k = TileTensor(acts.k, row_major(Coord(Index(seq, kv_out))))
    var v = TileTensor(acts.v, row_major(Coord(Index(seq, kv_out))))
    var attn_out = TileTensor(acts.attn_out, row_major(Coord(Index(seq, q_out))))
    var proj = TileTensor(acts.proj, row_major(Coord(Index(seq, hidden))))
    var gate = TileTensor(acts.gate, row_major(Coord(Index(seq, inter))))
    var up = TileTensor(acts.up, row_major(Coord(Index(seq, inter))))
    var act = TileTensor(acts.act, row_major(Coord(Index(seq, inter))))

    for i in range(n_layers):
        ref layer = weights.layers[i]

        # --- attention block ------------------------------------------------
        _rms_norm(acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), layer.attn_norm, seq, hidden, ctx)
        linear(q, xn, layer.q_proj, seq, q_out, hidden, ctx)
        linear(k, xn, layer.k_proj, seq, kv_out, hidden, ctx)
        linear(v, xn, layer.v_proj, seq, kv_out, hidden, ctx)

        # QK norm: per-head RMSNorm over head_dim. Rows = seq*heads, and the
        # kernel derives cols from gamma.dim[0] = head_dim. V is NOT normed.
        _rms_norm(acts.q.unsafe_ptr(), acts.q.unsafe_ptr(), layer.q_norm, seq * n_heads, head_dim, ctx)
        _rms_norm(acts.k.unsafe_ptr(), acts.k.unsafe_ptr(), layer.k_norm, seq * n_kv_heads, head_dim, ctx)

        # RoPE after QK-norm (HF order). Split-half convention.
        apply_rope_inplace[head_dim, theta](
            acts.q.unsafe_ptr(), n_heads, seq, past, ctx
        )
        apply_rope_inplace[head_dim, theta](
            acts.k.unsafe_ptr(), n_kv_heads, seq, past, ctx
        )

        # Append K/V for these tokens to this layer's cache slice.
        var layer_off = i * MAX_KEYS * kv_out
        var k_cache = cache.k.unsafe_ptr() + layer_off
        var v_cache = cache.v.unsafe_ptr() + layer_off
        cache_write(k_cache, acts.k.unsafe_ptr(), kv_out, seq, past, ctx)
        cache_write(v_cache, acts.v.unsafe_ptr(), kv_out, seq, past, ctx)

        attention[head_dim, group](
            acts.attn_out.unsafe_ptr(),
            acts.q.unsafe_ptr(),
            k_cache,
            v_cache,
            n_heads,
            n_kv_heads,
            seq,
            past,
            ctx,
        )

        linear(proj, attn_out, layer.o_proj, seq, hidden, q_out, ctx)
        residual_add(
            acts.x.unsafe_ptr(), acts.proj.unsafe_ptr(), seq * hidden, ctx
        )

        # --- FFN block ------------------------------------------------------
        _rms_norm(acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), layer.ffn_norm, seq, hidden, ctx)
        linear(gate, xn, layer.gate_proj, seq, inter, hidden, ctx)
        linear(up, xn, layer.up_proj, seq, inter, hidden, ctx)
        silu_mul(
            acts.act.unsafe_ptr(),
            acts.gate.unsafe_ptr(),
            acts.up.unsafe_ptr(),
            seq * inter,
            ctx,
        )
        linear(proj, act, layer.down_proj, seq, hidden, inter, ctx)
        residual_add(
            acts.x.unsafe_ptr(), acts.proj.unsafe_ptr(), seq * hidden, ctx
        )

    # --- final norm + LM head (tied: embed_tokens IS the head) --------------
    _rms_norm(acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), weights.output_norm, seq, hidden, ctx)
    var logits = TileTensor(acts.logits, row_major(Coord(Index(seq, vocab))))
    linear(logits, xn, weights.embed_tokens, seq, vocab, hidden, ctx)

    cache.length = past + seq
