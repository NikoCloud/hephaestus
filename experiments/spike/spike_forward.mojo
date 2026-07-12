# ===----------------------------------------------------------------------=== #
# DIAGNOSTIC ONLY -- instrumented copy of src/hephaestus/forward.mojo.
#
# Identical arithmetic, op for op. The ONLY additions are dump_slot() calls that
# snapshot one chosen row of the residual stream at 4 cut points per layer, so a
# layerwise bisect can find where the row-67 hidden state diverges from HF.
#
# It imports the REAL kernels from hephaestus.kernels -- nothing is reimplemented,
# so a bug in the production kernels reproduces here exactly.
#
# Dump slot layout (dump_row >= 0; pass -1 to disable):
#   slot 0                 embeddings (after gather, before layer 0)
#   slot 1 + 4*i + 0       o_proj output   (attention contribution, pre-residual)
#   slot 1 + 4*i + 1       x after attention residual add
#   slot 1 + 4*i + 2       down_proj output (FFN contribution, pre-residual)
#   slot 1 + 4*i + 3       x after FFN residual add  (= layer i output)
#   slot 1 + 4*n_layers    final normed hidden (input to the LM head)
# ===----------------------------------------------------------------------=== #

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from nn.gather_scatter import gather

from hephaestus.forward import Activations, KVCache, _rms_norm
from hephaestus.kernels import (
    BF16,
    MAX_KEYS,
    apply_rope_inplace,
    cache_write,
    linear,
    residual_add,
    silu_mul,
)
from hephaestus.model import Qwen3Weights
from spike_kernels import spike_attention


def n_slots(n_layers: Int) -> Int:
    return 1 + 4 * n_layers + 1


def dump_kernel(
    dst: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    src: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    row: Int,
    hidden: Int,
    slot: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= hidden:
        return
    dst[slot * hidden + i] = src[row * hidden + i]


def dump_slot(
    mut dst: DeviceBuffer[BF16],
    src: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    row: Int,
    hidden: Int,
    slot: Int,
    ctx: DeviceContext,
) raises:
    """Snapshot src[row, :] into dst[slot, :]. No-op if row < 0."""
    if row < 0:
        return
    comptime TPB = 256
    ctx.enqueue_function[dump_kernel](
        dst.unsafe_ptr(),
        src,
        row,
        hidden,
        slot,
        grid_dim=(ceildiv(hidden, TPB),),
        block_dim=(TPB,),
    )


def forward_dump[
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
    prob_bf16: Bool = True,   # production default (kernels.mojo:262)
    score_bf16: Bool = False,  # production default (kernels.mojo:242)
](
    weights: Qwen3Weights[
        _, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers
    ],
    mut acts: Activations[hidden, q_out, kv_out, inter, vocab],
    mut cache: KVCache[n_layers, kv_out],
    token_ids: DeviceBuffer[DType.int32],
    seq: Int,
    mut dump: DeviceBuffer[BF16],
    dump_row: Int,
    ctx: DeviceContext,
) raises:
    """Byte-for-byte the production forward pass, plus residual-stream snapshots.
    dump_row is a row index into THIS call's `seq` rows (-1 = no dump)."""
    comptime group = n_heads // n_kv_heads
    var past = cache.length

    var x = TileTensor(acts.x, row_major(Coord(Index(seq, hidden))))
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(token_ids, row_major(Coord(Index(seq)))),
        context=ctx,
    )
    dump_slot(dump, acts.x.unsafe_ptr(), dump_row, hidden, 0, ctx)

    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, hidden))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, q_out))))
    var k = TileTensor(acts.k, row_major(Coord(Index(seq, kv_out))))
    var v = TileTensor(acts.v, row_major(Coord(Index(seq, kv_out))))
    var attn_out = TileTensor(
        acts.attn_out, row_major(Coord(Index(seq, q_out)))
    )
    var proj = TileTensor(acts.proj, row_major(Coord(Index(seq, hidden))))
    var gate = TileTensor(acts.gate, row_major(Coord(Index(seq, inter))))
    var up = TileTensor(acts.up, row_major(Coord(Index(seq, inter))))
    var act = TileTensor(acts.act, row_major(Coord(Index(seq, inter))))

    for i in range(n_layers):
        ref layer = weights.layers[i]

        _rms_norm(
            acts.xn.unsafe_ptr(),
            acts.x.unsafe_ptr(),
            layer.attn_norm,
            seq,
            hidden,
            ctx,
        )
        linear(q, xn, layer.q_proj, seq, q_out, hidden, ctx)
        linear(k, xn, layer.k_proj, seq, kv_out, hidden, ctx)
        linear(v, xn, layer.v_proj, seq, kv_out, hidden, ctx)

        _rms_norm(
            acts.q.unsafe_ptr(),
            acts.q.unsafe_ptr(),
            layer.q_norm,
            seq * n_heads,
            head_dim,
            ctx,
        )
        _rms_norm(
            acts.k.unsafe_ptr(),
            acts.k.unsafe_ptr(),
            layer.k_norm,
            seq * n_kv_heads,
            head_dim,
            ctx,
        )

        apply_rope_inplace[head_dim, theta](
            acts.q.unsafe_ptr(), n_heads, seq, past, ctx
        )
        apply_rope_inplace[head_dim, theta](
            acts.k.unsafe_ptr(), n_kv_heads, seq, past, ctx
        )

        var layer_off = i * MAX_KEYS * kv_out
        var k_cache = cache.k.unsafe_ptr() + layer_off
        var v_cache = cache.v.unsafe_ptr() + layer_off
        cache_write(k_cache, acts.k.unsafe_ptr(), kv_out, seq, past, ctx)
        cache_write(v_cache, acts.v.unsafe_ptr(), kv_out, seq, past, ctx)

        spike_attention[head_dim, group, prob_bf16, score_bf16](
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
        dump_slot(
            dump, acts.proj.unsafe_ptr(), dump_row, hidden, 1 + 4 * i + 0, ctx
        )
        residual_add(
            acts.x.unsafe_ptr(), acts.proj.unsafe_ptr(), seq * hidden, ctx
        )
        dump_slot(
            dump, acts.x.unsafe_ptr(), dump_row, hidden, 1 + 4 * i + 1, ctx
        )

        _rms_norm(
            acts.xn.unsafe_ptr(),
            acts.x.unsafe_ptr(),
            layer.ffn_norm,
            seq,
            hidden,
            ctx,
        )
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
        dump_slot(
            dump, acts.proj.unsafe_ptr(), dump_row, hidden, 1 + 4 * i + 2, ctx
        )
        residual_add(
            acts.x.unsafe_ptr(), acts.proj.unsafe_ptr(), seq * hidden, ctx
        )
        dump_slot(
            dump, acts.x.unsafe_ptr(), dump_row, hidden, 1 + 4 * i + 3, ctx
        )

    _rms_norm(
        acts.xn.unsafe_ptr(),
        acts.x.unsafe_ptr(),
        weights.output_norm,
        seq,
        hidden,
        ctx,
    )
    dump_slot(
        dump, acts.xn.unsafe_ptr(), dump_row, hidden, 1 + 4 * n_layers, ctx
    )
    var logits = TileTensor(acts.logits, row_major(Coord(Index(seq, vocab))))
    linear(logits, xn, weights.embed_tokens, seq, vocab, hidden, ctx)

    cache.length = past + seq
