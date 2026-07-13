# Layer-by-layer activation dump harness (Phase 1b debug tool).
#
# Diagnostic copy of src/hephaestus/forward.mojo — does NOT modify the engine.
# After each cut point, copies the live device tensor to host and writes a raw
# payload + a manifest line. run.sh packs those into .npy via diff_layers.py.
#
# Cut points match the forward op order (per layer i, then final):
#   attn_norm, q_proj, k_proj, v_proj, q_norm, k_norm, rope_q, rope_k,
#   attention, o_proj_residual, ffn_norm, gate_proj, up_proj, silu_mul,
#   down_proj_residual; then final output_norm, lm_head.
#
# Naming: layer{L}_step{S}_{name}.raw  (S=0 prefill)
#         final_step{S}_{name}.raw
#
# Usage (from repo root):
#   pixi run mojo run -I $KERNELS -I src \
#       experiments/exp5_layer_diff/dump_activations.mojo \
#       tiny <dump_dir> [prompt_idx]
#   pixi run mojo run -I $KERNELS -I src \
#       experiments/exp5_layer_diff/dump_activations.mojo \
#       4b <dump_dir> <ids.txt>
#
# Tiny: dump every layer. 4B: dump layers 0,1, mid-1, mid, last-2, last-1 only.

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from nn.gather_scatter import gather
from nn.normalization import rms_norm_gpu

from hephaestus.constants import (
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    K_PROJ_OUT,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_PROJ_OUT,
    ROPE_THETA,
    VOCAB_SIZE,
)
from hephaestus.forward import Activations, KVCache
from hephaestus.kernels import (
    BF16,
    F32,
    MAX_KEYS,
    apply_rope_qk_inplace,
    attention,
    linear,
    linear_add_residual,
    silu_mul,
)
from hephaestus.loader import build_weights, load_arena, verify_manifest
from hephaestus.model import Qwen3Weights

comptime EPS = Float32(1e-6)

# --- tiny dims (must match fixtures/tiny_random) -----------------------------
comptime TINY_VOCAB = 256
comptime TINY_HIDDEN = 128
comptime TINY_N_HEADS = 4
comptime TINY_N_KV = 2
comptime TINY_HEAD_DIM = 32
comptime TINY_Q_OUT = TINY_N_HEADS * TINY_HEAD_DIM
comptime TINY_KV_OUT = TINY_N_KV * TINY_HEAD_DIM
comptime TINY_INTER = 256
comptime TINY_LAYERS = 2
comptime TINY_THETA = 10000.0


# ===----------------------------------------------------------------------=== #
# Dump helpers: raw payload + manifest line (packed to .npy by diff_layers.py)
# ===----------------------------------------------------------------------=== #


def should_dump_layer(i: Int, n_layers: Int, dump_all: Bool) -> Bool:
    """Tiny: all layers. 4B: start (0,1), middle pair, end pair."""
    if dump_all:
        return True
    if i == 0 or i == 1:
        return True
    if i == n_layers - 1 or i == n_layers - 2:
        return True
    var mid = n_layers // 2
    if i == mid or i == mid - 1:
        return True
    return False


def dump_bf16_device(
    dump_dir: String,
    stem: String,
    buf: DeviceBuffer[BF16],
    offset: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
    mut catalog: List[String],
) raises:
    """Copy [offset, offset+rows*cols) from a BF16 device buffer → stem.raw."""
    var n = rows * cols
    ctx.synchronize()
    var host = List[Scalar[BF16]]()
    with buf.map_to_host() as h:
        for i in range(n):
            host.append(h[offset + i])
    var path = dump_dir + "/" + stem + ".raw"
    var f = open(path, "w")
    f.write_bytes(
        Span[Byte, origin_of(host)](
            ptr=host.unsafe_ptr().bitcast[Byte](), length=n * 2
        )
    )
    f.close()
    catalog.append(
        stem
        + "\tbf16\t"
        + String(rows)
        + "\t"
        + String(cols)
        + "\n"
    )
    print("  dumped", stem, "shape", rows, "x", cols, "bf16")


def dump_f32_device(
    dump_dir: String,
    stem: String,
    buf: DeviceBuffer[F32],
    offset: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
    mut catalog: List[String],
) raises:
    var n = rows * cols
    ctx.synchronize()
    var host = List[Float32]()
    with buf.map_to_host() as h:
        for i in range(n):
            host.append(h[offset + i])
    var path = dump_dir + "/" + stem + ".raw"
    var f = open(path, "w")
    f.write_bytes(
        Span[Byte, origin_of(host)](
            ptr=host.unsafe_ptr().bitcast[Byte](), length=n * 4
        )
    )
    f.close()
    catalog.append(
        stem + "\tf32\t" + String(rows) + "\t" + String(cols) + "\n"
    )
    print("  dumped", stem, "shape", rows, "x", cols, "f32")


def write_manifest(dump_dir: String, catalog: List[String]) raises:
    var path = dump_dir + "/manifest.tsv"
    var f = open(path, "w")
    # header comment line (ignored by pack if it starts with #)
    var hdr = String("# stem\tdtype\tdim0\tdim1\n")
    f.write(hdr)
    for i in range(len(catalog)):
        f.write(catalog[i])
    f.close()
    print("wrote", path, "entries", len(catalog))


# ===----------------------------------------------------------------------=== #
# RMSNorm (copied from forward.mojo — private there, not imported)
# ===----------------------------------------------------------------------=== #


def _rms_norm(
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    gamma: TileTensor[BF16, ...],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
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


# ===----------------------------------------------------------------------=== #
# Instrumented forward (same ops as hephaestus.forward.forward)
# ===----------------------------------------------------------------------=== #


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
](
    weights: Qwen3Weights[
        _, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers
    ],
    mut acts: Activations[hidden, q_out, kv_out, inter, vocab],
    mut cache: KVCache[n_layers, kv_out],
    token_ids: DeviceBuffer[DType.int32],
    seq: Int,
    ctx: DeviceContext,
    dump_dir: String,
    dump_all_layers: Bool,
    step: Int,
    use_wmma: Bool,
) raises:
    comptime group = n_heads // n_kv_heads
    var past = cache.length
    var catalog = List[String]()

    var x = TileTensor(acts.x, row_major(Coord(Index(seq, hidden))))
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(token_ids, row_major(Coord(Index(seq)))),
        context=ctx,
    )

    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, hidden))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, q_out))))
    var attn_out = TileTensor(
        acts.attn_out, row_major(Coord(Index(seq, q_out)))
    )
    var gate = TileTensor(acts.gate, row_major(Coord(Index(seq, inter))))
    var up = TileTensor(acts.up, row_major(Coord(Index(seq, inter))))
    var act = TileTensor(acts.act, row_major(Coord(Index(seq, inter))))

    for i in range(n_layers):
        ref layer = weights.layers[i]
        var do_dump = should_dump_layer(i, n_layers, dump_all_layers)
        var prefix = "layer" + String(i) + "_step" + String(step) + "_"

        # --- attention block ------------------------------------------------
        _rms_norm(
            acts.xn.unsafe_ptr().as_unsafe_any_origin(),
            acts.x.unsafe_ptr().as_unsafe_any_origin(),
            layer.attn_norm,
            seq,
            hidden,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "attn_norm",
                acts.xn,
                0,
                seq,
                hidden,
                ctx,
                catalog,
            )

        var layer_off = i * MAX_KEYS * kv_out
        var k_cache = cache.k.unsafe_ptr() + layer_off
        var v_cache = cache.v.unsafe_ptr() + layer_off
        var k_new_ptr = k_cache + past * kv_out
        var v_new_ptr = v_cache + past * kv_out
        var k_dst = TileTensor(
            ptr=k_new_ptr, layout=row_major(Coord(Index(seq, kv_out)))
        )
        var v_dst = TileTensor(
            ptr=v_new_ptr, layout=row_major(Coord(Index(seq, kv_out)))
        )
        # Host-side offset of the new K/V slice inside the full cache buffer.
        var k_host_off = layer_off + past * kv_out

        linear(q, xn, layer.q_proj, seq, q_out, hidden, ctx, use_wmma=use_wmma)
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "q_proj",
                acts.q,
                0,
                seq,
                q_out,
                ctx,
                catalog,
            )

        linear(
            k_dst, xn, layer.k_proj, seq, kv_out, hidden, ctx, use_wmma=use_wmma
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "k_proj",
                cache.k,
                k_host_off,
                seq,
                kv_out,
                ctx,
                catalog,
            )

        linear(
            v_dst, xn, layer.v_proj, seq, kv_out, hidden, ctx, use_wmma=use_wmma
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "v_proj",
                cache.v,
                k_host_off,
                seq,
                kv_out,
                ctx,
                catalog,
            )

        # QK norm: rows = seq*heads, cols = head_dim
        _rms_norm(
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            layer.q_norm,
            seq * n_heads,
            head_dim,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "q_norm",
                acts.q,
                0,
                seq * n_heads,
                head_dim,
                ctx,
                catalog,
            )

        _rms_norm(
            k_new_ptr.as_unsafe_any_origin(),
            k_new_ptr.as_unsafe_any_origin(),
            layer.k_norm,
            seq * n_kv_heads,
            head_dim,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "k_norm",
                cache.k,
                k_host_off,
                seq * n_kv_heads,
                head_dim,
                ctx,
                catalog,
            )

        apply_rope_qk_inplace[head_dim, theta](
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_new_ptr.as_unsafe_any_origin(),
            n_heads,
            n_kv_heads,
            seq,
            past,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "rope_q",
                acts.q,
                0,
                seq,
                q_out,
                ctx,
                catalog,
            )
            dump_bf16_device(
                dump_dir,
                prefix + "rope_k",
                cache.k,
                k_host_off,
                seq,
                kv_out,
                ctx,
                catalog,
            )

        attention[head_dim, group](
            acts.attn_out.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_cache.as_unsafe_any_origin(),
            v_cache.as_unsafe_any_origin(),
            n_heads,
            n_kv_heads,
            seq,
            past,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "attention",
                acts.attn_out,
                0,
                seq,
                q_out,
                ctx,
                catalog,
            )

        linear_add_residual(x, attn_out, layer.o_proj, seq, hidden, q_out, ctx)
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "o_proj_residual",
                acts.x,
                0,
                seq,
                hidden,
                ctx,
                catalog,
            )

        # --- FFN block ------------------------------------------------------
        _rms_norm(
            acts.xn.unsafe_ptr().as_unsafe_any_origin(),
            acts.x.unsafe_ptr().as_unsafe_any_origin(),
            layer.ffn_norm,
            seq,
            hidden,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "ffn_norm",
                acts.xn,
                0,
                seq,
                hidden,
                ctx,
                catalog,
            )

        linear(
            gate, xn, layer.gate_proj, seq, inter, hidden, ctx, use_wmma=use_wmma
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "gate_proj",
                acts.gate,
                0,
                seq,
                inter,
                ctx,
                catalog,
            )

        linear(
            up, xn, layer.up_proj, seq, inter, hidden, ctx, use_wmma=use_wmma
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "up_proj",
                acts.up,
                0,
                seq,
                inter,
                ctx,
                catalog,
            )

        silu_mul(
            acts.act.unsafe_ptr().as_unsafe_any_origin(),
            acts.gate.unsafe_ptr().as_unsafe_any_origin(),
            acts.up.unsafe_ptr().as_unsafe_any_origin(),
            seq * inter,
            ctx,
        )
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "silu_mul",
                acts.act,
                0,
                seq,
                inter,
                ctx,
                catalog,
            )

        linear_add_residual(x, act, layer.down_proj, seq, hidden, inter, ctx)
        if do_dump:
            dump_bf16_device(
                dump_dir,
                prefix + "down_proj_residual",
                acts.x,
                0,
                seq,
                hidden,
                ctx,
                catalog,
            )

    # --- final norm + LM head -----------------------------------------------
    var fprefix = "final_step" + String(step) + "_"
    _rms_norm(
        acts.xn.unsafe_ptr().as_unsafe_any_origin(),
        acts.x.unsafe_ptr().as_unsafe_any_origin(),
        weights.output_norm,
        seq,
        hidden,
        ctx,
    )
    dump_bf16_device(
        dump_dir,
        fprefix + "output_norm",
        acts.xn,
        0,
        seq,
        hidden,
        ctx,
        catalog,
    )

    var logits = TileTensor(acts.logits, row_major(Coord(Index(seq, vocab))))
    linear(
        logits,
        xn,
        weights.embed_tokens,
        seq,
        vocab,
        hidden,
        ctx,
        use_wmma=use_wmma,
    )
    dump_f32_device(
        dump_dir,
        fprefix + "lm_head",
        acts.logits,
        0,
        seq,
        vocab,
        ctx,
        catalog,
    )

    cache.length = past + seq
    write_manifest(dump_dir, catalog)


# ===----------------------------------------------------------------------=== #
# Drivers: tiny / 4b
# ===----------------------------------------------------------------------=== #


def prompt_ids_tiny(idx: Int) raises -> List[Int32]:
    var ids = List[Int32]()
    if idx == 1:
        ids.append(0)
        ids.append(1)
        ids.append(2)
        ids.append(3)
    elif idx == 2:
        ids.append(10)
        ids.append(20)
        ids.append(30)
        ids.append(40)
        ids.append(50)
    else:
        ids.append(100)
        ids.append(200)
        ids.append(255)
        ids.append(5)
        ids.append(10)
    return ids^


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def run_tiny(dump_dir: String, prompt_idx: Int, use_wmma: Bool) raises:
    var ids = prompt_ids_tiny(prompt_idx)
    var seq = len(ids)
    var path_name = String("wmma") if use_wmma else String("naive")
    print(
        "dump tiny prompt",
        prompt_idx,
        "seq",
        seq,
        "path",
        path_name,
        "->",
        dump_dir,
        "(all layers)",
    )

    var ctx = DeviceContext()
    var arena = load_arena(ctx, "staged/tiny")
    verify_manifest[
        TINY_VOCAB,
        TINY_HIDDEN,
        TINY_Q_OUT,
        TINY_KV_OUT,
        TINY_HEAD_DIM,
        TINY_INTER,
        TINY_LAYERS,
    ](arena.entries, arena.index)
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=TINY_VOCAB,
        hidden=TINY_HIDDEN,
        q_out=TINY_Q_OUT,
        kv_out=TINY_KV_OUT,
        head_dim=TINY_HEAD_DIM,
        inter=TINY_INTER,
        n_layers=TINY_LAYERS,
    ](base_ptr, arena)

    var acts = Activations[
        TINY_HIDDEN, TINY_Q_OUT, TINY_KV_OUT, TINY_INTER, TINY_VOCAB
    ](ctx, 32)
    var cache = KVCache[TINY_LAYERS, TINY_KV_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    forward_dump[
        vocab=TINY_VOCAB,
        hidden=TINY_HIDDEN,
        q_out=TINY_Q_OUT,
        kv_out=TINY_KV_OUT,
        head_dim=TINY_HEAD_DIM,
        inter=TINY_INTER,
        n_layers=TINY_LAYERS,
        n_heads=TINY_N_HEADS,
        n_kv_heads=TINY_N_KV,
        theta=TINY_THETA,
    ](weights, acts, cache, dev_ids, seq, ctx, dump_dir, True, 0, use_wmma)
    ctx.synchronize()
    print("tiny dump complete")


def run_4b(dump_dir: String, ids_path: String, use_wmma: Bool) raises:
    var ids = read_ids(ids_path)
    var seq = len(ids)
    var path_name = String("wmma") if use_wmma else String("naive")
    print(
        "dump 4b seq",
        seq,
        "path",
        path_name,
        "->",
        dump_dir,
        "(selective layers 0,1,mid,end)",
    )

    var ctx = DeviceContext()
    var arena = load_arena(ctx, "staged/qwen3-4b")
    verify_manifest[
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ](arena.entries, arena.index)
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=VOCAB_SIZE,
        hidden=HIDDEN_SIZE,
        q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT,
        head_dim=HEAD_DIM,
        inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
    ](base_ptr, arena)

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, 32)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    forward_dump[
        vocab=VOCAB_SIZE,
        hidden=HIDDEN_SIZE,
        q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT,
        head_dim=HEAD_DIM,
        inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        theta=ROPE_THETA,
    ](weights, acts, cache, dev_ids, seq, ctx, dump_dir, False, 0, use_wmma)
    ctx.synchronize()
    print("4b dump complete")


def main() raises:
    if len(argv()) < 4:
        print(
            "usage: dump_activations tiny|4b <naive|wmma> <dump_dir>"
            " [prompt_idx|ids.txt]"
        )
        raise Error("bad args")

    var mode = String(argv()[1])
    var path = String(argv()[2])
    var dump_dir = String(argv()[3])
    var use_wmma: Bool
    if path == "wmma":
        use_wmma = True
    elif path == "naive":
        use_wmma = False
    else:
        raise Error("path must be naive or wmma")

    if mode == "tiny":
        var prompt_idx = 1
        if len(argv()) > 4:
            prompt_idx = Int(String(argv()[4]))
        run_tiny(dump_dir, prompt_idx, use_wmma)
    elif mode == "4b":
        if len(argv()) < 5:
            raise Error("4b mode needs <ids.txt>")
        run_4b(dump_dir, String(argv()[4]), use_wmma)
    else:
        raise Error("mode must be tiny or 4b")
