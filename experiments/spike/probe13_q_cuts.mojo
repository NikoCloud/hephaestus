# PROBE 13 -- layer-0 Q cut-point dumps + replace-and-continue.
#
# Exact-prefix (prompt1 + oracle[:67] = 77 tokens). Target row = 76.
#
# Modes (argv[1]):
#   dump              Run production path; write Hephaestus Q at three cuts
#                     as float32: ${out}_q_proj.f32, _q_norm.f32, _q_rope.f32
#                     plus final normed hidden + logits for the target row.
#   control           Same as dump but only final hidden/logits (no cut dumps).
#   inject_q_proj     After q_proj, overwrite acts.q with HF float32 dump
#                     (argv[6] path, cast to bf16), then continue Hephaestus
#                     q_norm + RoPE + layers 0..35.
#   inject_q_norm     Overwrite after q_norm.
#   inject_q_rope     Overwrite after RoPE.
#
# Inject files are float32 [seq * Q_PROJ_OUT], layout [seq, heads, head_dim]
# row-major -- identical to acts.q.
#
# Usage:
#   /tmp/spike_q_cuts dump|control|inject_* \
#       p1_prompt.txt p1_oracle.txt staged/qwen3-4b /tmp/spike13 [hf_q.f32]

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from nn.gather_scatter import gather

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
from hephaestus.forward import Activations, KVCache, _rms_norm
from hephaestus.kernels import (
    BF16,
    MAX_KEYS,
    apply_rope_inplace,
    attention,
    cache_write,
    linear,
    residual_add,
    silu_mul,
    F32,
)
from hephaestus.loader import build_weights, load_arena, verify_manifest

comptime STEP = 67
comptime TARGET_TOK = 96874
comptime GROUP = NUM_HEADS // NUM_KV_HEADS


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def write_f32(path: String, buf: List[Float32]) raises:
    var f = open(path, "w")
    f.write_bytes(
        Span[Byte, origin_of(buf)](
            ptr=buf.unsafe_ptr().bitcast[Byte](), length=len(buf) * 4
        )
    )
    f.close()


def read_f32(path: String, n: Int) raises -> List[Float32]:
    """Read exactly n float32 values from a little-endian binary file."""
    var host = List[Float32](capacity=n)
    for _ in range(n):
        host.append(Float32(0))
    var f = open(path, "r")
    var byte_ptr = host.unsafe_ptr().bitcast[Scalar[DType.uint8]]()
    var total = n * 4
    var done = 0
    while done < total:
        var got = f.read(
            Span[Scalar[DType.uint8]](ptr=byte_ptr + done, length=total - done)
        )
        if got <= 0:
            raise Error(
                "truncated f32 file "
                + path
                + ": expected "
                + String(total)
                + " got "
                + String(done)
            )
        done += got
    f.close()
    return host^


def copy_q_from_f32(
    mut q_buf: DeviceBuffer[BF16],
    host_f32: List[Float32],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Cast host f32 -> bf16 into device acts.q."""
    with q_buf.map_to_host() as h:
        for i in range(n):
            h[i] = host_f32[i].cast[BF16]()
    ctx.synchronize()


def dump_q_f32(
    path: String, q_buf: DeviceBuffer[BF16], n: Int, ctx: DeviceContext
) raises:
    var out = List[Float32]()
    with q_buf.map_to_host() as h:
        for i in range(n):
            out.append(h[i].cast[DType.float32]())
    write_f32(path, out)
    _ = ctx


def main() raises:
    var mode = String(argv()[1])
    var prompt = read_ids(String(argv()[2]))
    var oracle = read_ids(String(argv()[3]))
    var wprefix = String(argv()[4])
    var out = String(argv()[5])

    var inject_path = String("")
    if (
        mode == "inject_q_proj"
        or mode == "inject_q_norm"
        or mode == "inject_q_rope"
    ):
        inject_path = String(argv()[6])

    var ids = List[Int32]()
    for i in range(len(prompt)):
        ids.append(prompt[i])
    for i in range(STEP):
        ids.append(oracle[i])
    var seq = len(ids)
    var target = seq - 1
    var q_elems = seq * Q_PROJ_OUT
    print("mode =", mode, " seq =", seq, " target =", target)

    var ctx = DeviceContext()
    var arena = load_arena(ctx, wprefix)
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
    ](ctx, seq + 1)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    cache.length = 0
    var past = 0

    var dev = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    # Optional inject payload (loaded once).
    var inj = List[Float32]()
    if inject_path.byte_length() > 0:
        inj = read_f32(inject_path, q_elems)
        print("loaded inject", inject_path, "n =", q_elems)

    # --- embedding ---
    var x = TileTensor(acts.x, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(dev, row_major(Coord(Index(seq)))),
        context=ctx,
    )

    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, Q_PROJ_OUT))))
    var k = TileTensor(acts.k, row_major(Coord(Index(seq, K_PROJ_OUT))))
    var v = TileTensor(acts.v, row_major(Coord(Index(seq, K_PROJ_OUT))))
    var attn_out = TileTensor(
        acts.attn_out, row_major(Coord(Index(seq, Q_PROJ_OUT)))
    )
    var proj = TileTensor(acts.proj, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    var gate = TileTensor(acts.gate, row_major(Coord(Index(seq, INTERMEDIATE_SIZE))))
    var up = TileTensor(acts.up, row_major(Coord(Index(seq, INTERMEDIATE_SIZE))))
    var act = TileTensor(acts.act, row_major(Coord(Index(seq, INTERMEDIATE_SIZE))))

    var do_dump = mode == "dump"

    for i in range(NUM_LAYERS):
        ref layer = weights.layers[i]

        _rms_norm(
            acts.xn.unsafe_ptr(),
            acts.x.unsafe_ptr(),
            layer.attn_norm,
            seq,
            HIDDEN_SIZE,
            ctx,
        )
        linear(q, xn, layer.q_proj, seq, Q_PROJ_OUT, HIDDEN_SIZE, ctx)
        linear(k, xn, layer.k_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        linear(v, xn, layer.v_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)

        if i == 0:
            if mode == "inject_q_proj":
                copy_q_from_f32(acts.q, inj, q_elems, ctx)
            if do_dump:
                dump_q_f32(out + "_q_proj.f32", acts.q, q_elems, ctx)

        _rms_norm(
            acts.q.unsafe_ptr(),
            acts.q.unsafe_ptr(),
            layer.q_norm,
            seq * NUM_HEADS,
            HEAD_DIM,
            ctx,
        )
        _rms_norm(
            acts.k.unsafe_ptr(),
            acts.k.unsafe_ptr(),
            layer.k_norm,
            seq * NUM_KV_HEADS,
            HEAD_DIM,
            ctx,
        )

        if i == 0:
            if mode == "inject_q_norm":
                copy_q_from_f32(acts.q, inj, q_elems, ctx)
            if do_dump:
                dump_q_f32(out + "_q_norm.f32", acts.q, q_elems, ctx)

        apply_rope_inplace[HEAD_DIM, ROPE_THETA](
            acts.q.unsafe_ptr(), NUM_HEADS, seq, past, ctx
        )
        apply_rope_inplace[HEAD_DIM, ROPE_THETA](
            acts.k.unsafe_ptr(), NUM_KV_HEADS, seq, past, ctx
        )

        if i == 0:
            if mode == "inject_q_rope":
                copy_q_from_f32(acts.q, inj, q_elems, ctx)
            if do_dump:
                dump_q_f32(out + "_q_rope.f32", acts.q, q_elems, ctx)

        var layer_off = i * MAX_KEYS * K_PROJ_OUT
        var k_cache = cache.k.unsafe_ptr() + layer_off
        var v_cache = cache.v.unsafe_ptr() + layer_off
        cache_write(k_cache, acts.k.unsafe_ptr(), K_PROJ_OUT, seq, past, ctx)
        cache_write(v_cache, acts.v.unsafe_ptr(), K_PROJ_OUT, seq, past, ctx)

        attention[HEAD_DIM, GROUP](
            acts.attn_out.unsafe_ptr(),
            acts.q.unsafe_ptr(),
            k_cache,
            v_cache,
            NUM_HEADS,
            NUM_KV_HEADS,
            seq,
            past,
            ctx,
        )

        linear(proj, attn_out, layer.o_proj, seq, HIDDEN_SIZE, Q_PROJ_OUT, ctx)
        residual_add(
            acts.x.unsafe_ptr(), acts.proj.unsafe_ptr(), seq * HIDDEN_SIZE, ctx
        )

        _rms_norm(
            acts.xn.unsafe_ptr(),
            acts.x.unsafe_ptr(),
            layer.ffn_norm,
            seq,
            HIDDEN_SIZE,
            ctx,
        )
        linear(gate, xn, layer.gate_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx)
        linear(up, xn, layer.up_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx)
        silu_mul(
            acts.act.unsafe_ptr(),
            acts.gate.unsafe_ptr(),
            acts.up.unsafe_ptr(),
            seq * INTERMEDIATE_SIZE,
            ctx,
        )
        linear(proj, act, layer.down_proj, seq, HIDDEN_SIZE, INTERMEDIATE_SIZE, ctx)
        residual_add(
            acts.x.unsafe_ptr(), acts.proj.unsafe_ptr(), seq * HIDDEN_SIZE, ctx
        )

    # final norm + LM head
    _rms_norm(
        acts.xn.unsafe_ptr(),
        acts.x.unsafe_ptr(),
        weights.output_norm,
        seq,
        HIDDEN_SIZE,
        ctx,
    )
    var logits = TileTensor(acts.logits, row_major(Coord(Index(seq, VOCAB_SIZE))))
    linear(logits, xn, weights.embed_tokens, seq, VOCAB_SIZE, HIDDEN_SIZE, ctx)
    ctx.synchronize()

    # dump target-row final hidden (normed) + logits
    var hid = List[Float32]()
    with acts.xn.map_to_host() as h:
        var base = target * HIDDEN_SIZE
        for j in range(HIDDEN_SIZE):
            hid.append(h[base + j].cast[DType.float32]())
    write_f32(out + "_hidden.f32", hid)

    var best = 0
    var best_val = Float32(-3.4e38)
    var tgt = Float32(0)
    var lg = List[Float32]()
    with acts.logits.map_to_host() as h:
        var base = target * VOCAB_SIZE
        for j in range(VOCAB_SIZE):
            var val = h[base + j]
            lg.append(val)
            var r = val.cast[DType.bfloat16]().cast[DType.float32]()
            if r > best_val:
                best_val = r
                best = j
        tgt = h[base + TARGET_TOK]
    write_f32(out + "_logits.f32", lg)

    print(
        "argmax =",
        best,
        " logit[96874] =",
        tgt,
        " ||h|| =",
        # crude L2
        String("see hidden file"),
    )
    # compute L2 for the log line
    var nrm = Float64(0)
    for j in range(HIDDEN_SIZE):
        var vj = Float64(hid[j])
        nrm += vj * vj
    print("||h_final|| =", nrm ** 0.5)
    print("wrote", out, "_{hidden,logits}.f32 (+ q cuts if dump)")
