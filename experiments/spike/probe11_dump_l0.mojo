# PROBE 11 companion -- dump layer-0 post-RoPE Q/K/V and o_proj attn output
# for the exact-prefix sequence (prompt1 + oracle[:67]).
#
# Usage:
#   cd experiments/spike
#   PIXI_PROJECT_ROOT=../.. pixi run mojo run probe11_dump_l0.mojo \
#     ../../experiments/spike/out/p1_prompt.txt \
#     ../../experiments/spike/out/p1_oracle.txt \
#     ../../staged/qwen3-4b \
#     /tmp/spike_l0
#
# Writes: ${out}_q.f32 ${out}_k.f32 ${out}_v.f32 ${out}_attn.f32 (float32 host dumps)

from std.gpu.host import DeviceContext
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
)
from hephaestus.loader import build_weights, load_arena, verify_manifest

comptime STEP = 67


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def dump_f32(
    path: String,
    src: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    var buf = ctx.enqueue_create_buffer[BF16](n)
    # copy device -> host via temporary: src may already be device memory on acts
    # We accept a DeviceBuffer path instead in main; this helper is unused.
    _ = path
    _ = src
    _ = n
    _ = buf


def main() raises:
    var prompt = read_ids(String(argv()[1]))
    var oracle = read_ids(String(argv()[2]))
    var wprefix = String(argv()[3])
    var out = String(argv()[4])

    var ids = List[Int32]()
    for i in range(len(prompt)):
        ids.append(prompt[i])
    for i in range(STEP):
        ids.append(oracle[i])
    var seq = len(ids)
    var target = seq - 1
    print("seq =", seq, " target =", target)

    var ctx = DeviceContext()
    var arena = load_arena(ctx, wprefix)
    verify_manifest[
        VOCAB_SIZE, HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, HEAD_DIM,
        INTERMEDIATE_SIZE, NUM_LAYERS,
    ](arena.entries, arena.index)
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
    ](base_ptr, arena)

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, seq + 1)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    # embedding
    var x = TileTensor(acts.x, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(dev, row_major(Coord(Index(seq)))),
        context=ctx,
    )

    ref layer = weights.layers[0]
    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, Q_PROJ_OUT))))
    var k = TileTensor(acts.k, row_major(Coord(Index(seq, K_PROJ_OUT))))
    var v = TileTensor(acts.v, row_major(Coord(Index(seq, K_PROJ_OUT))))
    var attn_out = TileTensor(acts.attn_out, row_major(Coord(Index(seq, Q_PROJ_OUT))))
    var proj = TileTensor(acts.proj, row_major(Coord(Index(seq, HIDDEN_SIZE))))

    _rms_norm(
        acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), layer.attn_norm, seq, HIDDEN_SIZE, ctx
    )
    linear(q, xn, layer.q_proj, seq, Q_PROJ_OUT, HIDDEN_SIZE, ctx)
    linear(k, xn, layer.k_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
    linear(v, xn, layer.v_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
    _rms_norm(
        acts.q.unsafe_ptr(), acts.q.unsafe_ptr(), layer.q_norm, seq * NUM_HEADS, HEAD_DIM, ctx
    )
    _rms_norm(
        acts.k.unsafe_ptr(), acts.k.unsafe_ptr(), layer.k_norm, seq * NUM_KV_HEADS, HEAD_DIM, ctx
    )
    apply_rope_inplace[HEAD_DIM, ROPE_THETA](acts.q.unsafe_ptr(), NUM_HEADS, seq, 0, ctx)
    apply_rope_inplace[HEAD_DIM, ROPE_THETA](acts.k.unsafe_ptr(), NUM_KV_HEADS, seq, 0, ctx)

    var k_cache = cache.k.unsafe_ptr()
    var v_cache = cache.v.unsafe_ptr()
    cache_write(k_cache, acts.k.unsafe_ptr(), K_PROJ_OUT, seq, 0, ctx)
    cache_write(v_cache, acts.v.unsafe_ptr(), K_PROJ_OUT, seq, 0, ctx)
    comptime group = NUM_HEADS // NUM_KV_HEADS
    attention[HEAD_DIM, group](
        acts.attn_out.unsafe_ptr(),
        acts.q.unsafe_ptr(),
        k_cache,
        v_cache,
        NUM_HEADS,
        NUM_KV_HEADS,
        seq,
        0,
        ctx,
    )
    linear(proj, attn_out, layer.o_proj, seq, HIDDEN_SIZE, Q_PROJ_OUT, ctx)
    ctx.synchronize()

    # host dumps as float32
    var qn = seq * Q_PROJ_OUT
    var kn = seq * K_PROJ_OUT
    var vn = seq * K_PROJ_OUT
    var an = HIDDEN_SIZE  # target row o_proj only

    var qh = List[Float32]()
    var kh = List[Float32]()
    var vh = List[Float32]()
    var ah = List[Float32]()
    with acts.q.map_to_host() as h:
        for i in range(qn):
            qh.append(h[i].cast[DType.float32]())
    with acts.k.map_to_host() as h:
        for i in range(kn):
            kh.append(h[i].cast[DType.float32]())
    with acts.v.map_to_host() as h:
        for i in range(vn):
            vh.append(h[i].cast[DType.float32]())
    with acts.proj.map_to_host() as h:
        var base = target * HIDDEN_SIZE
        for i in range(HIDDEN_SIZE):
            ah.append(h[base + i].cast[DType.float32]())

    var fq = open(out + "_q.f32", "w")
    fq.write_bytes(
        Span[Byte, origin_of(qh)](ptr=qh.unsafe_ptr().bitcast[Byte](), length=qn * 4)
    )
    fq.close()
    var fk = open(out + "_k.f32", "w")
    fk.write_bytes(
        Span[Byte, origin_of(kh)](ptr=kh.unsafe_ptr().bitcast[Byte](), length=kn * 4)
    )
    fk.close()
    var fv = open(out + "_v.f32", "w")
    fv.write_bytes(
        Span[Byte, origin_of(vh)](ptr=vh.unsafe_ptr().bitcast[Byte](), length=vn * 4)
    )
    fv.close()
    var fa = open(out + "_attn.f32", "w")
    fa.write_bytes(
        Span[Byte, origin_of(ah)](ptr=ah.unsafe_ptr().bitcast[Byte](), length=an * 4)
    )
    fa.close()
    print("wrote", out, "_{q,k,v,attn}.f32")
