# Attention QK / softmax / PV sub-split for 512-token prefill.
# Host-timed three launches per layer with phase masks; each launch redoes
# earlier phases so SM ≈ (QK+SM)-QK and PV ≈ ALL-(QK+SM).
#
# Usage (nightly):
#   mojo build -I $KERNELS -I src src/qwen_attn_phase_profile.mojo -o /tmp/attn_phases
#   /tmp/attn_phases [ids.txt]

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.time import perf_counter_ns
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
    ATTN_PHASE_ALL,
    ATTN_PHASE_QK,
    ATTN_PHASE_SOFTMAX,
    BF16,
    MAX_KEYS,
    apply_rope_qk_inplace,
    attention,
    linear,
    linear_add_residual,
    silu_mul,
)
from hephaestus.loader import build_weights, load_arena, verify_manifest


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def main() raises:
    var ids_path = String("bench/ab_prompt_long_ids.txt")
    if len(argv()) > 1:
        ids_path = String(argv()[1])
    var ids = read_ids(ids_path)
    var seq = len(ids)
    print("attn phase profile seq=", seq)

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
    ](ctx, seq + 8)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    comptime group = NUM_HEADS // NUM_KV_HEADS
    var past = 0

    var x = TileTensor(acts.x, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, Q_PROJ_OUT))))
    var attn_out = TileTensor(
        acts.attn_out, row_major(Coord(Index(seq, Q_PROJ_OUT)))
    )
    var gate = TileTensor(
        acts.gate, row_major(Coord(Index(seq, INTERMEDIATE_SIZE)))
    )
    var up = TileTensor(
        acts.up, row_major(Coord(Index(seq, INTERMEDIATE_SIZE)))
    )
    var act = TileTensor(
        acts.act, row_major(Coord(Index(seq, INTERMEDIATE_SIZE)))
    )

    # Warmup
    print("warmup...")
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(dev_ids, row_major(Coord(Index(seq)))),
        context=ctx,
    )
    attention[HEAD_DIM, group](
        acts.attn_out.unsafe_ptr().as_unsafe_any_origin(),
        acts.q.unsafe_ptr().as_unsafe_any_origin(),
        cache.k.unsafe_ptr().as_unsafe_any_origin(),
        cache.v.unsafe_ptr().as_unsafe_any_origin(),
        NUM_HEADS,
        NUM_KV_HEADS,
        min(seq, 16),
        0,
        ctx,
        parallel=True,
        phases=ATTN_PHASE_ALL,
    )
    ctx.synchronize()

    cache.length = 0
    past = 0
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(dev_ids, row_major(Coord(Index(seq)))),
        context=ctx,
    )
    ctx.synchronize()

    var sum_qk: Int = 0
    var sum_qk_sm: Int = 0
    var sum_all: Int = 0

    print("measure phases over", NUM_LAYERS, "layers...")
    for i in range(NUM_LAYERS):
        ref layer = weights.layers[i]
        _rms_norm(
            acts.xn.unsafe_ptr().as_unsafe_any_origin(),
            acts.x.unsafe_ptr().as_unsafe_any_origin(),
            layer.attn_norm,
            seq,
            HIDDEN_SIZE,
            ctx,
        )
        var layer_off = i * MAX_KEYS * K_PROJ_OUT
        var k_cache = cache.k.unsafe_ptr() + layer_off
        var v_cache = cache.v.unsafe_ptr() + layer_off
        var k_new = k_cache + past * K_PROJ_OUT
        var v_new = v_cache + past * K_PROJ_OUT
        var k_dst = TileTensor(
            ptr=k_new, layout=row_major(Coord(Index(seq, K_PROJ_OUT)))
        )
        var v_dst = TileTensor(
            ptr=v_new, layout=row_major(Coord(Index(seq, K_PROJ_OUT)))
        )
        linear(q, xn, layer.q_proj, seq, Q_PROJ_OUT, HIDDEN_SIZE, ctx)
        linear(k_dst, xn, layer.k_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        linear(v_dst, xn, layer.v_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        _rms_norm(
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            layer.q_norm,
            seq * NUM_HEADS,
            HEAD_DIM,
            ctx,
        )
        _rms_norm(
            k_new.as_unsafe_any_origin(),
            k_new.as_unsafe_any_origin(),
            layer.k_norm,
            seq * NUM_KV_HEADS,
            HEAD_DIM,
            ctx,
        )
        apply_rope_qk_inplace[HEAD_DIM, ROPE_THETA](
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_new.as_unsafe_any_origin(),
            NUM_HEADS,
            NUM_KV_HEADS,
            seq,
            past,
            ctx,
        )
        ctx.synchronize()

        var t0 = perf_counter_ns()
        attention[HEAD_DIM, group](
            acts.attn_out.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_cache.as_unsafe_any_origin(),
            v_cache.as_unsafe_any_origin(),
            NUM_HEADS,
            NUM_KV_HEADS,
            seq,
            past,
            ctx,
            parallel=True,
            phases=ATTN_PHASE_QK,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        attention[HEAD_DIM, group](
            acts.attn_out.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_cache.as_unsafe_any_origin(),
            v_cache.as_unsafe_any_origin(),
            NUM_HEADS,
            NUM_KV_HEADS,
            seq,
            past,
            ctx,
            parallel=True,
            phases=ATTN_PHASE_QK | ATTN_PHASE_SOFTMAX,
        )
        ctx.synchronize()
        var t2 = perf_counter_ns()
        attention[HEAD_DIM, group](
            acts.attn_out.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_cache.as_unsafe_any_origin(),
            v_cache.as_unsafe_any_origin(),
            NUM_HEADS,
            NUM_KV_HEADS,
            seq,
            past,
            ctx,
            parallel=True,
            phases=ATTN_PHASE_ALL,
        )
        ctx.synchronize()
        var t3 = perf_counter_ns()
        sum_qk += t1 - t0
        sum_qk_sm += t2 - t1
        sum_all += t3 - t2

        linear_add_residual(
            x, attn_out, layer.o_proj, seq, HIDDEN_SIZE, Q_PROJ_OUT, ctx
        )
        _rms_norm(
            acts.xn.unsafe_ptr().as_unsafe_any_origin(),
            acts.x.unsafe_ptr().as_unsafe_any_origin(),
            layer.ffn_norm,
            seq,
            HIDDEN_SIZE,
            ctx,
        )
        linear(
            gate, xn, layer.gate_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx
        )
        linear(
            up, xn, layer.up_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx
        )
        silu_mul(
            acts.act.unsafe_ptr().as_unsafe_any_origin(),
            acts.gate.unsafe_ptr().as_unsafe_any_origin(),
            acts.up.unsafe_ptr().as_unsafe_any_origin(),
            seq * INTERMEDIATE_SIZE,
            ctx,
        )
        linear_add_residual(
            x, act, layer.down_proj, seq, HIDDEN_SIZE, INTERMEDIATE_SIZE, ctx
        )

    # Each launch redoes earlier phases → marginal costs:
    var sm_est = sum_qk_sm - sum_qk
    var pv_est = sum_all - sum_qk_sm
    print("=== attention phase sub-split (ms, all layers) ===")
    print("QK_ms:", Float64(sum_qk) / 1e6)
    print("softmax_ms (est):", Float64(sm_est) / 1e6)
    print("PV_ms (est):", Float64(pv_est) / 1e6)
    print("ALL_ms (fused launch):", Float64(sum_all) / 1e6)
    print(
        "raw intervals ms: QK=",
        Float64(sum_qk) / 1e6,
        " QK+SM=",
        Float64(sum_qk_sm) / 1e6,
        " ALL=",
        Float64(sum_all) / 1e6,
    )
