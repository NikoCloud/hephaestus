# G1a-2 profiling harness: per-category timing breakdown of one decode step.
#
# Mirrors hephaestus.forward.forward()'s exact call sequence (copy, not the
# production function -- production forward() is untouched by this file) but
# synchronizes and times around each named kernel group. This is a DIAGNOSTIC
# tool: the sync calls add host round-trip overhead the unstrumented decode
# step doesn't pay, so the TOTAL time here will be higher than production
# tok/s implies. The point is the RELATIVE breakdown between categories, not
# an absolute number -- each category pays one sync, so the bias is roughly
# uniform across categories and the proportions stay informative.
#
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/qwen_profile.mojo [n_steps]

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from std.utils.index import Index
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
    apply_rope_qk_inplace,
    attention,
    linear,
    linear_add_residual,
    silu_mul,
    BF16,
    MAX_KEYS,
)
from hephaestus.loader import build_weights, load_arena, verify_manifest
from hephaestus.model import Qwen3Weights

comptime N_CATEGORIES = 12
comptime CAT_EMBED = 0
comptime CAT_ATTN_NORM = 1
comptime CAT_QKV_PROJ = 2
comptime CAT_QK_NORM = 3
comptime CAT_ROPE = 4
comptime CAT_ATTENTION = 5
comptime CAT_O_PROJ = 6
comptime CAT_FFN_NORM = 7
comptime CAT_GATE_UP = 8
comptime CAT_SILU = 9
comptime CAT_DOWN_PROJ = 10
comptime CAT_FINAL = 11

def category_names() -> List[String]:
    var names = List[String]()
    names.append("embed")
    names.append("attn_norm")
    names.append("qkv_proj")
    names.append("qk_norm")
    names.append("rope")
    names.append("attention")
    names.append("o_proj+residual")
    names.append("ffn_norm")
    names.append("gate_up_proj")
    names.append("silu")
    names.append("down_proj+residual")
    names.append("final_norm+lm_head")
    return names^


def profiled_decode_step(
    weights: Qwen3Weights[
        _, VOCAB_SIZE, HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, HEAD_DIM,
        INTERMEDIATE_SIZE, NUM_LAYERS,
    ],
    mut acts: Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ],
    mut cache: KVCache[NUM_LAYERS, K_PROJ_OUT],
    token_ids: DeviceBuffer[DType.int32],
    seq: Int,
    ctx: DeviceContext,
    mut totals_ns: List[Int],
) raises:
    comptime group = NUM_HEADS // NUM_KV_HEADS
    var past = cache.length

    var t = perf_counter_ns()
    var x = TileTensor(acts.x, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    gather[axis=0, target="gpu"](
        x,
        weights.embed_tokens,
        TileTensor(token_ids, row_major(Coord(Index(seq)))),
        context=ctx,
    )
    ctx.synchronize()
    totals_ns[CAT_EMBED] = totals_ns[CAT_EMBED] + (perf_counter_ns() - t)

    var xn = TileTensor(acts.xn, row_major(Coord(Index(seq, HIDDEN_SIZE))))
    var q = TileTensor(acts.q, row_major(Coord(Index(seq, Q_PROJ_OUT))))
    var attn_out = TileTensor(
        acts.attn_out, row_major(Coord(Index(seq, Q_PROJ_OUT)))
    )
    var gate = TileTensor(
        acts.gate, row_major(Coord(Index(seq, INTERMEDIATE_SIZE)))
    )
    var up = TileTensor(acts.up, row_major(Coord(Index(seq, INTERMEDIATE_SIZE))))
    var act = TileTensor(acts.act, row_major(Coord(Index(seq, INTERMEDIATE_SIZE))))

    for i in range(NUM_LAYERS):
        ref layer = weights.layers[i]

        t = perf_counter_ns()
        _rms_norm(
            acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), layer.attn_norm, seq,
            HIDDEN_SIZE, ctx,
        )
        ctx.synchronize()
        totals_ns[CAT_ATTN_NORM] = totals_ns[CAT_ATTN_NORM] + (perf_counter_ns() - t)

        var layer_off = i * MAX_KEYS * K_PROJ_OUT
        var k_cache = cache.k.unsafe_ptr() + layer_off
        var v_cache = cache.v.unsafe_ptr() + layer_off
        var k_new_ptr = k_cache + past * K_PROJ_OUT
        var v_new_ptr = v_cache + past * K_PROJ_OUT
        var k_dst = TileTensor(
            ptr=k_new_ptr, layout=row_major(Coord(Index(seq, K_PROJ_OUT)))
        )
        var v_dst = TileTensor(
            ptr=v_new_ptr, layout=row_major(Coord(Index(seq, K_PROJ_OUT)))
        )

        t = perf_counter_ns()
        linear(q, xn, layer.q_proj, seq, Q_PROJ_OUT, HIDDEN_SIZE, ctx)
        linear(k_dst, xn, layer.k_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        linear(v_dst, xn, layer.v_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        ctx.synchronize()
        totals_ns[CAT_QKV_PROJ] = totals_ns[CAT_QKV_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        _rms_norm(
            acts.q.unsafe_ptr(), acts.q.unsafe_ptr(), layer.q_norm,
            seq * NUM_HEADS, HEAD_DIM, ctx,
        )
        _rms_norm(
            k_new_ptr, k_new_ptr, layer.k_norm, seq * NUM_KV_HEADS, HEAD_DIM,
            ctx,
        )
        ctx.synchronize()
        totals_ns[CAT_QK_NORM] = totals_ns[CAT_QK_NORM] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        apply_rope_qk_inplace[HEAD_DIM, ROPE_THETA](
            acts.q.unsafe_ptr(), k_new_ptr, NUM_HEADS, NUM_KV_HEADS, seq,
            past, ctx,
        )
        ctx.synchronize()
        totals_ns[CAT_ROPE] = totals_ns[CAT_ROPE] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        attention[HEAD_DIM, group](
            acts.attn_out.unsafe_ptr(), acts.q.unsafe_ptr(), k_cache,
            v_cache, NUM_HEADS, NUM_KV_HEADS, seq, past, ctx,
        )
        ctx.synchronize()
        totals_ns[CAT_ATTENTION] = totals_ns[CAT_ATTENTION] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear_add_residual(
            x, attn_out, layer.o_proj, seq, HIDDEN_SIZE, Q_PROJ_OUT, ctx
        )
        ctx.synchronize()
        totals_ns[CAT_O_PROJ] = totals_ns[CAT_O_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        _rms_norm(
            acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), layer.ffn_norm, seq,
            HIDDEN_SIZE, ctx,
        )
        ctx.synchronize()
        totals_ns[CAT_FFN_NORM] = totals_ns[CAT_FFN_NORM] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear(gate, xn, layer.gate_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx)
        linear(up, xn, layer.up_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx)
        ctx.synchronize()
        totals_ns[CAT_GATE_UP] = totals_ns[CAT_GATE_UP] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        silu_mul(
            acts.act.unsafe_ptr(), acts.gate.unsafe_ptr(),
            acts.up.unsafe_ptr(), seq * INTERMEDIATE_SIZE, ctx,
        )
        ctx.synchronize()
        totals_ns[CAT_SILU] = totals_ns[CAT_SILU] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear_add_residual(
            x, act, layer.down_proj, seq, HIDDEN_SIZE, INTERMEDIATE_SIZE, ctx
        )
        ctx.synchronize()
        totals_ns[CAT_DOWN_PROJ] = totals_ns[CAT_DOWN_PROJ] + (perf_counter_ns() - t)

    t = perf_counter_ns()
    _rms_norm(
        acts.xn.unsafe_ptr(), acts.x.unsafe_ptr(), weights.output_norm, seq,
        HIDDEN_SIZE, ctx,
    )
    var logits = TileTensor(acts.logits, row_major(Coord(Index(seq, VOCAB_SIZE))))
    linear(logits, xn, weights.embed_tokens, seq, VOCAB_SIZE, HIDDEN_SIZE, ctx)
    ctx.synchronize()
    totals_ns[CAT_FINAL] = totals_ns[CAT_FINAL] + (perf_counter_ns() - t)

    cache.length = past + seq


def main() raises:
    var n_steps = 30
    if len(argv()) > 1:
        n_steps = Int(String(argv()[1]))

    var ctx = DeviceContext()
    var arena = load_arena(ctx, "staged/qwen3-4b")
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
    ](ctx, 8)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](1)
    with dev_ids.map_to_host() as h:
        h[0] = 785  # arbitrary valid token; content doesn't matter for timing

    var cat_names = category_names()
    var totals_ns = List[Int]()
    for _ in range(N_CATEGORIES):
        totals_ns.append(0)

    # Warmup (first decode step pays one-time costs: JIT/codegen caches etc.)
    profiled_decode_step(weights, acts, cache, dev_ids, 1, ctx, totals_ns)
    for i in range(N_CATEGORIES):
        totals_ns[i] = 0
    cache.length = 1

    var t_wall0 = perf_counter_ns()
    for _ in range(n_steps):
        profiled_decode_step(weights, acts, cache, dev_ids, 1, ctx, totals_ns)
    var t_wall1 = perf_counter_ns()

    var wall_ms_per_step = Float64(t_wall1 - t_wall0) / Float64(n_steps) / 1e6
    print("=== per-category breakdown (", n_steps, "steps, synced timing) ===")
    var category_sum_ns = 0
    for i in range(N_CATEGORIES):
        category_sum_ns += totals_ns[i]
    for i in range(N_CATEGORIES):
        var avg_ms = Float64(totals_ns[i]) / Float64(n_steps) / 1e6
        var pct = 100.0 * Float64(totals_ns[i]) / Float64(category_sum_ns)
        print(" ", cat_names[i], ":", avg_ms, "ms/step  (", pct, "% )")
    print("profiled (synced) wall time/step:", wall_ms_per_step, "ms")
    print(
        "  ( production, unsynced tok/s implies ~20.4ms/step; this run's"
        " total is higher because every category pays its own sync )"
    )
