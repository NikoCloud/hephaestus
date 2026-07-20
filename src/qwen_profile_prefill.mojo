# v3a prefill profiling harness: per-category timing for one 512-token prefill.
#
# Diagnostic copy of forward() with synchronize + host timer around each group.
# Sync bias is roughly uniform across categories; use proportions, not absolute
# tok/s (production is unsynced). Barriers inside WMMA kernels are not
# separately measurable from host — included in GEMM category times.
#
# Usage (nightly):
#   pixi run mojo build -I $KERNELS -I src src/qwen_profile_prefill.mojo \
#       -o /tmp/qwen_profile_prefill
#   /tmp/qwen_profile_prefill [ids.txt]   # default: bench/ab_prompt_long_ids.txt

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
    BF16,
    MAX_KEYS,
    apply_rope_qk_inplace,
    attention,
    linear,
    linear_add_residual,
    silu_mul,
)
from hephaestus.loader import build_weights, load_arena, verify_manifest
from hephaestus.model import Qwen3Weights

# Fine-grained categories for GEMM vs non-GEMM split.
comptime N_CAT = 16
comptime C_EMBED = 0
comptime C_ATTN_NORM = 1
comptime C_Q_PROJ = 2
comptime C_K_PROJ = 3
comptime C_V_PROJ = 4
comptime C_QK_NORM = 5
comptime C_ROPE = 6
comptime C_ATTENTION = 7
comptime C_O_PROJ = 8
comptime C_FFN_NORM = 9
comptime C_GATE_PROJ = 10
comptime C_UP_PROJ = 11
comptime C_SILU = 12
comptime C_DOWN_PROJ = 13
comptime C_OUT_NORM = 14
comptime C_LM_HEAD = 15


def cat_names() -> List[String]:
    var n = List[String]()
    n.append("embed")
    n.append("attn_norm")
    n.append("q_proj")
    n.append("k_proj")
    n.append("v_proj")
    n.append("qk_norm")
    n.append("rope")
    n.append("attention")
    n.append("o_proj+residual")
    n.append("ffn_norm")
    n.append("gate_proj")
    n.append("up_proj")
    n.append("silu_mul")
    n.append("down_proj+residual")
    n.append("output_norm")
    n.append("lm_head")
    return n^


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def profiled_prefill(
    weights: Qwen3Weights[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
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
    totals_ns[C_EMBED] = totals_ns[C_EMBED] + (perf_counter_ns() - t)

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

    for i in range(NUM_LAYERS):
        ref layer = weights.layers[i]

        t = perf_counter_ns()
        _rms_norm(
            acts.xn.unsafe_ptr().as_unsafe_any_origin(),
            acts.x.unsafe_ptr().as_unsafe_any_origin(),
            layer.attn_norm,
            seq,
            HIDDEN_SIZE,
            ctx,
        )
        ctx.synchronize()
        totals_ns[C_ATTN_NORM] = totals_ns[C_ATTN_NORM] + (perf_counter_ns() - t)

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
        ctx.synchronize()
        totals_ns[C_Q_PROJ] = totals_ns[C_Q_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear(k_dst, xn, layer.k_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        ctx.synchronize()
        totals_ns[C_K_PROJ] = totals_ns[C_K_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear(v_dst, xn, layer.v_proj, seq, K_PROJ_OUT, HIDDEN_SIZE, ctx)
        ctx.synchronize()
        totals_ns[C_V_PROJ] = totals_ns[C_V_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        _rms_norm(
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            layer.q_norm,
            seq * NUM_HEADS,
            HEAD_DIM,
            ctx,
        )
        _rms_norm(
            k_new_ptr.as_unsafe_any_origin(),
            k_new_ptr.as_unsafe_any_origin(),
            layer.k_norm,
            seq * NUM_KV_HEADS,
            HEAD_DIM,
            ctx,
        )
        ctx.synchronize()
        totals_ns[C_QK_NORM] = totals_ns[C_QK_NORM] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        apply_rope_qk_inplace[HEAD_DIM, ROPE_THETA](
            acts.q.unsafe_ptr().as_unsafe_any_origin(),
            k_new_ptr.as_unsafe_any_origin(),
            NUM_HEADS,
            NUM_KV_HEADS,
            seq,
            past,
            ctx,
        )
        ctx.synchronize()
        totals_ns[C_ROPE] = totals_ns[C_ROPE] + (perf_counter_ns() - t)

        t = perf_counter_ns()
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
        )
        ctx.synchronize()
        totals_ns[C_ATTENTION] = totals_ns[C_ATTENTION] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear_add_residual(
            x, attn_out, layer.o_proj, seq, HIDDEN_SIZE, Q_PROJ_OUT, ctx
        )
        ctx.synchronize()
        totals_ns[C_O_PROJ] = totals_ns[C_O_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        _rms_norm(
            acts.xn.unsafe_ptr().as_unsafe_any_origin(),
            acts.x.unsafe_ptr().as_unsafe_any_origin(),
            layer.ffn_norm,
            seq,
            HIDDEN_SIZE,
            ctx,
        )
        ctx.synchronize()
        totals_ns[C_FFN_NORM] = totals_ns[C_FFN_NORM] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear(
            gate, xn, layer.gate_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx
        )
        ctx.synchronize()
        totals_ns[C_GATE_PROJ] = totals_ns[C_GATE_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear(
            up, xn, layer.up_proj, seq, INTERMEDIATE_SIZE, HIDDEN_SIZE, ctx
        )
        ctx.synchronize()
        totals_ns[C_UP_PROJ] = totals_ns[C_UP_PROJ] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        silu_mul(
            acts.act.unsafe_ptr().as_unsafe_any_origin(),
            acts.gate.unsafe_ptr().as_unsafe_any_origin(),
            acts.up.unsafe_ptr().as_unsafe_any_origin(),
            seq * INTERMEDIATE_SIZE,
            ctx,
        )
        ctx.synchronize()
        totals_ns[C_SILU] = totals_ns[C_SILU] + (perf_counter_ns() - t)

        t = perf_counter_ns()
        linear_add_residual(
            x, act, layer.down_proj, seq, HIDDEN_SIZE, INTERMEDIATE_SIZE, ctx
        )
        ctx.synchronize()
        totals_ns[C_DOWN_PROJ] = totals_ns[C_DOWN_PROJ] + (perf_counter_ns() - t)

    t = perf_counter_ns()
    _rms_norm(
        acts.xn.unsafe_ptr().as_unsafe_any_origin(),
        acts.x.unsafe_ptr().as_unsafe_any_origin(),
        weights.output_norm,
        seq,
        HIDDEN_SIZE,
        ctx,
    )
    ctx.synchronize()
    totals_ns[C_OUT_NORM] = totals_ns[C_OUT_NORM] + (perf_counter_ns() - t)

    t = perf_counter_ns()
    var logits = TileTensor(
        acts.logits, row_major(Coord(Index(seq, VOCAB_SIZE)))
    )
    linear(logits, xn, weights.embed_tokens, seq, VOCAB_SIZE, HIDDEN_SIZE, ctx)
    ctx.synchronize()
    totals_ns[C_LM_HEAD] = totals_ns[C_LM_HEAD] + (perf_counter_ns() - t)

    cache.length = past + seq


def main() raises:
    var ids_path = String("bench/ab_prompt_long_ids.txt")
    if len(argv()) > 1:
        ids_path = String(argv()[1])

    var ids = read_ids(ids_path)
    var seq = len(ids)
    print("profile prefill seq=", seq, " layers=", NUM_LAYERS)

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

    # max_seq must cover prefill length
    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, seq + 8)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    var names = cat_names()
    var totals_ns = List[Int]()
    for _ in range(N_CAT):
        totals_ns.append(0)

    # Warmup: one full prefill (JIT / first-launch cost).
    print("warmup...")
    profiled_prefill(weights, acts, cache, dev_ids, seq, ctx, totals_ns)
    for i in range(N_CAT):
        totals_ns[i] = 0
    # Reset cache for a clean measured prefill (same as cold-ish prefill).
    cache.length = 0

    print("measure (1× synced prefill)...")
    var t0 = perf_counter_ns()
    profiled_prefill(weights, acts, cache, dev_ids, seq, ctx, totals_ns)
    var t1 = perf_counter_ns()

    var wall_ms = Float64(t1 - t0) / 1e6
    var cat_sum: Int = 0
    for i in range(N_CAT):
        cat_sum += totals_ns[i]

    print("=== per-category (full prefill, synced) ===")
    for i in range(N_CAT):
        var ms = Float64(totals_ns[i]) / 1e6
        var pct = 100.0 * Float64(totals_ns[i]) / Float64(cat_sum)
        print(" ", names[i], ":", ms, "ms  (", pct, "% )")

    # Rollups for the report table.
    var gemm_ns = (
        totals_ns[C_Q_PROJ]
        + totals_ns[C_K_PROJ]
        + totals_ns[C_V_PROJ]
        + totals_ns[C_O_PROJ]
        + totals_ns[C_GATE_PROJ]
        + totals_ns[C_UP_PROJ]
        + totals_ns[C_DOWN_PROJ]
        + totals_ns[C_LM_HEAD]
    )
    var norm_ns = (
        totals_ns[C_ATTN_NORM]
        + totals_ns[C_QK_NORM]
        + totals_ns[C_FFN_NORM]
        + totals_ns[C_OUT_NORM]
    )
    var attn_ns = totals_ns[C_ATTENTION]
    var rope_ns = totals_ns[C_ROPE]
    var silu_ns = totals_ns[C_SILU]
    var embed_ns = totals_ns[C_EMBED]
    var other_ns = cat_sum - gemm_ns - norm_ns - attn_ns - rope_ns - silu_ns - embed_ns

    print("")
    print("=== rollup ===")
    print("wall_ms (synced prefill):", wall_ms)
    print("category_sum_ms:", Float64(cat_sum) / 1e6)
    print("seq:", seq)
    print("implied_tok_s_synced:", Float64(seq) * 1e9 / Float64(t1 - t0))
    print(
        "WMMA_GEMM_ms:",
        Float64(gemm_ns) / 1e6,
        " pct:",
        100.0 * Float64(gemm_ns) / Float64(cat_sum),
    )
    print(
        "  q_proj_ms:",
        Float64(totals_ns[C_Q_PROJ]) / 1e6,
        " k_proj_ms:",
        Float64(totals_ns[C_K_PROJ]) / 1e6,
        " v_proj_ms:",
        Float64(totals_ns[C_V_PROJ]) / 1e6,
    )
    print(
        "  o_proj_ms:",
        Float64(totals_ns[C_O_PROJ]) / 1e6,
        " gate_ms:",
        Float64(totals_ns[C_GATE_PROJ]) / 1e6,
        " up_ms:",
        Float64(totals_ns[C_UP_PROJ]) / 1e6,
    )
    print(
        "  down_proj_ms:",
        Float64(totals_ns[C_DOWN_PROJ]) / 1e6,
        " lm_head_ms:",
        Float64(totals_ns[C_LM_HEAD]) / 1e6,
    )
    print(
        "Attention_ms:",
        Float64(attn_ns) / 1e6,
        " pct:",
        100.0 * Float64(attn_ns) / Float64(cat_sum),
    )
    print(
        "RMSNorm_ms:",
        Float64(norm_ns) / 1e6,
        " pct:",
        100.0 * Float64(norm_ns) / Float64(cat_sum),
    )
    print(
        "RoPE_ms:",
        Float64(rope_ns) / 1e6,
        " pct:",
        100.0 * Float64(rope_ns) / Float64(cat_sum),
    )
    print(
        "silu_mul_ms:",
        Float64(silu_ns) / 1e6,
        " pct:",
        100.0 * Float64(silu_ns) / Float64(cat_sum),
    )
    print(
        "embed_ms:",
        Float64(embed_ns) / 1e6,
        " pct:",
        100.0 * Float64(embed_ns) / Float64(cat_sum),
    )
    print(
        "other_ms:",
        Float64(other_ns) / 1e6,
        " pct:",
        100.0 * Float64(other_ns) / Float64(cat_sum),
    )
    print(
        "non_GEMM_ms:",
        Float64(cat_sum - gemm_ns) / 1e6,
        " pct:",
        100.0 * Float64(cat_sum - gemm_ns) / Float64(cat_sum),
    )
    print(
        "NOTE: in-kernel barriers (v3a LDS) are inside GEMM times; not split out."
    )
