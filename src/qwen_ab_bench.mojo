# A/B benchmark harness (bench/1a-ab.md): reports prefill tok/s, TTFT,
# decode tok/s, and total wall time for a single run -- same measurement
# scope as an end-to-end llama-cli run (model already loaded/staged; load
# time is excluded here exactly as it is from llama-cli's "load time" stat).
#
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/qwen_ab_bench.mojo <ids.txt> [n_new]

from std.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

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
from hephaestus.forward import Activations, KVCache, forward
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
    var ids_path = String(argv()[1])
    var n_new = 256
    if len(argv()) > 2:
        n_new = Int(String(argv()[2]))

    var ids = read_ids(ids_path)
    var seq = len(ids)

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

    # max_seq must cover the largest prompt this binary will see (512-token
    # A/B prompt + headroom for a few decode steps beyond it).
    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, 600)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    var t_prefill_ns = Int(0)
    var t_decode_ns = Int(0)
    var t_argmax_prefill_ns = Int(0)
    var t_argmax_decode_ns = Int(0)
    var decode_steps = 0

    var t_run_start = perf_counter_ns()
    for step in range(n_new):
        var n = seq if step == 0 else 1
        var t0 = perf_counter_ns()
        forward[
            vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
            kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
            n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
            theta=ROPE_THETA,
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if step == 0:
            t_prefill_ns = t1 - t0
        else:
            t_decode_ns += t1 - t0
            decode_steps += 1

        var t_am0 = perf_counter_ns()
        var best = Int32(0)
        var best_val = Float32(-3.4e38)
        with acts.logits.map_to_host() as h:
            var base = (n - 1) * VOCAB_SIZE
            for i in range(VOCAB_SIZE):
                var val = h[base + i].cast[DType.bfloat16]().cast[DType.float32]()
                if val > best_val:
                    best_val = val
                    best = Int32(i)
        with dev_ids.map_to_host() as h:
            h[0] = best
        var t_am1 = perf_counter_ns()
        if step == 0:
            t_argmax_prefill_ns = t_am1 - t_am0
        else:
            t_argmax_decode_ns += t_am1 - t_am0
    var t_run_end = perf_counter_ns()

    var prefill_s = Float64(t_prefill_ns) / 1e9
    var argmax_prefill_s = Float64(t_argmax_prefill_ns) / 1e9
    var total_s = Float64(t_run_end - t_run_start) / 1e9
    print("prompt_tokens:", seq)
    print("prefill_s (forward-pass only):", prefill_s)
    print("prefill_s (incl. host argmax):", prefill_s + argmax_prefill_s)
    print("prefill_tok_s:", Float64(seq) / prefill_s)
    print("ttft_ms (forward-pass only):", prefill_s * 1000.0)
    print("ttft_ms (incl. host argmax):", (prefill_s + argmax_prefill_s) * 1000.0)
    if decode_steps > 0:
        var decode_s = Float64(t_decode_ns) / 1e9
        var argmax_decode_s = Float64(t_argmax_decode_ns) / 1e9
        print("decode_tok_s (forward-pass only):", Float64(decode_steps) / decode_s)
        print("decode_tok_s (incl. host argmax):", Float64(decode_steps) / (decode_s + argmax_decode_s))
        print("host_argmax_s (decode steps only):", argmax_decode_s)
        print("host_argmax_ms_per_step:", argmax_decode_s * 1000.0 / Float64(decode_steps))
    print("total_s:", total_s)
    print("total_tokens_generated:", n_new)
