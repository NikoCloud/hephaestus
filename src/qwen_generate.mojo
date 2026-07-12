# Qwen3-4B greedy decode -- the G1a-1 gate.
# Prefill + 256 greedy tokens with KV cache reuse, per prompt.
# Writes generated IDs to a file for scripts/check_g1a1.py to diff against
# fixtures/oracle/promptN_output_ids.json.
#
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/qwen_generate.mojo <ids.txt> <out_ids.txt> [n_tokens]

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
    var ids_path = String("prompt_ids.txt")
    var out_path = String("out_ids.txt")
    var n_new = 256
    if len(argv()) > 1:
        ids_path = String(argv()[1])
    if len(argv()) > 2:
        out_path = String(argv()[2])
    if len(argv()) > 3:
        n_new = Int(String(argv()[3]))

    var ids = read_ids(ids_path)
    var seq = len(ids)

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
    ](ctx, 64)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    var f = open(out_path, "w")
    var t_decode = Int(0)
    var decode_steps = 0

    for step in range(n_new):
        var n = seq if step == 0 else 1
        var t0 = perf_counter_ns()
        forward[
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
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if step > 0:  # exclude prefill from decode timing
            t_decode += t1 - t0
            decode_steps += 1

        var best = Int32(0)
        var best_val = Float32(-3.4e38)
        with acts.logits.map_to_host() as h:
            var base = (n - 1) * VOCAB_SIZE
            for i in range(VOCAB_SIZE):
                # Greedy = argmax over the REFERENCE's logit dtype. HF's lm_head
                # emits bf16, and torch.argmax returns the FIRST max, so ties
                # break to the lower id. Comparing in fp32 resolves ties torch
                # never saw and picks differently (prompt3 step7). Round first.
                var val = h[base + i].cast[DType.bfloat16]().cast[DType.float32]()
                if val > best_val:
                    best_val = val
                    best = Int32(i)
        f.write(String(best) + "\n")
        with dev_ids.map_to_host() as h:
            h[0] = best

    f.close()
    var secs = Float64(t_decode) / 1e9
    print("generated", n_new, "tokens ->", out_path)
    if decode_steps > 0:
        print("decode:", Float64(decode_steps) / secs, "tok/s (", decode_steps, "steps )")
