# Full 256-step teacher-forced check, ONE forward call per prompt.
#
# Input = prompt + oracle_output_ids[:255] (length prompt_len+255). Since
# attention is causal, running this whole sequence through forward() once
# yields, at row (prompt_len-1+k) of acts.logits, exactly the logits that
# predict oracle_output_ids[k] -- for all k in 0..255 simultaneously. This is
# the same computation as 256 sequential decode steps with the oracle's own
# tokens fed back (teacher forcing), just batched into a single prefill.
#
# Writes:
#   <prefix>_logits.f32   256 x VOCAB_SIZE float32, row k = step k's logits
#   <prefix>_argmax.txt   256 lines: our bf16-rounded argmax pick per step
#
# Usage: pixi run mojo run ... src/qwen_teacher_forced_full.mojo \
#            <ids.txt> <oracle_out_ids.txt> <prefix>

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

comptime N_STEPS = 256
comptime MAX_SEQ = 300


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
    var oracle_path = String(argv()[2])
    var prefix = String(argv()[3])

    var prompt = read_ids(ids_path)
    var oracle = read_ids(oracle_path)  # 256 tokens
    var prompt_len = len(prompt)

    # Full input = prompt + oracle's first 255 generated tokens.
    var full = List[Int32]()
    for i in range(prompt_len):
        full.append(prompt[i])
    for i in range(255):
        full.append(oracle[i])
    var full_len = len(full)  # prompt_len + 255

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
    ](ctx, MAX_SEQ)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](full_len)
    with dev_ids.map_to_host() as h:
        for i in range(full_len):
            h[i] = full[i]

    var t0 = perf_counter_ns()
    forward[
        vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
        theta=ROPE_THETA,
    ](weights, acts, cache, dev_ids, full_len, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("single prefill of", full_len, "tokens took", Float64(t1 - t0) / 1e9, "s")

    # Rows (prompt_len-1) .. (prompt_len-1+255) predict oracle steps 0..255.
    var f = open(prefix + "_logits.f32", "w")
    var fa = open(prefix + "_argmax.txt", "w")
    var mismatches = 0
    with acts.logits.map_to_host() as h:
        for k in range(N_STEPS):
            var row = prompt_len - 1 + k
            var base = row * VOCAB_SIZE
            var best = 0
            var best_val = Float32(-3.4e38)
            var buf = List[Float32]()
            for i in range(VOCAB_SIZE):
                var val = h[base + i]
                buf.append(val)
                var rounded = val.cast[DType.bfloat16]().cast[DType.float32]()
                if rounded > best_val:
                    best_val = rounded
                    best = i
            f.write_bytes(
                Span[Byte, origin_of(buf)](
                    ptr=buf.unsafe_ptr().bitcast[Byte](),
                    length=VOCAB_SIZE * 4,
                )
            )
            fa.write(String(best) + "\n")
            if best != Int(oracle[k]):
                mismatches += 1
    f.close()
    fa.close()
    print("prompt done:", N_STEPS - mismatches, "/", N_STEPS, "argmax matches vs oracle")
