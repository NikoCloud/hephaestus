# Decode-path teacher-forced check for FP8 W8A8 (G1b-4 path).
#
# Same protocol as qwen_teacher_forced_decode.mojo: 256 real M=1 decode
# steps, feeding the ORACLE's tokens (not our argmax). Compares our argmax
# to the oracle token at each step.
#
# Usage:
#   mojo run -I $KERNELS -I src src/qwen_teacher_forced_decode_fp8.mojo \
#       <ids.txt> <oracle_out_ids.txt> <prefix>

from std.gpu.host import DeviceContext
from std.sys import argv

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
from hephaestus.forward import Activations, KVCache, forward_fp8
from hephaestus.kernels import argmax_logits
from hephaestus.loader import build_weights_fp8, load_arena_bytes

comptime N_STEPS = 256


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
    var oracle = read_ids(oracle_path)
    if len(oracle) < N_STEPS:
        raise Error("oracle shorter than N_STEPS")
    var seq0 = len(prompt)

    var ctx = DeviceContext()
    print("loading FP8 staged/qwen3-4b-fp8 ...")
    var arena = load_arena_bytes(ctx, "staged/qwen3-4b-fp8")
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights_fp8[
        vocab=VOCAB_SIZE,
        hidden=HIDDEN_SIZE,
        q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT,
        head_dim=HEAD_DIM,
        inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
    ](base_ptr, arena)
    print("FP8 weights ready, tensors=", len(arena.entries))

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, 64)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq0)
    with dev_ids.map_to_host() as h:
        for i in range(seq0):
            h[i] = prompt[i]

    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    var fa = open(prefix + "_argmax.txt", "w")
    var mismatches = 0

    for step in range(N_STEPS):
        var n = seq0 if step == 0 else 1
        forward_fp8[
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

        var logits_base = (
            acts.logits.unsafe_ptr() + (n - 1) * VOCAB_SIZE
        ).as_unsafe_any_origin()
        var best = argmax_logits(
            logits_base, argmax_bf16, argmax_idx, VOCAB_SIZE, ctx
        )
        fa.write(String(best) + "\n")
        if Int(best) != Int(oracle[step]):
            mismatches += 1
            if mismatches <= 16:
                print(
                    "mismatch step",
                    step,
                    "got",
                    best,
                    "oracle",
                    oracle[step],
                )

        # Teacher forcing: feed ORACLE token (not our argmax).
        with dev_ids.map_to_host() as h:
            h[0] = oracle[step]

    fa.close()
    var matches = N_STEPS - mismatches
    print(
        matches,
        "/",
        N_STEPS,
        "argmax matches vs oracle (FP8 W8A8 decode teacher-forced)",
    )
    print("match_rate:", Float64(matches) / Float64(N_STEPS))
    print("mismatch_count:", mismatches)
