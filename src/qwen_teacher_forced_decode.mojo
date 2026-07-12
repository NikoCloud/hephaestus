# Decode-path teacher-forced check: unlike qwen_teacher_forced_full.mojo
# (one big prefill, M>1, exercises matmul_kernel_naive only), this steps
# through 256 REAL decode calls (M=1 each, the gemv_gpu + linear_add_residual
# path) feeding the ORACLE's own tokens back at every step (not our argmax).
#
# This is the check that actually exercises what the G1a-2 gemv/residual-
# fusion optimizations changed. The one-shot prefill check does not: it
# never calls the M=1 code path at all.
#
# Usage: pixi run mojo run ... src/qwen_teacher_forced_decode.mojo \
#            <ids.txt> <oracle_out_ids.txt> <prefix>

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
from hephaestus.forward import Activations, KVCache, forward
from hephaestus.loader import build_weights, load_arena, verify_manifest

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
    var oracle = read_ids(oracle_path)  # 256 tokens
    var seq0 = len(prompt)

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
    ](ctx, 64)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq0)
    with dev_ids.map_to_host() as h:
        for i in range(seq0):
            h[i] = prompt[i]

    var fa = open(prefix + "_argmax.txt", "w")
    var mismatches = 0

    for step in range(N_STEPS):
        var n = seq0 if step == 0 else 1
        forward[
            vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
            kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
            n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
            theta=ROPE_THETA,
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()

        var best = 0
        var best_val = Float32(-3.4e38)
        with acts.logits.map_to_host() as h:
            var base = (n - 1) * VOCAB_SIZE
            for i in range(VOCAB_SIZE):
                var val = h[base + i]
                var rounded = val.cast[DType.bfloat16]().cast[DType.float32]()
                if rounded > best_val:
                    best_val = rounded
                    best = i
        fa.write(String(best) + "\n")
        if best != Int(oracle[step]):
            mismatches += 1

        # Teacher forcing: feed the ORACLE's token, not our own argmax.
        # This makes every one of the 256 steps a decode call (M=1), which
        # is the code path the gemv/residual-fusion changes actually touch.
        with dev_ids.map_to_host() as h:
            h[0] = oracle[step]

    fa.close()
    print(N_STEPS - mismatches, "/", N_STEPS, "argmax matches vs oracle (decode-path teacher-forced)")
