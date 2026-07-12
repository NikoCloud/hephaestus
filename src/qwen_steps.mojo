# Teacher-forced step diff: feed the ORACLE's tokens (not our own samples) and
# dump logits at each of the 10 saved steps.
#
# This separates two failure modes that look identical from the outside:
#   - forward-pass bug   -> a step's logits are wrong even with correct history
#   - numerical drift    -> every step's logits are right, but greedy flips a
#                           near-tie somewhere and the histories diverge after
#
# Usage: pixi run mojo run ... src/qwen_steps.mojo <ids.txt> <oracle_out_ids.txt> <prefix>
# Writes <prefix>_step{K}.f32 for K in 0..9.

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

comptime N_STEPS = 10


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

    var ids = read_ids(ids_path)
    var oracle = read_ids(oracle_path)
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

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, 64)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    for step in range(N_STEPS):
        var n = seq if step == 0 else 1
        forward[
            vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
            kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
            n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
            theta=ROPE_THETA,
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()

        var best = 0
        var best_val = Float32(-3.4e38)
        var f = open(prefix + "_step" + String(step) + ".f32", "w")
        with acts.logits.map_to_host() as h:
            var base = (n - 1) * VOCAB_SIZE
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
        f.close()
        print(
            "step", step, "argmax", best, "oracle", Int(oracle[step]),
            "MATCH" if best == Int(oracle[step]) else "*** MISMATCH ***",
        )

        # Teacher forcing: feed the ORACLE's token, not ours.
        with dev_ids.map_to_host() as h:
            h[0] = oracle[step]
