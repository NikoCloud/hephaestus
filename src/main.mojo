# Hephaestus CLI -- Phase 1a loader milestone.
# Loads the staged Qwen3-4B-Instruct-2507 weights onto GPU 0 and verifies.
#
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/main.mojo [staged_prefix]
#
# Staging pre-pass (one-time): pixi run python scripts/stage_weights.py \
#            /mnt/models/models/qwen3-4b-instruct-2507 staged/qwen3-4b

from std.sys import argv
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from hephaestus.constants import (
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    K_PROJ_OUT,
    NUM_LAYERS,
    Q_PROJ_OUT,
    VOCAB_SIZE,
)
from hephaestus.loader import build_weights, load_arena, verify_manifest


def main() raises:
    var prefix = String("staged/qwen3-4b")
    if len(argv()) > 1:
        prefix = String(argv()[1])

    var t0 = perf_counter_ns()
    var ctx = DeviceContext()  # device 0 = R9700
    var arena = load_arena(ctx, prefix)
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
    var t1 = perf_counter_ns()

    var secs = Float64(t1 - t0) / 1e9
    print("qwen3-4b loaded and verified")
    print("  tensors:", len(arena.entries))
    print("  bytes:  ", arena.total_bytes)
    print("  layers: ", len(weights.layers))
    print("  wall:   ", secs, "s")
    print("  device round-trip: PASS (sampled compare)")
    if secs < 30.0:
        print("  G1a-3 (<30s): PASS")
    else:
        print("  G1a-3 (<30s): FAIL")
