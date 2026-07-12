# Tiny-random debug loop: load the 2-layer fixture model and verify.
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/load_tiny.mojo [staged_prefix]
#
# Dims from fixtures/tiny_random/config.json:
#   vocab=256 hidden=128 heads=4 kv_heads=2 head_dim=32 inter=256 layers=2

from std.sys import argv
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from hephaestus.loader import build_weights, load_arena, verify_manifest

comptime TINY_VOCAB = 256
comptime TINY_HIDDEN = 128
comptime TINY_Q_OUT = 4 * 32
comptime TINY_KV_OUT = 2 * 32
comptime TINY_HEAD_DIM = 32
comptime TINY_INTER = 256
comptime TINY_LAYERS = 2


def main() raises:
    var prefix = String("staged/tiny")
    if len(argv()) > 1:
        prefix = String(argv()[1])

    var t0 = perf_counter_ns()
    var ctx = DeviceContext()  # device 0
    var arena = load_arena(ctx, prefix)
    verify_manifest[
        TINY_VOCAB,
        TINY_HIDDEN,
        TINY_Q_OUT,
        TINY_KV_OUT,
        TINY_HEAD_DIM,
        TINY_INTER,
        TINY_LAYERS,
    ](arena.entries, arena.index)
    # Detach the pointer's origin from `arena` so passing both doesn't trip
    # the exclusivity checker (the arena outlives the weights struct here).
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=TINY_VOCAB,
        hidden=TINY_HIDDEN,
        q_out=TINY_Q_OUT,
        kv_out=TINY_KV_OUT,
        head_dim=TINY_HEAD_DIM,
        inter=TINY_INTER,
        n_layers=TINY_LAYERS,
    ](base_ptr, arena)
    var t1 = perf_counter_ns()

    print("tiny_random loaded and verified")
    print("  tensors:", len(arena.entries))
    print("  bytes:  ", arena.total_bytes)
    print("  layers: ", len(weights.layers))
    print("  wall:   ", Float64(t1 - t0) / 1e9, "s")
    print("  device round-trip: PASS (full byte compare)")
