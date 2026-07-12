# Phase 1a — Loader Timing (G1a-3)
## Date: 2026-07-12
## Hardware: GPU 0 = AMD Radeon AI Pro R9700 32GB (gfx1201), see bench/hardware.md
## Host: 31GB RAM, blob on /home NVMe (nvme0n1p2)

## What was measured

Wall time of the full loader path: parse `.offsets` manifest → read 8.04GB
`.weights` blob into pinned HostBuffer → copy to single DeviceBuffer arena on
GPU 0 → copy back to a second HostBuffer and byte-compare (first/last 64B +
4096 strided samples) → construct all TileTensor weight structs (398 tensors,
36 layers). Timed inside the binary with `perf_counter_ns` from before
`DeviceContext()` to after struct construction.

## Commands (exact)

```
pixi run mojo build -I ~/projects/modular/max/kernels/src -I src src/main.mojo -o /tmp/heph_main
/tmp/heph_main staged/qwen3-4b
```

Staging pre-pass (one-time, not part of the measured load):
```
pixi run python scripts/stage_weights.py /mnt/models/models/qwen3-4b-instruct-2507 staged/qwen3-4b
# 398 tensors, 8,044,936,192 bytes, ~10.8s
```

## Results (3 reps)

| rep | wall (s) |
|---|---|
| 1 | 6.601 |
| 2 | 6.248 |
| 3 | 6.353 |

Page-cache state: partially warm (blob written shortly before rep 1; host has
31GB RAM vs 8GB blob + 12GB buff/cache). Cold-cache bound measured separately:

```
dd if=staged/qwen3-4b.weights of=/dev/null bs=64M iflag=direct
# 8044936192 bytes copied, 1.867 s, 4.3 GB/s
```

Worst-case cold load ≈ 6.6s + 1.9s ≈ 8.5s.

## Gate verdict

**G1a-3 (loads in under 30s from cold): PASS** — ~6.3s warm, ≤ ~8.5s cold bound.

Note: measured wall includes the device round-trip verification copy (8GB
device→host + compare), which is loader self-check overhead, not part of a
production load. The gate passes with it included.
