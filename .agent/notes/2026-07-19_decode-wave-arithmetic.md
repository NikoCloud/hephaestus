# Why M=1 decode cannot be fixed by kernel tuning (wave arithmetic)

**Date:** 2026-07-19. **Status:** derived + measured. This is the unifying theory for
four separate experimental results that previously looked like unrelated failures.

## The arithmetic

The FP8 decode kernel launches one wave per 16-column output tile: `grid = (N/16,)`.

| projection | N | waves launched |
|---|---:|---:|
| q_proj | 4096 | 256 |
| k_proj / v_proj | 1024 | **64** |
| o_proj / down_proj | 2560 | 160 |
| gate / up_proj | 9728 | 608 |

The R9700 has 64 CUs and can host on the order of **thousands** of concurrent waves.
At M=1 we occupy a few percent of the machine. This is not a tuning deficiency --
the available parallelism is bounded by N/16, which is a property of the problem
shape at M=1, not of the kernel.

## Little`s Law consequence (why per-wave tuning cannot rescue it)

To sustain B bytes/s at memory latency L you need B x L bytes in flight.
At the measured achievable **569 GB/s** and ~400 ns GDDR6 latency that is
**~228 KB in flight**.

- 256 waves x 1 outstanding 8-byte load = ~2 KB. We are ~100x short.
- Unrolling the K-loop 4x = ~8 KB. Still ~28x short.
- Saturating from 256 waves would need ~110 outstanding loads **per wave**.

**Conclusion: no amount of per-wave unroll/prefetch reaches the roofline at M=1.**
It is worth at best single-digit-to-teens percent. The binding constraint is wave count.

## What this explains (one frame, four results)

| Result | Explanation |
|---|---|
| Split-K regressed (56.6 -> 45.8) | It *did* add waves, but a global partials buffer + a second reduce launch cost more than the added parallelism returned |
| LDS staging regressed (56.6 -> 36.8) | No reuse at M=1, so barriers are pure overhead (see LDS asymmetry below) |
| Weight pre-swizzle regressed (56.6 -> 39.8) | Coalescing was never the bottleneck; latency/occupancy was |
| 8 concurrent streams scaled 8.0x linearly | 8 x 256 = ~2000 waves finally fills the machine. The GPU was idle, not saturated |

## The only wave-adding lever not yet tried

**In-block split-K** -- what llama.cpp actually does (`mmvq.cu`: `nwarps=8` for simple
`vec_dot` at `ncols_dst=1`, tuned for RDNA4). Grid stays `(N/16,)`, each block becomes
8 waves splitting K, reduced in **LDS** rather than through a global partials buffer
and a second kernel launch. That is 256 x 8 = ~2048 waves, i.e. actually filling the
machine, without the cost that made our global split-K regress.

**Deferred deliberately.** It targets single-stream M=1 decode, which the project has
concluded is the wrong target: prefill is a 16x win and batching solves decode
structurally in Phase 2. Recorded so it is not re-derived from scratch later.

## LDS asymmetry (named trap)

LDS staging **helps prefill and hurts decode**, and the reason is reuse:

- **M > 1 (prefill):** each staged weight tile is reused across M output rows, which
  amortizes the barrier. v3a`s BF16 LDS structure is proven at 793 tok/s.
- **M = 1 (decode):** every weight element is used exactly once. Barriers are pure
  overhead. Measured 36.8 vs 56.6 without.

**Do not port the decode kernel`s no-LDS shape into the prefill GEMM.** The FP8
prefill GEMM must mirror v3a *including* LDS staging.

## Measurement hygiene established this session

- **569.2 GB/s** is the measured achievable bandwidth on gfx1201 (RDNA4 llama.cpp
  fork), not the 640 GB/s spec. Our FP8 decode at ~227 GB/s is **40% of achievable**.
- **`rocm-smi` UMC%% is a duty-cycle sample, not bandwidth.** A 6%% reading alongside
  ~227 GB/s of real traffic is a counter artifact. Do not use it for roofline claims.
- Navi 48 has a **64 MB Infinity Cache**: any microbenchmark that re-reads a small
  weight tile measures LLC, not VRAM, and will report impossible numbers.
- **Graph capture is unavailable from Mojo on AMD** (no `hipGraph` in the Mojo GPU
  stdlib; MAX`s CUDA-graph support sits in the Python pipeline layer). Launch-overhead
  reduction must therefore come from kernel fusion, which has broken correctness twice
  in this repo (QK-norm, gate+up) -- one fusion at a time, teacher-forced after each.
