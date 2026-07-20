# HANDOFF — Phase 1b close-out + Phase 2 baseline (for GLM -> Grok prompt)

**Date:** 2026-07-20. `main` = 94db405, in sync, compiles, contains all four lines
(v3a, parallel attention, FP8 decode, FP8 prefill).

## 1. Phase 1b: final measured state (all co-measured, same session, GPU0 free)

| | Hephaestus | llama.cpp Q8_0 ROCm | ratio |
|---|---|---|---|
| Prefill pp512 | **1702.6** FP8 / 1376.9 BF16 | 7839 | **0.22x** |
| Decode tg | **66.1** FP8 / 62.37 BF16 | 109.8 | **0.60x** |

- FP8 prefill: **87 -> 1703 tok/s = 19.6x** (the row-looped gemv -> v3a-shaped FP8 WMMA GEMM)
- **FP8 beats BF16 on BOTH regimes**: prefill 1.24x end-to-end / ~1.47x at the GEMM;
  decode 66.1 vs 62.37 (first time -- the halved-weight-bytes win on the memory-bound path)
- Teacher-forced argmax: **748/768 = 97.4%** (back to the historical figure)
- Ragged M=137: correct, no cliff
- G1b-4 (no weight dequant in any hot loop): **MET**, verified at the kernel source

**Proven claims, all defensible:** native FP8 E4M3 on gfx1201 via direct
`llvm_intrinsic` (ROCm returns NotImplementedError; vLLM silently dequantizes to FP32;
AITER has zero tuned configs for this card) - 97.4% argmax parity - FP8 > BF16 on both
regimes - no silent dequant anywhere.

**Honest competitive line:** behind on speed (0.22x prefill / 0.60x decode), alone on
the FP8 path. The gap is kernel maturity, not architecture. Not a gate -- prefill
parity was retired deliberately.

## 2. Corrected batching economics (NEW -- read before scoping Phase 2)

A common intuition -- "dequant/translation/format costs are hidden at M=1 and will
surface under concurrency" -- is **inverted**. Correct arithmetic:

Batched GEMM intensity = `2M/b` FLOP/byte (M = batch, b = bytes/weight).
FP8 roofline crossover = 383 TFLOP / 569 GB/s = **~673 FLOP/byte**.

| M | intensity | regime |
|---|---|---|
| 1 | 2 | deeply memory-bound |
| 8 | 16 | **still ~40x from compute-bound** |
| ~336 | ~673 | crossover |

**So 6-12 concurrent agents remains firmly memory-bound.** Compute-bound arrives in
the hundreds, not the teens.

**What amortizes away with batching (does NOT surface):**
- Dequant happens once per weight *read*, reused across all M rows -> cost/token = cost/M
- Translation-layer and launch overhead is per-forward -> divided by M
- File format is load-time -> invisible at any M

**What actually persists or compounds:**
- **Bytes-per-weight** -- 2x at every M. Already visible (FP8 decode now beats BF16)
- **Capacity** -- FP8 frees VRAM for KV cache, and KV cache caps concurrent slots.
  This one genuinely compounds with agent count
- At batch ~300+, FP8`s 2x matrix throughput becomes binding

**The size of the Phase 2 prize:**
- One fused forward reads 4.02 GB / 569 GB/s = **7.06 ms**, producing **M** tokens
- At M=8: **~1133 tok/s aggregate ceiling**
- The process-concurrency probe measured **453** (8 processes = 8 weight copies,
  amortizing nothing) -> **fused batching is worth ~2.5x over the probe**
- Mechanism is amortization + utilization (M=1 uses ~3% of wave slots), NOT a regime change
- llama Q8 at M=8 has a ~1058 ceiling (4.3 GB) -> **~7% byte edge only.** Any larger
  advantage must be earned in implementation (fused + paged KV vs their slot cache)

## 3. Settled -- do not reopen

- **Shape-matched decode question: ANSWERED by measurement.** FP8 > BF16 on both
  regimes, so there is no BF16-compute compromise to rule on. The pillar holds.
- **M=1 decode kernel tuning: dead end.** grid=(N/16,) gives 256 waves (q_proj) / 64
  (k,v) on a machine hosting thousands; Little`s Law says per-wave tuning cannot reach
  the roofline. Split-K, LDS staging, weight pre-swizzle all measured regressions.
- **FP8 WMMA reachability**: proven 2026-07-13 (exp3g).
- **Prefill gate (1.5x llama Q8)**: retired as mis-specified.

## 4. UNMEASURED -- must not be claimed publicly

**llama.cpp batched serving has never been measured.** The "we win at concurrency"
claim has now been asserted twice (Frank, Kimi) on the strength of an 8-process probe
that used 8 weight copies -- not a product config, and not how llama serves concurrency.

**Blocker found:** the ROCm build contains **only `llama-bench`**. There is no
`llama-server` and no `llama-batched-bench` binary. They must be built first:
`cmake --build build --target llama-batched-bench -j` (and/or `llama-server`).

Also unmeasured: the GEMM`s 1.47x -> ~2.0x gap, attributed (not profiled) to the
per-matmul activation-quant launch tax.

## 5. Ranked next tasks

**1. Measure llama.cpp batched throughput -- BEFORE building Phase 2.**
Build `llama-batched-bench`, sweep batch 1/2/4/8/16 on Q8_0, ROCm, GPU0, same session.
Produces llama`s concurrency scaling curve = the Phase 2 baseline.
**Rationale: this is exactly the 11,532 phantom-gate lesson.** We spent a week
optimizing toward an extrapolated prefill target that was wrong by 5.5x. Phase 2`s
entire premise is "we beat llama at concurrency." Measure the target before building
toward it. Cheap, and it either validates or kills the Phase 2 narrative up front.

**2. Phase 1b writeup / Modular forum post.** Lead with the **toolchain contribution**,
not the benchmark: Mojo`s `mma_amd_rdna` emits RDNA3 16-element fragments on gfx12; the
correct ABI is 8-element (`<2 x i32>` FP8, `<8 x i16>` BF16) with **no i1 operands** on
the f32-accum forms, verified against `IntrinsicsAMDGPU.td` and the clang gfx12
builtins (`experiments/exp3g_RESULTS.md`). Upstream issue modular/modular#6722 already
has a drafted comment at `.agent/notes/upstream-6722-comment.md`. The engine is proof
the fix works. Then state numbers honestly. **Do not include any batching claim.**

**3. Act-quant fusion** (the 1.47 -> 2.0 GEMM gap). One fusion at a time,
teacher-forced re-verified after each -- fusion has broken correctness twice here
(QK-norm, gate+up).

**4. Phase 2 scoping**: fused batching (one weight copy, M sequences) + paged KV.
Ceiling maths in section 2.

## 6. Traps that carry forward

- **LDS asymmetry**: keep for prefill (M>1 reuse amortizes barriers), drop for decode
- **Infinity Cache**: 64 MB LLC -- microbenchmarks must rotate >64 MB or they measure cache
- **`rocm-smi` UMC%% is a duty-cycle sample, not bandwidth.** Achievable is 569.2 GB/s
- **Backend matters**: llama F16 prefill is 1431 (ROCm) vs 8134 (Vulkan) -- 5.7x. Q8 is
  ~identical across backends (7839 / 7688). Always state the backend
- **rocprof is NOT installed** (only rocprofiler-register)
- **Merge promptly.** Branch-only work has manufactured two phantom problems in two days
