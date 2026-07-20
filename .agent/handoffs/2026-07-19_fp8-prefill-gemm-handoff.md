# HANDOFF — FP8 WMMA Prefill GEMM (for GLM -> Grok prompt construction)

**Date:** 2026-07-19. **Audience:** GLM (prompt author), Grok (implementer).
**One-line task:** merge v3a into main, then build the FP8 WMMA prefill GEMM on top.

## 1. Why this, and why now

FP8 prefill is **87 tok/s**. BF16 v3a prefill is **~1400**. That 16x gap is a
**missing kernel**, not a dtype problem: `linear_fp8` row-loops the gemv for M>1;
the FP8 matrix path for prefill was never built.

- Prefill is **compute-bound**, which is where FP8 genuinely doubles throughput
  (383 vs 191 TFLOPS). Decode is not (see below).
- **Pillar-neutral**: no BF16 compute, no upconversion, no architecture ruling needed.
  It can proceed before the shape-matched decode decision is made.
- It is the **Odysseus TTFT pain** directly.

**Do not spend effort on M=1 decode.** It is structurally under-parallelized:
`grid=(N/16,)` launches 256 waves (q_proj) / 64 (k,v) on a machine hosting thousands.
By Little`s Law, saturating 569 GB/s needs ~228 KB in flight; 256 waves gives ~2 KB.
Split-K, LDS staging and weight pre-swizzle all regressed. Batching solves decode in
Phase 2. Full derivation: `.agent/notes/2026-07-19_decode-wave-arithmetic.md`.

## 2. Repo state

- `main` = **132cb8c**, in sync with origin, **compiles clean**. Contains the whole
  FP8 decode line, `exp3g` (FP8 WMMA proof), corrected docs, wave-arithmetic note.
- `main` does **NOT** contain v3a (the 64x64 tile). That lives on **`v3a-profiling`**
  (superset: v3a + o_proj/down_proj WMMA residual routing + profiling).
- All branches are pushed to origin. Nothing is single-copy.

## 3. STEP 0 — merge `v3a-profiling` into main (union map)

Both `wmma_gfx12.mojo` and `kernels.mojo` conflict. It is **not** concatenation --
both branches edited shared dispatch functions.

| symbol | take from |
|---|---|
| `WMMA_TILE/LANES/FRAG`, `LDS_STRIDE`, `wmma_bf16`, `wmma_gemm_kernel_v1`, `wmma_gemm_kernel`, `wmma_gemm_bf16_v1` | identical either side |
| `BM/BN/BK`, `V3A_THREADS/WAVES/SC` | **v3a** |
| `wmma_gemm_kernel_v3a`, `wmma_gemm_kernel_residual`, `wmma_gemm_bf16_v3a`, `wmma_gemm_bf16_residual`, `wmma_gemm_bf16_v3a_residual` | **v3a** |
| `FP8`, `FP8_E4M3_MAX`, `wmma_fp8`, `wmma_fp8_decode_kernel`, `wmma_gemm_fp8_decode` | **main** |
| **`wmma_gemm_bf16`** (differs) | **v3a** -- superset dispatch `M,N%64 -> v3a else v2` |
| kernels.mojo `quantize_act_*`, `gemv_fp8`, `linear_fp8`, `linear_add_residual_fp8`, `embed_*_fp8` | **main** |
| kernels.mojo `linear_add_residual` o/down -> WMMA routing (commit 70c4074) | **v3a** |

**DANGER:** the last two rows are functions *both* sides edited. A bad resolution
does not fail to compile -- it silently mis-routes. Verify with §6 gates, not a build.

**Bonus:** `70c4074` routes o_proj/down_proj through WMMA with fused residual-add --
the Amdahl fix (o+down are 35%% of linear FLOPs and were still naive). Comes free.

## 4. STEP 1 — the FP8 prefill GEMM: three new pieces

Everything else is a dtype substitution into the proven v3a structure.

1. **Per-row activation quant.** `A[M,K]` BF16 -> E4M3, one F32 scale per row ->
   `act_scale[M]`. Same `__shfl_xor` butterfly absmax as the decode quant, with a row
   index added. (Decode`s existing quant is single-row + pad-to-16; prefill needs M
   real rows, no padding.)
2. **2D rank-1 scale epilogue.** `C[m,n] = act_scale[m] * w_scale[n] * acc[m,n]`.
   At the G1b-0 store mapping each lane has `m = MB + w*16 + (l/16)*8 + j` and
   `n = NB + sc*16 + l%16`, so it indexes `act_scale[m]` per `j` instead of decode`s
   single broadcast scalar. Slots in exactly where v3a`s residual-add sits.
3. **LDS staging KEPT.** See trap §5.1.

**Geometry:** v3a 64x64, 4 waves, `V3A_THREADS=128`, BK=16, G1b-0 lane mappings.
**Packing:** A and B as `bitcast[int32,2]` of 8 E4M3 bytes ->
`llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8`, F32 accumulator. Proven in the decode kernel.
**Wire into:** `linear_fp8` M>1 path, replacing the row-loop.

## 5. Traps (each has already cost a cycle)

1. **LDS asymmetry.** LDS **helps prefill** (M>1 reuses each staged tile across output
   rows, amortizing the barrier -- v3a proven at 793 tok/s) and **hurts decode** (M=1
   has no reuse; measured 36.8 vs 56.6). **Do not port the decode kernel no-LDS shape
   into prefill.**
2. **Ragged M.** v3a asserts `M%64==0`. All N qualify (4096/1024/2560/9728/151936 are
   all divisible by 64) but **M = sequence length is arbitrary** -- a 137-token prompt
   is ragged. ESDMAX measured a **3x cliff** on ragged remainders. Edge-masking is
   mandatory and **must be tested at a non-multiple of 64**, or it passes at M=512 and
   falls over in production.
3. **Infinity Cache.** Navi 48 has 64 MB LLC. Any microbenchmark re-reading a small
   tile measures LLC, not VRAM, and reports impossible numbers. Rotate over >64 MB.
4. **`rocm-smi` UMC%% is a duty-cycle sample, not bandwidth.** Do not use it for
   roofline claims. Achievable bandwidth on gfx1201 is **569.2 GB/s** (not the 640 spec).
5. **Settled regressions -- do not re-run:** decode LDS staging, weight pre-swizzle,
   global split-K. All measured, all worse.

## 6. Verification gates, in order

1. **Build** (the merged tree must compile; ~3s).
2. **Merge validation** -- this is what catches a silent mis-route:
   - teacher-forced FP8 4B must still be **97.4%** (748/768) -> FP8 decode routing survived
   - M=512 prefill bench must still be **~1400** BF16 -> v3a dispatch survived
3. **FP8 prefill correctness**: CPU F32 reference on one 64x64 tile, then layer-diff.
4. **Teacher-forced again** after wiring (97.4% must hold).
5. **Prefill bench**, M=512, 3 reps, co-measured.
6. **Ragged test**: M=137 (or any non-multiple of 64) must be correct, not just fast.

**Correctness before any speed claim.** FP8 divergence from BF16 is expected
arithmetic, not a bug -- the bar is argmax/perplexity preserved, never bit-identity.

## 7. Targets

| | tok/s |
|---|---|
| FP8 prefill now (row-looped) | 87 |
| BF16 v3a prefill | ~1400 |
| Target: kernel parity | ~1400 |
| Target: + FP8 compute edge | up to ~2800 |
| llama.cpp Vulkan | 8134 -- **not a gate** (retired; residual is kernel maturity) |

## 8. Starting material and environment

- Kimi drafts (design-complete, **never compiled**): `/tmp/quant_rows.mojo`,
  `/tmp/fp8_v3a_gemm.mojo` on CachyOS. Use or discard.
- v3a source: `wmma_gemm_kernel_v3a` in `v3a-profiling:src/hephaestus/wmma_gfx12.mojo`.
- FP8 packing reference: `wmma_fp8` / `wmma_fp8_decode_kernel` on `main`.
- **GPU 0 only**, `HIP_VISIBLE_DEVICES=0`, check `rocm-smi` first.
- **rocprof is NOT installed** (only `rocprofiler-register`). Any profiling step needs
  it installed first, or use `mojo build --emit asm` metadata.
- Branch from `main` (single branch -- the whole point of step 0).

## 9. Out of scope

Fused batching (Phase 2). M=1 decode optimization (structural dead end). Chasing
llama`s 8134 prefill (not a gate). Kernel fusion for launch reduction -- it has broken
correctness twice here (QK-norm, gate+up); separate task, one fusion at a time.
