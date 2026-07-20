# G1b-2 — FP8 E4M3 decode path (co-measured)

**Date:** 2026-07-13  
**Branch:** `fp8-decode`  
**Hardware:** GPU 0 = AMD Radeon AI PRO R9700 (gfx1201), both GPUs free before co-measure  
**Env:** `~/projects/hephaestus-wmma-nightly` (Mojo `1.0.0b3.dev2026071206`)

## What landed

| Workstream | Status |
|------------|--------|
| FP8 loader (mixed-dtype arena: F8_E4M3 + F32 scales + BF16 norms) | **Done** — 651 tensors, asserts pass |
| `stage_weights.py` dtype field | **Already present** — zero changes |
| `gemv_fp8` decode kernel (M=1) + scale epilogue | **Done** — MAX `gemv_kernel_vector` @ FP8 simd_width=16 + scale |
| `forward_fp8` integration | **Done** — all projs + embed + lm_head |
| FP8 WMMA prefill GEMM | **Out of scope** (per plan) |

Staged blob: `staged/qwen3-4b-fp8.{weights,offsets}`  
- 253 × F8_E4M3, 253 × F32 scales, 145 × BF16 norms = **4.03 GB** (vs 8.04 GB BF16)

## Co-measured decode (G1b-2 gate)

Protocol: free both GPUs → Hephaestus FP8 10-tok prompt × 256 gen × 3 reps → immediately `llama-bench -p 0 -n 128` on the same GPU with Q8_0 GGUF.

### Hephaestus FP8 (forward-pass only)

| rep | decode tok/s (fwd) | decode +argmax | total_s |
|----:|-------------------:|---------------:|--------:|
| 1 | 55.024 | 54.108 | 4.865 |
| 2 | 55.032 | 54.125 | 4.862 |
| 3 | 55.088 | 54.196 | 4.856 |
| **mean** | **55.05** | **54.14** | **4.861** |

Config: `qwen_ab_bench_fp8.mojo`, `staged/qwen3-4b-fp8`, `ab_prompt_short_ids.txt`, n_new=256.

Short-context note: 10×128 gen measured **~61.6 tok/s** fwd-only (attention cheaper at short past). Gate table uses 256-gen to match BF16 co-measure methodology.

### llama.cpp Q8_0 (same session, ROCm backend)

| test | tok/s |
|------|------:|
| tg128 (5 reps) | **109.53 ± 0.69** |

```
llama-bench -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf -p 0 -n 128 -r 5 -d 0
# backend: ROCm, build 33ca0dcb9
```

**Note:** Prior Q8 baseline **127.25 ± 0.21** was Vulkan (`bench/commeasured-baseline.md`). This co-measure used ROCm and got **109.5**. Gate ratio below uses **this session’s** llama number; also report vs the published Vulkan 127 for continuity.

### Ratios

| Reference | tok/s | Heph FP8 / ref |
|-----------|------:|---------------:|
| **llama Q8_0 (this co-measure, ROCm)** | **109.53** | **0.50×** |
| llama Q8_0 (prior Vulkan baseline) | 127.25 | 0.43× |
| Hephaestus BF16 (prior co-measure, 256-gen) | 62.37 | 0.88× (FP8 slower here*) |

\*FP8 at 55 tok/s on 256-gen is slightly **below** BF16 62 — not the expected ~2× bandwidth win. See “Why not 2×” below.

## G1b-2 gate

| Criterion | Target | Measured | Met? |
|-----------|--------|----------|------|
| Match llama.cpp Q8_0 decode | ≥ ~127 (Vulkan) / ≥ this-session Q8 | **55.0** vs **109.5** (ROCm) | **NO** |
| Ratio to co-measured Q8 | ≥ 1.0× | **0.50×** | **NO** |

**G1b-2: NOT MET.** Path is integrated and correct-loading; kernel still leaves ~2× on the table vs Q8.

## Why not ~2× BF16 (physics expectation)

1. **FP8 gemv effective BW ~150 GB/s** on a 2560×2560 microbench (weight bytes / wall) — far from ~960 GB/s peak. Path is still latency/convert bound, not pure weight-bandwidth bound.
2. **MAX `gemv_gpu` split-K cannot be used as-is** for mixed BF16 acts + FP8 weights: it `rebind`s weight vectors to `a_type` (bit reinterpret, not cast). We use `gemv_kernel_vector` with proper casts + scale epilogue instead.
3. **BF16 gemv uses AMD `v_dot2_f32_bf16`**; FP8 goes cast→FMA. Conversion + missing fused DP can cancel the 2× weight traffic cut.
4. **Attention / norms / launches** grow with decode position (55 tok/s @ 256-gen vs 62 @ 128-gen) and are dtype-agnostic.

## Correctness (smoke)

| Check | Result |
|-------|--------|
| Loader: 651 tensors, contiguous offsets, every F8 has `_scale` F32 | **PASS** |
| 4B FP8 load + decode runs without HIP errors | **PASS** |
| Microbench `scale * sum(fp8(1)*bf16(1))` for N=K=2560 → 1280 | **PASS** |
| Layer-diff / teacher-forced vs BF16 oracle | **Not run this session** (follow-up) |

Wider FP8 tolerance for layer-diff (when run): `1e-3 + 5e-2 * |ref|`; expect first divergence at `q_proj`.

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8

# 3 reps (GPUs free)
for r in 1 2 3; do /tmp/qwen_ab_fp8 bench/ab_prompt_short_ids.txt 256; done

# Immediate co-measure
~/projects/llama.cpp/build/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf \
  -p 0 -n 128 -r 5 -d 0
```

## Next kernel work (toward 2×)

- Port AMD split-K with **cast** (not rebind) for mixed BF16×FP8; non-temporal weight loads; tile_n selection from `_amd_gemv_config`.
- Confirm whether ROCm FP8→F32 is hardware-assisted; if not, evaluate packed conversion paths.
- Profile one decode step (rocprof) to split gemv vs attention vs launch overhead.
- Optional: PDL across the full forward graph (not just gemv).

## Files touched

- `src/hephaestus/model_fp8.mojo` — FP8 weight/scale structs  
- `src/hephaestus/loader.mojo` — `WeightArenaBytes`, `load_arena_bytes`, `build_weights_fp8`  
- `src/hephaestus/kernels.mojo` — `gemv_fp8`, `linear_fp8`, `linear_add_residual_fp8`, `embed_lookup_fp8`  
- `src/hephaestus/forward.mojo` — `forward_fp8`  
- `src/qwen_ab_bench_fp8.mojo` — A/B bench entry  
- `docs/fp8-checkpoint-format.md` — format notes  
- `staged/qwen3-4b-fp8.*` — staged checkpoint  
