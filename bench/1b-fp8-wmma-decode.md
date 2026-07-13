# G1b-2 / G1b-4 — FP8 W8A8 WMMA decode

**Date:** 2026-07-13  
**Branch:** `fp8-wmma-decode` (from `fp8-decode`)  
**Hardware:** GPU 0 = AMD Radeon AI PRO R9700 (gfx1201); both GPUs free before co-measure  
**Env:** `~/projects/hephaestus-wmma-nightly` (Mojo `1.0.0b3.dev2026071206`)

## Thesis

RDNA4 has **no scalar FP8 dot**. FP8 compute is **WMMA-only**, both operands 8-bit. The previous `gemv_fp8` dequant path (FP8→F32 × BF16 in a vector gemv) never touched FP8 matrix hardware — off-thesis and off G1b-4.

### Decode path (this commit)

1. **Act quant (per-token absmax):** BF16 `x[1,K]` → `scale_act = max(|x|)/448`, `x_fp8 = cast(x/scale_act)`, pad to **`[16,K]`** zeros.
2. **FP8 WMMA:** `C[16,N] = A_fp8[16,K] @ W_fp8[N,K]^T` via  
   `llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8` (exp3g PASS; A/B = `v2i32`, C/D = `v8f32`).
3. **Dual-scale epilogue:** `out[n] = scale_act * weight_scale[n] * acc[0,n]` (+ residual); cast BF16 or F32 (lm_head).
4. **Tiling:** grid = `N/16` blocks; extract **row 0** only. Shared quant for q/k/v and gate/up.

Weights stay FP8 in the arena for the entire matmul. **No weight dequant in the K-loop.**

## G1b-4 grep (weight hot path)

| Location | F32 usage | OK? |
|----------|-----------|-----|
| `wmma_fp8_decode_kernel` K-loop | WMMA F32 **accumulator only**; A/B packed FP8→`v2i32` | **Yes** |
| Dual-scale epilogue | `acc * scales` (+ residual cast) | **Yes** (hardware C is F32) |
| `quantize_act_absmax_kernel` | BF16→F32 for absmax + quant → FP8 | **Yes** (activation path, not weight dequant) |
| `embed_fp8_kernel` | FP8→F32 on **gather** into BF16 residual | **Yes** (embed, not matmul) |
| Attention / RoPE / silu | BF16↔F32 elementwise | **Yes** (unchanged non-FP8 ops) |

**Removed:** MAX `gemv_kernel_vector` dequant gemv (`FP8` weights cast to F32 × BF16 acts).

**G1b-4: MET** for the matmul path (no FP8 weight→F32 dequant for compute).

## Co-measured decode (G1b-2)

Protocol: free GPUs → Hephaestus FP8 WMMA 10-tok × 256 gen × 3 → immediate `llama-bench -p 0 -n 128` Q8_0 on GPU 0.

### Hephaestus FP8 W8A8 WMMA (forward-only)

| rep | decode tok/s (fwd) | +argmax | total_s |
|----:|-------------------:|--------:|--------:|
| 1 | 56.481 | 55.512 | 4.727 |
| 2 | 56.683 | 55.730 | 4.700 |
| 3 | 56.513 | 55.542 | 4.716 |
| **mean** | **56.56** | **55.59** | **4.714** |

(Short-context 10×128 earlier in session: ~59.3 tok/s fwd — attention cheaper.)

### llama.cpp Q8_0 (same session, ROCm)

| test | tok/s |
|------|------:|
| tg128 (5 reps) | **109.61 ± 0.62** |

### Ratios

| Reference | tok/s | Heph / ref |
|-----------|------:|-----------:|
| **llama Q8_0 (this co-measure)** | **109.61** | **0.52×** |
| Prior Vulkan Q8 baseline | 127.25 | 0.44× |
| Hephaestus BF16 co-measure | 62.37 | 0.91× |
| Roofline ceiling (doc) | ~159 | 0.36× |

**G1b-2: NOT MET** (need ≥ gate ~110–127; measured **56.6**).

## Correctness

| Check | Result |
|-------|--------|
| Loader 651 tensors / scale pairing | PASS (unchanged) |
| Decode runs (no HIP errors) | PASS |
| exp3g FP8 WMMA all-ones → 16.0 | Prior PASS (intrinsic) |
| Tiny layer-diff / teacher-forced argmax ≥95% non-ties | **Not run this session** — follow-up |
| Expected first divergence | layer0 `q_proj` (quantization noise) |

## Why not ≥ gate yet

1. **Decode still pays 15/16 idle WMMA rows** (pad) + act quant every unique activation (4 quants/layer after sharing).
2. **Direct-global FP8 fragment loads** (v1 geometry); BF16 prefill already moved to LDS v2 — same optimization not yet applied to FP8.
3. **Attention / launches** still dominate late positions (56 vs 59 tok/s at 256 vs 128 gen).
4. **Bandwidth win is real in the weight stream**, but quant + pad + non-matmul work keep wall time near BF16.

Next: LDS coalesced FP8 fragment load; fuse quant into fewer launches; profile one step (rocprof).

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8_wmma

for r in 1 2 3; do /tmp/qwen_ab_fp8_wmma bench/ab_prompt_short_ids.txt 256; done
~/projects/llama.cpp/build/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf -p 0 -n 128 -r 5 -d 0
```

## Files

| File | Change |
|------|--------|
| `src/hephaestus/wmma_gfx12.mojo` | `wmma_fp8`, `wmma_fp8_decode_kernel`, `wmma_gemm_fp8_decode` |
| `src/hephaestus/kernels.mojo` | Replace dequant gemv with quant + W8A8 WMMA; shared-quant flag |
| `src/hephaestus/forward.mojo` | `a_fp8_pad` / `act_scale` scratch; qkv/gate-up quant share |
| `bench/1b-fp8-wmma-decode.md` | This writeup |
