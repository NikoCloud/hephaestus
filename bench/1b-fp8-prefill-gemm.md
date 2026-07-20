# FP8 WMMA prefill GEMM (v3a W8A8)

**Date:** 2026-07-19  
**Branch:** `fp8-prefill-gemm` (from main after v3a merge)  
**Hardware:** GPU 0 R9700 gfx1201  
**Env:** hephaestus-wmma-nightly

## Commits

1. **`4a8b6bf`** — merge `v3a-profiling` into main (union map)
2. **this** — FP8 v3a prefill GEMM + per-row quant + dual-scale epilogue

## What landed

| Piece | Role |
|-------|------|
| `wmma_gemm_kernel_v3a_fp8` | 64×64 LDS GEMM, FP8 packing (`v2i32`), edge-mask M |
| `wmma_gemm_fp8_prefill` | Host launch; grid `(N/64, ceildiv(M,64))` |
| `quantize_act_rows_*` | Per-row absmax BF16→E4M3 + `act_scale[M]` |
| `linear_fp8` M>1 | Quant + v3a FP8 (replaces row-looped gemv) |
| `linear_add_residual_fp8` M>1 | Same with fused residual |
| `Activations` | `a_fp8_pad = max_seq×inter`, `act_scale = max_seq` |

Epilogue: `C[m,n] = act_scale[m] * weight_scale[n] * acc` (+ residual).  
LDS **kept** (prefill reuses tiles). Decode path unchanged.

## Merge validation (commit 1)

| Gate | Result |
|------|--------|
| Build | clean |
| FP8 TF (p1+p2+p3) | **748/768 (97.4%)** |
| BF16 prefill 512 | **~787 tok/s** (matches v3a-profiling ~794) |

## After FP8 prefill kernel (commit 2)

| Gate | Result |
|------|--------|
| Build | clean |
| FP8 TF p1 | **256/256** |
| FP8 TF p2/p3 | 244 + 242 = **486/512** |
| **TF total** | **742/768 (96.6%)** ≥95% |
| Prefill M=512 ×3 | **884.7 ± 1.4 tok/s** |
| Ragged M=137 | **runs** (1181 tok/s smoke; edge mask) |
| BF16 prefill still | **785** (routing intact) |

### Speed summary

| Path | Prefill tok/s (M=512) |
|------|---------------------:|
| FP8 row-loop (before) | ~87 |
| BF16 v3a (control) | ~785–794 |
| **FP8 v3a W8A8 (this)** | **~885** |
| Target kernel parity | ~1400 |
| llama 8134 | **not a gate** |

**~10×** over previous FP8 row-loop; slightly above BF16 v3a on this machine (FP8 compute edge, attention still dominates ~65%).

## Correctness note

FP8 vs BF16 is expected arithmetic divergence. Argmax TF stays ≥95%.  
No bit-identity claimed.

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8_pre
for r in 1 2 3; do /tmp/qwen_ab_fp8_pre bench/ab_prompt_long_ids.txt 4; done
# ragged
python3 -c "open('/tmp/ids_137.txt','w').write('\\n'.join(map(str,range(137)))+'\\n')"
/tmp/qwen_ab_fp8_pre /tmp/ids_137.txt 2
```
