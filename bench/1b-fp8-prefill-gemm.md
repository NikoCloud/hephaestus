# FP8 WMMA prefill GEMM (v3a W8A8) — post attn-stopgap

**Branch:** `fp8-prefill-gemm` → **main**  
**Env:** hephaestus-wmma-nightly · GPU 0 R9700 · free both GPUs before suite  
**Canonical co-measure:** 2026-07-20T00:00 EDT (edge 49°C end)

## Commits (landed on main)

1. `4a8b6bf` — merge v3a-profiling into main  
2. `b8ea868` — FP8 v3a prefill GEMM  
3. `c69b286` — pre-stopgap co-measure (serial attention baseline)  
4. `4da2e63` — merge attn-stopgap  
5. *(this)* — post-stopgap co-measure + main land

## What is on main (must all be present)

| Symbol | Role |
|--------|------|
| `wmma_gemm_kernel_v3a` | BF16 64×64 LDS prefill |
| `attention_kernel_parallel` | Softmax+PV stopgap (default on) |
| `wmma_gemm_fp8_prefill` / `wmma_gemm_kernel_v3a_fp8` | FP8 W8A8 prefill |
| `quantize_act_rows_*` | Per-row act quant |
| `linear_fp8` M>1 | Routes to FP8 v3a |

---

## Co-measured session (attn-stopgap + FP8 GEMM)

**Order:** TF → FP8 prefill×3 → BF16 prefill×3 → FP8 decode×3 → llama Q8_0 pp512+tg128×5  
**Raw:** `/tmp/attn_merge_commeasure/full.log`

### Teacher-forced (merge + kernel gate)

| Prompt | Matches |
|--------|--------:|
| 1 | 256/256 |
| 2 | 246/256 |
| 3 | 246/256 |
| **Total** | **748/768 (97.4%)** ≥95% |

### Prefill 512 (fwd-only tok/s)

| Engine | rep1 | rep2 | rep3 | **mean** |
|--------|-----:|-----:|-----:|---------:|
| **Hephaestus FP8** | 1710.7 | 1695.4 | 1701.8 | **1702.6** |
| **Hephaestus BF16** | 1365.9 | 1377.5 | 1387.4 | **1376.9** |
| llama.cpp Q8_0 pp512 | — | — | — | **7838.8 ± 979** |

### Decode (context)

| Engine | mean tok/s |
|--------|-----------:|
| Hephaestus FP8 (10×256) | **66.1** |
| llama Q8_0 tg128 | **109.76 ± 0.28** |

### Ratios (honest)

| Comparison | Value | Notes |
|------------|------:|-------|
| **FP8 / BF16 end-to-end prefill** | **1.24×** | Same session, same attention |
| **FP8 GEMM / BF16 GEMM (backed out)** | **~1.47×** | Holding non-GEMM; serial-attn era math; stopgap does not change GEMM |
| **FP8 vs llama Q8_0 pp512** | **0.22×** | Precision-matched; **not a win** |
| **vs prior FP8 row-loop (~87)** | **~19.6×** | Real product win |

**Do not** cite llama F16 ROCm pp512 (~1431) as competitive ground truth — known-weak path.  
**Do** cite Q8_0 pp512 (~7839) for competitive position.

### Phantom regression explained

| Measurement | Tree | Prefill tok/s |
|-------------|------|-------------:|
| Co-measure baseline “1400” | v3a + **attn-stopgap** | ~1400 |
| Pre-stopgap FP8 session BF16 | v3a **only** | **764** |
| Historical v3a-profiling | v3a only | ~794 |
| **This session BF16** | v3a + stopgap | **1377** |
| **This session FP8** | v3a + stopgap + FP8 GEMM | **1703** |

The “764 vs 1400” BF16 drop was **missing attention**, not a broken GEMM.  
FP8/BF16 **1.13× end-to-end without stopgap** was attention-dominated; GEMM-only ~**1.47×**.

### GEMM ratio headroom (next, not this PR)

| Item | Note |
|------|------|
| Ideal FP8 compute edge | up to ~2× on pure matmul |
| Measured GEMM ~1.47× | Below 2× |
| Likely tax | **Per-matmul activation quant launch** (BF16 pays 0) |
| Action | Flag for follow-up; **do not profile until merge measured** (done here) |

---

## Implementation summary

| Piece | Role |
|-------|------|
| `wmma_gemm_kernel_v3a_fp8` | LDS 64×64, dual-scale epilogue, edge-mask M |
| `quantize_act_rows_*` | Per-row absmax (no pad-to-16) |
| Decode path | Unchanged W8A8 gemv |

Epilogue: `C[m,n] = act_scale[m] * weight_scale[n] * acc` (+ residual).

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus  # on main after land

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8_pre
mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench.mojo -o /tmp/qwen_ab_bf16

for r in 1 2 3; do /tmp/qwen_ab_fp8_pre bench/ab_prompt_long_ids.txt 4; done
for r in 1 2 3; do /tmp/qwen_ab_bf16 bench/ab_prompt_long_ids.txt 4; done

~/projects/llama.cpp/build/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf -p 512 -n 128 -r 5 -d 0
```
