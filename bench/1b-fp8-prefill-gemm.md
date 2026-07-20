# FP8 WMMA prefill GEMM (v3a W8A8)

**Date:** 2026-07-19 (implementation) / **co-measure re-run 2026-07-19T23:35–23:37 EDT**  
**Branch:** `fp8-prefill-gemm` (from main after v3a merge)  
**Hardware:** GPU 0 R9700 gfx1201; both GPUs free before suite  
**Env:** hephaestus-wmma-nightly  
**GPU end temps:** edge 52°C / junction 56°C / mem 62°C (GPU0)

## Commits

1. **`4a8b6bf`** — merge `v3a-profiling` into main (union map)
2. **`b8ea868`** — FP8 v3a prefill GEMM + per-row quant + dual-scale epilogue

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

---

## Co-measured session (canonical numbers)

**Procedure:** rebuild binaries → free GPUs → Hephaestus FP8/BF16 prefill → ragged → TF → **immediately** `llama-bench` Q8_0 then F16 on same GPU (`HIP_VISIBLE_DEVICES=0`, ROCm backend).  
Raw log: `/tmp/fp8_prefill_commeasure/full.log`.

### Hephaestus

| Test | rep1 | rep2 | rep3 | mean |
|------|-----:|-----:|-----:|-----:|
| **FP8 prefill 512** (fwd tok/s) | 828.3 | 882.8 | 884.7 | **865.3** (warm 2–3: **883.7**) |
| **BF16 prefill 512** (fwd tok/s) | 767.6 | 756.3 | 769.1 | **764.4** |
| FP8 ragged M=137 prefill | — | — | — | **1161.3** (smoke) |

| Teacher-forced FP8 (256 steps, oracle feed) | Matches |
|---------------------------------------------|---------|
| prompt1 | 256/256 |
| prompt2 | 244/256 |
| prompt3 | 242/256 |
| **Total** | **742/768 (96.6%)** ≥95% |

### llama.cpp (same session, ROCm, build `33ca0dcb9`)

| Model | test | tok/s |
|-------|------|------:|
| **Q8_0** | pp512 | **7745 ± 1177** |
| **Q8_0** | tg128 | **109.80 ± 0.36** |
| **F16** | pp512 | **1431 ± 50** |
| **F16** | tg128 | **61.99 ± 0.14** |

**Backend note:** prior cross-day F16 prefill **~8134** was **Vulkan**. This co-measure is **ROCm**, where F16 pp512 is only **~1431**. Q8_0 pp512 remains high (~7745). Do not mix Vulkan and ROCm prefill numbers.

### Ratios (this session only)

| Comparison | Heph | llama (ROCm) | Ratio |
|------------|-----:|-------------:|------:|
| Prefill FP8 vs BF16 (Heph) | 865 | 764 | **1.13×** |
| Prefill FP8 warm vs BF16 | 884 | 764 | **1.16×** |
| Prefill FP8 vs llama F16 pp512 | 865 | 1431 | 0.60× |
| Prefill FP8 vs llama Q8 pp512 | 865 | 7745 | 0.11× (not a gate) |

llama prefill is **context**, not a gate (handoff: residual is kernel maturity / attention).

---

## Merge validation (earlier same day, pre co-measure)

| Gate | Result |
|------|--------|
| Build | clean |
| FP8 TF | 748/768 (97.4%) |
| BF16 prefill 512 | ~787 tok/s (v3a-profiling baseline ~794) |

## Implementation gates (earlier same day)

| Gate | Result |
|------|--------|
| Prefill M=512 (warm) | ~885 |
| Ragged M=137 | runs |
| BF16 routing after FP8 kernel | intact |

### Speed rollup

| Path | Prefill tok/s (M=512) |
|------|---------------------:|
| FP8 row-loop (before kernel) | ~87 |
| BF16 v3a (this co-measure) | **764** |
| **FP8 v3a W8A8 (this co-measure, all reps)** | **865** |
| **FP8 warm (reps 2–3)** | **884** |
| Target kernel parity | ~1400 |
| llama ROCm F16 / Q8 pp512 | 1431 / 7745 — **not gates** |

**~10×** over previous FP8 row-loop; **~1.13×** Hephaestus BF16 in the same thermal window.

## Correctness note

FP8 vs BF16 is expected arithmetic divergence. Argmax TF stays ≥95%.  
No bit-identity claimed.

## GPU condition takeaway

| Variable | This session |
|----------|----------------|
| Idle before suite | yes (0% both GPUs) |
| llama backend | ROCm (not Vulkan) |
| End edge temp | 52°C |
| Heph FP8 prefill cold rep1 | 828 (then ~884) |
| llama Q8 pp512 σ | large (±1177) — first-rep / clock ramp |

Re-running llama in the same session is required for fair ratios; historical Vulkan 8k prefill is a different stack.

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8_pre
mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench.mojo -o /tmp/qwen_ab_bf16

for r in 1 2 3; do /tmp/qwen_ab_fp8_pre bench/ab_prompt_long_ids.txt 4; done
for r in 1 2 3; do /tmp/qwen_ab_bf16 bench/ab_prompt_long_ids.txt 4; done

~/projects/llama.cpp/build/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf -p 512 -n 128 -r 5 -d 0
~/projects/llama.cpp/build/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf -p 512 -n 128 -r 5 -d 0
```
