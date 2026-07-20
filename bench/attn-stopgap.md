# ATTN-STOPGAP — Parallel softmax + PV (2026-07-13)

**Hardware:** GPU 0 Radeon AI PRO R9700 (gfx1201)  
**Env:** hephaestus-wmma-nightly (Mojo 1.0.0b3.dev2026071206)  
**Branch:** `attn-stopgap` (from `v3a-profiling` stack: v3a WMMA + residual)  
**Change:** `attention_kernel_parallel` — block-parallel softmax + 128-way PV; **QK unchanged**.  
Default `attention(..., parallel=True)`. Serial kernel kept as `attention_kernel` / `parallel=False`.

---

## Prefill 512 tok/s (3 reps, M=512, unsynced forward)

| rep | prefill_s | **tok/s** |
|-----|----------:|----------:|
| 1 | 0.369 | **1389** |
| 2 | 0.363 | **1409** |
| 3 | 0.367 | **1395** |

**Mean ≈ 1398 tok/s** (was **~794** with serial softmax/PV + v3a GEMM).

| Threshold | Meaning | Verdict |
|-----------|---------|---------|
| ≥ ~1700 | gate via attention alone; vendored WMMA = Phase-2 | not reached |
| ~1000–1300 | vendored WMMA 1b-urgent | **above this band** |
| **~1400 (actual)** | **strong stopgap; still short of G1b-3 (~2100)** | **between bands** |

**~1.76×** end-to-end vs pre-stopgap. Attention alone does **not** clear G1b-3; remaining headroom is mostly **QK** (see sub-split).

---

## Attention phase sub-split (512 prefill, all 36 layers, host-timed)

Marginal costs from three launches (QK / QK+SM / ALL); each redoes earlier phases:

| Phase | ms (total prefill) | % of fused attention |
|-------|-------------------:|---------------------:|
| **QK** | **82.5** | **58%** |
| **Softmax** | **3.5** | **2%** |
| **PV** | **56.5** | **40%** |
| **ALL (fused)** | **142.6** | 100% |

Previously attention was **~413 ms** (serial softmax+PV). Now **~143 ms** fused.

- Softmax was the serial disaster; parallelized → **negligible**.
- **QK is now the largest slice** of attention (unchanged multi-warp dots).
- PV is second; already column-parallel and bit-near-exact.

---

## VGPR / LDS (mojo `--emit asm` metadata)

| Kernel | VGPR/lane | LDS/wg | Spills |
|--------|----------:|-------:|--------|
| `attention_kernel` (serial) | 22 | 16896 B | 0 |
| **`attention_kernel_parallel`** | **25** | **16912 B** | **0** |

LDS: `scores[MAX_KEYS]` + `q_sh[head_dim]` + `red[NUM_WARPS]` (parallel only).

---

## Correctness

| Check | Result |
|-------|--------|
| Serial vs parallel microbench (SEQ=64, 32 heads) | bit_mismatches **8/262144**, max_abs=**0.0039**, tol_exceed **0** |
| Softmax sum order | expected small F32 diffs; BF16 prob round-trip preserved for PV |
| Teacher-forced decode (4B, 256 steps) | **256/256** |

---

## Decision routing (spec §8)

| Number | Spec routing |
|--------|----------------|
| ≥1700 | vendored WMMA attention = Phase-2 investment |
| 1000–1300 | vendored port **1b-urgent** |
| **~1400 (this run)** | **Stopgap works hard; G1b-3 still needs more.** Sub-split says next attack is **QK** (58% of remaining attention) and/or full flash/WMMA attention for the rest of the gap to ~2100. |

**Recommendation:** treat vendored/flash-style attention as **1b-relevant** (not optional Phase-2 only) if G1b-3 is hard; the stopgap proved softmax/PV were free wins, and **QK is the residual attention bottleneck**.

---

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
NIGHTLY=~/projects/hephaestus-wmma-nightly
KERNELS=~/projects/modular/max/kernels/src
REPO=~/projects/hephaestus

(cd $NIGHTLY && pixi run mojo build -I $KERNELS -I $REPO/src \
  $REPO/src/qwen_ab_bench.mojo -o /tmp/qwen_ab_attn)
/tmp/qwen_ab_attn $REPO/bench/ab_prompt_long_ids.txt 8

(cd $NIGHTLY && pixi run mojo build -I $KERNELS -I $REPO/src \
  $REPO/src/qwen_attn_phase_profile.mojo -o /tmp/attn_phases)
/tmp/attn_phases $REPO/bench/ab_prompt_long_ids.txt
```
