# G1b-2 / G1b-4 — FP8 W8A8 WMMA decode (correctness + LDS)

**Date:** 2026-07-13  
**Branch:** `fp8-wmma-decode`  
**Hardware:** GPU 0 = R9700 gfx1201; both GPUs free before co-measure  
**Env:** `~/projects/hephaestus-wmma-nightly` (Mojo `dev2026071206`)

---

## Fix 1 — Correctness (DONE FIRST)

### Teacher-forced decode (4B, 256 steps, oracle tokens)

Binary: `src/qwen_teacher_forced_decode_fp8.mojo` → `/tmp/tf_fp8`  
Weights: `staged/qwen3-4b-fp8`  
Protocol: real M=1 decode path; feed **oracle** token each step (not our argmax).

| Prompt | Matches | Match rate | Mismatches |
|--------|--------:|-----------:|-----------:|
| 1 | 256/256 | **100.0%** | 0 |
| 2 | 247/256 | **96.5%** | 9 |
| 3 | 245/256 | **95.7%** | 11 |
| **Total** | **748/768** | **97.4%** | **20** |

**Target ≥95% on non-ties:** overall match rate is **97.4%** vs the BF16 HF oracle.  
Prior BF16 Hephaestus vs same oracle was ~756/768 (~98.4%) with most diffs on ties (`.agent/notes/768-step-teacher-forced-results.md`).

FP8 adds ~8 extra mismatches vs that BF16 baseline — consistent with W8A8 quantization noise, **not** a kernel structural failure.

- **No kernel fix indicated.** Residual gap vs oracle is checkpoint/quantization territory (weight-only FP8, no act-outlier protection), not “fix the WMMA.”
- Non-tie-only breakdown not run (no top-2 oracle logits for all 256 steps). Raw match rate already clears 95%.

### Tiny layer-diff (BF16 path vs FP8 W8A8)

- Staged synthetic tiny FP8 from BF16 via `scripts/stage_tiny_fp8_from_bf16.py` (per-row absmax).
- Dumps: `dump_activations.mojo` (BF16) vs `dump_activations_fp8.mojo` (W8A8).
- Tolerance: `1e-3 + 5e-2 * |ref|`.

| Finding | Detail |
|---------|--------|
| First divergent cut | **`layer0_step0_attn_norm`** (before any matmul) |
| Cause | FP8 **embed** dequant on gather → residual differs → attn_norm differs |
| First matmul diverge | **`layer0_step0_q_proj`** (as expected for W8A8 matmul) |
| q_proj noise shape | max_abs=0.026, mean=0.006, **max/median err ≈ 4.5** → **uniform quant noise**, not sparse outliers |
| Wide-tol “fail” count | All later tensors fail (error accumulates) — expected for W8A8 |

**Interpretation:** divergence is **structural quantization** (embed + every matmul), not a few wild tokens → **not** a kernel bug; **not** a “re-quant for outliers” emergency from noise shape alone. Teacher-forced already shows usable argmax fidelity on 4B production FP8.

### Correctness gate

| Check | Result |
|-------|--------|
| TF match ≥95% | **PASS (97.4%)** |
| Layer-diff first matmul = q_proj | **PASS** (after expected embed noise) |
| Outlier-shaped errors | **No** (uniform) |
| Checkpoint re-quant required? | **Not forced by TF** — optional for higher fidelity |

→ **Proceeded to Fix 2.**

---

## Fix 2 — LDS-staged weight loads

Implemented BF16-v2-style cooperative coalesced global→LDS for A and W tiles per 16-K strip, then fragment from LDS → WMMA (`FP8_DECODE_USE_LDS` in `wmma_gfx12.mojo`).

Also tried multi-strip LDS (4×16 K panels) — **worse**.

### Measured decode (10-tok prompt × 256 gen, fwd-only, 3 reps)

| Kernel variant | tok/s (mean) | Est. effective GB/s* | Notes |
|----------------|-------------:|---------------------:|-------|
| **Direct global** (G1b-0) | **56.58** | ~227 | Pre-LDS best |
| **LDS v2** (1 strip, barriers) | **36.84** | ~148 | Thesis staging |
| Multi-strip LDS (4 strips) | **19.16** | ~77 | Barrier/panel overhead |
| Target (LDS “fixed”) | — | **≥400** | Not reached |
| Roofline (640 GB/s × 4.02 GB) | ~159 | 640 | Ceiling |

\* `tok/s × 4.02e9 / 1e9` assuming full weight stream once per token.

**LDS did not help decode.** Reason: **no LDS reuse** on M=1 (each weight element used once); barriers dominate. Coalescing via LDS is real, but the barrier tax exceeds the uncoalesced-global penalty on this shape.

`FP8_DECODE_USE_LDS = True` by default (requested staging). Flip to `False` for the faster direct path. Both share the same WMMA + scale math (TF still 256/256 on p1 with LDS).

---

## Co-measure after both fixes (LDS default)

| | tok/s |
|--|------:|
| **Hephaestus FP8 WMMA + LDS** (fwd-only, mean of 3) | **36.84** |
| **llama.cpp Q8_0** tg128 ROCm (same session) | **109.49 ± 0.70** |
| **Ratio** | **0.34×** |

Direct-global ablation (same session window): **56.58** tok/s → **0.52×** vs Q8.

### Gates

| Gate | Status |
|------|--------|
| **G1b-4** (no weight dequant in matmul) | **MET** |
| **G1b-2** (≥ Q8_0 decode) | **NOT MET** (0.34× with LDS; 0.52× direct) |
| Correctness TF ≥95% | **MET (97.4%)** |

---

## Files

| File | Role |
|------|------|
| `src/qwen_teacher_forced_decode_fp8.mojo` | 4B FP8 TF harness |
| `experiments/exp5_layer_diff/dump_activations_fp8.mojo` | Tiny FP8 activation dump |
| `scripts/stage_tiny_fp8_from_bf16.py` | Tiny BF16→FP8 staging |
| `src/hephaestus/wmma_gfx12.mojo` | FP8 WMMA + optional LDS (`FP8_DECODE_USE_LDS`) |
| `experiments/exp5_layer_diff/diff_layers.py` | `--atol` / `--rtol` for W8A8 |

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus

# Teacher-forced
mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_teacher_forced_decode_fp8.mojo -o /tmp/tf_fp8
/tmp/tf_fp8 /tmp/prompt1_input_ids.txt /tmp/prompt1_oracle_out.txt /tmp/tf_p1

# Co-measure
mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8_wmma
for r in 1 2 3; do /tmp/qwen_ab_fp8_wmma bench/ab_prompt_short_ids.txt 256; done
~/projects/llama.cpp/build/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf -p 0 -n 128 -r 5 -d 0
```
