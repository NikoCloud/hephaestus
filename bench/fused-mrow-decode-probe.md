# Fused M-row decode probe — does weight amortization work?

**Date:** 2026-07-20  
**Branch:** `probe/fused-mrow-decode` (from `main` @ `76905c8`)  
**Harness:** `src/qwen_fused_mrow_probe.mojo` (measurement only)  
**GPU:** Device 0 — AMD Radeon AI PRO R9700 (gfx1201). GPU 1 empty throughout.  
**ROCm:** 7.2.4  
**Env:** `hephaestus-wmma-nightly` Mojo `1.0.0b3.dev2026071206`  
**Weights:** one FP8 arena `staged/qwen3-4b-fp8`  
**Roofline denominator:** **569 GB/s only** (460 retired, `3072b4a`)  
**Rule:** pure measurement. Temporary weight-byte instrumentation in `linear_fp8` / `gemv_fp8` paths **reverted** after the run. No scheduler, no optimization, no `main` edits.

---

## Null-result note: prior serial-launch “concurrency-shape” probe

The prior probe (`bench/concurrency-shape-probe.md`) looped **N independent `forward_fp8` calls on one HIP stream**. A HIP stream is **in-order**, so the harness **serialized by construction**: step times 20→40→81→161 ms for N=1/2/4/8 (identity), aggregate flat ~49.5 tok/s.

**Cause:** submission model (single in-order queue), **not** a kernel/memory floor.

**Contradicting concurrent-submission evidence already on record (DECISIONS 2026-07-19):** 8 OS processes scaled **8.0×** (56.6 tok/s each, **453 aggregate**) on the same silicon/kernels. 453 vs 49.5 is **how work is submitted**, not whether kernels can scale. File the serial-launch result as a **null with cause**; do not treat it as a finding about our kernels.

This probe measures the **opposite shape**: **one** fused call with **M activation rows** (Phase 2 Multiplier thesis).

---

## What was measured

**Full `forward_fp8(seq=M)`** — one forward step with M distinct token rows through the real Qwen3-4B FP8 stack (all projections + attention + lm_head). Not N serial single-row forwards. Not a scheduler.

| M | GEMM path |
|--:|-----------|
| 1 | `gemv_fp8` / `wmma_gemm_fp8_decode` (decode M=1) |
| ≥2 | `wmma_gemm_fp8_prefill` / v3a W8A8 (same path as 1702.6 tok/s prefill @ M=512) |

Real projection shapes (hidden 2560): q 4096×2560, k/v 1024×2560, o 2560×4096, gate/up 9728×2560, down 2560×9728, lm_head 151936×2560.

**Protocol:** past=0 each timed step (constant work); 3 warmup + 32 timed steps; M ∈ {1,2,4,8,16}; recheck M=1 and M=8 at end.

**Not claimed:** multi-request continuous batching, ragged lengths, or batched attention across independent sequences. This is fused **M-row GEMM weight amortization** inside one forward.

---

## Arithmetic anchors (before the table)

### M=1 reference (known, product)

| Item | Value |
|------|------:|
| FP8 decode (prior co-measure, longer context) | ≈ **66.1** tok/s |
| Weights ≈ 4.02 GB / step @ 66.1 | ≈ **266 GB/s** ≈ **46.7%** of 569 |
| Older direct-global datapoint | 56.6 tok/s ≈ 227 GB/s |

This run’s short past=0 M=1 step is **faster** (~75 tok/s) — less attention than 512-context decode. Use it as the **self-relative** M=1 anchor for this session; the 66.1 number remains the product reference.

### One denominator

**569 GB/s.** Vulkan has measured 530.7 GB/s (93.3% of 569) on mixed weights+KV — 460 was ROCm’s implementation limit, not hardware.

### Sobering perfect-amortization ceiling

If fusion amortizes weights perfectly and step time ≈ M=1:

| Ideal | tok/s |
|-------|------:|
| 8 × 66.1 | **≈ 529** |
| llama ROCm npl=8 (this session) | **≈ 550** |
| llama Vulkan npl=8 (prior) | **≈ 678** |

A **perfect 8× still sits just under ROCm and well under Vulkan**. Matching llama at npl=8 needs near-ideal amortization **plus** further wins. **Fused path is necessary, not sufficient.** This framing prevents a good curve from being over-read.

### Bytes/step model

```
fused:   bytes/step = 4.02 GB + M × (576 × 147456 / 1024³)   # weights once
unfused: bytes/step = M × (4.02 + KV_per_seq)                 # do NOT use for fused BW
```

Effective BW (fused model) = `(aggregate_tok_s × bytes/step_fused) / M`  
% roofline = `eff_BW / 569 × 100`

---

## Negative controls

### Primary — weight amortization (must be able to fail)

**Method:** Temporary host-side attribution at each FP8 linear launch (`gemv_fp8`, `linear_fp8` prefill, `linear_add_residual_fp8` prefill): record **`n × k` FP8 weight bytes once per launch**, not ×M. Row-loop fallback would call `gemv_fp8` per row and **sum to ×M** — that path would fail this control.

**Expected if fused:** ≈ 4.022 GB projection bodies (36 layers × all projs + lm_head) = **4 022 272 000 bytes**, **flat vs M**.  
**Expected if unfused:** ×M.

| M | weight bytes attributed | GB | launches | ratio to fused (~4.022e9) | ratio to unfused (×M) | verdict |
|--:|------------------------:|---:|---------:|--------------------------:|----------------------:|---------|
| 1 | **4 022 272 000** | 3.746 | 253 | **1.000** | 1.0 | baseline |
| 2 | **4 022 272 000** | 3.746 | 253 | **1.000** | 0.50 | **PASS** |
| 4 | **4 022 272 000** | 3.746 | 253 | **1.000** | 0.25 | **PASS** |
| 8 | **4 022 272 000** | 3.746 | 253 | **1.000** | 0.125 | **PASS** |
| 16 | **4 022 272 000** | 3.746 | 253 | **1.000** | 0.0625 | **PASS** |

253 launches = 36×7 projs + lm_head — one launch per weight tensor per forward, independent of M.

**PASS:** weight traffic is the fused model, not M×. Tok/s gains (below) are not explained by re-reading weights M times.

### Secondary — distinct rows (must be able to fail)

**Method:** Feed M **different** token ids (`1000 + i×97`). After forward, sample logits rows vs row0 every 1024 vocab columns. Fail if any row is elementwise-identical on all samples.

| M | result |
|--:|--------|
| 1 | n/a |
| 2–16 | **0 / 149 samples identical** row-to-row0 for every other row → **PASS** |

Rows are real distinct activations, not M copies of one row.

---

## Results table

| M | step ms | tok/s agg | tok/s per-row | bytes/step (fused GB) | eff BW GB/s | % of 569 | weight-bytes scale? | rows distinct? |
|--:|--------:|----------:|--------------:|----------------------:|------------:|---------:|---------------------:|---------------:|
| 1 | 13.41 | **74.56** | 74.56 | 4.099 | 305.6 | **53.7** | n/a (baseline) | n/a |
| 2 | 28.63 | **69.85** | 34.93 | 4.178 | 145.9 | **25.6** | **no (flat)** | **yes** |
| 4 | 28.62 | **139.75** | 34.94 | 4.336 | 151.5 | **26.6** | **no (flat)** | **yes** |
| 8 | 29.23 | **273.68** | 34.21 | 4.653 | 159.2 | **28.0** | **no (flat)** | **yes** |
| 16 | 34.29 | **466.67** | 29.17 | 5.286 | 154.2 | **27.1** | **no (flat)** | **yes** |

**Recheck (end of session):** M=1 → 75.9 tok/s; M=8 → 273.7 tok/s (stable).

**Weight GB per token (instrument):** 3.75 → 1.87 → 0.94 → 0.47 → 0.23 — falls as **1/M**.

---

## Curve shape (one sentence)

**Weight traffic is perfectly flat vs M (fused); aggregate tok/s rises steeply from M=2→16 on the v3a prefill GEMM path (~70→467), while step time stays ~29–34 ms — amortization works, but M=1 decode-GEMV and M≥2 prefill-WMMA are different kernels so M=1→2 is not a pure free lunch.**

Notes:

- **M=1 → M=2 step time doubles** (~13→29 ms): path switch decode-GEMV → v3a prefill (BM=64 tile; M=2 is heavily edge-masked). Not evidence against amortization — instrument still shows one weight read.
- **M=2 → M=8:** step time flat (~28.6→29.2 ms), aggregate **~3.9×** — classic weight amortization inside the fused GEMM.
- **M=8 aggregate 274** vs perfect 8× from this M=1 (596) or from 66.1 (529): **~0.5× of ideal 8×**. Necessary, not sufficient vs llama ~550.

---

## Co-measure (same session, GPU 0, ROCm)

| Item | Value |
|------|-------|
| Backend | **ROCm/HIP** |
| llama.cpp | `33ca0dcb9` |
| Model | Q8_0 qwen3-4b-instruct-2507 |
| Flags | `-npp 512 -ntg 128 -npl 1,8 -ngl 99 -fa on` |

| npl | S_TG (tok/s) |
|----:|-------------:|
| 1 | 105.83 |
| 8 | **549.81** |

Hephaestus fused M=8 aggregate **273.7** ≈ **0.50×** llama ROCm npl=8 in the same window. Gap remains; fusion alone does not close it.

---

## Decision (per rule)

| Criterion | Result |
|-----------|--------|
| Weight bytes ≈ fused model (not ×M) | **PASS** (exact 4 022 272 000 for all M) |
| Eff. cost per token / aggregate climbs with M | **PASS** for M≥2 on fused path (70→467 tok/s; weight GB/tok 1.87→0.23) |
| Flat / weight scales with M | **No** |

### **Fused path works. Phase 2 may proceed to scheduler design on a measured basis.**

Multiplier thesis (weights once, amortized across M rows) is **supported by direct weight-byte attribution**, not only by tok/s.

### Warrant (required)

**This is the kernel result for fused M-row work. It says nothing yet about scheduler batch formation, continuous batching, or ragged sequence lengths. A good curve is not “Phase 2 de-risked.”**

Phase 2 is **neither de-risked nor at-risk** from the null serial-launch probe; **this** probe unblocks kernel-side fusion as a real lever and still leaves serving shape unproven.

---

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=$HOME/projects/hephaestus-wmma-nightly/.pixi/envs/default
export PATH=$CONDA_PREFIX/bin:$PATH
export MODULAR_HOME=$CONDA_PREFIX/share/max
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
KERNELS=$HOME/projects/modular/max/kernels/src

cd ~/projects/hephaestus
# temporary weight-log instrumentation was used for NC then reverted —
# re-apply from session history only if re-validating the control
mojo build -I $KERNELS -I src src/qwen_fused_mrow_probe.mojo -o /tmp/fused_mrow_probe
/tmp/fused_mrow_probe 32
```

Raw: `/tmp/fused_mrow_out/hephaestus.log`, `/tmp/fused_mrow_out/llama.log`.

---

## One-line verdict

**Weights are read once per fused M-row forward (not ×M); aggregate scales to ~274 tok/s at M=8 and ~467 at M=16 on the prefill GEMM path — fused amortization works; still ~0.5× llama npl=8 and below the ~529 perfect-8× ceiling; Phase 2 scheduler work is justified on this kernel result, not “de-risked.”**
