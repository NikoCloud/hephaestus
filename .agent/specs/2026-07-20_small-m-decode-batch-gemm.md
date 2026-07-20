# Small-M decode-batch FP8 GEMM — build + measure spec

**Status:** DRAFT 2026-07-20 — Frank.  
**Owner of build:** Grok.  
**Depends on:** `bench/fused-mrow-decode-probe.md` (amortization PASS; LDS small-M DEFICIT localized).  
**Branch:** from `main` (or current truth tip) → `probe/small-m-decode-gemm` (do not land on main until PASS).  
**Why now:** Fused weight amortization is confirmed. Competitiveness gap at M=8 is **not** batching logic — it is the **v3a prefill LDS GEMM run at decode-batch M**, holding ~27% of 569 while the barrier-free M=1 GEMV holds ~54%. Scheduler work on a 27% kernel caps us near half of llama.

---

## 0. What the last probe proved (do not re-litigate)

From `bench/fused-mrow-decode-probe.md`:

| Fact | Evidence |
|------|----------|
| Weights read **once** per step for every M | Exactly 4,022,272,000 weight bytes for M=1..16; fusion ratio 1.0 |
| Rows are distinct | Different token ids → different logits rows |
| Amortization works | M=2→8 step time flat (~28.6→29.2 ms); aggregate ~4× |
| Efficiency collapses at the kernel switch | M=1 GEMV **53.7%** of 569; M≥2 v3a prefill GEMM **~27%** flat |
| Perfect amortization ceiling | 8 × 66.1 ≈ **529** tok/s (product decode anchor) |
| Measured M=8 | **~274** tok/s ≈ **0.5×** ideal ≈ **0.50×** llama ROCm npl=8 (550) |

**Causal read (Opus, ratified):** M≥2 dispatches `wmma_gemm_fp8_prefill` (v3a, LDS weight-staging tuned for deep M-row reuse at prefill). At M=8 the reuse is too thin to pay for barriers. This is the **mirror** of DECISIONS 2026-07-19 LDS asymmetry:

> keep LDS for prefill, drop it for decode — reverse the mistake at small M.

**Arithmetic that localizes the prize:**

- GEMV moves ~300 GB/s (4.02 GB / 13.4 ms in the probe window).
- Fused v3a moves ~140 GB/s (4.02 GB / 29 ms).
- If fused held GEMV’s ~300 GB/s at M=8: step ≈ 13 ms → aggregate ≈ **600** tok/s → **above** llama ROCm npl=8 (550).

So the entire 0.5×-of-ideal gap is kernel bandwidth efficiency, not failed amortization.

---

## 1. Goal

Build and measure a **small-M FP8 W8A8 GEMM path** for decode-batch M ∈ **{2, 4, 8, 16[, 32]}** that:

1. Keeps **fused** weight load (once per step) — already proven.
2. Recovers **≥ 53% of 569 GB/s** effective bandwidth on the fused M=8 full-forward step (same accounting as the fused-mrow probe).
3. Does **not** regress large-M prefill (M≥64 or M=512 path stays on v3a LDS).

This is a **kernel tier**, not a scheduler.

---

## 2. Pre-registered prediction (disconfirmable)

**Prediction (Opus / physics):** An 8-row barrier-free (or thin-LDS) kernel does **8 MACs per weight byte** vs GEMV’s 1 — *less* memory-bound than GEMV. The GEMV already sustains ~53–54% barrier-free; a barrier-free M=8 path should hold **at least ~53%**.

| If M=8 full-forward eff BW… | Interpretation |
|-----------------------------|----------------|
| **≥ 53% of 569** | Prediction holds. Small-M kernel is the right lever; proceed to wire + re-measure vs llama; then scheduler. |
| **40–53%** | Partial recovery. Ship if multi-projection full-forward still net-positive; profile remainder. |
| **≤ ~30%** (still v3a-like) | Prediction fails. LDS is not the dominant cost — rescope (launch tax, attention, quant) before more tile work. |

Gate is on **full `forward_fp8(seq=M)` effective bandwidth %**, same formula as fused-mrow probe (roofline 569 only). Optional secondary: isolated down_proj/q_proj microbench for diagnosis only — **does not pass the gate alone**.

---

## 3. Dispatch contract (surgical)

Current (`linear_fp8` / `linear_add_residual_fp8`):

```
m == 1                    → gemv_fp8 / decode WMMA
m > 1 && n%BN==0 && k%16  → wmma_gemm_fp8_prefill (v3a LDS)   // BN=64 today
else                      → row-loop gemv
```

**Target:**

```
m == 1                              → unchanged decode GEMV
2 ≤ m ≤ M_SMALL_MAX && n%<tile> && k%16 → NEW small-M kernel (this spec)
m > M_SMALL_MAX && n%64 && k%16     → existing v3a prefill (LDS) UNCHANGED
else                                → row-loop fallback
```

Recommended defaults:

- `M_SMALL_MAX = 32` (covers decode-batch and small chunks; leaves true prefill on v3a).
- Tile: start **BM=16 or 32**, **BN=64 or 128**, **BK=16** (must remain WMMA 16-aligned on K). Prefer matching existing `BN=64` cut so N divisibility for model dims is unchanged (q/k/v/o/gate/down/lm_head already satisfy %64 where v3a runs).

**Non-negotiable:** M=512 prefill path must still hit v3a. Smoke: one M=512 (or M=256) prefill bench ≥ prior ~1700 tok/s class (same protocol as `bench/1b-fp8-prefill-gemm.md`); any large regression = FAIL.

---

## 4. Kernel design — what to build

### 4.1 Shape

`C[M,N] = scale_row[M] * scale_col[N] * (A_fp8[M,K] @ W_fp8[N,K]^T)`  
Accumulate F32 (or same as current prefill), then apply dual scales + optional residual like existing `wmma_gemm_fp8_prefill`.

- A already quantized by `quantize_act_rows_fp8` (reuse — do not invent a second quant).
- W is existing FP8 arena weights (swizzle policy: match whatever prefill path uses for those projections; decode swizzle defaults stay for M=1 only unless measurements deman same layout).

### 4.2 LDS policy (the actual lever)

**Primary attempt (pre-registered): no LDS staging on B (weights)** — direct global fragment loads, decode-style, multi-row A.

Rationale: LDS paid at deep M; at M=2..32 barriers dominate. DECISIONS LDS asymmetry is the design law.

**Allowed fallback if no-LDS under-delivers (second iteration only):** thin single-buffered LDS on A only, or double-buffer only if counters show clear win — do **not** start with full v3a B-staging.

### 4.3 Parallelism

- Grid over N-tiles (and M-tiles if BM < M).
- Prefer **multiple waves per block** only if VGPR allows (v3a was 4-wave / 64×64; small-M may want fewer waves, more N-parallelism).
- Do **not** reintroduce global split-K + host partials buffer (measured regression class).

### 4.4 Correctness

Same W8A8 math as prefill path. Gate:

1. **Bit-exact or near-exact vs row-loop gemv reference** on a frozen (M,N,K) fixture for M=2,8 (max abs err policy: match existing FP8 prefill tolerances / layer-diff prefs).
2. **Teacher-forced** after wiring into `forward_fp8`: ≥95% absolute; aim to hold ~97.4% class (± noise). One structural change at a time.
3. Negative control: path **not taken** for M=1 (still GEMV) and M=512 (still v3a) — print or counter which kernel is used.

---

## 5. Measurement protocol (full theorem, not microbench-only)

Reuse `src/qwen_fused_mrow_probe.mojo` pattern (or extend it):

- `forward_fp8(seq=M)`, past=0, 3 warmup + 32 timed steps  
- M ∈ {1, 2, 4, 8, 16}  
- Roofline **569 only**  
- bytes/step fused model = `4.02e9 + M * KV_term`  
- Co-measure: short llama ROCm npl=1 and npl=8 in the **same thermal window** (sandwich)  
- GPU 0 only; both GPUs freed first; `rocm-smi --showpids` before/after  

**Table (required):**

| M | kernel | step ms | agg tok/s | % of 569 | vs old v3a % | vs M=1 % | weight bytes once? |
|--:|--------|--------:|----------:|---------:|-------------:|---------:|--------------------|
| 1 | gemv | | | | n/a | 100% | yes |
| 2 | small-M | | | | | | yes |
| 4 | small-M | | | | | | yes |
| 8 | small-M | | | | | | yes |
| 16 | small-M | | | | | | yes |
| 512 smoke | v3a | | prefill tok/s | — | prior ~1700 | — | — |

**Pass:** M=8 **% of 569 ≥ 53%** on full forward, weights still once, TF OK, prefill smoke OK.  
**Also report:** agg tok/s vs llama npl=8 (550 ROCm / 678 Vulkan envelope) as headline, not the gate.

---

## 6. Out of scope

- Scheduler / continuous batching / paged KV implementation  
- Changing FP8-KV probe / act-quant math  
- Graph capture  
- Splitting / rewriting large prefill v3a for M=512  
- Claiming Phase 2 de-risked from this alone  

---

## 7. Sequencing after this

| Outcome | Next |
|---------|------|
| PASS (≥53% @ M=8) | Land kernel behind dispatch; free DECISIONS row; **then** scheduler design has a kernel that can actually win |
| Soft miss (40–53%) | Profile: act-quant launch tax vs GEMM; one lever only |
| FAIL (≤~30%) | Stop tile thrash; reverse Amdahl — attention + quant share of step |

---

## 8. Deliverables

1. Kernel + dispatch in tree on branch `probe/small-m-decode-gemm`  
2. `bench/small-m-decode-gemm.md` — curve, controls, gate pass/fail, co-measure, warrant  
3. Optional: DECISIONS draft line (do not write until PASS)  
4. Temporary instrumentation reverted before commit  

---

## 9. Warrant (every writeup ends with this)

This measures whether a **decode-batch-sized FP8 GEMM** can recover GEMV-class memory efficiency while keeping fused weight loads.  
It does **not** prove continuous batching, ragged batching, paged KV, or that Phase 2 is done.  
If it PASS, Phase 2’s kernel foundation is honest. If it FAIL, we learned the expensive thing before building a scheduler on a 27% kernel.

---

*Frank 2026-07-20. Pre-registered gate: **M=8 full-forward ≥53% of 569 GB/s**. Preferred mechanism: no-LDS / decode-style loads for 2≤M≤32; v3a LDS retained for large prefill only.*
