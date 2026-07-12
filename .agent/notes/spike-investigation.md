# Prompt 1 / step 67 logit-spike investigation

**Date:** 2026-07-12  
**Status:** **CLOSED** — baseline fully understood; RoPE defect fixed on main; residual is irreducible BF16 matmul ambiguity.

## Final verdict (Phase 1b entry)

**Root cause identified. Two layers:**

| layer | nature | action |
|---|---|---|
| **RoPE** f32-accum vs HF stepwise bf16 | **Real engine defect** | **Fixed** in `rope_kernel` / `rope_kernel_qk` (main). Cut-verified: post-RoPE Q rel 1.64e-3 → 1.07e-4. |
| Residual full-vocab spikes (e.g. abs ~12 at tok 96874) | **Irreducible BF16 matmul reduction-order ambiguity** | **Not a defect.** Heph `q_proj` is bit-exact sequential f32-accum of identical `xn`; HF differs in **110/315392** ULPs (torch blocked/tree reduction). Hybrid inject of those 110 alone flips logit 16.38→4.71 — ill-conditioned amplification of legitimate precision-regime disagreement. |

**Characterization:**

- **Not benign under strict FP8 argmax-parity** with the measured residual delta (probe 12: dozens of non-tie rows with `s_flip ≤ 16`). Do not claim “harmless under FP8 widening.”
- **Not an implementation bug** in the remaining path: sequential f32-accum is the correct textbook reduction; matching HF bit-for-bit would mean matching torch’s blocked matmul, not “fixing” Hephaestus arithmetic.
- **Phase 1b entry gate CLEARS** because the BF16 baseline is now **fully understood**: one real bug found and fixed (RoPE); residual spikes characterized as inherent to BF16 matmul non-associativity / reduction order under ill-conditioned rows. Phase 1b gates use **tolerance-based** metrics (perplexity, speed) and layer-by-layer logit diffs against this known BF16 reference — not full-vocab bit identity.

**Argmax / G1a-1:** unchanged — 0 non-tie flips across 768 teacher-forced steps (decision boundary clean).

## Determinism result (read this first)

**DETERMINISTIC.** Five independent processes wrote full `[256, 151936]` float32 logit artifacts for the teacher-forced prompt-1 run. All five SHA-256 hashes are identical:

```
9618d6846a352682c6cc2f2af37c6b0a1c61769ba778f5d6cfb12d972ae39e00
```

At prompt 1, teacher-forced step 67, token 96874 (pre-RoPE-fix artifact):

| path | value | float32 bits |
|---|---:|---:|
| Hephaestus | 16.31208610534668 | `0x41827f27` |
| HF SDPA | 4.25 | `0x40880000` |
| signed difference | **+12.06208610534668** | — |

Post-RoPE-fix (same step): logit **16.380**, abs **12.130** — RoPE fix removed the elevated cut-point error but not the residual spike (see probe 15).

Re-verified from `/tmp/spike-det-1783875368/rep{1..5}_logits.f32`. Source: `experiments/spike/probe0_determinism.py`.

**Cut of the search space:** races, uninitialized memory, and nondeterministic reductions are **ruled out**.

## Corrected anomaly shape

This is **not** an isolated tail-token spike.

For the entire step-67 vocabulary row (Hephaestus vs HF SDPA):

| statistic | value |
|---|---:|
| mean abs error | 1.83245 |
| median | 1.62881 |
| p99 | 5.62833 |
| p99.9 | 7.35287 |
| tokens with abs err > 1.0 | 103,951 / 151,936 |
| tokens with abs err > 5.0 | 3,396 / 151,936 |
| maximum | **12.06209** at token 96874 |

Token 96874: HF rank **2282** → Hephaestus rank **12**. Argmax remains **15678** on both sides with a large decision margin (Hephaestus top1−top2 ≈ 10.53; argmax−96874 ≈ 17.11). Neighboring steps 66 and 68 are ordinary (~0.07 / ~0.18 mean abs). The same token's median error across other rows is ~0.075, with other large hits on the same phase family.

Evidence: `probe0_determinism.json`, `probe1_rowspace.json`.

## Alignment / execution routes

| route | logit[96874] | notes |
|---|---:|---|
| HF exact-prefix vs full recompute | bit-identical at target | no future leakage |
| Hephaestus exact-prefix vs full | **16.312086** both | bit-identical |
| Hephaestus sequential cached | 16.149622 | max row delta 1.55; argmax preserved |

Future-token leakage and row misalignment are **ruled out**. Cached vs one-shot reduction order moves the value but does not remove the anomaly. Evidence: `probe2_modes.mojo`, `run_modes.sh`.

## Hypothesis table

| hypothesis | falsifiable prediction | probe / result | status |
|---|---|---|---|
| Nondeterminism / race / uninit | raw bits or artifact SHA vary across processes | 5/5 SHA identical; value bits `0x41827f27` | **RULED OUT** |
| Row/token misalignment or future leakage | exact prefix differs from full causal recompute | HF and Hephaestus prefix/full both bit-identical | **RULED OUT** |
| Corrupt tied LM head / bad projection | logit row not representable as `E @ h` | relative residual `6.99e-7`; `E @ delta_h` predicts 99.94% of the spike | **RULED OUT** |
| Pure final-hidden scale error | one scalar explains most of `delta_h` | best scale explains ~7.5% of hidden delta | **RULED OUT** |
| One broken layer/kernel (discontinuity) | layerwise error shows a discrete first jump | embeddings exact; layer-0 attn starts at **0.00174** rel; grows to ~0.33 final hidden; largest growth mid/late net, not one cliff | **NOT FOUND as single broken layer** |
| RoPE GPU `cos`/`sin` precision alone | trig error large enough to explain 12-unit gap | prior `exp6_gpu_trig_probe.mojo`: ~1e-6…1e-8 | **RULED OUT as sole cause** |
| BF16 probability rounding is the main/sole cause | HF eager ≪ SDPA gap to Hephaestus; turning off prob-BF16 collapses to HF | eager target 4.375, median err 1.573 vs SDPA 1.629 (marginal). Intervention B: target → 8.289 (not 4.25), median → 0.541 | **CONTRIBUTOR; sole cause RULED OUT** |
| BF16 score rounding missing | adding score rounding converges to HF | target stays 11.86; median 1.267 | **RULED OUT as fix** |
| Seed is inside attention kernel only | post-RoPE Q/K/V match; only attn_out diverges | Q tgt rel **0.00159** ≈ attn_out **0.00174**; V all rel **0.000115** (~14× smaller) | **Attention-only seed RULED OUT**; seed is **mostly Q-path**, already present pre-attention |
| Layer-0 seed is abnormal vs HF self-variation | Hephaestus L0 ≫ HF eager-vs-SDPA | Heph L0 attn rel 0.00174 vs HF self-spread **0.00191** (ratio 0.91) | **Seed magnitude is within HF self-spread** |
| Deterministic amplification / ill-conditioning | independent perturbations + HF self-spread peak on same phase families; error grows with depth | phase 3/7 dominate; corr(log amp, log HF self-spread)=**0.9066**; layerwise growth observed | **SUPPORTED** (mechanism after seed) |
| Hephaestus inside a defensible BF16 ensemble | measured independent impls bracket Hephaestus | HF eager−SDPA hidden spread at row 67: **0.046**; Heph−SDPA: **0.335** | **NOT ESTABLISHED** at final hidden (seed is normal-sized; amplification is not) |
| Seed is `q_proj` matmul | cut error large at post-`q_proj`; inject collapses | cut rel **9.88e-5**; inject ratio 0.36 (not collapsed) | **RULED OUT as primary** |
| Seed is `q_norm` | cut error jumps at post-`q_norm`; inject collapses | cut rel **1.05e-4** (×1.06 vs proj); inject **identical** to `q_proj` inject | **RULED OUT as primary** |
| Seed is RoPE on Q | cut error jumps at post-RoPE; inject collapses final spike | cut rel **1.64e-3** (**×15.7**); inject hidden ratio **0.114**, logit gap 12.06→**0.20** | **SUPPORTED** |
| cos/sin / inv_freq construction wrong | heph cos/sin differ from HF enough to explain dump gap | cos **bit-identical**; sin 1 pair ×1 ulp; inv_freq rel 1e-16 | **RULED OUT** |
| Wrong rotate pairing / sign | interleaved or conjugate matches heph dump | interleaved/conjugate rel ~0.85–0.99 vs dump | **RULED OUT** |
| f32-accum rotate vs HF bf16 stepwise | f32-accum gap matches dump; strict bf16 matches HF | f32-accum vs HF **1.91e-3** ≈ dump **1.64e-3**; strict bf16 **2.07e-5** | **ROOT CAUSE** |
| Benign under FP8 widening | no clear argmax flips when error is coarsened to E4M3-class | probe 8: 3 / 6 clear flips; probe 12: **81 non-tie** rows with `s_flip ≤ 16`; spike row `s_flip ≈ 2.18` | **BENIGN REJECTED** |

### Hidden-state / projection evidence

- `||delta_h|| / ||h_HF|| = 0.3345` at row 67 (control row 202: 0.0229)
- Recovered target-token dot predicts 99.94% of the observed spike
- Median all-row hidden divergence 0.0198; phase means: phase 7 = 0.1125, phase 3 = 0.0765, others ≤ 0.0381

Evidence: `probe1_rowspace.json`, `probe4_bisect.json`, `probe6_rowdiv.py`.

### Layer-0 seed localization (probes 11)

Hephaestus dumps: `probe11_dump_l0.mojo` → `/tmp/spike_l0_{q,k,v,attn}.f32`.  
HF post-rotary Q/K/V captured from the real `apply_rotary_pos_emb` path (not a reimplementation).

| tensor | rel L2 (Heph vs HF SDPA) |
|---|---:|
| Q (all tokens) | 0.001639 |
| Q (target row) | 0.001587 |
| K (all) | 0.000146 |
| V (all) | 0.000115 |
| o_proj attention out (target) | 0.001740 |

HF eager vs SDPA attention-out rel at the same cut: **0.001913**.

**Interpretation (updated by probe 13):** the first divergence is a normal-sized BF16 delta concentrated on **post-RoPE Q** (not `q_proj`/`q_norm`). It is **not** a catastrophic attention-kernel bug at layer 0. The pathology is **amplification** of that RoPE seed through 36 layers on sensitive rows — final hidden error is ~7× the HF eager/SDPA ensemble spread. Replace-and-continue with HF post-RoPE Q collapses the spike (see Probe 13).

Evidence: `probe11_layer0_seed.json`, `probe11_qkv_hfrope.json`, `probe4_bisect.json`, `probe13_q_cuts.json`.

### Conditioning evidence and its limit

Small embedding perturbations in HF produce large, phase-correlated responses; HF eager vs SDPA shows the same phase structure (`corr = 0.9066`). Affected rows are intrinsically sensitive.

This does **not** prove Hephaestus correct: row 67 ranks only 26/256 under synthetic perturbation response, and final Hephaestus spread still dwarfs HF's self-spread.

Evidence: `probe5_conditioning.json`, `probe10_hf_variants.json`.

## FP8 bar (the deliverable standard)

"Benign" is **not** "doesn't affect BF16 greedy today."  
It is: **provably cannot affect decode output even when FP8's coarser mantissa widens the error.**

### Probe 8 — concrete E4M3 candidate

Per-output-channel absmax weight scales; optional per-token activation scales. Aligned 3×256 teacher-forced rows:

| path | flips | near-tie | **clear** |
|---|---:|---:|---:|
| BF16-sized noise (3 seeds) | 10 / 10 / 8 | all | **0** |
| E4M3 weights | 16 | 13 | **3** |
| E4M3 weights + activations | 17 | 11 | **6** |

`SPEC.md` allows per-tensor or per-channel scales but does not freeze the Phase 1b recipe. These results are **not** proof that every FP8 design fails; they **are** negative evidence against claiming the current anomaly harmless at FP8 precision.

### Probe 12 — magnification of the measured BF16 delta

Model: `logits(S) = HF + S · (Hephaestus − HF)`. Search smallest `S` that changes BF16-truncated argmax.

Rough mantissa ratio E4M3/BF16 ≈ `2^-3 / 2^-7 = 16`.

| fact | value |
|---|---:|
| Spike row (p1/s67) `s_flip` | **2.18** |
| Hephaestus margin top1−top2 at spike | 10.53 (≈42 bf16 ulps at mag 33) |
| Non-tie rows with `s_flip ≤ 16` (all prompts) | **81** |
| Concrete E4M3 clear flips (probe 8) | 3 / 6 |

Even without inventing new quantization noise, **simply magnifying the existing BF16 anomaly pattern by less than the E4M3/BF16 mantissa ratio flips dozens of clear decisions.** The canonical spike row needs only ~2.2×. That fails the FP8 benign bar on its own terms.

Evidence: `probe8_fp8.json`, `probe12_fp8_margin.json`.

## Probe 13 — Q cut-points + replace-and-continue (2026-07-12)

Exact-prefix seq=77, target row=76 (prompt1 step 67). HF SDPA vs eager Q cuts are **bit-identical** (pre-attention path shared).

### Cut-point relative L2 (Hephaestus vs HF SDPA, full `[77,32,128]`)

| cut | all_rel | tgt_rel | all_max_abs | growth vs previous |
|---|---:|---:|---:|---:|
| post `q_proj` | **9.88e-5** | 1.89e-5 | 0.00195 | — |
| post `q_norm` | **1.05e-4** | 2.18e-5 | 0.03125 | ×1.06 |
| post RoPE | **1.64e-3** | 1.59e-3 | 0.0625 | **×15.7** |

**First elevated cut: `q_rope`.**

### Replace-and-continue (inject HF SDPA Q at cut, continue Hephaestus)

Predeclared collapse rule: final hidden rel < 25% of control (control = 0.3345).

| mode | hidden_rel vs HF | collapse_ratio | logit[96874] | abs diff vs HF 4.25 |
|---|---:|---:|---:|---:|
| control | 0.3345 | 1.000 | **16.312** | **12.062** |
| inject_q_proj | 0.1199 | 0.358 | 6.363 | 2.113 |
| inject_q_norm | 0.1199 | 0.358 | 6.363 | 2.113 |
| **inject_q_rope** | **0.0382** | **0.114** | **4.054** | **0.196** |

`inject_q_proj` and `inject_q_norm` are **numerically identical** end-to-end (same `||h||`, same logits) — `q_norm` adds no independent defect. Both leave the broken Hephaestus RoPE in the path, so they cannot fully clear the spike. Only forcing post-RoPE Q to the HF reference collapses it.

**Causal reading:** the 12-unit spike is the amplification of a **RoPE-composed Q mismatch**, not of `q_proj`/`q_norm`. Fixing RoPE (and validating K similarly) is the engine-side remediation path; do not chase matmul or RMSNorm for this anomaly.

Evidence: `experiments/spike/out/probe13_q_cuts.json`, `probe13_q_cuts.mojo`, `probe13_hf_q_cuts.py`, `run_probe13.sh`.

## Probe 14 — RoPE sub-step isolation (2026-07-12)

**ROOT CAUSE:** `rope_kernel` applies the rotate as an **f32-accumulated**  
`(re*cos)−(im*sin)` / `(im*cos)+(re*sin)` after casting cos/sin to bf16.  
HF applies **stepwise bf16** `(q*cos)+(rotate_half(q)*sin)` (mul then add, both in bf16).  
Algebraically identical in f64; **not** identical under bf16 rounding. The f32 path is more accurate and therefore diverges from the reference.

| test | result |
|---|---|
| cos heph vs hf | **bit-identical** |
| sin heph vs hf | 1/4928 pairs differ by 1 ulp (negligible) |
| inv_freq max_rel | **1.7e-16** |
| closed form ≡ rotate_half (f64) | **max abs 0** |
| HF stepwise bf16 on hf_pre → hf_post | rel **2.07e-5** (self-check) |
| strict bf16 closed form → hf_post | rel **2.07e-5** (matches HF) |
| **f32-accum closed form → hf_post** | rel **1.91e-3** |
| Hephaestus dump → hf_post | rel **1.64e-3** (ratio dump/model **0.86**) |
| HF rope on heph_pre → hf_post | rel **1.07e-4** (pre-rope noise only) |
| interleaved / wrong-sign / conjugate | rel ~0.85–0.99 — **not** the dump |

Element view (target row): q_rope mean abs 0.00049, max 0.03125; **not** concentrated on highest-freq pair 0 (mean 0.00033) — consistent with systematic rounding-mode difference, not a trig-domain failure.

**Engine locus (fixed):** `src/hephaestus/kernels.mojo` `rope_kernel` — stepwise bf16 casts after each mul and after add/sub.

Evidence: `out/probe14_rope_substep.json`, `probe14_rope_substep.py`.

## Verified fix — stepwise BF16 RoPE (2026-07-12)

**Applied** on `investigate/logit-spike` only (not main).

### Patch

```mojo
# Before (f32-accum, one store cast):
x[base + i_re] = (re * cos_v) - (im * sin_v)
x[base + i_im] = (im * cos_v) + (re * sin_v)

# After (HF-matching stepwise bf16):
var re_c = (re.cast[F32]() * cos_v.cast[F32]()).cast[BF16]()
var im_s = (im.cast[F32]() * sin_v.cast[F32]()).cast[BF16]()
var im_c = (im.cast[F32]() * cos_v.cast[F32]()).cast[BF16]()
var re_s = (re.cast[F32]() * sin_v.cast[F32]()).cast[BF16]()
x[base + i_re] = (re_c.cast[F32]() - im_s.cast[F32]()).cast[BF16]()
x[base + i_im] = (im_c.cast[F32]() + re_s.cast[F32]()).cast[BF16]()
```

### Cut-point verification (probe 13 dump)

| cut | pre-fix all_rel | post-fix all_rel |
|---|---:|---:|
| q_norm | 1.05e-4 | 1.05e-4 (unchanged) |
| **q_rope** | **1.64e-3** | **1.07e-4** |

Target-row q_rope rel: **1.59e-3 → 2.18e-5**. RoPE sub-step defect is **gone**.

### Spike verification (probe 13 control — NOT cleared)

| metric | pre-fix | post-fix | hoped |
|---|---:|---:|---:|
| logit[96874] | 16.312 | **16.380** | ~4.25 |
| abs vs HF 4.25 | 12.062 | **12.130** | ~0.20 |
| hidden rel vs HF | 0.3345 | **0.3216** | ≪0.1 |
| step-67 row median abs | 1.629 | **1.564** | small |

**Spike does not drop.** Expectation that rope-only fix would yield inject_q_rope’s ~0.20 abs was wrong: that inject replaced the **entire** post-RoPE Q (killing pre-rope error too).

### Post-fix inject re-ablation (causal residual)

| mode | logit[96874] | abs vs 4.25 | hidden_rel |
|---|---:|---:|---:|
| control (fixed rope) | 16.380 | 12.130 | 0.322 |
| **inject_q_proj** | **4.713** | **0.463** | **0.046** |
| inject_q_norm | 4.713 | 0.463 | 0.046 |
| inject_q_rope | 4.842 | 0.592 | 0.054 |

With rope fixed, **`inject_q_proj` collapses the spike.** Remaining seed = **`q_proj` matmul** (naive GPU vs torch reduction/rounding), amplified through depth on ill-conditioned rows.

### Probe 12 FP8 margin (post-fix)

| | pre-fix | post-fix |
|---|---:|---:|
| target abs diff | 12.062 | 12.130 |
| target s_flip | 2.176 | 2.164 |
| non-tie rows s_flip≤16 | **81** | **74** |

Slight improvement; **not** cleared. `out/probe12_fp8_margin_post_rope_fix.json`.

### Teacher-forced max_abs (post-fix)

| prompt | max_abs | at |
|---|---:|---|
| 1 | **12.13** | step 67, tok 96874 |
| 2 | 8.81 | step 215 |
| 3 | **0.90** | step 205 (was 1.75 pre-fix) |

Prompt 3 improved; prompt 1 spike intact.

## Probe 15 — q_proj accumulation (2026-07-12)

**Hypothesis under test:** gemv / matmul f32 accum is “too accurate”; force BF16 accum to match HF.

### Path fact (non-negotiable)

Prefill anomaly uses **`m = 77` → `matmul_kernel_naive`**, not `gemv_gpu` (`m == 1` decode only). Both use **f32 accumulation** then cast to bf16 (`get_accum_type[bf16]`).

### Elementwise dumps (layer 0, exact-prefix)

| tensor | Heph vs HF | n_bf16_diff |
|---|---|---:|
| input_layernorm `xn` | **bit-identical** (rel 0) | **0** |
| `q_proj` output | rel **9.88e-5**, max abs **0.00195** | **110 / 315392** |
| target-row `q_proj` | rel **1.89e-5** | **2** cols (105, 890) |

### Recompute on identical `xn` and `W`

| recompute | vs HF q_proj | vs Heph q_proj |
|---|---:|---:|
| sequential **f32-accum** then bf16 cast | rel 9.88e-5 | **rel 0 (bit-identical)** |
| bf16 stepwise accum (target row) | rel **0.126** | rel **0.126** |
| bf16 chunk-64 accum | rel 0.0093 | rel 0.0093 |

**Hephaestus is bit-exact sequential f32-accum.** HF Linear disagrees with that reference in the same 110 elements. Pure bf16 accum is **farther from both**, not closer to HF.

### Hybrid inject (causal)

| inject payload | logit[96874] | abs vs HF 4.25 |
|---|---:|---:|
| control (native) | 16.380 | 12.130 |
| full HF q_proj | 4.713 | 0.463 |
| **hybrid: only the 110 differing elems → HF** | **4.713** | **0.463** |
| heph-self re-inject | 16.380 | 12.130 |

Those **110 ULP-level elements alone** collapse the spike. Not a global mis-accumulation mode.

### Verdict

| claim | status |
|---|---|
| Bug is gemv inner loop | **RULED OUT** (wrong kernel for this path) |
| Force BF16 accum to match HF | **RULED OUT** (wrong direction; worsens vs HF) |
| Heph q_proj mis-implemented vs f32 sum | **RULED OUT** (bit-exact match) |
| Residual seed = sparse torch-vs-sequential f32 matmul ULPs, amplified | **SUPPORTED** |

**Closed as irreducible precision-regime ambiguity** (not a Hephaestus defect). Optional future work if full-vocab HF identity is ever required: match torch/ROCm blocked matmul reduction for prefill — engineering cost, not “more correct” sequential f32.

Evidence: `out/probe15_qproj_accum.json`, `probe15_qproj_accum.{mojo,py}` (investigate/logit-spike branch).

## Closure / main merge

- **RoPE fix** merged to `main` from investigation `eabf42c` (kernels only): stepwise bf16 in `rope_kernel` **and** `rope_kernel_qk` (main’s fused Q+K path).
- **Phase 1b entry gate CLEARED** (see `DECISIONS.md` 2026-07-12): baseline fully understood; residual spikes characterized; 1b uses tolerance metrics vs this known BF16 reference.
- Investigation probes remain on `investigate/logit-spike` under `experiments/spike/`.

## Artifact map

Full probe suite on `investigate/logit-spike` (`experiments/spike/`). Compact JSON in `experiments/spike/out/`.

| probe | purpose | committed output |
|---|---|---|
| 0 | determinism + row statistics | `out/probe0_determinism.json` |
| 1 | row-space / hidden recovery | `out/probe1_rowspace.json` |
| 2 | full / prefix / sequential routes | log under `/tmp/spike_modes_run.log` |
| 3–4 | HF slots + layerwise bisect | `out/probe4_bisect.json` |
| 5 | conditioning + HF self-spread | `out/probe5_conditioning.json` |
| 8 | concrete E4M3 candidate | `out/probe8_fp8.json` |
| 9 | attention rounding intervention | `/tmp/spike_iv_*` |
| 10 | HF eager vs SDPA vs Hephaestus | `out/probe10_hf_variants.json` |
| 11 | layer-0 Q/K/V seed localization | `out/probe11_layer0_seed.json` |
| 12 | FP8 widening / margin bar | `out/probe12_fp8_margin.json` |
| 13 | Q cut-points + replace-and-continue | `out/probe13_q_cuts.json` |
| 14 | RoPE sub-step root cause | `out/probe14_rope_substep.json` |
| **15** | **q_proj ULP residual** | **`out/probe15_qproj_accum.json`** |
