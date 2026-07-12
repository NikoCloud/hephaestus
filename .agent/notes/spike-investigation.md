# Prompt 1 / step 67 logit-spike investigation

**Date:** 2026-07-12  
**Gate moved:** Phase 1b entry gate recorded in `DECISIONS.md`: explain the 12.06 logit anomaly **or** establish an FP8-relevant benign bound.

## Determinism result (read this first)

**DETERMINISTIC.** Five independent processes wrote full `[256, 151936]` float32 logit artifacts for the teacher-forced prompt-1 run. All five SHA-256 hashes are identical:

```
9618d6846a352682c6cc2f2af37c6b0a1c61769ba778f5d6cfb12d972ae39e00
```

At prompt 1, teacher-forced step 67, token 96874:

| path | value | float32 bits |
|---|---:|---:|
| Hephaestus | 16.31208610534668 | `0x41827f27` |
| HF SDPA | 4.25 | `0x40880000` |
| signed difference | **+12.06208610534668** | — |

Re-verified from the five on-disk `/tmp/spike-det-1783875368/rep{1..5}_logits.f32` files (independent of the original run shell). Source: `experiments/spike/probe0_determinism.py`, summary `experiments/spike/out/probe0_determinism.json`.

**Cut of the search space:** races, uninitialized memory, and nondeterministic reductions are **ruled out**. The anomaly is a deterministic indexing / arithmetic / reduction-order class of bug (or a deterministic implementation difference that amplifies).

## Verdict

**UNRESOLVED as to a single initiating arithmetic defect. NOT benign under the FP8 bar. Search space is materially narrowed. Do not clear the Phase 1b entry gate.**

What is established:

1. The event is **deterministic, row-wide, and upstream of the LM head**, then amplified through depth on phase-locked ill-conditioned rows.
2. The layer-0 **seed** is small (~0.17% relative on attention out) and **within HF's own eager-vs-SDPA self-spread** at that cut (~0.19%). Most of that seed already lives in the **Q path** (post-RoPE Q relative error ~0.16%); V is ~14× cleaner.
3. No single kernel has been shown to be *the* cause. Probability-BF16 rounding is a **contributor**, not the sole cause. RoPE trig precision alone was previously ruled out.
4. **Benign is rejected** under the deliverable's FP8 standard (not "harmless at BF16 today"):
   - Concrete E4M3 candidate (probe 8): **3 clear** non-tie flips (weights) / **6 clear** (weights+activations) across 768 rows.
   - Magnification model (probe 12): magnifying the measured Hephaestus−HF logit delta by the rough E4M3/BF16 mantissa ratio (~16×) produces **81 non-tie** rows that flip. The canonical spike row itself flips at only **S ≈ 2.18**.

What is **not** established: a single line of engine code that, if changed, collapses the row-wide error to the HF ensemble. That remains the next discriminating experiment (component-matched intervention continuing from the layer-0 Q seed).

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
| Deterministic amplification / ill-conditioning | independent perturbations + HF self-spread peak on same phase families; error grows with depth | phase 3/7 dominate; corr(log amp, log HF self-spread)=**0.9066**; layerwise growth observed | **SUPPORTED** (mechanism class, not a line-of-code root cause) |
| Hephaestus inside a defensible BF16 ensemble | measured independent impls bracket Hephaestus | HF eager−SDPA hidden spread at row 67: **0.046**; Heph−SDPA: **0.335** | **NOT ESTABLISHED** — Hephaestus is an outlier vs the HF ensemble at the final hidden, despite a normal-sized layer-0 seed |
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

**Interpretation:** the first divergence is a normal-sized BF16 implementation delta concentrated on the **Q path** (q_proj / q_norm / RoPE composition). It is **not** a catastrophic attention-kernel bug at layer 0. The pathology is **amplification** of that seed through 36 layers on sensitive rows — final hidden error is ~7× the HF eager/SDPA ensemble spread.

Evidence: `probe11_layer0_seed.json`, `probe11_qkv_hfrope.json`, `probe4_bisect.json`.

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

## Narrowest next discriminating probe

The layer-0 seed is now localized to the **Q path** at normal BF16 magnitude. Next:

1. Dump intermediate Q after `q_proj`, after `q_norm`, and after RoPE separately (Hephaestus vs HF) for the target row and a control row.
2. Predeclared predictions:
   - if error appears at `q_proj` → matmul reduction-order (naive GPU vs torch) is the initiator;
   - if error appears at `q_norm` → RMSNorm path (despite exp4 bit-exact on pure norms — check in-place / layout interaction);
   - if error appears only after RoPE → freq/cast composition (not raw `cos` accuracy);
   - if all three match and only full-stack Q differs → dump-path bug.
3. Then run a **replace-and-continue** intervention: swap Hephaestus's layer-0 Q (or attn out) for HF's, run layers 1…35 unchanged, and test whether final row-67 hidden collapse occurs.

That is the shortest path from "narrowed" to "positive root cause."

## Artifact map

See `experiments/spike/README.md`. Compact committed evidence in `experiments/spike/out/*.json`. Full vocab / hidden matrices live under `/tmp` and are intentionally not in git.

| probe | purpose | committed output |
|---|---|---|
| 0 | determinism + row statistics | `out/probe0_determinism.json` |
| 1 | row-space / hidden recovery | `out/probe1_rowspace.json` |
| 2 | full / prefix / sequential routes | log under `/tmp/spike_modes_run.log` |
| 3–4 | HF slots + layerwise bisect | `out/probe4_bisect.json` |
| 5 | conditioning + HF self-spread | `out/probe5_conditioning.json` |
| 8 | concrete E4M3 candidate | `out/probe8_fp8.json` |
| 9 | attention rounding intervention | `/tmp/spike_iv_*` (A=16.312, B=8.289, C=11.859) |
| 10 | HF eager vs SDPA vs Hephaestus | `out/probe10_hf_variants.json` |
| 11 | layer-0 Q/K/V seed localization | `out/probe11_layer0_seed.json`, `out/probe11_qkv_hfrope.json` |
| 12 | FP8 widening / margin bar | `out/probe12_fp8_margin.json` |
