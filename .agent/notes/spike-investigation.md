# Prompt 1 / step 67 logit-spike investigation

**Date:** 2026-07-12  
**Gate moved:** Phase 1b entry gate recorded in `DECISIONS.md`: explain the 12.06 logit anomaly or establish an FP8-relevant benign bound.

## Verdict

**UNRESOLVED, with the search space materially narrowed. Do not call this benign and do not clear the Phase 1b entry gate.**

The anomaly is a deterministic, row-wide divergence caused upstream of the tied LM head. The affected rows are intrinsically sensitive: independent HF attention implementations and controlled perturbations diverge most strongly in the same phase families, while the Hephaestus layerwise error grows rather than appearing as one corrupt output. That supports deterministic numerical amplification.

It does **not** identify the initiating arithmetic difference. HF eager is only marginally closer to Hephaestus than HF SDPA, and changing Hephaestus probability rounding moves the row substantially but does not collapse it to either reference. The narrowest remaining explanation is therefore **compound reduction/rounding-order differences entering at layer 0 attention and amplified through ill-conditioned rows**, not a demonstrated single kernel defect and not a proven harmless reference-path difference.

The FP8 bar fails provisionally: one concrete E4M3 scaling candidate produces clear argmax flips. The exact Phase 1b scaling scheme is not fixed, so this is not a verdict on every possible FP8 design; it is enough to reject a benignness claim.

## Frozen baseline and reproducibility

Canonical anomaly source: commit `d60630d`. Investigation worktree: `fe3c65a`. The full teacher-forced driver is byte-identical to `d60630d` (`8ed4aad5...e0a78e`). Between those commits, the only numerical source change is the `m == 1` GEMV route; this anomaly uses the one-shot `m > 1` path. The instrumented forward uses the production path plus snapshots and a parameterized copy of attention; its production control reproduces the anomaly exactly.

Five independent processes wrote full `[256,151936]` float32 logit artifacts. All five SHA-256 hashes are:

`9618d6846a352682c6cc2f2af37c6b0a1c61769ba778f5d6cfb12d972ae39e00`

At prompt 1, teacher-forced step 67, token 96874:

| path | value | float32 bits |
|---|---:|---:|
| Hephaestus | 16.31208610534668 | `0x41827f27` |
| HF SDPA | 4.25 | `0x40880000` |
| signed difference | +12.06208610534668 | — |

Command shape and per-run checksums are recorded in `experiments/spike/out/probe0_determinism.json`; source is `experiments/spike/probe0_determinism.py`.

**Finding:** races, uninitialized reads, and nondeterministic reductions are ruled out for this artifact.

## The corrected anomaly shape

This is not an isolated tail-token spike.

For the entire step-67 vocabulary row:

- mean absolute error: **1.83245**
- median: **1.62881**
- p99: **5.62833**
- p99.9: **7.35287**
- tokens above 1.0: **103,951 / 151,936**
- tokens above 5.0: **3,396 / 151,936**
- maximum: **12.06209** at token 96874

Token 96874 changes rank from **2282** in HF to **12** in Hephaestus, while argmax remains token 15678 with a large decision margin. The same token's median error across the other 255 rows is 0.0751, but reaches 7.323 elsewhere. Row 67 is the second-worst row by median error; row 93 is worse by median. Errors are phase-locked to the prompt's repeating token cycle, especially phases 7 and 3. Evidence: `probe0_determinism.json`, `probe1_rowspace.json`.

## Alignment and execution routes

- HF exact prefix (77 tokens) and full 265-token recompute are bit-identical at the target row.
- Hephaestus exact prefix and full recompute are also bit-identical, including target value 16.312086.
- Hephaestus sequential cached execution changes the target to 16.149622 and the row by max 1.54587, but preserves argmax.

Therefore future-token leakage and row misalignment are ruled out. Cached versus one-shot reduction order matters, but does not remove the anomaly. Evidence: `probe2_modes.mojo`, `run_modes.sh`, `/tmp/spike_modes_run.log`.

## Hypothesis table

| hypothesis | falsifiable prediction | probe / result | status |
|---|---|---|---|
| Nondeterminism, race, uninitialized memory | raw float32 bits or artifact checksum varies across independent processes | 5/5 complete artifacts byte-identical | **RULED OUT** |
| Row/token misalignment or future leakage | exact prefix differs from full causal recompute | HF prefix/full and Hephaestus prefix/full both bit-identical | **RULED OUT** |
| Corrupt tied LM head / bad projection | logit row cannot be represented as `E @ h`, or recomputed target dot fails | Hephaestus row relative residual `6.99e-7`; `E @ delta_h = 12.0549` versus observed 12.0621 | **RULED OUT** |
| Pure final-hidden scale error | one scalar explains most of `delta_h` | best scale explains only 7.5% of hidden delta | **RULED OUT** |
| One broken layer/kernel | layerwise error shows a discrete first jump inconsistent with incoming error | embeddings exact; layer-0 attention starts at 0.00174 relative error; error grows across layers to 0.3345 final hidden, with no unique discontinuity | **NOT FOUND** |
| RoPE trigonometric precision | measured trig error is large enough or changing trig path removes anomaly | prior direct gfx1201 trig probe measured about `1e-6` to `1e-8`, far below the observed effect | **RULED OUT as sole cause** |
| BF16 probability rounding is the main cause | HF eager should be materially closer than SDPA, and disabling Hephaestus probability rounding should collapse the row | HF eager target is 4.375 and median Heph error 1.573 vs SDPA 1.629—only marginally closer. Probability-FP32 intervention improves median to 0.541 and target error to 4.039 but does not eliminate it | **CONTRIBUTOR; main/sole cause RULED OUT** |
| BF16 score rounding is missing | adding score rounding converges to HF | median improves only to 1.267 and target error remains 7.609 | **RULED OUT as fix** |
| Deterministic numerical amplification / ill-conditioning | small independent perturbations and independent reference paths peak on the same rows; error grows through depth rather than appearing at projection | phase-3/7 rows dominate Hephaestus hidden divergence; correlation between perturbation amplification and HF eager-vs-SDPA spread is 0.9066; layerwise growth observed | **SUPPORTED, not a root cause** |
| Hephaestus is within a defensible BF16 implementation ensemble | measured independent implementations bracket Hephaestus, not a synthetic model chosen after observation | HF eager-SDPA hidden spread at row 67 is 0.0461; Hephaestus-SDPA is 0.3345 and Hephaestus-eager 0.3258 | **NOT ESTABLISHED** |

### Hidden-state and projection evidence

The tied embedding has shape `[151936,2560]`. Least-squares recovery shows the Hephaestus logits are an internally consistent projection of a different final hidden state, not sporadic vocabulary corruption:

- `||delta_h|| / ||h_HF|| = 0.33451` at row 67
- typical control row 202: 0.02286
- recovered target-token dot predicts 99.94% of the observed spike
- median all-row hidden divergence: 0.0198; row 67: 0.3345
- phase means: phase 7 = 0.1125, phase 3 = 0.0765, all others <= 0.0381

Evidence: `probe1_rowspace.json`, `probe4_bisect.json`, `probe6_rowdiv.py`.

### Conditioning evidence and its limit

A small embedding perturbation in HF produces large, phase-correlated output responses. HF eager versus SDPA exhibits the same phase structure; `corr(log amplification, log HF self-spread) = 0.9066`. This falsifies the claim that the affected rows are stable in the reference and only Hephaestus is anomalous.

It does not prove Hephaestus correct. Row 67 ranks only 26/256 under the synthetic perturbation response, and the measured HF implementation spread is much smaller than the Hephaestus spread. The random-noise ensemble in `probe7_ensemble.py` is retained as exploratory code but is not used for the verdict: its perturbation distribution is a model, not a measured implementation bound.

Evidence: `probe5_conditioning.json`, `probe10_hf_variants.json`.

## FP8 bar

`probe8_fp8.py` was corrected to use each prompt's real prompt length. Post-fix HF BF16 one-shot agreement with the saved cached-generation oracle is 255/256, 250/256, and 252/256, consistent with the already-documented reference-path ambiguity.

The probe applies real E4M3 round-trip quantization using one concrete candidate:

- weights: per-output-channel absmax scale
- activations: per-token absmax scale at each Linear input

Across 768 aligned teacher-forced rows:

| path | flips | near-tie | clear |
|---|---:|---:|---:|
| BF16-sized perturbation, seeds 0/1/2 | 10 / 10 / 8 | all | 0 / 0 / 0 |
| E4M3 weights | 16 | 13 | **3** |
| E4M3 weights + activations | 17 | 11 | **6** |

At target row 67, final-hidden divergence is 0.1408 for quantized weights and 0.3515 for weights+activations; no target-row argmax flip occurs. However, clear flips elsewhere mean this candidate fails the benignness bar.

`SPEC.md` permits per-tensor or per-channel scales but does not yet specify the exact Phase 1b scaling recipe. Consequently these results are **not definitive proof that every FP8 design fails**. They are definitive negative evidence against claiming the current anomaly harmless at FP8 precision.

Evidence: `probe8_fp8.py`, `probe8_fp8.json`.

## Narrowest next discriminating probe

Run a **component-matched layer-0 attention intervention**, not another end-to-end noise experiment:

1. Dump Q, K, V, scores, softmax probabilities, and attention output for the target query at layer 0 from Hephaestus.
2. Feed those exact BF16 Q/K/V tensors into a small CPU/reference implementation with selectable reduction order and rounding after each documented operation.
3. Compare against HF eager and SDPA component dumps.
4. Replace only the Hephaestus layer-0 attention output with each matched reference output, continue the unchanged remaining 35 layers, and test the predeclared prediction:
   - if one arithmetic variant collapses final hidden divergence and the row-wide error, that variant is causal;
   - if no layer-0 replacement collapses it, repeat at the earliest layer where contribution error exceeds its input-relative error.

This is the shortest path to positive causal evidence. More random perturbations can demonstrate sensitivity but cannot identify the initiating difference.

## Artifact map

See `experiments/spike/README.md`. Compact committed evidence lives in `experiments/spike/out/*.json`; full-vocabulary and hidden-state matrices remain in `/tmp` and are intentionally excluded from git.
