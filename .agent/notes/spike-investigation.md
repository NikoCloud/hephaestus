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

**Root cause identified (probe 14): `rope_kernel` f32-accumulates the rotate expression after bf16 cos/sin, while HF applies stepwise bf16 `(q*cos)+(rotate_half(q)*sin)`. Engine fix is a deliberate separate change to `kernels.mojo`. NOT benign under the FP8 bar until fixed and re-validated.**

What is established:

1. Deterministic, row-wide, upstream of LM head; amplified on ill-conditioned rows.
2. **Probe 13:** seed appears at post-RoPE Q (×15.7 cut jump); **`inject_q_rope` collapses** the spike (logit 16.31→4.05 vs HF 4.25).
3. **Probe 14 sub-step isolation:**
   - **cos/sin construction matches** (cos bit-identical; sin 1/4928 pairs off by 1 bf16 ulp; inv_freq max_rel 1.7e-16).
   - Closed form `re*cos±im*sin` **≡** HF `rotate_half` form in f64 (max abs 0) — **not** a pairing/sign bug.
   - HF stepwise bf16 self-checks against HF post-RoPE dump (rel **2.07e-5**).
   - **f32-accum closed form** (Mojo's effective path: BF16 muls promote to F32, single cast on store) vs HF post: rel **1.91e-3** ≈ dump gap **1.64e-3** (ratio **0.86**; corr pred/obs err **0.76**).
   - **Strict bf16** closed form matches HF stepwise (same 2.07e-5).
   - Applying HF rope to Hephaestus pre-RoPE Q recovers HF post within pre-rope noise.
4. Locus: `src/hephaestus/kernels.mojo` `rope_kernel` store lines — `(re * cos_v) - (im * sin_v)` without intermediate bf16 casts.
5. **Fix hint (not applied):** cast to BF16 after each mul and after add/sub (or emit the rotate_half form in bf16). More accurate f32 accum is **wrong** for HF matching.
6. **Benign rejected** under FP8 bar (probes 8/12) until a fix is measured.

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
| Seed is RoPE on Q | cut error jumps at post-RoPE; inject collapses final spike | cut rel **1.64e-3** (**×15.7**); inject hidden ratio **0.114**, logit gap 12.06→**0.20** | **SUPPORTED — positive causal evidence** |
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

## Narrowest next discriminating probe

Inside `rope_kernel` / HF `apply_rotary_pos_emb`, isolate which sub-step diverges:

1. Dump per-pair `cos`/`sin` (bf16) for positions 0…76 and pair indices 0…63 from both sides.
2. Dump one head's pre-RoPE Q and post-RoPE Q elementwise; check whether the closed-form  
   `out_re = re*cos - im*sin`, `out_im = im*cos + re*sin` matches HF on the **same** cos/sin.
3. Predeclared:
   - cos/sin tensors differ → freq construction or cast of angle;
   - cos/sin match, outputs differ → rotate pairing / multiply order / dtype of intermediates;
   - both match for layer-0 position 76 but full-seq dump still diverges → layout/indexing bug in dump only (unlikely given inject success).

Only after that one-line diagnosis should `src/hephaestus/kernels.mojo` be patched (separate decision; this investigation still does not modify engine code).

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
| **13** | **Q cut-points + replace-and-continue** | **`out/probe13_q_cuts.json`** |
