# Handoff — Vulkan baseline + FP8-KV verdict (Phase 1b → Phase 2 boundary)

**Date:** 2026-07-20
**Author:** Opus. **Sources:** `bench/fp8-kv-verification.md`, `bench/llamacpp-vulkan-baseline.md`, `.agent/specs/2026-07-20_phase2-scope-skeleton.md` (all committed, `85779cf`).
**Purpose:** two results landed at the Phase 1b/2 boundary. One clears FP8-KV for Phase 2 architecture; the other moves the competitive goalpost and changes how we state gates. This doc is the record so neither has to be re-derived.

---

## 1. FP8-KV: verified PASS

The quality probe (`85879df`) simulated FP8-KV — E4M3 quant→dequant at cache-write time, BF16 storage — rather than implementing it. That separated the *question* (does quality hold?) from the *build* (storage format, read path, slot accounting) and made the answer cheap.

Two verifications were held open before the PASS was allowed into any doc. Both closed:

| check | flag | result | verdict |
|---|---|---|---|
| counter calibration, 16 tok | ON | 1,152 | PASS |
| counter calibration, 32 tok | ON | 2,304 | PASS (linear) |
| token counter @ 4K | ON | 294,912 = 4096×36×2 | PASS |
| token counter @ 4K | OFF | 0 | PASS |
| E4M3 grid assert | ON | 0 violations | PASS |
| E4M3 grid assert | OFF | 282,047,598 violations | PASS (assert is not a no-op) |

**Verdict: the 99.93% is a real FP8-KV result. The flag did not no-op.**

Two design points worth preserving, because both were nearly missed:

- **The counter measures tokens covered, not invocations.** Call granularity is per-layer × per-{K,V} × per-forward-chunk — a 4K walk is only 4,608 host launches. An invocation counter would have read three orders of magnitude low and looked exactly like a no-op. False negatives in a gate are cheaper than false positives but still cost a day.
- **The grid assertion has its own negative control.** 282M violations with the flag OFF (93.4% of 301,989,888 values; the 6.6% that pass are zeros, small integers, exact powers of two) is what makes 0-violations-ON mean anything. An assertion that passes in both states is itself a no-op.

**Convention adopted: every gate gets a negative control that can fail.**

### Divergence and the anomaly

| length | divergent | rate |
|---|---|---|
| 512 self-A/B | 2 / 512 | 0.391% |
| 4K self-A/B | 3 / 4096 | 0.073% |

4K diverges ~5.3× *less* than 512 on the same axis — backwards from naive error accumulation. Recorded with its resolution so it isn't re-derived:

- **Cross-axis non-comparability (kills a phantom):** the earlier 768-token oracle run (748/768 → 743/768, a 5-token delta) uses a *different token set* and a *different axis* (net-of-oracle) than the 512 self-A/B (a prefix of the 4K probe sequence). The 5 and the 2 were never comparable and their failure to reconcile is not a contradiction.
- **The 512 window is a prefix of the 4K sequence,** so this is a density comparison *within one sequence*: 2/512 in the early region, 1/3584 in the remainder. The mismatches at positions 71 and 149 are literally the first two of the three 4K mismatches; 3071 is the only late one.
- **Hypothesis, labeled as hypothesis:** longer-context text is lower-entropy, argmax margins wider, fewer near-ties flip. Consistent with the data, not proven. **n=3 is far too small to generalize** — the honest statement is that error accumulation is *not the dominant pattern here*, not that divergence decreases with context.

Divergence shape is healthy: sparse, deterministic near-tie flips at specific low-margin tokens, no tail clustering, no snowball. That is what a correct per-token scaling scheme looks like.

### Carried forward

- **Capacity: 1.992×** (147,456 → 74,016 B/token). FP8 *weights* by comparison contribute ~5% and ~1.7 slots — marginal. KV is where the capacity thesis lives.
- **The simulation is the bitwise oracle.** A real FP8-KV cache is correct iff it reproduces `85879df` exactly. This is the durable payoff of simulating first, and it is a Phase 2 acceptance criterion.

---

## 2. Vulkan baseline: the goalpost moved, and one anchor is now obsolete

All prior llama baselines were ROCm/HIP. Co-measured (ROCm → Vulkan → ROCm, +0.21% thermal drift), same commit `33ca0dcb9`, same Q8 model, RADV GFX1201 confirmed 37/37 offload, FA on both, KV self 2304 MiB identical.

| npl | ROCm agg | Vulkan agg | VK/ROCm | ROCm %roof | VK %roof |
|---|---|---|---|---|---|
| 1 | 106.11 | 121.98 | 1.150× | 81% | 93% |
| 2 | 185.69 | 224.24 | 1.208× | 72% | 87% |
| 4 | 345.12 | 417.76 | 1.210× | 70% | 84% |
| 8 | 549.65 | 677.83 | 1.233× | 59% | **73%** |
| 16 | 962.30 | 873.52 | 0.908× | **59%** | 53% |

Single-stream: Vulkan tg128 127.06 vs ROCm 109.39 = **+16%**. Batched prefill: Vulkan trails 7–10% at every npl. No feature gap — batched-bench, FA, and KV footprint are identical.

### Three findings

**(a) The competitive target moved 23% at our gate point.** Pillar 2's gate is aggregate throughput at 8 concurrent. Best-in-class there is now Vulkan's 677.83, not ROCm's 549.65. Our decode position widened from 0.60× (vs ROCm) to **0.52×** (vs Vulkan). Prefill stays 0.22×.

**(b) The scaling *shapes* differ structurally, and that is the reusable finding.** ROCm: 81 → 72 → 70 → 59 → **59** — hits a bound and holds. Vulkan: 93 → 87 → 84 → 73 → **53** — never bounds, keeps degrading, surrenders a 23% lead between npl 8 and 16. Read: RADV is the better *kernel* runtime; ROCm has the better *batching* story. Phase 2 is the batching story, so ROCm's flat tail is the behavior to study and Vulkan's collapse is the behavior not to reproduce.

**(c) Best-in-class is an envelope across two backends that no user gets from one build.** Vulkan owns decode ≤8; ROCm owns decode at 16 and batched prefill throughout. State it as an envelope — more accurate than picking whichever flatters us, and more defensible when someone re-tests.

### The 460 GB/s anchor is superseded — action required on the skeleton

Earlier sizing used **460 GB/s** as the "practical ceiling" (llama ROCm at npl=1, where KV is 1.9% of bytes — effectively a pure weight sweep), on the reasoning that the synthetic 569 is unreachable on a mixed access pattern.

**Vulkan measured 530.7 GB/s at npl=1 — 93.3% of the synthetic 569, on that same mixed workload.** So 460 was never a hardware-practical ceiling; it was *ROCm's implementation limit*. The anchor is obsolete.

Consequences:

1. **`.agent/specs/2026-07-20_phase2-scope-skeleton.md` §3 has a live contradiction.** Its gate bullet quotes 569-based percentages (73% @ npl=8, 59% @ npl=16 — the figures in both bench tables), while its calibration bullet instructs renormalizing everything to 460. Two denominators in one gate. **Fix: use 569 throughout**, one denominator, consistent with both bench docs. Record 530.7 as best-measured-achievable rather than as a divisor.
2. **This strengthens rather than weakens the M=1 thesis.** Vulkan at 93.3% of synthetic roofline single-stream means there is essentially nothing left there for anyone. "No 2× hiding at M=1" is now measured, not argued.
3. **It also raises the concurrency prize.** If ~531 GB/s is demonstrably reachable and the best batched result is 416.5 (Vulkan @ npl=8), the gap between what the memory system can do and what batched serving extracts is ~22% — and ROCm gives up even more. That gap is the Phase 2 opportunity, stated in measured terms.

### Measurement caveat

ROCm pp512 stddev is ±864 and ±1310 (11–17%, cold first rep per process); Vulkan's is ±138 (1.8%). **Do not quote the llama-bench prefill "tie"** — it is entirely inside ROCm's noise. The batched prefill numbers, where Vulkan trails 7–10% consistently, are the trustworthy read.

---

## 3. What this changes for Phase 2

- **FP8-KV is cleared to enter Phase 2 as designed-in architecture** — storage format, dequant-on-read, slot accounting — with its quality gate satisfied rather than pending. Validate first, then scope; that sequencing held.
- **Gates are stated as roofline percentage, not tok/s.** A tok/s target moves when upstream lands a patch; a percentage does not. Report the tok/s ratio as headline, the percentage as the progress metric. This is the durable fix for the moving-goalpost problem.
- **Two gates, labeled, not merged.** Pillar gate: ≥3× aggregate at npl=8 vs our own npl=1 — *does the Multiplier multiply*. Competitive gate: beat the envelope per point, 73% @ npl=8 and 59% @ npl=16 — *is it implemented well*. These can diverge; 3× self-relative at 40% of roofline is a real possible outcome and would mean something specific.
- **The honest anchor stays in the room:** llama serves 16 streams on ~6.5 GB; our 8-process probe used 32 GB. That is the distance fused batching has to cover.
- **A cheap probe runs before the batching build.** See skeleton §4. It has no scheduler in it, so it establishes the *floor*: collapse with no scheduler present is decisive and no scheduler rescues it; flat clears one of two failure modes, not both. A flat curve must not be written up as "Phase 2 de-risked."

---

## 4. Open items

1. Skeleton §3: resolve the 569/460 double denominator per §2 above. **Blocking** — the probe spec inherits this gate.
2. Phase 2 pre-build concurrency-shape probe → sets the real gate from a measured curve.
3. Phase 1b writeup / Modular forum post, led by the **ABI contribution**, not benchmarks. Drafted comment at `.agent/notes/upstream-6722-comment.md`. No batching claim in it.
4. Act-quant fusion (GEMM 1.47 → 2.0 gap). One fusion at a time, teacher-forced after each.

---

*Every number here traces to a committed bench file. Anomalies are recorded with their resolutions, and hypotheses are labeled as hypotheses.*
