# 768-step teacher-forced check — results
## 2026-07-12

Requested check: for each of 3 prompts, 256 teacher-forced steps (history =
prompt + oracle's own generated tokens up to that point, so no error
propagation), does Hephaestus's argmax ever flip on a clear non-tie? What is
the measured max_abs_diff?

Method: single forward call per prompt over the full sequence
(prompt + oracle_output[:255]) — causal attention makes each row's logits
correct regardless of how far into the sequence it sits, so this is
equivalent to 256 sequential teacher-forced decode steps but ~100x cheaper
(one prefill vs 256 forward calls).

## Finding 0 (methodological): HF disagrees with itself

Computed HF's logits the same way (single-shot batched forward, sdpa). Before
even comparing to Hephaestus: **HF's own single-shot recompute does not
reproduce its own cached-autoregressive generation** — 255/256, 250/256,
252/256 self-agreement on prompts 1/2/3. Batched vs KV-cached attention
produces different rounding order within the same HF codebase. This means
"vs the committed oracle tokens" and "vs HF computed the same way we compute"
are two different (both legitimate) comparisons — reported both below.

## Results

| prompt | vs oracle (cached gen) | vs HF single-shot (apples-to-apples) | full-vocab max_abs_diff |
|---|---|---|---|
| 1 | 0/256 mismatch | 1/256 | 12.0621 (step 67, vocab id 96874) |
| 2 | 5/256 mismatch | 5/256 | 5.1435 (step 218, vocab id 21132) |
| 3 | 7/256 mismatch | 7/256 | 1.7485 (step 205, vocab id 33464) |
| **total** | 12/768 | **13/768** | **12.0621** |

## Decisive question: does argmax ever flip on a non-tie?

For every one of the 13 apples-to-apples mismatches, classified HF's own
top-1/top-2 gap at that step:

| prompt | exact ties (gap=0.0) | 1-bf16-ulp near-ties | clear non-tie flips |
|---|---|---|---|
| 1 | 1 | 0 | **0** |
| 2 | 3 | 2 | **0** |
| 3 | 7 | 0 | **0** |

**Answer: NO. Zero non-tie flips across all 768 steps.** Every single
argmax disagreement occurs where HF's own bf16 logits are either exactly
tied (bit-identical top-1 and top-2) or separated by exactly one bf16 ulp at
that magnitude (0.0625 at ~12, 0.125 at ~21–25 — matches `2^(exponent-7)`
precisely). At an exact tie there is no more information in bf16 to
disambiguate; the "correct" answer is implementation-defined.

## The max_abs_diff number needs a caveat, not a pass/fail on its own

12.06 sounds alarming reported bare. Investigated before including it:

- It occurs on **vocab id 96874, HF rank 2281** (i.e., a token HF itself
  considers wildly improbable) at **one single step (67)**. The same token
  at every *other* step in the same prompt differs by ~0.075 — ordinary
  noise. This is not a growing/systemic drift; it's an isolated (step, token)
  event.
- Per-step max_abs_diff, sampled every 8 steps across all 3 prompts, shows a
  gentle background (mean 0.02–0.1, max typically 0.1–0.8) consistent with
  ordinary bf16 accumulation over 36 layers, plus a handful of isolated
  spikes (this one, and four others: 4.15, 5.14, 1.75, 1.55) that don't fit
  a smooth trend.
- Ruled out one candidate root cause: GPU `cos`/`sin` precision at the large
  raw angles RoPE's highest-frequency term produces (up to ~265 radians at
  late positions). Measured directly against Python's `math.cos`/`sin`:
  error is 1e-6 to 1e-8 — six orders of magnitude too small to explain a
  12-unit logit gap. (`experiments/exp6_gpu_trig_probe.mojo`)
- Root cause of the spikes themselves is **not yet identified**. They do not
  affect any argmax decision found so far (the divergent tokens are always
  far outside the top-1 contest), but they are a real, unexplained
  numerical event and are logged here rather than smoothed over.

## What this means for the gate, factually

- The "token-identical" framing is confirmed unanswerable past ~4 tokens
  (previous session), and this check adds: even restated as "argmax never
  flips except at genuine ties," Hephaestus **passes** — 0/768 non-tie
  flips, measured, not assumed.
- A bound on *full-vocab* max_abs_diff cannot be set "tight" without
  incorrectly failing prompt 1 (which is the prompt that also achieved
  0-mismatch/token-exact-256/256 against the original oracle) — its 12.06
  outlier is on a token nobody would ever select. Full-vocab max_abs_diff is
  not the right instrument for a greedy-decode correctness gate; it would
  fail on an argmax-irrelevant tail token in the single case that behaved
  best by every other measure.
- A bound *at the decision boundary* (top-1 vs top-2 gap when they disagree)
  is well-supported by this data: every disagreement is within one bf16 ulp.

No SPEC.md edit made. Awaiting decision on how to state this in the exit
criteria.
