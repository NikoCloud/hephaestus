# Phase 2 Scope Skeleton — "The Multiplier" (for GLM → Grok spec)

**Status:** DRAFT 2026-07-20 — Frank. For GLM to expand into a Grok build spec.
**Inputs (all measured, all on main or cited branches):**
- FP8 WMMA prefill GEMM: 87 → 1703 tok/s (19.6×); FP8 > BF16 1.24× e2e, 1.47× GEMM.
- FP8 decode > BF16 decode (66.1 vs 62.37).
- FP8-KV quality probe: PASS, negative-controlled, 99.93% self-A/B @4K, 1.992× capacity (`bench/fp8-kv-verification.md`).
- llama batched baseline + Vulkan envelope (`bench/llamacpp-batched-baseline.md` + handoff).
- Opus's wave arithmetic + the ROCm-vs-Vulkan scaling-shape finding.

**This is a scope skeleton, not a build spec.** It names what's in, what's out, the
gates, and the one cheap probe that must run *before* the batching build. GLM expands
it into the executable spec.

---

## 0. The thesis, restated for Phase 2

Single-stream is a structural dead end. Three independent lines converged this
session: (a) Opus's wave arithmetic — the M=1 decode kernel uses ~3% of the machine's
wave slots; (b) three measured M=1 regressions (LDS, swizzle, split-K); (c) the
baseline — on single-stream decode the best anyone achieves is 93% of roofline, so
**there is no 2× hiding there for anyone.**

**The win is at concurrency.** Best-in-class batched decode is 73% of roofline *and
falling* (Vulkan's collapse). That is where a better architecture pays, and where
FP8-KV's 2× capacity compounds. Phase 2 is the batching story.

## 1. What's IN scope

1. **Continuous batching scheduler** — chunked prefill, per-request KV, concurrent
   decode. This is the engine's reason to exist (multi-agent, subagent fan-out).
2. **FP8-KV cache — designed in, not bolted on.** Storage format (E4M3, per-token
   absmax/448, separate K/V, post-RoPE K), dequant-on-read path, slot accounting.
   **The `fp8-kv-probe` simulation is the bitwise oracle**: the real cache is correct
   iff it reproduces `85879df`'s output exactly.
3. **Paged KV block allocator** — capacity is the FP8-KV win. Paging eliminates
   fragmentation/over-allocation (a *capacity* win, NOT a KV-read reduction — every
   sequence still reads its own cache).
4. **OpenAI-compatible `/v1/chat/completions` + streaming** — the serving surface.

## 2. What's OUT of scope (explicit non-goals this phase)

- Tensor parallelism / multi-GPU (one engine per card; GPU 0 only for dev).
- Chasing llama's single-stream decode or Q8 prefill (never a gate; kernel maturity).
- FP8 weights changes (validated, frozen).
- WMMA attention (candidate lever, separate gate, not required for Phase 2 exit).
- Further M=1 decode micro-optimization (measured dead end; batching supersedes).

## 3. THE GATE — roofline-percentage, not a tok/s number (adopt as standing convention)

The goalpost moved 23% when the Vulkan baseline landed. The fix that survives
upstream patches: **state the gate as % of roofline, report tok/s ratio as headline.**

- **Competitive gate (envelope roofline %): beat the envelope at each concurrency
  point — 73% @ npl=8 (Vulkan), 59% @ npl=16 (ROCm).** Not a single number: 73% is
  Vulkan at npl=8; at npl=16 Vulkan is 53% and ROCm is 59%, so best-in-class at
  npl=16 is ROCm's 59%. We must beat Vulkan where Vulkan is strong *and* not
  collapse where Vulkan collapses — winning at 8 while drooping at 16 would be
  reproducing Vulkan's failure with extra steps. ROCm's flat tail (59% floor) is the
  behavior to study; Vulkan's collapse (73→53) is the behavior to avoid.
- **Pillar gate (does the Multiplier multiply): ≥3× aggregate throughput at npl=8
  vs our own npl=1.** This is the *original* Pillar 2 gate and it is NOT the same
  test as the roofline % — the 3× is a self-relative multiplier (do we actually
  multiply?); the roofline % is a competitive-efficiency measure (are we implemented
  well?). Both are kept, labeled, and they can diverge (3× self-relative while
  sitting at 40% of roofline). Pillar gate = does it multiply; competitive gate =
  is it good. (This reconciles, not silently replaces, the stated Pillar 2 gate —
  per the supersession convention.)
- **Denominator: 569 GB/s throughout. One denominator, no renormalization.**
  [SUPERSEDED 2026-07-20 — the earlier "460 GB/s practical ceiling" is retired.]
  460 came from llama *ROCm* at npl=1 and was taken as the achievable ceiling on a
  mixed weights+KV access pattern. The Vulkan co-measure disproves it: **Vulkan
  reaches 530.7 GB/s at npl=1 — 93.3% of the synthetic 569 — on that same workload.**
  460 was ROCm's *implementation limit*, not a hardware ceiling. All percentages in
  this spec and in both bench tables are % of 569; do not renormalize.
- **What 530.7 means (record it as a measurement, not a divisor):** (a) it
  strengthens the M=1 thesis — at 93.3% of roofline single-stream there is
  essentially nothing left there for anyone, now measured rather than argued;
  (b) it raises the concurrency prize — ~531 GB/s is demonstrably reachable while
  the best batched result is 416.5 GB/s (Vulkan @ npl=8), a ~22% gap that is the
  Phase 2 opportunity stated in measured terms.
- **Honest anchor (the distance to cover):** llama serves 16 streams on ~6.5 GB; our
  8-process probe used 32 GB. That gap is what fused batching must close.

## 4. THE PRE-BUILD PROBE — "simulate, don't implement" (Opus's instruction, load-bearing)

The FP8-KV probe was cheap because it separated the *question* (does quality hold?)
from the *build* (storage format, read path). Phase 2 has an analogous unknown, and
it is **the single most expensive discovery available if found late**:

> **Does our engine hold throughput as concurrency rises — do we flatten like
> ROCm, or droop like Vulkan?**

Before building the full paged-KV serving path, run a **cheap directional probe**:
a minimal concurrent-decode harness that ramps simulated concurrency (N synthetic
requests sharing one weight stream, per-request KV) and measures the effective-
bandwidth curve shape. We do NOT need the production scheduler to learn the curve's
*shape* — flat-tail vs collapse — only its direction. If our kernel/memory shape looks
like Vulkan's collapse, that is the most expensive thing to learn after building;
learn it now, crudely.

**Be precise about the probe's warrant — it has no scheduler in it.** "N synthetic
requests sharing one weight stream" measures whether our **kernels and memory
system** hold shape under concurrent work — not whether a *scheduler* does. The
ROCm-flat / Vulkan-collapse divergence may itself *be* a scheduler artifact (batch
formation, dispatch granularity, synchronization), which this probe cannot see. So
the probe establishes the **floor**: if the kernel/bandwidth curve already collapses
with no scheduler in the picture, that's decisive and cheap, and no scheduler can
rescue it. If it holds flat, we've cleared ONE of two failure modes, not both — a
flat curve here must NOT be read as "Phase 2 de-risked."

**Probe exit:** a measured efficiency-vs-concurrency curve on our own engine, with
the flat-vs-droop shape named AND its warrant stated (kernel/memory floor, not
scheduler). This sets the *real* Phase 2 gate from a measured number on our own
engine, rather than from any extrapolation off llama's curve — the reason the
earlier 460-renormalization had to be retired is exactly that a number borrowed
from one backend's implementation limit is not a property of the hardware.

## 5. FP8-KV integration gates (already satisfied, carried forward)

- Quality: ≥95% argmax parity @4K teacher-forced — **PASS (99.93%)**, negative-
  controlled (`bench/fp8-kv-verification.md`). Real-cache correctness = bit-exact
  match to the simulation oracle.
- Capacity: 1.992× slots (147456 → 74016 B/token). This is the capacity half of the
  thesis; FP8 *weights* contribute only ~5% and 1.7 slots (marginal).

## 6. Correctness protocol (institutional memory — do not weaken)

- Teacher-forced re-verify after EVERY structural change (two fusion regressions
  already cost us: QK-norm, gate+up). One change at a time.
- Every gate gets a negative control that can FAIL (282M violations with flag off is
  what made 0-violations-on mean something). Verification-of-the-verifier at every level.
- Record anomalies *with their resolution* in the doc (the 4K-quieter-than-512
  entropy hypothesis is labeled hypothesis, not asserted; the 768-vs-512 cross-axis
  non-comparability is stated so it can't be re-derived).

## 7. Sequencing for GLM → Grok

1. **Pre-build concurrency-shape probe** (§4) — cheap, directional, sets the real gate.
2. **FP8-KV real cache** (storage + read path + slot accounting) vs the simulation oracle.
3. **Continuous batching scheduler** on top of paged FP8-KV.
4. **Co-measured batched sweep** (npl 1/2/4/8/16, total-bytes accounting) vs llama
   ROCm + Vulkan envelope — gate = beat 73% roofline @ npl=16.
5. Land on main. main stays the truth (two phantom episodes this session proved why).

---

*Drafted by Frank 2026-07-20. Every load-bearing number traced to a committed bench
file or handoff. The one thing this skeleton deliberately does NOT do is set a tok/s
gate — that is the pre-build probe's job to measure, per §4.*
