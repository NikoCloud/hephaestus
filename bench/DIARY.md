# Bench diary — methodology lessons

Cautionary notes about *how we measure*, kept next to the benchmarks so the lessons
don't get re-learned. Each entry is a mistake or a near-miss in probe design, not a
result. Results live in their own `bench/*.md`; this file is why some of them can't
be trusted, or why some were never written.

---

## 2026-07-20 — The null concurrency probe: a design whose output was forced by its plumbing

**What it was.** A probe meant to answer: *does our engine hold throughput as concurrency
rises — flatten like ROCm or droop like Vulkan?* As specified and built, it fired **N
independent `forward_fp8` passes on one `DeviceContext` / one HIP stream**, then synchronized.

**Why it returned nothing.** A HIP stream is an in-order queue. N forwards on it execute
strictly sequentially, so step time was 20 → 40 → 81 → 161 ms (N × single-stream, ±1%) and
aggregate was pinned at the single-stream rate (~49.5 tok/s) **by construction** — independent
of anything about our kernels, occupancy, or memory system. The "curve" (efficiency 35% → 5%)
was the identity function wearing a graph.

**The tell we almost logged as a PASS.** One negative control was *"N=2 step time > N=1 step
time (40.5 ms > 20.4 ms, ≈2×)"*, recorded as confirming the harness works. But 2× time for 2×
work **is** serialization. That control confirmed the harness serializes; it did not confirm the
harness measures concurrency. A probe must be able to return a value *other than* the one its
submission model forces — this one could not.

**The evidence that contradicted it.** DECISIONS 2026-07-19: 8 *independent OS processes*
(8 HIP contexts, genuinely concurrent submission) scaled **8.0× linearly, 453 tok/s aggregate**,
same silicon, same kernels. 453 vs 49.5 is a 9× spread attributable entirely to the submission
model. The memory system overlaps fine when work is actually submitted concurrently; the probe
measured a queue, not a floor.

**Worse: it tested the wrong architecture anyway.** Phase 2 is *fused* batching — N requests
become M rows of one GEMM, weights read once. "N independent forwards" is the opposite shape,
and one we already documented as a dead end (the 8-process probe was explicitly "not a product
configuration"). Even a correctly-submitted multi-stream version would have answered a
secondary question.

**What replaced it.** The fused M-row decode probe (`bench/fused-mrow-decode-probe.md`), which
asked the right question — *does M-row fusion amortize weights?* — and answered yes, leading to
the small-M kernel (`bench/small-m-decode-gemm.md`).

**Lesson.** Before running a probe, ask: *what outcomes can this design produce, and is the one
I expect the only one its plumbing allows?* If the answer to "could this return anything else?"
is no, the probe is measuring itself. Review of the concurrency probe caught that it "had no
scheduler in it" but missed that it had no concurrency in it either — the flaw was one level
below the one that got flagged.
