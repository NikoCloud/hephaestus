# CONCURRENCY SHAPE PROBE — Does our kernel/memory system hold throughput as concurrency rises?

**Status:** [SUPERSEDED 2026-07-20 — NULL RESULT] The probe as specified fired N independent forwards on one in-order HIP stream, which serializes by construction; it measured a queue, not a kernel/memory floor. Replaced by the fused M-row probe (bench/fused-mrow-decode-probe.md). The design lesson is recorded in bench/DIARY.md. This spec is kept for the record; do not re-run it.
**Owner of spec:** Frank (GLM). **Implementer:** Grok (fresh session). **Target:** `bench/concurrency-shape-probe.md`.
**Depends on:** Phase 1b closeout on `main` (FP8 WMMA decode + prefill, attention stopgap). Does NOT depend on any Phase 2 build.
**Rule:** this probe evaluates our engine's kernel/memory floor with NO scheduler. Collapse here is decisive — no scheduler rescues it. Flat clears ONE of two failure modes, not both. A flat curve must not be written up as "Phase 2 de-risked."

---

## 0. Intent

Phase 2's premise is "we beat llama at concurrency." The llama batched baseline (`bench/llamacpp-batched-baseline.md`) shows two shapes:
- **ROCm:** roofline efficiency 81% → 72% → 70% → 59% → **59%** (flattens at npl=8, holds)
- **Vulkan:** roofline efficiency 93% → 87% → 84% → 73% → **53%** (never flattens, keeps degrading)

The question this probe answers: **does our engine's kernel/memory system flatten like ROCm or droop like Vulkan as concurrency rises — with NO scheduler, NO fused batching, just N independent forward passes sharing one weight read?**

This is the floor. If the floor collapses, Phase 2's scheduler is built on sand. If the floor holds flat, the scheduler is the remaining risk (not the kernel/memory system). Either answer is valuable and changes the build plan.

---

## 1. Design — crude is fine, direction not precision

N synthetic decode requests, each with its own KV cache and activations, sharing one weight arena (one copy in VRAM). Ramp N = 1, 2, 4, 8. Measure aggregate tok/s and per-stream tok/s.

**No scheduler.** No request queue, no dynamic batching, no paged KV. Just: launch N forward passes that share the same weight DeviceBuffer, each writing to its own activation/KV buffers. This can be sequential launches (N enqueue_function calls per layer) or interleaved — the point is measuring whether the GPU can overlap N independent decode streams reading the same weights.

**Implementation approach (simplest viable):**
- Load weights ONCE into the arena (4.02 GB FP8)
- Allocate N sets of activation buffers and N KV caches
- For each decode step: loop over N streams, calling `forward_fp8` for each with its own buffers but the shared weights
- Measure: wall time per decode step (all N streams), aggregate tok/s = N / step_time

This is NOT fused batching (one kernel launch processing M rows). It's N separate forward passes sharing a weight read. The question is whether the GPU's memory system overlaps the weight reads across streams — which is the prerequisite for fused batching to work.

---

## 2. Measurement protocol

**GPU hygiene (non-negotiable):**
- GPU 0 only, `HIP_VISIBLE_DEVICES=0`
- Check `rocm-smi` — kill everything on both GPUs before starting
- Named tmux session so the run is recoverable
- Confirm both GPUs at 0% util and baseline VRAM before first measurement

**Co-measured:** run llama-batched-bench at npl=1,2,4,8 in the same session, same thermal window. Report both curves side by side. The ratio is the number; cross-session absolutes are noise (proven twice this session).

**Byte accounting (corrected formula):**
- Model: FP8, 4.02 GB
- KV per token: 8 KV heads × 128 head_dim × 2 (K+V) × 36 layers × 2 bytes = 147,456 bytes
- Average context during generation: ~576 tokens (512 prefill + 64 average during 128-gen)
- Total bytes per forward = `4.02 GB + (npl × 576 × 147456 / 1024^3)` in GB
- Effective bandwidth = `(aggregate_tok_s × total_bytes_per_forward) / npl` in GB/s
- Efficiency vs roofline = `effective_bandwidth / 569 × 100` as %

**Calibration:** renormalize to our own npl=1 as the self-relative anchor. The absolute roofline % at npl=1 tells us our current single-stream efficiency (expected ~47% based on 66.1 tok/s). The SHAPE of the curve (flat vs droop) relative to our own npl=1 is the decision-relevant output.

**What to sweep:** N = 1, 2, 4, 8. (Not 16 — at N=8 with 4.02 GB weights + 8 KV caches, we're near the VRAM ceiling. Check VRAM at each N; stop if OOM.)

---

## 3. Negative control (mandatory — can fail)

Same class as the FP8-KV token-counter calibration. Before trusting any curve:

1. **Confirm the concurrency ramp actually ramps.** At N=1, the step time should be ~15ms (66 tok/s). At N=2, if step time is still ~15ms, the streams aren't independent — they're aliased to one buffer. Assert that N=2 step time > N=1 step time (each stream does real work).

2. **Confirm per-request KV is genuinely per-request.** Each stream must have its own KV cache at a distinct device pointer. Print the KV cache pointers for all N streams at startup — if any two are equal, the allocation is aliased and the measurement is bogus.

3. **Confirm weights are shared, not duplicated.** Print the weight arena pointer — there should be exactly one, used by all N streams. If each stream has its own copy, VRAM will OOM at N=4-8 and the measurement is the process-concurrency probe (which we already did), not the shared-weight probe.

If any of these fail: stop, report the failure, do NOT report throughput numbers. A failed negative control invalidates the entire run.

---

## 4. Report — `bench/concurrency-shape-probe.md`

Table format:

| N | per-stream tok/s | aggregate tok/s | scaling % | total bytes/fwd (GB) | eff BW (GB/s) | % roofline | VRAM (MB) |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | | 
| 4 | | | | | | | | 
| 8 | | | | | | | | 

Plus:
- **Shape named:** flat (holds within ~10% of npl=1 efficiency) or droop (degrades >15% from npl=1 efficiency)
- **Comparison to llama's curves:** our shape vs ROCm (flat-tail) vs Vulkan (collapse)
- **Negative control results:** all three checks, pass/fail
- **Warrant stated:** "kernel/memory floor, not scheduler" — a flat curve does NOT mean Phase 2 is de-risked, it means the floor holds and the scheduler is the remaining risk
- **Calibration note:** npl=1 self-relative anchor, renormalized to our own baseline
- **Co-measured:** llama batched numbers from the same session

**Decision rule:**
- **Flat (holds within ~10%):** Phase 2 proceeds to build spec. The scheduler is the remaining risk. State explicitly: "floor holds, scheduler untested."
- **Droop (degrades >15%):** Stop. No scheduler recovers a kernel/memory floor that already droops. The lever moves to whatever the collapse points at. Phase 2's shape changes before any build starts.

---

## 5. Constraints

- **Do NOT optimize anything.** This is measurement, not tuning.
- **Do NOT touch main.** Branch `concurrency-shape-probe` from main.
- **Do NOT build a scheduler, fused batching, or paged KV.** This is N independent forward passes sharing weights.
- **Fresh Grok session.** No prior-turn context needed; independence is cheap.
- **GPU 0 only. Free both GPUs first.**
- **Work in isolated env:** `~/projects/hephaestus-wmma-nightly`.
- **Always state the backend** for the llama co-measurement.

---

*Spec ends. The probe's curve sets the Phase 2 gate. Measure before building.*
