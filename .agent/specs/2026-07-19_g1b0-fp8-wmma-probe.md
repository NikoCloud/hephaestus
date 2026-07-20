# G1b-0 — FP8 WMMA Reachability Probe (Experiment Spec)

**Status:** DRAFT 2026-07-19 — for Opus + GLM review before ratification.
**Owner of spec:** Frank (Hermes). **Implementer:** TBD at ratification.
**Depends on:** `.agent/notes/2026-07-19_fp8-hardware-access-landscape.md` (the
research that motivates every constraint below), `src/hephaestus/wmma_gfx12.mojo`
(the BF16 v1/v2/v3a path this probe mirrors).
**Gate:** **G1b-0 — a binary yes/no: can ANY code path fire a correct FP8 E4M3 WMMA
matmul on gfx1201?** Not a performance gate. Not an engine gate. Reachability only.

---

## 0. Intent

Phase 1b's thesis (FP8-native compute) rests on an assumption that has **never been
demonstrated on this hardware by this project**: that an FP8 E4M3 WMMA instruction
can be made to execute and return numerically correct output on gfx1201. On
2026-07-12 the Mojo `llvm_intrinsic` FP8 path failed with
`LLVM ERROR: Do not know how to split this operator's operand!` on nightly
dev2026071006. That is a *toolchain* failure, not evidence the silicon can't do it —
and external prior art (ESDMAX, Triton, hipBLASLt) proves the instruction
**`v_wmma_f32_16x16x16_fp8_fp8` fires correctly on this exact card.**

This probe answers the reachability question with the cheapest experiments first,
and produces the **correctness oracle** all later FP8 work is validated against. It
builds *nothing* into the engine. Its only outputs are (a) a yes/no per path, (b) a
frozen reference output, (c) a decision line in DECISIONS.md selecting the 1b FP8
mechanism.

**Out of scope (do not build):** the layer-diff harness, the FP8 loader, the engine
integration, performance tuning, the 128×128 tile work. Those follow only after a
path passes here.

---

## 1. Constraints established by the landscape research

These are not optional; each is cited in the landscape doc.

1. **fp8×fp8→f32 ONLY.** RDNA4 WMMA has **no mixed bf16×fp8** operand mode (ESDMAX
   ISA finding, corroborated by AMD precision docs). Both A and B fragments must be
   E4M3. The probe's matmul is therefore **E4M3 × E4M3 → F32**, not "FP8 weights on
   BF16 activations."
2. **Activation quantization is mandatory downstream** — but NOT part of this probe.
   The probe feeds **pre-quantized E4M3 inputs** so the only variable is whether the
   WMMA instruction itself works. The bf16→e4m3 scale kernel is 1b scope, specced
   separately *after* G1b-0 passes. (Flagged for Opus/GLM: see §7 open question.)
3. **Instruction confirmed:** `v_wmma_f32_16x16x16_fp8_fp8`, 16×16×16 tile, fp8×fp8
   accumulate to f32. Same tile geometry as the working BF16 WMMA.
4. **Infinity Cache trap.** Navi 48 has a 64 MB LLC. Any microbenchmark that re-times
   the same small weight reads from LLC, not VRAM, and lies. This probe is
   correctness-only, so it does not benchmark — but the rule is recorded here so the
   *next* spec doesn't inherit a broken methodology.
5. **M ≥ 16.** WMMA requires M a multiple of 16 (or edge-masked). The probe uses
   M=N=K=16 exactly — one tile, no masking, no remainder. Smallest possible case.

---

## 2. The single test case

One 16×16×16 matmul, `C = A @ B`, all in E4M3, accumulate F32.

- **Inputs:** deterministic, non-random, chosen to expose mantissa errors. A = a
  fixed ramp of E4M3-representable values (1.0, 1.5, 2.0, … with a few subnormals
  and one value near E4M3 max ~448). B = a different fixed pattern (identity-ish
  plus off-diagonal entries) so C is not trivially symmetric. Exact values are
  frozen in the probe file so the CPU reference and every GPU path agree bit-for-bit
  on inputs.
- **CPU reference:** a host-side FP32 accumulator over the same A, B — plain nested
  loop, no GPU, no WMMA. This is the ground truth. Tolerance: FP8 WMMA accumulates
  in F32, so vs an F32 CPU reference the match should be **near-exact** (≤ a few F32
  ULPs on the largest-magnitude element; exact-zero on elements whose K-products
  cancel). Any argmax-relevant deviation fails the path.
- **Pass condition:** GPU output equals CPU reference within the stated tolerance,
  on the actual R9700 (GPU 0), verified by a device→host readback compare — the
  same round-trip-verify discipline the loader already uses (silent GPU failures are
  a known project hazard).

---

## 3. Path A — Mojo `llvm_intrinsic` on a newer nightly (try FIRST; cheapest)

The 2026-07-12 failure was on dev2026071006. Upstream moves nightly. The operand-
legalization error may already be fixed.

1. Create an isolated pixi env pinned to the **latest** Mojo nightly (mirror the
   existing `hephaestus-wmma-nightly` env pattern; do NOT disturb the repo default
   pin).
2. Write a minimal kernel calling `llvm_intrinsic` with the FP8 WMMA intrinsic name.
   The BF16 file already establishes the arity-3 `llvm_intrinsic` call shape; the
   FP8 analog is the same pattern with the fp8 intrinsic and 8-element fragments
   bitcast to the integer type the intrinsic expects. **Candidate intrinsic names to
   try, in order** (stop at the first that legalizes):
   - `llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8`
   - `llvm.amdgcn.wmma.f32.16x16x16.f8.f8`
   - (if neither exists in this LLVM) grep the Mojo stdlib / LLVM for the emitted
     name — `experiments/exp3f_fp8_wmma_probe.mojo` already probed this and its
     error output names what was attempted; start there.
3. **Outcome recording:**
   - If it compiles AND matches CPU reference → **G1b-0 PASS via Path A.** Record the
     exact nightly version + intrinsic name. Done; skip Path B.
   - If it compiles but produces wrong values → a legalization-quiet numerics bug;
     record and STILL try Path B (asm bypasses the legalizer entirely).
   - If it fails to compile (same or new LLVM error) → record the verbatim error,
     move to Path B.

**Timebox: one working session.** This is a toolchain probe, not a kernel project.

## 4. Path B — hand-written gfx12 asm (only if Path A fails)

Bypass LLVM instruction selection entirely: emit the raw `v_wmma_f32_16x16x16_fp8_fp8`
encoding by hand, packing operands per the RDNA4 ISA. This is the path Fable was
policy-blocked from attempting and the one IDEAS.md parks ("if upstream stalls").
ESDMAX proves a hand-rolled FP8 kernel on this exact card is achievable and beats
untuned AITER ~2×.

1. **Encoding source:** the RDNA4 ISA manual's `v_wmma_f32_16x16x16_fp8_fp8` entry,
   cross-checked against how Triton's gfx12 backend packs the same instruction
   (Triton is open-source; its AMD backend is the reference for operand packing).
   K3's demonstrated GPU-compiler work is the relevant capability here.
2. **Fragment layout:** mirror the G1b-0 lane mappings already locked for BF16 in
   `wmma_gfx12.mojo` (A-load / W-load / store), substituting E4M3 fragment widths.
   The 16×16×16 geometry is identical; only the per-element bit width (8 vs 16) and
   the fragment register count change.
3. **Pass condition:** identical to §2 (CPU reference within tolerance).
4. **Outcome recording:** PASS → record the working encoding + operand packing as
   the 1b mechanism. FAIL → record verbatim and escalate to a design review: if
   neither llvm_intrinsic nor hand-asm reaches the instruction, Phase 1b re-scopes
   around waiting for the upstream toolchain (modular/modular#6722 + the drafted
   FP8 comment in `.agent/notes/upstream-6722-comment.md`, which should then be
   posted).

**Timebox: bounded.** If hand-asm isn't passing within the agreed session count,
stop and re-scope — do not thrash. (This is the explicit anti-repeat of the
momentum-killing FP8 hammering that stalled the project.)

## 5. The correctness oracle (built regardless of path)

Independent of which path reaches the instruction, the probe stands up **one**
external FP8 GEMM on the dev box to serve as a cross-check oracle — the same role
the HF transformers oracle plays for the BF16 pass.

- **Preferred: Triton FP8 GEMM** (`tl.dot` on e4m3 inputs, gfx1201). Confirmed to
  auto-lower to `v_wmma_f32_16x16x16_fp8_fp8`. No vLLM, no AITER — a bare Triton
  kernel in the probe env. If Triton's gfx12 FP8 lowering is itself broken, fall
  back to:
- **hipBLASLt FP8 GEMM** via a tiny C++ test binary (NOT vendored into the engine —
  a standalone oracle executable). Caveat from research: gfx1201 hipBLASLt has live
  FP8-path regressions (issue #8242); if it crashes, the CPU reference (§2) remains
  the sole ground truth and that is acceptable.

The oracle answers "is my hand-built path computing the right thing" against an
independent implementation, not just against my own CPU loop.

## 6. Deliverables (all committed to the repo)

1. `experiments/g1b0_fp8_wmma_probe.mojo` — the single-tile test, all paths gated by
   comptime flags, CPU reference included.
2. `bench/g1b0.md` — verbatim results per path: compiles?, matches CPU?, the exact
   nightly version, intrinsic name, or asm encoding used. Honest losses recorded.
3. `DECISIONS.md` — one line selecting the 1b FP8 mechanism (Path A / Path B /
   re-scope), with evidence.
4. Update this spec's status to RATIFIED-with-results or RETIRED.

## 7. Open questions for Opus + GLM (attack these)

1. **Scope of the probe:** should the bf16→e4m3 activation-quant kernel be part of
   G1b-0, since a working FP8 WMMA is useless to the engine without it? Frank's
   position: NO — G1b-0 isolates the reachability variable; quant is 1b scope
   specced next. Counter-argument welcome: if quant is on the critical path and
   cheap, folding it in avoids a second probe cycle.
2. **The 8× prefill gap:** is Frank's read correct — that naive→v3a (3.5×)→v3b
   (~projected) is a ladder that closes most of it, and the FP8 prefill thesis is
   therefore about *topping* the ladder, not escaping a wall? Or is there a reason
   the naive→WMMA ladder can't reach llama.cpp-class prefill on this card?
3. **The fp8×fp8-only ISA claim:** ESDMAX asserts no mixed bf16×fp8 operand mode on
   RDNA4 WMMA. Is this corroborated anywhere but their write-up (AMD ISA manual,
   Triton source, hipBLASLt docs)? If mixed-mode DOES exist, the activation-quant
   requirement evaporates and 1b gets much simpler — this is the single highest-
   value fact to verify.
4. **Path ordering:** is llvm_intrinsic-first correct, or should hand-asm go first
   because it's the confirmed-working instruction and the llvm path already failed
   once on a near-current nightly?

---

*Drafted by Frank 2026-07-19. Methodology: every constraint traced to
`.agent/notes/2026-07-19_fp8-hardware-access-landscape.md` or to an existing repo
artifact (`wmma_gfx12.mojo`, `experiments/exp3f_fp8_wmma_probe.mojo`). No code is
built by this document; it selects and bounds an experiment.*
