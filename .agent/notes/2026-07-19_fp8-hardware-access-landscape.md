# RDNA4 FP8 Hardware-Access Landscape — Research (2026-07-19)

> ## REVIEW STATUS (added 2026-07-19 on merge to main)
>
> **This research is retained because most of it is sound and several findings were not
> previously of record.** Three corrections apply, all stemming from the author reading
> `main`, which at the time still carried the 2026-07-12 WMMA blocker as standing fact
> and did not contain `exp3g`:
>
> 1. **FP8 WMMA reachability is NOT an open question.** It was proven on 2026-07-13 via
>    direct Mojo `llvm_intrinsic` (`experiments/exp3g_fp8_wmma_gfx12.mojo`, PASS), and a
>    full W8A8 decode kernel has been running on branch `fp8-wmma-decode` since. **Section 5
>    ("newer nightly, or hand-asm") is superseded** -- neither is needed. No hand-written
>    assembly is required.
> 2. **Activation quantization is built, not pending scope.** Per-token absmax bf16->e4m3
>    is implemented and validated at 97.4% teacher-forced argmax parity. The ISA constraint
>    in Section 1 is correct and independently confirmed by enumerating the LLVM gfx12 WMMA
>    builtins -- it is simply already handled.
> 3. **ESDMAX absolute numbers are not comparable to ours.** Their llama.cpp Q8_0 baseline
>    is 16 tok/s; ours co-measured on the same card is 109.5 tok/s -- they are running a far
>    larger model. **Treat their ratios as the signal, never the absolutes.**
>
> **What stands and is valuable:** the fp8xfp8-only ISA constraint; the finding that batch-1
> decode is memory-bound and FP8 WMMA is therefore *not* a decode speedup (independently
> corroborates our own measured LDS/swizzle regressions); the 64 MB Infinity Cache
> benchmark trap; the prefill `M % 64` cliff; the AITER gfx1201 tuning gap; hipBLASLt
> immaturity; and the MTP speculative-decoding result now tracked in IDEAS.md.


**Purpose.** Answer one question with external evidence: *by what mechanism does any
existing software reach the FP8 WMMA units on gfx1201 (RDNA4)?* This document exists
because that research was done once before and lived only in a chat transcript — it
was never committed, so per the repo rule ("if it only happened in your terminal, it
didn't happen") it didn't happen. It is now of record.

**Scope.** This is intelligence for the G1b-0 probe (see IDEAS.md / DECISIONS.md
2026-07-19 entries). It is *not* a design doc and *not* a commitment to any
implementation. No code is proposed here. Sources are linked inline; nothing here is
asserted without a citation.

---

## 0. The one-paragraph answer

There are exactly **three** demonstrated ways to fire an FP8 WMMA instruction on
gfx1201, and every working community effort uses one of them:

1. **Triton `tl.dot()` auto-lowering.** The Triton compiler detects FP8
   (`float8_e4m3fn`) operands + gfx1201 and emits
   `v_wmma_f32_16x16x16_fp8_fp8` natively. This is how the vLLM community patch
   (issue #28649) and AITER's Triton kernels reach the hardware. **Nobody wrote asm.**
2. **A hand-written kernel bypassing the toolchain's instruction selection.** The
   ESDMAX vLLM fork (mininmaxim/vllm, branch `esdmax-r9700-fp8`) — written by "Codex"
   per its own README — runs a custom FP8 block-scaled GEMM on the R9700 and beats
   untuned AITER by ~2× on decode shapes. Mechanism: custom kernel, M-dependent
   dispatch behind an opaque custom op.
3. **hipBLASLt.** AMD's own GEMM library declares `hipblaslt_f8` (E4M3) supported on
   "gfx950 and gfx12." This is the *maintained, first-party* path — but it is immature
   on gfx1201 (open SIGSEGV regressions) and, critically, it is a **C++ ABI**, which
   Mojo cannot call without a shim.

**The instruction is confirmed real and documented:** `v_wmma_f32_16x16x16_fp8_fp8`
(16×16×16 tile, fp8×fp8→f32). Triton emits it; it is the same tile geometry as the
BF16 WMMA already working in `src/hephaestus/wmma_gfx12.mojo`.

---

## 1. The ISA constraint that redefines Phase 1b (most important finding)

From the ESDMAX write-up (source §4):

> **"RDNA4 WMMA only supports fp8×fp8 → f32, not bf16×fp8 mixed operands. So you
> cannot feed bf16 activations straight into the FP8 matrix cores; you must quantize
> activations to fp8 first."**

This is a hardware fact, not a software limitation, and it has a hard architectural
consequence for Hephaestus:

**The 1a BF16 forward pass cannot reuse its activation path for FP8 matmuls.** FP8
WMMA requires *both* operands in E4M3. That means Phase 1b is not "load FP8 weights,
swap the matmul." It is:

- FP8 weights **and** an **activation-quantization step** (bf16 → e4m3, per-row or
  per-block scale) inserted before every FP8 matmul.
- The activation-quant kernel is new work with its own cost. ESDMAX's *first* attempt
  at it was a **30% regression** (10 → 7 tok/s) because a serial per-row scale scan
  left 255 of 256 threads idle; fixing it required a butterfly `__shfl_xor`
  wave reduction. This is a real kernel, not a footnote.
- This matches SPEC.md §3 Phase 1b scope ("per-tensor / per-channel scales") but the
  spec does not currently name *activation* quantization as a required kernel. It
  should — see the DECISIONS.md 2026-07-19 entry.

## 2. Mechanism 1 — Triton auto-lowering (the community patch)

**Source:** vllm-project/vllm issue #28649 + vLLM forum thread 1900.

The popular "RDNA4 FP8 patch" contains **zero GPU code.** It is three plumbing
changes: (a) add `gfx1201` to `on_mi3xx()` in `vllm/platforms/rocm.py`; (b) patch
AITER's `_ARCH_TO_DEVICE` to map `gfx1201 → MI350X` so AITER's Triton kernels stop
`KeyError`-ing; (c) add JSON tuning configs naming tile sizes. The actual WMMA
emission is done by the **Triton compiler** when `tl.dot()` sees FP8 operands on
gfx1201.

**Caveat that weakens this source's authority:** the issue author (Rob-P-Smith)
later *retracted* much of it in a follow-up comment — an existing fall-through case
already reached the same W8A8 path, the uplift was likely from CUDA-graph
improvements in a nightly, and "no patch is needed, only model configs." So treat
the *benchmark numbers* (63% faster, etc.) as unverified. The **mechanism** (Triton
auto-lowers FP8 `tl.dot` to `v_wmma_f32_16x16x16_fp8_fp8` on gfx1201) is
independently corroborated by the ESDMAX author and by AMD's own precision docs, and
stands.

**Relevance to Hephaestus: low for the engine, high as proof.** Hephaestus has no
Triton dependency (permanent design position). But this mechanism is the cleanest
existing proof that *the instruction fires correctly on gfx1201 silicon* — which is
the exact binary question G1b-0 must answer. It also hands us the instruction name
and tile geometry for free.

## 3. Mechanism 2 — ESDMAX custom kernel (the most relevant prior art)

**Source:** github.com/mininmaxim/vllm, branch `esdmax-r9700-fp8`
(`ESDMAX_KERNEL_README.md`, `summary.md`). **Same hardware as Hephaestus dev box:
2× R9700, gfx1201, ROCm 7.2.**

Headline results (their numbers, end-to-end measured):
- Stock vLLM (FP8→FP32 fallback): 6–7 tok/s
- Custom FP8 WMMA kernel + split-K + CUDA graphs: **26.5 tok/s** single stream
- + MTP speculative decoding: **55 tok/s** single stream; **188 tok/s** at 8-way
- Their own llama.cpp Q8 baseline on the same box: 16 tok/s

**Findings in their write-up that Hephaestus should treat as established:**

| Finding | Why it matters to us |
|---|---|
| fp8×fp8→f32 only (§1 above) | Forces activation-quant kernel into 1b scope |
| Batch-1 decode GEMV is ~2 FLOP/byte, **100% memory-bound; matrix cores idle regardless** | Confirms SPEC.md's "decode is bandwidth-bound" position. **FP8 WMMA is not a decode speedup** — the decode win is *bytes read* (8.06 bits/weight vs 16), which FP8 *loading* gives you with or without WMMA. WMMA shows up in **prefill** (compute-bound), exactly as SPEC.md G1b-3 already assumes. |
| AITER on gfx1201: imports, detects arch, **0 tuned configs, no prebuilt ASM** (`hsa/` only for gfx942/950/1250) | AITER cannot be vendored for FP8 on our card. Do not waste a probe on it. |
| Navi 48 has a **64 MB Infinity Cache** — microbenchmarks that re-time the same weight read from LLC, not VRAM, and report ">100% of roofline" (physically impossible) | Any Hephaestus FP8 microbenchmark must rotate over >64 MB of cold buffers or it lies. The 1b layer-diff harness must account for this. |
| Prefill GEMM needed `M % 64 == 0`; ragged remainders fell off a 3× cliff (404 → 1242 tok/s when fixed) | Our G1b-0 tile edge-masking (already in the BF16 WMMA kernel) is load-bearing for real prompts, not a nicety. |
| TP=2 on dual R9700 needs `NCCL_PROTO=Simple` + `GPU_MAX_HW_QUEUES=1` to avoid a deadlock | Only relevant to Phase 2 dual-server, but cheap to record now. |

**Caveat on authority:** ESDMAX's README says the kernel was written by "Codex" and
is a fork, not upstreamed — exactly the "fragile hack" category this project exists
to replace. Its *engineering measurements* (roofline math, cache behavior, AITER
tuning gaps) are the valuable part and are internally consistent; its code is not
something we would vendor.

## 4. Mechanism 3 — hipBLASLt (the first-party door, currently immature)

**Sources:** ROCm hipBLASLt data-type docs (7.0.0 / 7.2.0 / 7.2.2); rocm-libraries
issue #8242.

AMD's official docs state `hipblaslt_f8` (E4M3, OCP) is supported on **"gfx950 and
gfx12"** — i.e. RDNA4 is nominally covered by the *maintained* library. This is the
only mechanism that is a genuine "first-party citizen" path.

Two problems for Hephaestus:
1. **Maturity.** gfx1201 hipBLASLt has live regressions — issue #8242 documents a
   SIGSEGV in `hipblaslt_f8::is_inf` reachable from a plain *FP32* GEMM heuristic
   query (the FP8 type-check path executes on non-FP8 ops). Workaround
   `ROCBLAS_USE_HIPBLASLT=0` exists, but this signals the gfx12 FP8 path is young.
2. **FFI.** hipBLASLt is a C++ library. Mojo has no C++ interop; calling it would
   require a C shim layer, which cuts against the "no translation layers" design
   position.

**Verdict:** worth a *probe* (a tiny C++ test that one `hipblasLtMatmul` FP8 call
returns correct numbers on gfx1201) as a *reference oracle* — if it works, its
output can validate the hand-written path. Not a candidate to vendor into the engine.

## 5. What this means for the G1b-0 probe (not a design — constraints only)

The landscape narrows the honest options for reaching FP8 WMMA from Mojo to two:

- **(a) Mojo `llvm_intrinsic` with the FP8 WMMA intrinsic name.** The BF16 path
  already works this way. The open question from 2026-07-12 was FP8's
  operand-legalization error (`Do not know how to split this operator's operand`).
  Whether a newer Mojo nightly than dev2026071006 legalizes the FP8 intrinsic is a
  *toolchain* question, testable in minutes.
- **(b) Hand-written gfx12 asm for `v_wmma_f32_16x16x16_fp8_fp8`.** The confirmed
  instruction. This is the path Fable was policy-blocked from attempting and the one
  IDEAS.md already parks ("if upstream stalls"). ESDMAX proves a hand-rolled kernel
  on this exact card is achievable and can beat untuned AITER 2×.

Triton and hipBLASLt are **out of scope as engine paths** (no Triton dependency; no
C++ FFI), but both serve as **correctness oracles**: if either can be made to run one
FP8 GEMM on the dev box, its output is the reference the hand-written path must match
bit-for-bit, in the same spirit as the existing HF oracle.

The activation-quantization requirement (§1) is the single biggest scope addition
this research surfaces and is recorded in DECISIONS.md 2026-07-19.

---

## 6. Sources (all retrieved 2026-07-19)

1. vllm-project/vllm **issue #28649** — "Someone please upstream this gfx1201/RDNA4
   FP8 Patch." Mechanism 2 (Triton path). Note author's later partial retraction.
2. vLLM forum **thread 1900** — same patch, discussion + maintainer confirmation of
   the Triton auto-WMMA mechanism and the M≥16 WMMA constraint.
3. **mininmaxim/vllm @ esdmax-r9700-fp8** — `ESDMAX_KERNEL_README.md`, `summary.md`.
   Mechanism 3 (custom kernel). fp8×fp8-only ISA fact, roofline analysis, AITER
   tuning-gap table, Infinity Cache benchmark-trap, all measured on 2× R9700.
4. **ROCm hipBLASLt data-type support docs** (7.0.0/7.2.0/7.2.2) —
   `hipblaslt_f8`/`bf8` "gfx950 and gfx12" support claim.
5. ROCm/rocm-libraries **issue #8242** — gfx1201 hipBLASLt FP8-path SIGSEGV
   regression (maturity caveat).
6. ROCm **precision-support matrix** (7.2.2) — RDNA4 matrix-core row confirms
   float8 (E4M3) and float8 (E5M2) = ✅, bfloat16 = ✅.

*Methodology note: every claim above is traced to one of these six sources. Where a
source contradicted itself (#28649's retraction), that is stated rather than
smoothed over. Where a number is a vendor/community claim and not independently
verified, it is labeled as such.*
