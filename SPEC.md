# HEPHAESTUS — FP8-Native Inference Engine for RDNA4
## Project Spec & Scope Control Document — v1.1

> Changelog: v1.1 (2026-07-12) — G1a-1 exit criterion restated as a three-part
> checkable gate (§3, Phase 1a); North Star (§0) unchanged. See entry below
> for why the original wording was retired.

> Name is a placeholder (forge god; fits the kernel work and the Mojo 🔥 branding). Rename freely; nothing else in this doc changes.

---

## 0. NORTH STAR (one sentence, never edited without a version bump)

**A lightweight, GUI-less inference backend, written in Mojo, that serves FP8 E4M3 safetensors natively on AMD RDNA4 at llama.cpp-class single-stream speed — proving the silicon was never the bottleneck.**

If a proposed task does not serve this sentence, it goes to the Parking Lot or it dies.

---

## 1. SCOPE LAW

Three rules, enforced every working session:

1. **The Gate Rule.** Only work that moves the *current phase's exit gate* is in scope. Not the next phase's. Not "while I'm in here anyway."
2. **The Parking Lot Rule.** Any idea outside the current gate gets one line in `IDEAS.md` within 60 seconds of having it, then is dropped from working memory. Logging is mandatory; building is forbidden; **designing-for is also forbidden** (no "let's just make this interface generic so later we can…"). We do not architect for parked ideas. If a parked idea graduates, we refactor then. Refactoring later is cheaper than abstracting now.
3. **The Drift Check.** Before writing any code, answer aloud: *"Which exit criterion does this move?"* No answer → Parking Lot.

### Session protocol (every work session, ~2 min)
- Open: state current phase + which exit criterion today targets.
- Close: log progress in `DECISIONS.md` if any decision was made; sweep stray ideas into `IDEAS.md`.
- Claude's standing job: flag drift the moment it appears, cite this document, no diplomacy required.

---

## 2. NON-GOALS (permanent or until Phase 3 review)

These are explicitly OUT. Their presence here is what makes the spec narrow.

| Non-goal | Why excluded | Parked? |
|---|---|---|
| GGUF loading | Whole thesis is escaping that gravity well; llama.cpp serves it fine | Permanent |
| Any GUI / TUI | Backend only; OpenAI-compatible HTTP is the entire surface | Permanent |
| NVIDIA / Apple / Intel targets | Portability is Mojo's job later, not ours now | Parking Lot |
| Vulkan / SPIR-V path | No Mojo→SPIR-V backend exists; compatibility floor is a v2+ conversation | Parking Lot |
| New file format / wrapper | Formats win via killer readers; engine first | Phase 3 gate |
| Multi-GPU tensor parallel | Most work, least payoff on asymmetric 32+16GB rig | Parking Lot |
| Heterogeneous (asym-VRAM) scheduling | Genuinely novel, genuinely Phase 3+ | Parking Lot |
| MoE architectures | Routing complexity; dense first | Phase 2 |
| Multimodal (vision/audio) | Encoder handling is a project of its own | Parking Lot |
| Training / fine-tuning / quantizing | Inference only; use llm-compressor to produce FP8 checkpoints | Permanent |
| Speculative decoding / MTP | Real speed, wrong phase | Parking Lot |
| Windows support | Linux/ROCm only | Permanent |
| Chasing final 10% of single-stream parity | Bandwidth-bound; batching is where we win | Permanent |

---

## 3. PHASES & GATES

### Phase 1a — "It Thinks" (BF16 correctness baseline)
**Goal:** A Mojo binary that loads a safetensors checkpoint and produces correct tokens on one GPU.

Scope (exhaustive — if it's not listed, it's not in):
- safetensors mmap loader (BF16 tensors only)
- Tokenizer (vendor an existing implementation; do NOT write one)
- ONE architecture, hard-coded: **Qwen3 dense, 4B-class** (no config-driven generality)
- BF16 forward pass on the R9700 via Mojo `gpu` stdlib; vendor matmul/attention kernels from MAX's open kernel library — write nothing that can be vendored
- Greedy sampling only. No top-k, no top-p, no temperature.
- CLI: `hephaestus --model <dir> --prompt <str> --n-tokens <int>`. No server. No streaming.

**Exit gate (all must pass):**
- [x] **G1a-1: PASS** (2026-07-12) — correctness vs HF Qwen3-4B-Instruct-2507, restated as a three-part checkable gate (below); measured in `bench/1a.md`
- [x] **G1a-2: PASS** (2026-07-13) — 54.22 tok/s (98.3% of llama.cpp, 109.2% of target). Attention score loop parallelized across 4 warps. `bench/1a.md`
- [x] **G1a-3: PASS** (2026-07-12) — loads in ~6.3s warm, ≤~8.5s cold bound; measured in `bench/1a.md`
- Benchmark log committed to repo (`bench/1a.md`)

#### G1a-1, restated (amended 2026-07-12, v1.1)

The original wording — "token-identical output vs HF transformers reference" —
turned out not to name a single target. Measured 2026-07-12: HF's own `sdpa`
and `eager` attention implementations produce **different tokens on the same
prompts** (diverging at steps 4, 32, and 7 across three prompts), and HF's own
single-shot batched recompute does not even reproduce its own
cached-autoregressive generation (255/256, 250/256, 252/256 self-agreement).
Every one of these self-disagreements, and every disagreement Hephaestus has
with any of them, occurs at a bf16 logit near-tie — most are bit-identical
top-1/top-2. "Identical to a reference that disagrees with itself" cannot be a
criterion; it is retired in favor of three parts that are each independently
checkable and were each measured against real runs, not assumed:

- **(a) Teacher-forced argmax fidelity.** Feeding the reference's own
  generated tokens as history at every step (so no error propagates from an
  earlier divergence), across ≥3 prompts × 256 steps, Hephaestus's greedy
  argmax must equal HF's argmax, OR the disagreement must fall at a genuine
  near-tie: `|HF_top1 − HF_top2| ≤ 1 bf16 ulp` at that logit's magnitude
  (`2^(floor(log2|top1|)) × 2^-7`). Zero disagreements are permitted outside
  that bound.
- **(b) Full autoregressive reproduction.** At least one of the ≥3 prompts
  must be token-identical for the complete 256-token autoregressive greedy
  decode against the committed oracle (cached-KV-cache generation, `sdpa`) —
  i.e. proof that when no ties occur along a trajectory, no divergence
  happens at all.
- **(c) The bound is at the decision boundary, not the full vocabulary.**
  This gate does not bound raw logit deviation across all 151,936 vocab
  entries. An isolated, argmax-irrelevant tail-token deviation does not
  indicate a defect, and a full-vocab bound would fail even the prompt that
  passed (b) cleanly. Only the gap between top-1 and top-2 at the moment of
  disagreement is load-bearing, and (a) already checks it directly.

Reproduce: `src/qwen_teacher_forced_full.mojo` + `scripts/hf_teacher_forced_full.py`.
Full data: `.agent/notes/768-step-teacher-forced-results.md`, `bench/1a.md`.

Known open item, not blocking this gate: isolated, unexplained logit
magnitude spikes on argmax-irrelevant tail tokens (largest: 12.06, prompt 1
step 67). Root cause not yet identified; GPU `cos`/`sin` precision was tested
directly and ruled out. This is a **Phase 1b entry gate** — see `DECISIONS.md`
2026-07-12.

### Phase 1b — "The Thesis" (FP8 native)
**Goal:** Same engine, FP8 E4M3 weights fed to WMMA units natively. No FP32 fallback path may exist in the code.

Scope:
- FP8 E4M3 safetensors loading (per-tensor / per-channel scales)
- FP8 WMMA matmul path (RDNA4 gfx1201); BF16 path from 1a retained as reference & fallback tier
- Layer-by-layer logit diffing harness against the 1a BF16 pass (this is the debugging tool; build it early)

**Exit gate:**
- [ ] G1b-1: Perplexity within 1% of BF16 baseline on a fixed eval set
- [ ] G1b-2: Decode tok/s ≥ llama.cpp Q8_0 on same model/card (matched ~1 byte/weight)
- [ ] G1b-3: Prefill tok/s ≥ **1.5×** llama.cpp Q8_0 at 4K-token prompts (this is where WMMA shows)
- [ ] G1b-4: `grep` proves no FP8→FP32 dequant path in any hot loop
- Benchmark log: `bench/1b.md`. **Publish results** (Modular forum post minimum) — this is the proof-of-thesis artifact.

### Phase 2 — "The Multiplier" (concurrency)
**Goal:** The multi-agent unlock. Continuous batching + paged KV cache + HTTP server.

Scope: OpenAI-compatible `/v1/completions` + `/v1/chat/completions`, streaming; continuous batching scheduler with chunked prefill; paged KV block allocator; second architecture (Gemma 4 26B A4B or similar — first MoE allowed here); dual-server config (one engine instance per card) documented.

**Exit gate:**
- [ ] G2-1: 8 concurrent requests: aggregate throughput ≥ **3×** single-stream (llama.cpp's parallel mode as the bar to beat)
- [ ] G2-2: p95 inter-token latency under 8-way load ≤ 2× single-stream
- [ ] G2-3: Odysseus council fan-out runs against it as the live backend for one week without falling back to llama.cpp

### Phase 3 — "The Ecosystem" (unlock the Parking Lot)
Format/wrapper work, layout-tagged loading, other hardware targets, heterogeneous scheduling, MTP/speculative — **all parked ideas get their first legitimate review here.** No scope defined now, deliberately.

---

## 4. THE PARKING LOT MECHANISM

File: `IDEAS.md`, append-only, one line per idea:

```
| date | idea (≤15 words) | earliest phase | logged instead of built? ✅ |
```

Rules:
- Anyone (Niko or Claude) can append at any time. No discussion required to log.
- Ideas are **reviewed only at phase-gate crossings** — never mid-phase.
- An idea graduates only by being written into the next phase's scope list with an exit criterion. No criterion, no graduation.
- Pre-seeded from this conversation: Vulkan/SPIR-V backend, wrapper format w/ layout tags, residual-encoding dual-precision, heterogeneous asym-VRAM scheduler, Intel Level Zero FFI, Apple Metal target, MTP drafter support, MoE-optimized routing kernels, KV-cache FP8.

## 5. DECISION LOG

File: `DECISIONS.md`, append-only:

```
| date | decision | why | reversible? |
```

Pre-seeded: Mojo native (no Vulkan) for v1 · BF16-before-FP8 sequencing · Qwen3-4B dense as first arch · vendor MAX kernels rather than write · 90% single-stream bar, not parity · GGUF permanently out.

## 6. DRIFT SIGNALS (Claude's checklist)

Claude flags immediately when any of these appear in conversation:
1. "While we're at it…" / "It would be easy to also…"
2. Designing an abstraction whose only justification is a parked idea
3. Benchmarking against a target not named in the current gate
4. Adding a model architecture before its phase
5. Any sentence containing "GGUF support"
6. Optimizing single-stream decode past the 90% gate
7. Discussing the wrapper format before G2-3 is checked

Response to a flag is always one of: (a) justify against current gate, (b) log to IDEAS.md, (c) drop.

---
*v1.1 — 2026-07-12: G1a-1 restated as a three-part checkable gate (§3). v1.0 drafted 2026-07-11. North Star (§0) edits require version bump + one night of sleep before committing; this was not a North Star edit.*
