# HEPHAESTUS — FP8-Native Inference Engine for RDNA4
## Project Spec & Scope Control Document — v1.0

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
- [ ] G1a-1: Token-identical output vs HF transformers reference (greedy, same prompt, ≥3 prompts × 256 tokens)
- [ ] G1a-2: ≥ **90%** of llama.cpp single-stream decode tok/s, same model converted to **F16 GGUF** via `convert_hf_to_gguf.py` (we make the F16 GGUF ourselves), same card, ≥3 runs each
- [ ] G1a-3: Loads in under 30s from cold
- Benchmark log committed to repo (`bench/1a.md`)

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
*v1.0 — drafted 2026-07-11. North Star edits require version bump + one night of sleep before committing.*
