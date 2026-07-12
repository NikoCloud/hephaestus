# CLAUDE.md — Hephaestus Project Instructions

You are the resident coding agent for **Hephaestus**: a GUI-less LLM inference engine in Mojo, serving FP8 E4M3 (and BF16) safetensors natively on AMD RDNA4 (gfx1201). You work directly on the dev machine (both GPUs are local to you).

## Read these before any work, in order
1. `SPEC.md` — the governance document. **It is law.** Phase gates, scope rules, non-goals.
2. `docs/architecture-dossier.md` — every model constant, sourced. Do not recall constants from memory; use this file.
3. `docs/kernel-interface-report.md` — the vendoring contract for MAX kernels (TileTensor requirements, WMMA constraints).
4. `docs/environment-notes.md` — toolchain and machine quirks.
5. `.agent/specs/2026-07-11_safetensors-loader.md` — the current implementation spec.

## Current state (2026-07-11)
- **Phase: 1a "It Thinks."** Goal: BF16 forward pass of Qwen3-4B-Instruct-2507, token-exact vs oracle, ≥ 49.63 tok/s decode on the R9700.
- Feasibility proven: Mojo 1.0.0b3 compiles and runs correct GPU kernels on gfx1201 (via pixi).
- Baseline measured: llama.cpp F16 = 55.14 tok/s decode (`bench/0-baseline-llamacpp-f16.md`).
- Oracle fixtures exist (`fixtures/oracle/`): 3 prompts × 256 greedy tokens + first-10-step logits. **G1a-1 = token-identical against these.** Feed the saved token IDs directly; do not re-tokenize text.
- Tiny-random oracle exists (2-layer real architecture, 644KB): your millisecond debug loop. Always validate against tiny before touching the 4B model.
- **Current task: implement the safetensors loader per the spec.** Two open questions are resolved first, by experiment (see spec §open-questions): TileTensor type-erasure approach, and mmap vs HostBuffer for host→device copy. Timebox each to ~15 minutes, record the answer + evidence in `DECISIONS.md`.

## Scope law (from SPEC.md — enforced, not advisory)
- Only work that moves the **current phase's exit gate**. Before writing code, answer: *which exit criterion does this move?* No answer → don't write it.
- Ideas outside the gate go in `IDEAS.md` (one line, append-only) — logged, **never built, never designed-for**. No speculative abstractions justified by parked ideas.
- **Non-goals, permanent:** GGUF anything, GUI/TUI, Windows, training/quantization tooling, CUDA. **Non-goals, this phase:** FP8 (that's 1b), sampling beyond greedy, HTTP serving, batching, MoE, multimodal, second architectures, multi-GPU.
- Do not optimize single-stream decode past the 90% gate. Bandwidth-bound; Phase 2 is where we win.
- The BF16 path being built now is permanent (it's the numerics reference for 1b), not throwaway.

## Engineering rules
- **Oracle over vibes.** "It produces plausible text" is not evidence. Token-exact diff or it isn't validated. GPU kernels fail silently — wrong results, no errors (we already saw a kernel return all zeros without complaint).
- **Raw output over recall.** Hardware facts come from `rocminfo`/`rocm-smi` verbatim. Model facts come from the dossier or the safetensors header, never from memory. (Both GPUs are gfx1201 — a prior agent misremembered this; the correction cost an audit.)
- **Vendor, don't write.** Matmul, attention, RMSNorm, RoPE, softmax, sampling, KV-cache kernels exist in `~/projects/modular/` under `kernels/src/` (paths in the interface report). Write only the layer that doesn't exist: loader, model struct, forward-pass orchestration, engine loop.
- **Scripts are files in the repo**, committed, run as files. Never pipe code through shell quoting (login shell is fish; `$` mangles — see environment notes).
- **Measure before enshrining.** Benchmarks: 3 reps minimum, exact flags recorded, written to `bench/` with hardware provenance. Report effective bandwidth utilization (achieved ÷ ceiling) alongside raw tok/s.
- **Decisions get logged.** Any choice with alternatives → one line in `DECISIONS.md`: what, why, reversible?
- **Commit granularity:** each working step (loader parses header → commit; tiny model loads → commit). Bench results and fixtures are repo artifacts.

## Environment facts
- Machine: CachyOS Linux. GPU 0 = R9700 AI Pro 32GB (dev + bench target). GPU 1 = RX 9070 XT 16GB. Both gfx1201 — one compile target.
- **GPU 1 may be serving a live LLM for the agent harness. Never run experimental GPU work on a card serving an agent.** Default all dev work to GPU 0; check `rocm-smi` for activity before any run. If a run may stress the driver, say so before launching.
- Mojo via pixi (`pixi run mojo ...`) in the repo env. `fn` is deprecated in this build — use `def`. No inline eval (`mojo -c` doesn't exist); write files and `mojo run`.
- torch 2.9.1+rocm6.3 is system-wide (deliberately not in pixi — resolver can't handle ROCm torch). It exists for fixture/conversion tooling only; the engine has no Python runtime dependency.
- Model: `/mnt/models/models/qwen3-4b-instruct-2507/` (3 safetensors shards + index json). F16 GGUF for benching lives beside it.

## Working with the human
- Niko is the architect and final judge. He is often on mobile; keep progress reports compact — status table, what changed, what's blocked, next step. Flag drift candidly; he wants the pushback.
- An external auditor (Hermes harness) reviews the repo over SSH. Everything of record must be **in the repo** — if it only happened in your terminal, it didn't happen.
- When something fails, report the failure with the exact command and output before proposing fixes. Honest losses are project culture (see README).
