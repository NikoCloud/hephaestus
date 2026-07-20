# Hephaestus 🔥

**Native FP8 inference for AMD RDNA4 — the hardware you own, finally fed.**

A lightweight, GUI-less LLM inference engine written in [Mojo](https://www.modular.com/mojo). Loads FP8 E4M3 **safetensors directly** — no GGUF, no conversion step, no translation layers — and feeds them straight to RDNA4's WMMA matrix units.

> **Status: pre-alpha, Phase 1a complete (G1a-1/2/3 PASS), Phase 1b in progress.** BF16 Qwen3-4B forward path works on RDNA4 and clears the Phase 1a gates. **Native FP8 E4M3 WMMA is proven on gfx1201** (`experiments/exp3g_*`, via direct `llvm_intrinsic`) and a W8A8 FP8 decode path runs end-to-end at **97.4% argmax parity** vs the HF oracle -- but it is **not yet faster than llama.cpp Q8_0 decode** (0.36x; see `bench/1b-fp8-wmma-decode.md`). No HTTP server, no batching, no general model support -- not a drop-in llama.cpp replacement.

[![Stars](https://img.shields.io/github/stars/NikoCloud/hephaestus?style=for-the-badge&color=e94560)](https://github.com/NikoCloud/hephaestus/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)
[![Platform: Linux + ROCm](https://img.shields.io/badge/Platform-Linux_%2B_ROCm-blue?style=for-the-badge)](https://rocm.docs.amd.com)
[![Phase: 1a PASS](https://img.shields.io/badge/Phase-1a_PASS-2ecc71?style=for-the-badge)](https://github.com/NikoCloud/hephaestus)

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [What it is / what it isn't](#what-it-is--what-it-isnt)
- [Roadmap &amp; honest numbers](#roadmap--honest-numbers)
  - [Phase 1a numbers (honest, R9700, Qwen3-4B-Instruct-2507)](#phase-1a-numbers-honest-r9700-qwen3-4b-instruct-2507)
- [Design positions](#design-positions)
- [Hardware](#hardware)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Why this exists

RDNA4 consumer GPUs ship with real FP8 matrix hardware running at **double their FP16 throughput** — the same E4M3 format as AMD's datacenter Instinct cards. Almost nothing feeds it:

- The most common vLLM path on RDNA4 was found **silently dequantizing FP8 weights to FP32** for every operation — zero benefit from the format, no warning given.
- llama.cpp's quantized formats dequantize to compute; the WMMA FP8 path goes unused.
- Vendor frameworks optimize for enterprise silicon. Consumer ROCm gets whatever's left.

The result: people who bought RDNA4 cards for the VRAM-per-dollar own FP8 accelerators that sit idle. Every serving-layer improvement of the past three years — native low-precision compute, continuous batching, paged KV — arrives on these cards late, patched, or not at all.

Hephaestus is built by and for that underserved group. Not a llama.cpp replacement — llama.cpp is excellent and serves GGUF well. This is the engine for the other path: **safetensors-native, FP8-first, no silent downgrades anywhere in the code.**

## What it is / what it isn't

| Is | Isn't |
|---|---|
| FP8 E4M3 + BF16 safetensors, loaded directly | GGUF (permanently out of scope) |
| Single-GPU engine instances; run one per card | Tensor parallelism across mismatched cards |
| OpenAI-compatible HTTP endpoint (Phase 2) | A GUI, a launcher, a model manager |
| RDNA4 (gfx1201) first, tuned on real hardware | A portability layer (Mojo makes that possible later; it is not the mission now) |
| Continuous batching + paged KV (Phase 2) | Training, fine-tuning, or quantization tooling |

## Roadmap & honest numbers

Numbers are published as they're measured — including the losses. Baseline hardware: Radeon AI PRO R9700 (32GB, gfx1201). Reference model: Qwen3-4B-Instruct, same weights on both engines.

| Phase | Goal | Gate | Status |
|---|---|---|---|
| **1a — It Thinks** | BF16 forward pass of Qwen3-4B; teacher-forced argmax fidelity vs HF; load ≤30s | ≥ 90% of llama.cpp F16 decode vs **original** 55.14 baseline (**≥ 49.6 tok/s**) | ✅ **PASS** — see `bench/1a.md`, `bench/1a-ab.md` |
| **1b -- The Thesis** | Native FP8 WMMA path, no FP32 fallback in any hot loop | PPL within 1% of BF16; >= llama.cpp Q8_0 decode; >= 1.5x its prefill at 4K | in progress -- FP8 WMMA proven, W8A8 decode numerically correct (97.4%); speed gates not met |
| **2 — The Multiplier** | Continuous batching, paged KV, HTTP serving | ≥ 3× aggregate throughput at 8 concurrent requests | ⏳ |
| **3 — Ecosystem** | Layout-tagged loading, more architectures, the parking lot | — | ⏳ |

### Phase 1a numbers (honest, R9700, Qwen3-4B-Instruct-2507)

Measured 2026-07-13 post GPU-argmax fix (`bench/1a-ab.md`). llama.cpp F16 same card/build.

| Metric | Hephaestus BF16 | llama.cpp F16 | Notes |
|---|---:|---:|---|
| Decode tok/s (**forward only** — G1a-2 metric) | **53.7** | 62.0 | Gate used original llama **55.14** baseline → **98% / 109% of 90% target** |
| Decode tok/s (**incl. greedy sampling**) | **52.8** | 61.8 | Was ~14 tok/s with host argmax; GPU argmax ~0.3 ms/step |
| Total time (10-tok prompt × 256 gen) | **4.94 s** | 5.17 s | Was **18.0 s** before GPU argmax |
| Prefill tok/s (10 / 512 tok) | ~95 / ~121 | ~148 / ~1430 | Prefill is still weak; not a 1a gate |
| Peak VRAM | **~29.6 GB** | ~8.1 GB | MAX/AsyncRT reserves ~90% of card on first buffer — not loader bloat; not a leak |
| Load time | ~6–8 s | — | G1a-3 PASS |

**Correctness (G1a-1):** teacher-forced argmax matches HF on non-ties across 768 steps (0 non-tie flips). Full-vocab logit spikes vs HF exist and are **characterized** (RoPE stepwise-bf16 fixed; residual = irreducible BF16 matmul reduction-order ULPs) — not claimed bit-identical to HF across the whole vocab. Details: `.agent/notes/spike-investigation.md`.

**Caveats you should not miss:**

- G1a-2’s 90% bar is vs the **2026-07-11** llama.cpp 55.14 tok/s citation. Fresh llama.cpp on the same machine later measured ~62 tok/s; against *that* number Hephaestus is ~87% — see `bench/1a-ab.md` Finding 1.
- Single architecture hard-coded (Qwen3 dense 4B). CLI only. No server, no batching. The FP8 W8A8 decode path exists and is numerically correct, but is not yet competitive on speed.
- BF16 and FP8 WMMA both execute on gfx1201 via **direct `llvm_intrinsic`**; the Mojo stdlib RDNA WMMA path is gfx11-only and unusable here (see DECISIONS 2026-07-13). Attention is hand-written and parallelized. The FP8 WMMA decode path is correct but bandwidth-limited at M=1.

Why 90% and not parity in 1a: single-stream decode is memory-bandwidth-bound; the last 10% costs months and buys little. Phase 2 is where this engine is supposed to pull ahead — concurrency is the point, single-stream is the credential.

Why BF16 before FP8: a validated BF16 pass in the *same engine* is the numerics reference that makes FP8 kernels debuggable. Correctness first, thesis second.

## Design positions

- **Vendor where it compiles; write what must match.** MAX kernels are vendored when they work on gfx1201. On this Mojo nightly, WMMA paths do not compile for RDNA4, so Phase 1a matmul/attention/argmax are hand-written (and documented). Scheduler, HTTP, and multi-arch are still future work.
- **The FP8 tier is canonical, not a fallback.** When 1b lands, there will be no code path that dequantizes FP8 to FP32 for compute. If it can't run native, it fails loudly. (FP8 is not implemented yet.)
- **Validated, not vibed.** Forward-pass changes are checked against HuggingFace `transformers` with a restated gate (teacher-forced argmax fidelity + one clean full decode + decision-boundary ties) — not “it looks fluent.” A tiny-random model with the real architecture is the millisecond debug loop.
- **Effective bandwidth utilization is the honest metric.** Achieved tok/s divided by the memory-bandwidth ceiling for the model's byte-width. It's how we know whether a gap is kernel work or physics.
- **Placement is a first-class concern.** Asymmetric multi-GPU (e.g. 32GB + 16GB) is normal in consumer rigs and tensor parallelism wastes it. One engine per card now; smarter scheduling is parked, deliberately, for Phase 3.

## Hardware

Developed and benchmarked on consumer silicon, because that's the constituency:

- AMD Radeon AI PRO R9700 (32GB, gfx1201)
- AMD Radeon RX 9070 XT (16GB, gfx1201)
- ROCm/HIP on Linux. No Windows. No CUDA.

If you have RDNA4 hardware and want to contribute benchmarks when there's something to run, open an issue — real datapoints from real rigs are exactly what this project wants.

## Acknowledgments

- [Modular](https://github.com/modular/modular) for open-sourcing the kernel library and building a language where one person can write GPU engines.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the proof that community engines set the standard, and the baseline this project measures itself against.
- [colibrì](https://github.com/JustVugg/colibri) for the oracle-validation pattern and the honest-numbers README culture this project imitates.

## License

Apache 2.0. Vendored kernels retain their upstream Apache 2.0 license and attribution.
