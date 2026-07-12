# Hephaestus 🔥

**Native FP8 inference for AMD RDNA4 — the hardware you own, finally fed.**

A lightweight, GUI-less LLM inference engine written in [Mojo](https://www.modular.com/mojo). Loads FP8 E4M3 **safetensors directly** — no GGUF, no conversion step, no translation layers — and feeds them straight to RDNA4's WMMA matrix units.

> **Status: pre-alpha, Phase 1a in progress.** Nothing here is ready to use. The baseline is measured, the oracle fixtures exist, the loader is being written. Star and watch if the mission matters to you; come back when the numbers below start filling in.

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
| **1a — It Thinks** | BF16 forward pass, token-exact vs a `transformers` oracle | ≥ 90% of llama.cpp F16 single-stream (**≥ 49.6 tok/s** vs measured 55.1) | 🔨 in progress |
| **1b — The Thesis** | Native FP8 WMMA path, no FP32 fallback in any hot loop | ≥ llama.cpp Q8_0 decode; ≥ 1.5× its prefill at 4K prompts | ⏳ |
| **2 — The Multiplier** | Continuous batching, paged KV, HTTP serving | ≥ 3× aggregate throughput at 8 concurrent requests | ⏳ |
| **3 — Ecosystem** | Layout-tagged loading, more architectures, the parking lot | — | ⏳ |

Why 90% and not parity in 1a: single-stream decode is memory-bandwidth-bound; the last 10% costs months and buys little. Phase 2 is where this engine pulls ahead — concurrency is the point, single-stream is the credential.

Why BF16 before FP8: a token-exact BF16 pass in the *same engine* is the numerics reference that makes FP8 kernels debuggable. Correctness first, thesis second.

## Design positions

- **Vendored kernels, original engine.** Matmul/attention kernels come from [Modular's open kernel library](https://github.com/modular/modular) (Apache 2.0) — 450k lines of production Mojo, including RDNA-specific WMMA paths. We write the layer they don't ship: loader, scheduler, KV management, serving.
- **The FP8 tier is canonical, not a fallback.** There is no code path that dequantizes FP8 to FP32 for compute. If it can't run native, it fails loudly.
- **Validated, not vibed.** Every forward-pass change is diffed token-exact against a HuggingFace `transformers` oracle. A tiny-random model with the real architecture gives a millisecond debug loop.
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
