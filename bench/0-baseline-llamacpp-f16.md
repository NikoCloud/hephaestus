# Baseline Benchmark — llama.cpp F16 GGUF
## Date: 2026-07-11
## Model: Qwen3-4B-Instruct-2507 (F16 GGUF, 7.49 GiB)
## Device: AMD Radeon AI PRO R9700 (gfx1201, 32 GB VRAM) — Device 0
## llama.cpp build: 33ca0dcb9 (9906), ROCm/HIP backend

| test | t/s |
|---|---|
| pp512 | 1418.39 ± 29.10 |
| pp2048 | 1835.92 ± 5.70 |
| tg128 | **55.14 ± 0.13** |
| tg512 | **55.12 ± 0.04** |

**G1a-2 target (90% decode):** ≥ 49.63 t/s (tg128)

## Hardware Notes
- Both cards report as gfx1201 (confirmed via rocminfo)
  - Device 0: AMD Radeon AI PRO R9700, gfx1201, 64 CUs, 32624 MiB VRAM
  - Device 1: AMD Radeon RX 9070 XT, gfx1201, 64 CUs, 16304 MiB VRAM
- Both are RDNA4; gfx1201 is the correct arch identifier for both
- llama-bench ran on Device 0 only (-d 0)
