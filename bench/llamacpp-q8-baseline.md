# llama.cpp baseline — prefill (G1b-3) + decode (G1b-2) on R9700 Vulkan (2026-07-13)

**Hardware:** GPU 0 = AMD Radeon AI PRO R9700 32GB (gfx1201 / RADV)  
**Model:** Qwen3-4B-Instruct-2507  
**Tool:** `~/projects/llama.cpp/build-vulkan/bin/llama-bench` build `33ca0dcb9 (9906)`, **Vulkan** backend  
**Device:** `Vulkan0` = R9700 (verified via `llama-bench --list-devices`)  
**Methods:**
- Prefill: `-p 512 -n 0 -r 5 -ngl 99 -dev Vulkan0`
- Decode: `-p 0 -n 128 -r 5 -ngl 99 -dev Vulkan0` (tg128)
**GPU hygiene:** KFD empty; no compute on either GPU (monitors only: `nvtop`/`amdgpu_top`).

GGUFs:
- F16: `/mnt/models/models/qwen3-4b-instruct-2507-f16.gguf` (7.49 GiB, 4.02B)
- Q8_0: `/mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf` (3.98 GiB) — quantized this session from F16 via `llama-quantize … Q8_0`

---

## Results (pp512, tok/s)

| quant | size | pp512 (5 reps) |
|-------|------|----------------|
| **F16** | 7.49 GiB | **8133.90 ± 60.40** |
| **Q8_0** | 3.98 GiB | **7687.73 ± 131.87** |

Raw tables:

```
| qwen3 4B F16  | 7.49 GiB | 4.02 B | Vulkan | 99 | Vulkan0 | pp512 | 8133.90 ± 60.40 |
| qwen3 4B Q8_0 | 3.98 GiB | 4.02 B | Vulkan | 99 | Vulkan0 | pp512 | 7687.73 ± 131.87 |
```

Note: Q8_0 is slightly *below* F16 on this card/backend (dequant cost vs F16 coopmat path). The **gate still uses Q8_0 × 1.5** per SPEC, not F16.

---

## Computed G1b-3 gate

| Quantity | Value |
|----------|------:|
| Q8_0 pp512 | **7688 ± 132** tok/s |
| **G1b-3 = 1.5 × Q8_0** | **≈ 11532 tok/s** |
| Prior extrapolation (stale) | ~2100 tok/s |

**The real gate is ~11.5k tok/s**, not ~2.1k. Prior “~2100” came from outdated/mis-scaled baselines (~1400 Q8 × 1.5).

---

## Where Hephaestus BF16 (~1398 tok/s) sits

Current Hephaestus prefill (attn-stopgap + v3a WMMA, 512-token, mean of 3 reps ≈ **1398 tok/s** from `bench/attn-stopgap.md`):

| Comparison | Ratio | Comment |
|------------|------:|---------|
| vs llama.cpp **F16** (equal-ish precision) | **0.17×** (17%) | Not competitive at 16-bit |
| vs llama.cpp **Q8_0** | **0.18×** | |
| vs **G1b-3 gate (1.5× Q8)** | **0.12×** (12%) | Need **~8.2×** more e2e |

Equal-precision grade: at BF16/F16, we are **~6× behind** Vulkan llama.cpp F16 on the same card.

---

## Implications for FP8 / remaining roadmap (prefill)

| Target | tok/s | vs Hephaestus 1398 |
|--------|------:|-------------------:|
| Match F16 llama.cpp | ~8134 | **5.8×** |
| Clear G1b-3 (1.5× Q8) | **~11532** | **8.2×** |

- **G1b-3 prefill gate is being retired as primary** — unreachable without llama.cpp-class attention + more.
- Prefill still matters for product quality, but **decode (G1b-2) is the primary speed gate** going forward.

---

## Decode results (tg128, tok/s) — G1b-2 baseline

Measured 2026-07-13, same machine/backend/device as prefill (`-p 0 -n 128 -r 5 -ngl 99 -dev Vulkan0`).

| quant | size | tg128 (5 reps) |
|-------|------|----------------|
| **F16** | 7.49 GiB | **75.17 ± 0.10** |
| **Q8_0** | 3.98 GiB | **127.25 ± 0.21** |

Raw:

```
| qwen3 4B F16  | 7.49 GiB | 4.02 B | Vulkan | 99 | Vulkan0 | tg128 |  75.17 ± 0.10 |
| qwen3 4B Q8_0 | 3.98 GiB | 4.02 B | Vulkan | 99 | Vulkan0 | tg128 | 127.25 ± 0.21 |
```

**G1b-2 gate (match or beat Q8_0 decode):** **≈ 127.3 tok/s**

Unlike prefill, Q8_0 **beats** F16 on decode here (~1.7×) — memory-bound decode benefits from the smaller weight footprint.

---

## Where Hephaestus BF16 decode (~54 tok/s) sits

Phase 1a / G1a-2 Hephaestus decode (forward-only, ~**54 tok/s** from prior benches — e.g. `bench/1a-ab.md` ~52.8–54.4):

| Comparison | Ratio | Comment |
|------------|------:|---------|
| vs llama.cpp **F16** (matched precision) | **0.72×** (72%) | Within striking distance at 16-bit |
| vs llama.cpp **Q8_0** (**G1b-2 gate**) | **0.42×** (42%) | Need **~2.4×** for gate |

| Target | tok/s | vs Hephaestus ~54 |
|--------|------:|------------------:|
| Match F16 decode | ~75 | **1.4×** |
| **G1b-2 = match Q8_0 decode** | **~127** | **~2.4×** |

**Read for FP8 on decode:** closing **~2.4×** to Q8_0 is a plausible GEMV/weight-bandwidth win if FP8 decode path is solid; matching F16 (~1.4×) is an intermediate equal-precision bar. Unlike the retired prefill 8× cliff, **decode is a realistic primary gate**.

---

## Reproduce

```bash
# Free GPUs (no KFD compute)
rocm-smi --showpids

# Devices
~/projects/llama.cpp/build-vulkan/bin/llama-bench --list-devices
# expect Vulkan0 = R9700, Vulkan1 = 9070 XT

# Quantize once
~/projects/llama.cpp/build-vulkan/bin/llama-quantize \
  /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf \
  /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf Q8_0

# Prefill
HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build-vulkan/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf \
  -p 512 -n 0 -r 5 -ngl 99 -dev Vulkan0

HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build-vulkan/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf \
  -p 512 -n 0 -r 5 -ngl 99 -dev Vulkan0

# Decode (tg128)
HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build-vulkan/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf \
  -p 0 -n 128 -r 5 -ngl 99 -dev Vulkan0

HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build-vulkan/bin/llama-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf \
  -p 0 -n 128 -r 5 -ngl 99 -dev Vulkan0
```
