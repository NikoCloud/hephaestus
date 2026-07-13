# Co-measured baseline — Hephaestus BF16 vs llama.cpp (same session window)

**Date:** 2026-07-13  
**Hardware:** GPU 0 = AMD Radeon AI PRO R9700 (gfx1201)  
**GPU hygiene:** Free before both llama.cpp and Hephaestus runs (monitors only).

### llama.cpp (minutes earlier, same card)

| quant | metric | tok/s (5 reps) |
|-------|--------|----------------|
| F16 | tg128 decode | **75.17 ± 0.10** |
| Q8_0 | tg128 decode | **127.25 ± 0.21** |
| F16 | pp512 prefill | **8133.90 ± 60.40** |
| Q8_0 | pp512 prefill | **7687.73 ± 131.87** |

Source: `bench/llamacpp-q8-baseline.md` (Vulkan0, build `33ca0dcb9`).

### Hephaestus BF16 (this run, immediately after)

| Item | Value |
|------|--------|
| Tree | `attn-stopgap` stack (v3a WMMA + residual + parallel attention) |
| Env | `~/projects/hephaestus-wmma-nightly` |
| Binary | `qwen_ab_bench.mojo` |
| Decode config | 10-token prompt (`ab_prompt_short_ids.txt`), **256** tokens generated, 3 reps |
| Prefill config | 512-token prompt (`ab_prompt_long_ids.txt`), 8 tokens generated, 3 reps |

---

## Summary table (means)

| Metric | Hephaestus BF16 | llama.cpp F16 | Ratio (Heph/F16) | llama.cpp Q8_0 | Ratio (Heph/Q8_0) |
|--------|----------------:|--------------:|-----------------:|---------------:|------------------:|
| **Decode tok/s** (fwd-only) | **62.37 ± 0.01** | **75.17 ± 0.10** | **0.83×** | **127.25 ± 0.21** | **0.49×** |
| Decode tok/s (incl. GPU argmax) | 61.23 ± 0.01 | — | — | — | — |
| **Prefill tok/s** (fwd-only, pp512) | **1400.7 ± 2.5** | **8133.9 ± 60.4** | **0.17×** | **7687.7 ± 131.9** | **0.18×** |
| TTFT ms (fwd-only, 512) | 365.5 ± 0.7 | — | — | — | — |
| **Total wall 256-gen** (s) | **4.250 ± 0.001** | ~3.41† | 0.80× | ~2.01† | 0.47× |

† llama.cpp wall for 256 tokens estimated as `256 / tg128` (pure decode, no short prefill). Hephaestus total includes 10-token prefill + 256 decode steps + argmax.

**G1b-2 gate** = match Q8_0 decode ≈ **127.3 tok/s** → Hephaestus needs **~2.0×** from 62.4 (forward-only).

---

## Hephaestus per-rep detail

### Decode (10-tok prompt → 256 gen)

| rep | decode fwd-only | decode +argmax | total_s |
|----:|----------------:|---------------:|--------:|
| 1 | 62.379 | 61.246 | 4.251 |
| 2 | 62.374 | 61.230 | 4.249 |
| 3 | 62.360 | 61.228 | 4.249 |
| **mean** | **62.371** | **61.235** | **4.250** |

### Prefill (512-tok prompt → 8 gen)

| rep | prefill_tok_s | ttft_ms (fwd) |
|----:|--------------:|--------------:|
| 1 | 1397.9 | 366.3 |
| 2 | 1402.4 | 365.1 |
| 3 | 1401.9 | 365.2 |
| **mean** | **1400.7** | **365.5** |

---

## Honest read

1. **Decode (primary gate):** Co-measured **62.4 tok/s** vs llama F16 **75.2** (0.83× equal precision) and Q8 **127.3** (0.49× / G1b-2). Prior “~54” is slightly pessimistic vs this build; still **~2×** short of Q8 decode.
2. **Prefill:** Co-measured **1401 tok/s** vs F16 **8134** (0.17×) / Q8 **7688** (0.18×). Unchanged story — prefill is not the competitive surface.
3. **Use this table** for FP8 and future work, not cross-day mixes.

---

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
# Build (nightly + kernels; tree with attn-stopgap + v3a)
cd ~/projects/hephaestus-wmma-nightly
pixi run mojo build -I ~/projects/modular/max/kernels/src -I ~/projects/hephaestus/src \
  ~/projects/hephaestus/src/qwen_ab_bench.mojo -o /tmp/qwen_ab_com

# Decode 3×
for i in 1 2 3; do /tmp/qwen_ab_com ~/projects/hephaestus/bench/ab_prompt_short_ids.txt 256; done

# Prefill 3×
for i in 1 2 3; do /tmp/qwen_ab_com ~/projects/hephaestus/bench/ab_prompt_long_ids.txt 8; done
```
