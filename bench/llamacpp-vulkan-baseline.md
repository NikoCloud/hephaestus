# llama.cpp Vulkan vs ROCm baseline (Q8_0) — co-measure

**Date:** 2026-07-20  
**Purpose:** All prior llama baselines were ROCm/HIP. On RDNA, Vulkan often wins — measure the honest best-in-class line before any “we match llama” claim.  
**Scope:** Pure measurement. No engine code, no tuning, no quant/model change. One variable: **backend**.

## Environment

| Item | Value |
|------|-------|
| GPU | **GPU 0 only** — AMD Radeon AI PRO R9700 **gfx1201**, 32624 MiB |
| Model | `/mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf` (Q8_0, 3.98 GiB / ~4.27 GB, 4.02B) |
| llama.cpp | **`33ca0dcb9`** (build 9906) — same commit both backends |
| ROCm build | `~/projects/llama.cpp/build` (`GGML_HIP=ON`, `GGML_VULKAN=OFF`) |
| Vulkan build | `~/projects/llama.cpp/build-vulkan` (`GGML_VULKAN=ON`, `GGML_HIP=OFF`) — **separate dir, ROCm build untouched** |
| ROCm runtime | 7.2.4 |
| Vulkan driver | Mesa **RADV** 26.1.4 (`radv`, non-conformant warning printed) |
| Flash attention | **on** both (`-fa on` → `flash_attn = 1` / enabled) |
| Layers | `-ngl 99` → **37/37 offloaded** both |
| Device pin | ROCm: `HIP_VISIBLE_DEVICES=0 -dev ROCm0`; Vulkan: `-dev Vulkan0` |
| Roofline | 569 GB/s (same card reference as ROCm batched baseline) |
| Session | Intended named tmux `llamacpp-vk-baseline` (tmux server unavailable in this environment); ran as **named nohup job** `/tmp/llamacpp-vulkan-baseline/run_co_measure.sh` with full log |

### Device confirmation (void if wrong)

**ROCm** (`ggml_cuda_init` / prepare):
```
Device 0: AMD Radeon AI PRO R9700, gfx1201 (0x1201)
using device ROCm0 (AMD Radeon AI PRO R9700) (0000:11:00.0)
load_tensors: offloaded 37/37 layers to GPU
ROCm0 model buffer size = 4076.43 MiB
```

**Vulkan** (`ggml_vulkan` / prepare):
```
ggml_vulkan: 0 = AMD Radeon AI PRO R9700 (RADV GFX1201) (radv) | ... | matrix cores: KHR_coopmat
using device Vulkan0 (AMD Radeon AI PRO R9700 (RADV GFX1201)) (0000:11:00.0)
load_tensors: offloaded 37/37 layers to GPU
Vulkan0 model buffer size = 4076.43 MiB
```

Both on **gfx1201 R9700**, full GPU offload — **not CPU, not GPU1**. Run is valid.

### Feature parity (what worked)

| Feature | ROCm | Vulkan |
|---------|------|--------|
| `llama-bench` | yes | yes |
| `llama-batched-bench` + `-npl` concurrent slots | **yes** | **yes** (same binary family, same flags) |
| Flash attention `-fa on` | enabled | enabled |
| KV f16 default | yes | yes |
| KV self size @ n_seq_max=16, 1024 cells | **2304.00 MiB** | **2304.00 MiB** (identical) |

**No batched-bench / concurrency gap on Vulkan.** Same API, same allocation.

---

## 1. llama-bench — pp512 + tg128, 3 reps, ROCm → Vulkan → ROCm

Flags (identical except device):
```bash
# ROCm
HIP_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -m ...-q8_0.gguf -p 512 -n 128 -ngl 99 -fa on -r 3 -dev ROCm0

# Vulkan
build-vulkan/bin/llama-bench \
  -m ...-q8_0.gguf -p 512 -n 128 -ngl 99 -fa on -r 3 -dev Vulkan0
```

### Co-measure sandwich (primary)

| Order | Backend | pp512 (mean ± std) | tg128 (mean ± std) |
|------:|---------|-------------------:|-------------------:|
| 1 | **ROCm** | 7909.85 ± 863.78 | **109.39 ± 0.57** |
| 2 | **Vulkan** | 7678.12 ± 138.19 | **127.06 ± 0.04** |
| 3 | **ROCm** | 7511.34 ± 1309.74 | **109.62 ± 0.47** |

Thermal bound: ROCm tg128 **109.39 → 109.62** (+0.21%). Drift is noise; decode comparison is stable.

ROCm pp512 std is large (cold first rep in each ROCm process). Decode std is tiny on both backends.

Sanity vs prior ROCm-only notes (pp~7839, tg~109.8): this co-measure ROCm tg lands **~109.5**; pp is in the same ballpark once cold start is ignored.

### Per-rep samples (warm follow-up jsonl, same flags, after main sandwich)

| Backend | test | rep1 | rep2 | rep3 | mean |
|---------|------|-----:|-----:|-----:|-----:|
| ROCm | pp512 | 6419.54 | 8140.42 | 8343.88 | 7634.61 |
| ROCm | tg128 | 108.836 | 109.660 | 109.913 | 109.47 |
| Vulkan | pp512 | 7428.76 | 7669.14 | 7648.12 | 7582.01 |
| Vulkan | tg128 | 127.074 | 127.001 | 126.980 | 127.02 |

### llama-bench ratios (co-measure means: ROCm = avg of runs 1&3)

| Metric | ROCm | Vulkan | **Vulkan/ROCm** |
|--------|-----:|-------:|----------------:|
| pp512 | ~7711 | 7678 | **0.996×** (~flat; ROCm noisy) |
| tg128 | ~109.5 | **127.1** | **1.16×** (**Vulkan +16% decode**) |

**Honest single-stream line: Vulkan decode is the bar, not ROCm.**

---

## 2. llama-batched-bench — npl 1/2/4/8/16 (same params as ROCm baseline)

```bash
# identical except binary + -dev
…/llama-batched-bench -m …-q8_0.gguf -c 16384 -npp 512 -ntg 128 \
  -npl 1,2,4,8,16 -ngl 99 -fa on -lv 4 -dev ROCm0|Vulkan0
```

### Raw tables (co-measure order: ROCm then Vulkan)

**ROCm** (`flash_attn=1`, KV self **2304.00 MiB**):
```
| PP  | TG  | B  | N_KV  | S_PP t/s | S_TG t/s |
| 512 | 128 |  1 |   640 |  7148.74 |  106.11  |
| 512 | 128 |  2 |  1280 |  8183.16 |  185.69  |
| 512 | 128 |  4 |  2560 |  8031.47 |  345.12  |
| 512 | 128 |  8 |  5120 |  8045.03 |  549.65  |
| 512 | 128 | 16 | 10240 |  8039.82 |  962.30  |
```

**Vulkan** (`flash_attn=1`, KV self **2304.00 MiB**):
```
| PP  | TG  | B  | N_KV  | S_PP t/s | S_TG t/s |
| 512 | 128 |  1 |   640 |  6584.79 |  121.98  |
| 512 | 128 |  2 |  1280 |  7324.80 |  224.24  |
| 512 | 128 |  4 |  2560 |  7501.64 |  417.76  |
| 512 | 128 |  8 |  5120 |  7489.95 |  677.83  |
| 512 | 128 | 16 | 10240 |  7478.88 |  873.52  |
```

ROCm agg decode vs prior ROCm-only baseline (105.8 / 185.4 / 345.3 / 549.8 / 968.6): **within ~1%** — co-measure ROCm is consistent with the earlier doc.

### Derived: per-stream, scaling, BW (weights+KV), ratios

Accounting (same as `bench/llamacpp-batched-baseline.md`):
```
total_bytes_per_forward (GB) = 4.27 + (npl × 576 × 147 / 1024 / 1024)
effective_BW (GB/s)          = (S_TG × total_bytes_per_forward) / npl
% roofline                   = effective_BW / 569 × 100
scaling %                    = S_TG / (npl × S_TG@npl=1) × 100
```

| npl | ROCm tg/str | ROCm tg agg | VK tg/str | VK tg agg | **VK/ROCm tg** | ROCm scal% | VK scal% | tot B/fwd (GB) | ROCm eff BW | VK eff BW | ROCm %roof | VK %roof | ROCm pp agg | VK pp agg | **VK/ROCm pp** | KV self (MiB) |
|----:|------------:|------------:|----------:|----------:|---------------:|-----------:|---------:|---------------:|------------:|----------:|-----------:|---------:|------------:|----------:|---------------:|--------------:|
| 1 | 106.11 | 106.11 | 121.98 | 121.98 | **1.150×** | 100% | 100% | 4.3507 | 461.7 | **530.7** | 81.1% | **93.3%** | 7149 | 6585 | 0.921× | 2304 both |
| 2 | 92.85 | 185.69 | 112.12 | 224.24 | **1.208×** | 87.5% | 91.9% | 4.4315 | 411.4 | **496.9** | 72.3% | **87.3%** | 8183 | 7325 | 0.895× | 2304 |
| 4 | 86.28 | 345.12 | 104.44 | 417.76 | **1.210×** | 81.3% | 85.6% | 4.5930 | 396.3 | **479.7** | 69.6% | **84.3%** | 8031 | 7502 | 0.934× | 2304 |
| 8 | 68.71 | 549.65 | 84.73 | 677.83 | **1.233×** | 64.8% | 69.5% | 4.9160 | 337.8 | **416.5** | 59.4% | **73.2%** | 8045 | 7490 | 0.931× | 2304 |
| 16 | 60.14 | **962.30** | 54.60 | **873.52** | **0.908×** | 56.7% | 44.8% | 5.5620 | **334.5** | 303.7 | **58.8%** | 53.4% | 8040 | 7479 | 0.930× | 2304 |

---

## 3. What the numbers say

### Prefill

- **Single-stream llama-bench:** Vulkan ≈ ROCm (0.996× on co-measure means; both ~7.5–7.9k with ROCm high variance).
- **Batched S_PP:** Vulkan is **~7–10% behind** ROCm at every npl (ratios 0.895–0.934). ROCm holds ~8.0–8.2k aggregate; Vulkan ~7.3–7.5k after npl≥2.

### Decode (the claim that matters)

- **Single-stream:** Vulkan **+16%** (127 vs ~109.5). This is the honest **best-in-class single-stream** line on this card for Q8.
- **Concurrency npl 1–8:** Vulkan **holds and widens** the gap (~1.15× → **1.23×**). Not a single-stream-only win.
- **npl=16:** gap **collapses and reverses** — ROCm **962** vs Vulkan **874** (Vulkan **0.91×**). Vulkan scaling efficiency falls harder (VK 44.8% vs ROCm 56.7% of linear at npl=16).

### Efficiency curve (total-bytes accounting)

| | npl=1 | npl=8 | npl=16 |
|--|------:|------:|-------:|
| ROCm % of 569 | 81% | 59% | 59% |
| Vulkan % of 569 | **93%** | **73%** | **53%** |

Vulkan is closer to the synthetic roofline at low–mid concurrency, then **underperforms ROCm at npl=16**. Best bar is **backend- and npl-dependent**, not a single number.

### Capacity

KV self **identical** (2304 MiB / 144 MiB per 1k-cell slot). Backend does not change the capacity thesis.

---

## 4. Implications for “match llama”

| Claim surface | Honest target (this co-measure) |
|---------------|----------------------------------|
| Single-stream decode (tg128) | **Vulkan ~127 tok/s**, not ROCm ~110 |
| Prefill (pp512) | Either ~7.6–7.9k; ROCm/Vulkan within noise on llama-bench |
| Concurrent decode npl≤8 | **Vulkan** is the higher bar (+15–23%) |
| Concurrent decode npl=16 | **ROCm** is the higher bar (~962 vs ~874) |
| Capacity / KV MiB | Same both |

Do **not** claim “match llama” against ROCm-only numbers when Vulkan is available and faster on the metric you care about.

---

## 5. Artifacts

| Path | Content |
|------|---------|
| `/tmp/llamacpp-vulkan-baseline/full.log` | Full co-measure transcript |
| `…/rocm{1,2}_bench.txt`, `vulkan_bench.txt` | llama-bench tables |
| `…/rocm_bench.jsonl`, `vulkan_bench.jsonl` | Per-rep samples (warm follow-up) |
| `…/rocm_batched.txt`, `vulkan_batched.txt` | Batched tables |
| `…/*_stderr.txt` | Device / KV / offload confirmation |
| `…/run_co_measure.sh` | Repro script |

## 6. Reproduce

```bash
# builds already present; rebuild targets only if needed
cmake --build ~/projects/llama.cpp/build -j --target llama-bench llama-batched-bench
cmake --build ~/projects/llama.cpp/build-vulkan -j --target llama-bench llama-batched-bench

# GPU idle
rocm-smi; rocm-smi --showpids

bash /tmp/llamacpp-vulkan-baseline/run_co_measure.sh
# or step-by-step with -dev ROCm0 / -dev Vulkan0 as above
```

---

## Bottom line

1. **Vulkan is real on gfx1201** (RADV GFX1201, full offload, FA on, batched-bench works).  
2. **Decode bar moves up ~16%** single-stream (127 vs 110).  
3. **Vulkan lead holds through npl=8**, then **ROCm wins at npl=16**.  
4. Prefill: batched ROCm slightly ahead; single-stream roughly tied.  
5. No feature gap that voids the concurrency comparison — same KV footprint, same FA flag.

Measurement only. No Hephaestus / `src/` changes.
