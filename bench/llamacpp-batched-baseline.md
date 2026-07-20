# llama.cpp Batched Throughput Baseline (Q8_0)

**Date:** 2026-07-20  
**Purpose:** Phase 2 premise is "we beat llama at concurrency." Measure llama's actual batched throughput **before** building toward it. Pure measurement — no Hephaestus code, no optimization.

## Environment

| Item | Value |
|------|-------|
| Device | GPU 0 only — AMD Radeon AI PRO R9700 (gfx1201), 32624 MiB VRAM |
| Backend | **ROCm/HIP** (`ROCm0`) |
| ROCm | **7.2.4** |
| llama.cpp commit | **`33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519`** (`33ca0dcb9` — ggml-hip: `-fno-finite-math-only` alongside `-ffast-math` #25373) |
| Binary | `~/projects/llama.cpp/build/bin/llama-batched-bench` (built `cmake --build build --target llama-batched-bench`) |
| Model | `/mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf` — Q8_0, **3.98 GiB file / ~4.27 GB**, 4.02B params |
| GPU model buffer | ROCm0 **4076.43 MiB** (+ CPU_Mapped 394.12 MiB for residual) |
| Flash attention | **Available and enabled** (`-fa on` → `flash_attn = enabled` / `flash_attn = 1`) |
| Achievable BW (card) | **569 GB/s** (same roofline as prior decode baselines) |

**Note on Q8_0 stability:** Q8_0 is stable across backends (prior co-measure: ~7745 ROCm vs ~7688 Vulkan pp512). This sweep is ROCm only.

## Command

```bash
# GPUs free (rocm-smi: 0% VRAM, no KFD PIDs) before run
HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build/bin/llama-batched-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf \
  -c 16384 \
  -npp 512 \
  -ntg 128 \
  -npl 1,2,4,8,16 \
  -ngl 99 \
  -fa on
```

Flag notes from `--help` (this build):
- `-fa` / `--flash-attn [on|off|auto]` — not a bare boolean; used `-fa on`
- `-npl` = parallel sequences; `-npp` = prompt tokens; `-ntg` = gen tokens

Context construction (from load log, `-lv 4` required to see INFO — this tree maps `GGML_LOG_LEVEL_INFO` → verbosity TRACE):

```
llama_context: n_seq_max     = 16
llama_context: n_ctx         = 16384
llama_context: n_ctx_seq     = 1024
llama_context: flash_attn    = enabled
llama_context: kv_unified    = false
llama_kv_cache:      ROCm0 KV buffer size =  2304.00 MiB
llama_kv_cache: size = 2304.00 MiB (  1024 cells,  36 layers, 16/16 seqs), K (f16): 1152.00 MiB, V (f16): 1152.00 MiB
```

KV is allocated **once** for `n_seq_max = max(npl) = 16` at 1024 cells/seq — not re-sized per row.

## Metric definitions (do not double-count)

Batched-bench reports **aggregate** rates:

| Symbol | Meaning |
|--------|---------|
| `S_TG` | Aggregate decode tok/s = total tokens across all sequences ÷ time |
| `S_PP` | Aggregate prefill tok/s |
| per-stream TG | `S_TG / npl` |
| per-stream PP | `S_PP / npl` |
| scaling % | `S_TG / (npl × (S_TG at npl=1)) × 100` |

**Do not** multiply aggregate by npl — that double-counts.

### Bandwidth accounting (weights + KV)

| Constant | Value | Source |
|----------|-------|--------|
| Weight bytes | 4.27 GB | Q8_0 model size |
| KV bytes/token | ~147 KB | 8 KV heads × 128 dim × 2 (K+V) × 36 layers × 2 bytes (f16) |
| Avg context depth during gen | 576 | after 512-token prefill, mid-point of 128-token gen ≈ 512+64 |
| Roofline | 569 GB/s | measured achievable on this card |

```
total_bytes_per_forward (GB) = 4.27 + (npl × 576 × 147 / 1024 / 1024)
effective_BW (GB/s)          = (S_TG × total_bytes_per_forward) / npl
% roofline                   = effective_BW / 569 × 100
```

Why total bytes: weights-only accounting invents a fake droop as npl grows (KV reads grow linearly; weight reads stay fixed). Total-bytes denominator is flat ~100% for a perfectly saturated implementation — any real droop is genuine overhead.

## Results

Raw table from `llama-batched-bench` (representative run with full load log; three runs agreed within ~1%):

```
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   512 |    128 |    1 |    640 |    0.071 |  7177.30 |    1.210 |   105.81 |    1.281 |   499.59 |
|   512 |    128 |    2 |   1280 |    0.125 |  8169.78 |    1.381 |   185.40 |    1.506 |   849.87 |
|   512 |    128 |    4 |   2560 |    0.254 |  8056.49 |    1.483 |   345.31 |    1.737 |  1473.85 |
|   512 |    128 |    8 |   5120 |    0.508 |  8058.80 |    1.862 |   549.80 |    2.371 |  2159.64 |
|   512 |    128 |   16 |  10240 |    1.019 |  8040.20 |    2.114 |   968.58 |    3.133 |  3268.11 |
```

### Derived table

| npl | tg/stream | tg agg | pp/stream | pp agg | scaling % | total bytes/fwd (GB) | eff BW (GB/s) | % roofline | KV self (MiB) |
|-----|-----------|--------|-----------|--------|-----------|----------------------|---------------|------------|---------------|
| 1   | 105.81    | 105.81 | 7177.30   | 7177.30 | 100.0%    | 4.3507               | 460.4         | **80.9%**  | 2304.00       |
| 2   | 92.70     | 185.40 | 4084.89   | 8169.78 | 87.6%     | 4.4315               | 410.8         | **72.2%**  | 2304.00       |
| 4   | 86.33     | 345.31 | 2014.12   | 8056.49 | 81.6%     | 4.5930               | 396.5         | **69.7%**  | 2304.00       |
| 8   | 68.72     | 549.80 | 1007.35   | 8058.80 | 65.0%     | 4.9160               | 337.9         | **59.4%**  | 2304.00       |
| 16  | 60.54     | 968.58 | 502.51    | 8040.20 | 57.2%     | 5.5620               | 336.7         | **59.2%**  | 2304.00       |

KV self is the **exact** load-time line: `size = 2304.00 MiB (1024 cells, 36 layers, 16/16 seqs)`. Same allocation for every row (bench sets `n_seq_max = 16` once).

## Analysis

### 1. npl=1 calibration

| | tok/s |
|--|------:|
| This batched-bench npl=1 (after 512-token prefill) | **105.81** |
| Prior llama-bench `tg128` Q8_0 (near-empty context) | ~109.8 |

**Within expected band (95–105-ish / slightly above 105).** Batched-bench generates against a 512-deep KV; llama-bench tg128 is lighter. Not wildly off — no investigation needed.

### 2. Efficiency curve shape (total-bytes accounting) — **most important output**

| npl | % roofline (weights+KV) |
|-----|------------------------:|
| 1   | 80.9% |
| 2   | 72.2% |
| 4   | 69.7% |
| 8   | 59.4% |
| 16  | 59.2% |

**Shape: real droop, not flat.** From ~81% → ~59% as concurrency goes 1 → 16. With total-bytes in the denominator this is **genuine implementation overhead**, not the physics of larger KV. A mature, bandwidth-saturated multi-stream decode would hold ~90%+ flat across npl.

Contrast (weights-only, **do not use for Phase 2 claims** — invents extra droop):

| npl | % roofline (weights only) |
|-----|--------------------------:|
| 1   | 79.4% |
| 8   | 51.6% |
| 16  | 45.4% |

### 3. Scaling efficiency (throughput vs ideal linear)

- npl 1→2: **87.6%** (mild sublinear)
- npl 1→4: **81.6%**
- npl 1→8: **65.0%**
- npl 1→16: **57.2%** (saturating)

Aggregate decode still climbs (106 → 969 tok/s) but per-stream falls 106 → 61 tok/s. Prefill aggregate stays ~8k tok/s (GPU-compute bound); per-stream prefill divides cleanly.

### 4. KV-cache VRAM efficiency (capacity bar)

| Quantity | Value |
|----------|------:|
| KV self (printed) | **2304.00 MiB** |
| Sequences / cells | 16 seqs × 1024 cells (f16 K+V) |
| **Per concurrent slot** | **144.00 MiB** (2304/16) |
| Bytes/token (measured) | 144 KiB = 8×128×2×36×2 |

**This is the capacity number FP8 competes on.** Paged attention does **not** reduce KV **reads** — each sequence still reads its own cache. Paging eliminates fragmentation / over-allocation so more sequences fit in the same VRAM. Capacity win compounds with FP8 weight footprint.

At this config llama reserves **~2.25 GiB** of KV for 16 concurrent 1k-token slots on top of ~4.0 GiB Q8 weights — still comfortable on 32 GB, but the **per-slot 144 MiB/1k-ctx** is the density bar. FP8 weights free VRAM for more slots or longer ctx; paged layout must match or beat llama's packing (here: exact theoretical density, no obvious over-alloc beyond the 1024-cell ceiling vs 640-token working set).

Working-set note: each sequence only needs 640 tokens (512+128) but llama allocated 1024 cells/seq because `n_ctx / n_seq_max = 16384/16`. Spare ~37% cells per slot at this workload.

### 5. What this means for Phase 2

| Bar | llama position | Fused+paged must clear |
|-----|----------------|------------------------|
| **Efficiency** | ~81% roofline at npl=1; **droops to ~59%** by npl=8–16 under honest total-bytes accounting | Beat the **curve**, not a single point — especially npl≥8 where llama leaves ~40% of BW on the table |
| **Capacity** | **144 MiB / concurrent slot / 1k ctx** (f16 KV, exact theory) | Match density; FP8 weights buy more slots in the same VRAM; paging must not waste that headroom |
| **Aggregate decode** | 106 → 969 tok/s (npl 1→16) | Higher aggregate **and** higher/flatter % roofline |
| **Prefill concurrent** | Aggregate ~8.0–8.2k tok/s flat | Separate problem; Phase 1 prefill already co-measured |

Opening for fused+paged design: llama's multi-stream decode is **not** a flat 90%+ mature bar — it **degrades with concurrency** even after KV-bytes are accounted. That droop is the implementation gap Phase 2 targets. Capacity (KV self / slot) remains a hard floor FP8 must respect or beat.

### 6. Method notes

- Both GPUs idle before run (`rocm-smi`: 0% VRAM, no KFD PIDs); `HIP_VISIBLE_DEVICES=0` only.
- Three full sweeps reproduced S_TG within ~1% (npl=1: 105.44–105.81).
- `-fa on` confirmed in load log and bench banner (`flash_attn = 1`).
- INFO-level KV line requires `-lv 4` on this llama.cpp tree (INFO mapped to TRACE verbosity); value captured from that run.
- No Hephaestus binary, no engine changes, no optimization — measurement only.

## Raw artifact

Full load + table: `/tmp/llama-bb-v4.log` (session machine). Key excerpt:

```
I load_tensors:        ROCm0 model buffer size =  4076.43 MiB
I llama_kv_cache:      ROCm0 KV buffer size =  2304.00 MiB
I llama_kv_cache: size = 2304.00 MiB (  1024 cells,  36 layers, 16/16 seqs), K (f16): 1152.00 MiB, V (f16): 1152.00 MiB
llama_batched_bench: n_kv_max = 16384, ..., flash_attn = 1, ..., n_gpu_layers = 99
```
