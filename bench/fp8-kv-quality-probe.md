# FP8 KV-cache quality probe

**Date:** 2026-07-20  
**Branch:** `fp8-kv-probe`  
**Decides:** whether FP8-KV enters Phase 2 scope as architecture.  
**Rule:** validate quality via quantize→dequantize simulation **before** building real FP8 storage.

## Method (simulation, not storage)

| Item | Choice |
|------|--------|
| What | After cache-write values are ready, E4M3 quant→dequant **in place**; storage stays **BF16** |
| K | Roundtrip **after RoPE** (post-rotation distribution) |
| V | Roundtrip after `v_proj` (no RoPE) |
| Scale | Per-token row absmax / 448, **separate K vs V**, per layer (caller) |
| Flag | `forward_fp8[..., fp8_kv_cache=False]` — default path unchanged |
| GPU | Device 0 only (`HIP_VISIBLE_DEVICES=0`) |
| Env | `hephaestus-wmma-nightly` Mojo `1.0.0b3.dev2026071206` |
| Weights | FP8 E4M3 `staged/qwen3-4b-fp8` (same as 97.4% baseline) |

Capacity check (theory, including scale overhead):

| | bytes/token |
|--|------------:|
| BF16 KV | 147 456 (8×128×2×36×2) |
| FP8 KV + K/V f32 scales | 74 016 (73 728 + 288) |
| **Ratio** | **1.992×** (~2×) |

Scale tax is &lt;0.4% of the savings. Claiming ~2× capacity is honest.

## Gate

**PASS = ≥95% teacher-forced argmax parity at 4K** (self-A/B vs BF16-KV), with uniform noise shape preferred over outlier-shaped collapse.

Also report delta vs **97.4%** absolute baseline (FP8 weights + BF16 KV, 748/768 vs HF oracle).

## Results (co-measured, same session)

### 512 / existing HF oracle protocol (3 prompts × 256 steps)

Teacher-forced on real M=1 decode path; feed **oracle** tokens (not our argmax). Short prompts (~10–12 tok) + 256 gen — same protocol as the 97.4% baseline.

| Config | Match | Rate |
|--------|------:|-----:|
| **BF16-KV** (FP8 weights) | **748 / 768** | **97.40%** |
| **FP8-KV** (FP8 weights + E4M3 K/V roundtrip) | **743 / 768** | **96.74%** |
| Prior baseline (docs) | 748 / 768 | 97.4% |

| Prompt | BF16-KV | FP8-KV |
|--------|--------:|-------:|
| 1 | 256/256 | 256/256 |
| 2 | 246/256 | 242/256 |
| 3 | 246/256 | 245/256 |

**Delta vs 97.4%:** −5 tokens (−0.65 points). Absolute FP8-KV still **≥95%**.

Mismatch steps remain sparse and mostly overlapping with BF16-KV’s known hard steps (prompt 2/3); FP8-KV adds a few extra early steps on p2 — not a single catastrophic position.

### 4K self-A/B (deciding test)

Same 4096-token prompt (8× `ab_prompt_long_ids` concat), chunked teacher-forced walk (chunk=64). Both configs use FP8 weights; only KV write path differs. Compare argmax at every position.

| Metric | Value |
|--------|------:|
| **Match** | **4093 / 4096** |
| **Rate** | **99.927%** |
| Mismatches | **3** |

Mismatch positions (uniform sparse noise):

| pos | BF16-KV argmax | FP8-KV argmax |
|----:|---------------:|--------------:|
| 71 | 220 | 89673 |
| 149 | 1096 | 576 |
| 3071 | 785 | 279 |

Mismatch density by 512-token bin:

| bin | mismatches |
|-----|----------:|
| [0, 512) | 2 |
| [512, 1024) … [2048, 2560) | 0 |
| [2560, 3072) | 1 |
| [3072, 4096) | 0 |

**Shape: uniform sparse quantization noise — not outlier-shaped.** No late-context collapse; error does not snowball with depth under this scaling scheme. Scaling scheme is fine; no need to redesign scales before scoping storage.

## Verdict

| Criterion | Result |
|-----------|--------|
| ≥95% at 4K (self-A/B) | **PASS (99.93%)** |
| Divergence shape | **Uniform / sparse (healthy)** |
| 512 absolute vs HF | **96.74%** (still ≥95%) |
| Delta vs 97.4% baseline | **−0.65 points** (−5/768) |

### **PASS — FP8-KV enters Phase 2 scope as architecture**

It changes:

1. KV **storage format** (real E4M3 + per-token K/V scales)
2. **Dequant-on-read** (or fused attention dequant)
3. **Slot accounting** (~2× concurrent slots at every context length)

These must be designed into Phase 2, not bolted on later.

### Quality price for 2× capacity (Niko’s call)

| | |
|--|--|
| Absolute (512 oracle) | 97.4% → **96.7%** |
| Isolated FP8-KV delta (4K) | **0.07%** disagreement vs BF16-KV |

The isolated KV effect is tiny; most of the gap to HF is already FP8 **weights**. Paying ~0.7 points absolute for **2× concurrent capacity** is the tradeoff — probe does not auto-reject it.

## Implementation (probe only)

| File | Change |
|------|--------|
| `src/hephaestus/kernels.mojo` | `kv_fp8_roundtrip_kernel` / `kv_fp8_roundtrip_inplace` |
| `src/hephaestus/forward.mojo` | `forward_fp8[..., fp8_kv_cache=False]`; post-RoPE K + V roundtrip when True |
| `src/qwen_fp8_kv_quality_probe.mojo` | Co-measure harness (512 oracle + 4K A/B) |

**Not built:** real FP8 storage layout, dequant-on-read bandwidth path, paged slots. Those are Phase 2 work **after** this pass.

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=$HOME/projects/hephaestus-wmma-nightly/.pixi/envs/default
export PATH=$CONDA_PREFIX/bin:$PATH
export MODULAR_HOME=$CONDA_PREFIX/share/max
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd ~/projects/hephaestus
mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_fp8_kv_quality_probe.mojo -o /tmp/fp8_kv_quality_probe

# prepare ids (once) — see handoff / this session’s /tmp/fp8_kv_probe/
/tmp/fp8_kv_quality_probe \
  p1_in p1_ora p2_in p2_ora p3_in p3_ora prompt_4k_ids out_prefix
```

Raw log: `/tmp/fp8_kv_probe/run.log`, report `/tmp/fp8_kv_probe/out_report.txt`.

## Phase 2 framing (from handoff, confirmed)

- Capacity thesis lives on **FP8-KV (~2× slots)**, not FP8 weights (~1.7 extra slots).
- llama baseline: KV already **2304 MiB @ npl=16** vs ~4.3 GB weights — KV is the binding constraint at scale.
- This probe clears the **quality** bar for putting FP8-KV in the architecture. Storage + read path is the next build.
