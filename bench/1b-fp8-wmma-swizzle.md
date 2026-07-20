# FP8 WMMA weight pre-swizzle (fragment-order layout)

**Date:** 2026-07-13  
**Branch:** `fp8-wmma-decode`  
**Hardware:** GPU 0 = R9700 gfx1201; both GPUs free before co-measure  
**Env:** `~/projects/hephaestus-wmma-nightly`

## Goal

Row-major B-fragment loads scatter across 16 weight rows (uncoalesced).  
Pre-swizzle each 16×16 tile into the exact 256-byte order lanes consume so  
`b[j] = W_swz[tile_base + l*8 + j]` is a single coalesced wave load.

**No LDS** (regressed previously: no reuse at M=1).

## Implementation

| Piece | Change |
|-------|--------|
| `loader.mojo` | After arena upload, `swizzle_fp8_projection_weights`: host swizzle in-place for every F8_E4M3 2D tensor **except** `embed_tokens` (252 tensors on 4B) |
| `wmma_gfx12.mojo` | `swizzled_b=True` B-load: `tile_base + l*8 + j` via `load[width=8]`; LDS path removed |
| `kernels` / `forward` | Default `swizzled_b=True`; lm_head uses `linear_fp8[F32, False]` (embed stays row-major for gather) |

Swizzle formula (matches G1b-0 B mapping):

```
swizzled[n_tile*(K/16)*256 + k_tile*256 + lane*8 + j]
  = original[(n_tile*16 + lane%16)*K + k_tile*16 + (lane//16)*8 + j]
```

## Correctness

Teacher-forced 4B, 256 steps, oracle feed (same protocol as pre-swizzle):

| Prompt | Matches | Rate |
|--------|--------:|-----:|
| 1 | 256/256 | 100% |
| 2 | 247/256 | 96.5% |
| 3 | 245/256 | 95.7% |
| **Total** | **748/768** | **97.4%** |

**Same rates as pre-swizzle** (identical mismatch steps on p2/p3). Swizzle preserves math.

## Speed (co-measured)

| Config | Decode tok/s (fwd, 10×256, mean of 3) | Eff. GB/s* |
|--------|--------------------------------------:|-----------:|
| **Swizzled B (this work)** | **39.78** | **~160** |
| Unswizzled row-major (control, same day) | **56.52** | **~227** |
| llama.cpp Q8_0 tg128 ROCm (same session) | **109.49 ± 0.59** | — |
| Target | — | **≥400** |
| Roofline | ~159 tok/s | **640** |

\* `tok/s × 4.02` (GB weights / token).

### Ratios (swizzled)

| vs | Ratio |
|----|------:|
| llama Q8_0 | **0.36×** |
| Unswizzled control | **0.70×** (regression) |

**Gate G1b-2: still NOT MET.**

## Grade

| Criterion | Result |
|-----------|--------|
| Correctness vs prior TF | **PASS (97.4%)** |
| Eff. BW ≥ 400 GB/s | **FAIL (~160)** |
| Eff. BW ≥ 300 GB/s | **FAIL** |
| Swizzle helped vs unswizzled | **No — 30% slower** |

### Investigation notes (why not faster)

1. **Control re-measured at 56.5** with swizzle off + `swizzled_b=False` — same machine/session family as earlier 56.6. So the regression is real, not thermal noise.
2. Layout math matches G1b-0 and TF rates are unchanged → **not a wrong-tile bug**.
3. Hypotheses for regression (open):
   - RDNA global unit may not reward 8-byte/lane stride the same as 4-byte; need ISA-level load packing.
   - Destroying row-major spatial locality may hurt L2 for multi-block scheduling even if single-wave coalescing is ideal on paper.
   - `lm_head` remains unswizzled (embed dual-use); may dominate less than expected, but cannot explain *slower* projections alone.
4. **Next experiments:** swizzle + persist on disk (kill host swizzle noise — already outside timed region); try 4-byte fragment packing; profile with rocprof (mem wavefront coalescing counters); consider multi-wave N-tiling for B reuse across output tiles.

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=~/projects/hephaestus-wmma-nightly/.pixi/envs/default
export MODULAR_HOME=$CONDA_PREFIX/share/max
export PATH=$CONDA_PREFIX/bin:$PATH
cd ~/projects/hephaestus

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_ab_bench_fp8.mojo -o /tmp/qwen_ab_fp8_swz
# load prints: "swizzled 252 FP8 projection tensors..."
for r in 1 2 3; do /tmp/qwen_ab_fp8_swz bench/ab_prompt_short_ids.txt 256; done

mojo build -I ~/projects/modular/max/kernels/src -I src \
  src/qwen_teacher_forced_decode_fp8.mojo -o /tmp/tf_fp8_swz
/tmp/tf_fp8_swz /tmp/prompt1_input_ids.txt /tmp/prompt1_oracle_out.txt /tmp/tf_p1
```

Toggle: `DO_FP8_WMMA_SWIZZLE` in `loader.mojo` + default `swizzled_b` in kernels for A/B.
