# Fixed-M batched greedy generation — first end-to-end Multiplier numbers

**Date:** 2026-07-20  
**Branch:** `probe/batched-generation-m` (from `main` @ `7fd39ae`)  
**Harness:** `src/qwen_batched_gen_probe.mojo`  
**GPU:** Device 0 — AMD Radeon AI PRO R9700 (gfx1201). GPU 1 empty throughout (`rocm-smi --showpids` empty before/after).  
**ROCm:** 7.2.x · backend for llama co-measure = **ROCm**  
**Env:** `hephaestus-wmma-nightly` Mojo `1.0.0b3.dev2026071206`  
**Weights:** one FP8 arena `staged/qwen3-4b-fp8` (~4.02 GB) shared by all M  
**Prompt:** `bench/ab_prompt_long_ids.txt` (512 tokens) + 128 greedy gen  
**Roofline denominator:** **569 GB/s only** (460 retired)  
**Rule:** temporary weight-byte path tags in `linear_fp8` / `linear_add_residual_fp8` applied for NC3, **reverted before commit**. No scheduler / continuous batching / paged KV / FP8-KV store / HTTP.

---

## Anchors (before the table)

| Anchor | Value |
|--------|------:|
| Product single-stream FP8 decode (prior co-measure, full pipeline) | ~**66.1** tok/s |
| Forward-only past=0 M=8 small-M (`bench/small-m-decode-gemm.md`) | ~**519** tok/s @ ~**53%** of 569 — **kernel floor, not generation** |
| Perfect weight-amortization ceiling | 8×66.1 ≈ **529** tok/s — generation is harder (KV grows, sample) |
| llama ROCm this sandwich npl=8 S_TG | ~**549** tok/s |
| llama ~16 streams on ~6.5 GB vs 8-process probe ~32 GB | fused one-weight generation is the gap-closer under test |

---

## What was built

| Piece | Role |
|-------|------|
| `BatchedKVCache` | Per-sequence K/V slabs in one allocation: `[max_batch][n_layers][MAX_KEYS][kv_out]` — **not aliased** |
| `apply_rope_qk_fixed_pos` | RoPE for M independent rows at the **same** absolute position (not prefill `tok+offset`) |
| `forward_fp8_batched_decode` | **One** fused M-row GEMM call per projection (`linear_fp8` → gemv / small-M); then **M separate per-seq attentions** |
| Prefill | Token-at-a-time fused M-row steps (same decode path); writes each row’s K/V into its own cache |
| Sample | Existing GPU argmax (`argmax_logits`), greedy |
| Harness | Correctness gates + NC1–3 + M-sweep + timing |

### Fused / serial boundary (mandatory)

```
per decode step:
  embed M tokens
  for layer:
    fused M-row q/k/v  →  linear_fp8(m=M)     # weights once
    q_norm, k_norm on all rows
    rope fixed-pos (all rows at past)
    scatter K/V row i → cache[i][past]
    for i in 0..M-1: attention(seq=1, KV=cache[i])   # NOT fused across seq
    fused o_proj + residual
    fused gate/up/down
  fused lm_head
```

**Not** `M × forward_fp8()` — that was the superseded null probe (step times 20→40→81→161 ms).

Existing `forward_fp8(seq=M)` is **prefill causal** (M tokens of one sequence, one shared KV). It is **not** used for batched decode.

---

## Correctness

### Gate 1 — Isolation (bit-exact, same small-M kernel)

Row 0 in **M=8** vs row 0 in **M=2**, identical prompt for row 0, distinct companions, after short prefill + one decode step.

| Check | Result |
|-------|--------|
| `max_abs_diff` row0 logits | **0.0** |
| argmax row0 | both **525** |
| Verdict | **PASS** |

If prefill-style causal attention had been reused, row 0 would see other sequences’ KV → FAIL. Bit-exact match also satisfies **NC2** (KV not aliased).

### Gate 2 — Kernel-switch teacher-forced (gemv vs small-M)

**Teacher-forced** (not free-running). Reference trajectory from M=1 greedy; both M=1 gemv path and batched M=2 small-M row0 re-anchored on the same tokens each step. 512 prefill + 128 steps.

| Check | Result |
|-------|--------|
| Matches | **128 / 128** |
| Match rate | **100%** (≥95% gate; 97.4% class) |
| Divergences | **0** |
| Verdict | **PASS** |

### Gate 3 — TF smoke (forward/KV write path)

Free-run M=2 for 32 steps, then teacher-force the recorded tokens — must be bit-exact (same kernel).

| Check | Result |
|-------|--------|
| Matches | **32 / 32** |
| Verdict | **PASS** |

---

## Negative controls

### NC1 — Not M copies of one decode

M=4 distinct prompts, 16 gen tokens. Streams 1/2/3 all differ from stream 0.

**PASS.**

### NC2 — KV not aliased

Gate 1 under distinct companions is the strongest disproof: row 0 bit-exact while M−1 different sequences share the batch.

**PASS** (via Gate 1).

### NC3 — Weights read once per step (fused/serial control)

Temporary host-side attribution at each FP8 linear launch: **`n × k` FP8 body bytes once per launch**.  
Expected fused = **4 022 272 000** (flat vs M). Serial-M-forwards = **M × 4 022 272 000**.

| M | weight_bytes | launches | ratio to fused | verdict |
|--:|-------------:|---------:|---------------:|---------|
| 1 | **4 022 272 000** | 253 | **1.000** | baseline |
| 2 | **4 022 272 000** | 253 | **1.000** | **PASS** |
| 4 | **4 022 272 000** | 253 | **1.000** | **PASS** |
| 8 | **4 022 272 000** | 253 | **1.000** | **PASS** |

253 = 36×7 projs + lm_head. **PASS:** fusion is real; throughput is not from re-reading weights ×M.

Instrument **reverted** after measurement.

---

## Measurement table (512 prefill + 128 gen, decode-only aggregate)

**Protocol:** one weight load; exclude load and prefill from decode aggregate. Decode timer = `forward_fp8_batched_decode` + sync only (matches product `qwen_generate` convention; GPU argmax runs for real next tokens but is outside the timed window). Final context = 640 for all M.

| M | prefill_s | decode_agg_tok/s | per_stream | total_tokens | final_ctx | step_ms_late | notes |
|--:|----------:|-----------------:|-----------:|-------------:|----------:|-------------:|-------|
| 1 | 8.628 | **48.27** | 48.27 | 128 | 640 | 21.47 | gemv path |
| 2 | 11.055 | **67.38** | 33.69 | 256 | 640 | 31.09 | small-M |
| 4 | 14.930 | **88.46** | 22.11 | 512 | 640 | 47.94 | small-M |
| 8 | 22.662 | **104.67** | 13.08 | 1024 | 640 | 81.90 | small-M |
| 16 | 37.689 | **116.03** | 7.25 | 2048 | 640 | 149.32 | small-M; VRAM OK |

**Prefill note:** token-at-a-time fused steps (512× decode-shaped forwards). Correct and fused, but not a v3a long-seq prefill GEMM — prefill_s is high and **not** the Multiplier metric.

### Ratios (report-only)

| Ratio | Value | Notes |
|-------|------:|-------|
| **Pillar** `decode_agg(M=8) / decode_agg(M=1)` | **2.17×** | **&lt;3× → stop-and-diagnose** (below) |
| **Competitive** `Heph M=8 / llama npl=8 S_TG` | **0.19×** | 104.67 / 549.05 |
| Expected pillar if weights amortized & attention cheap | ~5–7× | past=0 floor was in that class |

### % of 569

**Omitted.** Honest bytes/step needs weights once **+ M × KV reads** at growing depth; KV traffic was not instrumented cleanly this run. Prefer wrong-absent over fake-present.

---

## llama sandwich (same thermal window, GPU 0, ROCm)

```text
llama-batched-bench -m qwen3-4b-instruct-2507-q8_0.gguf \
  -c 8192 -npp 512 -ntg 128 -npl 1,8 -ngl 99 -fa on
```

| side | npl | S_TG tok/s |
|------|----:|-----------:|
| pre  | 1 | 105.87 |
| pre  | 8 | 549.46 |
| post | 1 | 106.54 |
| post | 8 | 548.64 |
| **mean** | **1** | **106.2** |
| **mean** | **8** | **549.05** |

Thermal drift negligible (~0.15% on npl=8).

---

## Curve shape (one-liner)

**Aggregate climbs slowly with M (48→67→88→105→116) while step_ms_late scales nearly linearly (21→31→48→82→149) — weight fusion is real (NC3 flat 4.02 GB) but serial per-sequence attention at ctx=640 dominates step time, so the Multiplier does not get a free 5–7×.**

---

## Stop-and-diagnose (pillar &lt; 3×)

**Finding:** At 512+128 generation, pillar = **2.17×**, not the ~5–7× weight-amortization class from past=0 small-M (~519 agg @ M=8).

**Not a fusion bug.** NC3 proves weight bytes = 4.02 GB once for every M. Gate 1/2/3 and NC1 pass.

**Cause:** Attention is **per-sequence by necessity** (different KV). This harness launches **M independent `attention(seq=1)` calls** per layer per step on one in-order HIP stream. Work ≈ `M × O(ctx)` serializes. At past=0 attention is tiny → fused GEMM shows ~5–7× class aggregate; at ctx=640 attention is the bulk of the extra cost.

**Supporting diagnostic (same binary path, short ctx ~96, decode-only):**

| M | step ms | agg tok/s | pillar vs M=1 |
|--:|--------:|----------:|--------------:|
| 1 | 14.6 | 68.3 | 1.0× |
| 8 | 26.3 | **305** | **~4.5×** |

Short-context pillar recovers into the expected class. Long-context generation is the honest product number — and it is attention-bound across streams, not weight-bound.

**M=1 ~48 tok/s vs product ~66.1:** same order; late-step at ctx=640 is ~21 ms (attention-grown vs short-ctx ~15 ms / ~66). Not a separate correctness issue.

**What this does *not* license:** continuous batching, paged KV, or FP8-KV — those are later. A next leverage point for fixed-M itself is reducing serial attention cost (occupancy / batch-dimension launch / less host bounce), not re-proving GEMM fusion.

---

## Warrant (mandatory)

This measures **fixed-M greedy generation** with:

- shared FP8 weights  
- per-sequence growing KV  
- fused M-row GEMM + per-seq attention  
- real token counts over 128 decode steps after 512 prefill  

It is **not** continuous batching, **not** paged KV, **not** FP8-KV storage, **not** HTTP serving.  
A good (or sobering) curve here does **not** mean Phase 2 is de-risked.

---

## Repro

```bash
export CONDA_PREFIX=$HOME/projects/hephaestus-wmma-nightly/.pixi/envs/default
export PATH=$CONDA_PREFIX/bin:$PATH
export MODULAR_HOME=$CONDA_PREFIX/share/max
KERNELS=$HOME/projects/modular/max/kernels/src

mojo build -I $KERNELS -I src src/qwen_batched_gen_probe.mojo -o /tmp/batched_gen
HIP_VISIBLE_DEVICES=0 /tmp/batched_gen gates   # correctness + NC
HIP_VISIBLE_DEVICES=0 /tmp/batched_gen bench   # M sweep

HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build/bin/llama-batched-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf \
  -c 8192 -npp 512 -ntg 128 -npl 1,8 -ngl 99 -fa on
```

Raw logs: `/tmp/batched_gen_m/{gates,heph_bench2,llama_pre,llama_post}.log`.

---

## Done block

```text
EXIT:0
BRANCH: probe/batched-generation-m
SHA: 718f0c3
DELIVERABLE: bench/batched-generation-m.md
PILLAR_M8: 2.17x  (report-only; <3x = stop-and-diagnose — serial per-seq attn at ctx=640)
HEPH_M8_VS_LLAMA: 0.19x  (report-only)
CORRECTNESS: PASS
NC: PASS
ONE_LINE: Fixed-M greedy generation works (fused GEMM + per-seq KV, all gates/NC pass) but at 512+128 pillar is only 2.17× — weight fusion is real (4.02GB once), serial M×attention at growing ctx is the bottleneck; ~0.19× llama npl=8.
```
