# Small-M decode-batch FP8 GEMM — build + measure

**Date:** 2026-07-20  
**Branch:** `probe/small-m-decode-gemm` (from `main` @ `76905c8`)  
**Spec:** `.agent/specs/2026-07-20_small-m-decode-batch-gemm.md`  
**Prior:** `bench/fused-mrow-decode-probe.md` (amortization PASS; v3a at M=8 ~28% of 569)  
**GPU:** Device 0 — AMD Radeon AI PRO R9700 (gfx1201). GPU 1 empty throughout.  
**ROCm:** 7.2.x · `rocm-smi --showpids` empty before/after  
**Env:** `hephaestus-wmma-nightly` Mojo `1.0.0b3.dev2026071206`  
**Weights:** one FP8 arena `staged/qwen3-4b-fp8`  
**Roofline denominator:** **569 GB/s only** (460 retired)  
**Rule:** temporary weight-byte path tags in `linear_fp8` / `linear_add_residual_fp8` **reverted** after the run. No scheduler / paged KV / FP8-KV / large-prefill v3a rewrite.

---

## What was built

| Symbol | Role |
|--------|------|
| `wmma_fp8_small_m_kernel` | No-LDS W8A8 GEMM; direct global A/B fragments; dual-scale + residual |
| `wmma_gemm_fp8_small_m` | Host launch |
| Dispatch in `linear_fp8` / `linear_add_residual_fp8` | See below |

**Tile:** `BM_SM=16`, `BN=64`, `BK=16`, **1 wave / block**, `SM_SC=4` N-subcols (A fragment hoisted across sc).  
**LDS on B:** **none** (primary attempt per spec §4.2).  
**Edge-mask M** (valid for 2 ≤ M ≤ 32).

### Dispatch contract (as landed)

```
m == 1                              → gemv_fp8 / wmma_gemm_fp8_decode
2 ≤ m ≤ M_SMALL_MAX(=32) && n%64 && k%16 → wmma_gemm_fp8_small_m  (NEW)
m > 32 && n%64 && k%16              → wmma_gemm_fp8_prefill (v3a LDS) UNCHANGED
else                                → row-loop gemv
```

---

## Correctness

| Check | Result |
|-------|--------|
| Isolated GEMM small-M vs v3a (M=2,8,16,32; N∈{1024,2560,4096}) | **bit-exact** (`max_abs_diff=0`) — `src/qwen_small_m_gemm_smoke.mojo` |
| Teacher-forced decode M=1 (oracle, 3×256) | **256 + 246 + 246 = 748/768 (97.4%)** ≥95% |
| Path NC M=512 prefill | **253/253 launches tagged `v3a`** (small-M not taken) |
| Path NC M∈{2..16} full forward | **253 launches**, weight bytes = **4 022 272 000** once (tag `smallm` during TEMP instrument) |

M=1 decode TF holds prior class; multi-row math matches existing v3a epilogue (same W8A8 dual-scale).

---

## Gate results (full `forward_fp8(seq=M)`, past=0)

**Protocol:** 3 warmup + 32 timed steps; distinct token ids `1000+i*97`; fused bytes/step model  
`4.02 + M × (576 × 147456 / 1024³)` GB; eff BW = `(agg_tok_s × bytes/step) / M`; % = eff/569×100.

| M | kernel | step ms | agg tok/s | per-row | % of 569 | vs old v3a % | vs M=1 % | weight once? | rows distinct? |
|--:|--------|--------:|----------:|--------:|---------:|-------------:|---------:|:------------:|:--------------:|
| 1 | `gemv` / decode | 13.44 | **74.40** | 74.40 | **53.6** | n/a | 100 | yes | n/a |
| 2 | `small-M` | 14.62 | **136.77** | 68.39 | **50.2** | 25.6→ | 93.7 | **yes** | **yes** |
| 4 | `small-M` | 14.86 | **269.26** | 67.32 | **51.3** | 26.6→ | 95.7 | **yes** | **yes** |
| 8 | `small-M` | 15.42 | **518.75** | 64.84 | **53.0** | 28.0→ | 98.9 | **yes** | **yes** |
| 16 | `small-M` | 15.68 | **1020.20** | 63.76 | **59.2** | 27.1→ | 110.5 | **yes** | **yes** |

**Recheck (end of sweep):** M=1 → 76.1 tok/s (54.9%); M=8 → **523.4 tok/s (53.5%)**.

### Pre-registered gate (M=8 full-forward % of 569)

| Band | Criterion | Measured | Verdict |
|------|-----------|---------:|---------|
| **PASS** | ≥ 53% | **53.0%** (recheck **53.5%**) | **PASS** |
| Soft | 40–53% | — | — |
| FAIL | ≤ ~30% | — | — |

Old v3a-at-M=8 was **~28%** / **~274 agg tok/s**. New small-M: **~53%** / **~519 agg tok/s** — **~1.9×** aggregate, kernel efficiency back to GEMV class.

---

## Prefill smoke (M=512 must stay v3a / ~1700 class)

| rep | prefill tok/s (fwd-only) |
|----:|-------------------------:|
| 1 | 1695.2 |
| 2 | 1692.3 |
| 3 | 1690.0 |
| **mean** | **1692.5** |

Prior canonical: **1702.6** (`bench/1b-fp8-prefill-gemm.md`). Same class; no large regression. Path tags: all **v3a**.

---

## Co-measure — llama ROCm npl=1,8 (same session, GPU 0)

| Item | Value |
|------|-------|
| Tool | `llama-batched-bench` (`build/bin`) |
| Model | Q8_0 `qwen3-4b-instruct-2507-q8_0.gguf` |
| Flags | `-c 8192 -npp 512 -ntg 128 -npl 1,8 -ngl 99 -fa on` |

| npl | S_TG (tok/s) |
|----:|-------------:|
| 1 | 105.50 |
| 8 | **548.27** |

**Headline (not gate):** Hephaestus fused M=8 aggregate **~519 tok/s** ≈ **0.95×** llama ROCm npl=8 (548) in this window. Prior fused-v3a session was **0.50×** (274 vs 550).

---

## Negative controls

| Control | Result |
|---------|--------|
| Weights once (not ×M) | **PASS** — exactly **4 022 272 000** bytes, **253** launches for every M∈{1,2,4,8,16} |
| Distinct rows | **PASS** — 0/149 identical samples row-to-row0 for all M≥2 |
| M=1 still gemv | **PASS** — TF 97.4%; M=1 BW ~54% class |
| M=512 still v3a | **PASS** — 253 `v3a` tags; ~1693 tok/s prefill |

---

## Curve shape (one sentence)

**Removing LDS weight-staging for 2≤M≤32 recovers GEMV-class memory efficiency (~50–59% of 569) while keeping fused weight traffic flat; M=8 full-forward jumps from ~28%/274 tok/s (v3a) to ~53%/519 tok/s, clearing the pre-registered ≥53% gate.**

---

## Decision

| Criterion | Result |
|-----------|--------|
| M=8 % of 569 ≥ 53% | **PASS** (53.0 / recheck 53.5) |
| Weights once | **PASS** |
| TF ≥95% | **PASS** (97.4%) |
| Prefill M=512 ~1700 class | **PASS** (~1693) |
| Path isolation M=1 / M=512 | **PASS** |

### **Small-M no-LDS kernel is the right lever. Gate PASS — free to land dispatch; scheduler work now has a competitive M-batch GEMM floor.**

Optional DECISIONS row (not written until land on main):  
*2026-07-20 — small-M FP8 GEMM (no LDS B, BM=16/BN=64/1-wave) for 2≤M≤32; v3a retained for M>32; M=8 full-forward 53% of 569 / ~519 agg tok/s.*

---

## Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=$HOME/projects/hephaestus-wmma-nightly/.pixi/envs/default
export PATH=$CONDA_PREFIX/bin:$PATH
export MODULAR_HOME=$CONDA_PREFIX/share/max
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
KERNELS=$HOME/projects/modular/max/kernels/src

cd ~/projects/hephaestus
mojo build -I $KERNELS -I src src/qwen_small_m_gemm_smoke.mojo -o /tmp/small_m_smoke
/tmp/small_m_smoke

mojo build -I $KERNELS -I src src/qwen_fused_mrow_probe.mojo -o /tmp/small_m_mrow_probe
# optional TEMP weight log hooks (reverted in tree) for NC re-check
/tmp/small_m_mrow_probe 32

mojo build -I $KERNELS -I src src/qwen_ab_bench_fp8.mojo -o /tmp/ab_fp8_sm
for r in 1 2 3; do /tmp/ab_fp8_sm bench/ab_prompt_long_ids.txt 4; done

HIP_VISIBLE_DEVICES=0 \
  ~/projects/llama.cpp/build/bin/llama-batched-bench \
  -m /mnt/models/models/qwen3-4b-instruct-2507-q8_0.gguf \
  -c 8192 -npp 512 -ntg 128 -npl 1,8 -ngl 99 -fa on
```

Raw: `/tmp/small_m_out/hephaestus.log`, `prefill.log`, `llama.log`, `tf.log`.

---

## Warrant

This measures whether a **decode-batch-sized FP8 GEMM** can recover GEMV-class memory efficiency while keeping fused weight loads.  
It does **not** prove continuous batching, ragged batching, paged KV, or that Phase 2 is done.  
**PASS:** Phase 2’s kernel foundation for M-batch GEMM is honest; the prior 0.5×-of-llama gap at M=8 was the v3a LDS path at thin reuse, not failed amortization.

---

## One-line verdict

**M=8 full-forward: 53.0% of 569 GB/s (PASS), ~519 agg tok/s ≈ 0.95× llama ROCm npl=8; no-LDS small-M kernel for 2≤M≤32 with v3a retained for prefill.**
