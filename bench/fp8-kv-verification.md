# FP8-KV verification — negative control + 512 self-A/B

**Date:** 2026-07-20  
**Branch:** `fp8-kv-probe`  
**Commit:** `85879df` (`85879df93b054236c5931b8b3c19d1a538a77cce`)  
**Message:** `feat(fp8-kv): quality probe via E4M3 quant→dequant at cache-write`  
**GPU:** Device 0 only — AMD Radeon AI PRO R9700 (gfx1201). GPU 1 (RX 9070 XT) empty throughout (`rocm-smi --showpids` before/after: no KFD PIDs after run).  
**ROCm:** 7.2.4  
**HIP:** 7.2.53211  
**Env:** `hephaestus-wmma-nightly` Mojo `1.0.0b3.dev2026071206`  
**Weights:** FP8 E4M3 `staged/qwen3-4b-fp8`  
**Prompt:** `/tmp/fp8_kv_probe/prompt_4k_ids.txt` (same 4K sequence as the quality probe)  
**Rule:** pure measurement — no quant-math / flag / default-path changes. Temporary token-counter + lattice-check instrumentation was used for this run and **reverted** afterward (tree back to `85879df` probe sources).

---

## Verdict (the only question)

### **The 99.93% is a real FP8-KV result — the flag did not no-op.**

Negative control (token counter + E4M3 grid assert, each with its own OFF control) passes. Proceeding to harden the Phase 2 handoff is allowed from this gate’s perspective.

---

## Flag, round-trip path, call granularity (verbatim from code)

| Item | Record |
|------|--------|
| **Flag** | Compile-time parameter on `forward_fp8`: `fp8_kv_cache: Bool = False` — **not** an env var. Default `False` leaves the production path unchanged. Call sites pass `fp8_kv_cache=True` only from the probe/verification harness. |
| **Round-trip functions** | `kv_fp8_roundtrip_kernel` + host launcher `kv_fp8_roundtrip_inplace` in `src/hephaestus/kernels.mojo` |
| **Call sites** | `src/hephaestus/forward.mojo` inside the layer loop, under `comptime if fp8_kv_cache:` — once on **K** (post-RoPE) and once on **V** (post-`v_proj`), each with `n_tokens=seq`, `row_width=kv_out` |
| **Quant math** | Per-token row: `scale = max(\|row\|) / FP8_E4M3_MAX` (448); in-place BF16 → E4M3 → BF16. Separate K vs V. Storage stays BF16. |
| **Call granularity** | **Per-layer × per-{K,V} × per-forward-chunk**, not one host call per token. Each host call covers the whole new `seq` chunk for that tensor (`grid_dim=(n_tokens,)` — one GPU block per token inside the kernel). With `CHUNK=64`, a 4K walk does `4096/64 = 64` forwards × 36 layers × 2 (K,V) = **4 608 host launches**, covering **294 912 token-rows**. |

Expected token-counter total (independent of chunking):

```
tokens × NUM_LAYERS × 2 = tokens × 36 × 2
```

---

## Results table

| check | flag | metric | result | verdict |
|-------|------|--------|-------:|---------|
| counter calibration | ON, 16 tok | tokens round-tripped | **1 152** (expect 1 152) | **PASS** |
| counter calibration | ON, 32 tok | tokens round-tripped | **2 304** (expect 2 304) | **PASS** |
| token counter | ON, 4K | tokens×layers×2 | **294 912** (expect 294 912) | **PASS** |
| token counter | OFF, 4K | tokens×layers×2 | **0** | **PASS** |
| E4M3 grid assert | ON, 4K cache | violations | **0** | **PASS** |
| E4M3 grid assert | OFF, 4K cache | violations | **282 047 598** (>0) | **PASS** |
| self-A/B | 512 | divergence / rate | **2 / 512 = 0.390625%** (match 510/512 = 99.609%) | report |
| self-A/B | 4K | divergence / rate | **3 / 4096 = 0.073242%** (match **4093/4096 = 99.927%**) | **re-confirmed** |

Calibration scales linearly with token count (2× tokens → 2× counter) → instrument measures **tokens covered**, not host invocations.

Counter and grid assert **agree** (both engaged on ON, both inert/violating on OFF). No scale-drift disagreement to flag.

---

## Counter calibration detail

| n_tokens | expected = n × 36 × 2 | measured | ratio measured/expected |
|---------:|----------------------:|---------:|------------------------:|
| 16 | 1 152 | 1 152 | 1.0 |
| 32 | 2 304 | 2 304 | 1.0 |
| 512 (ON) | 36 864 | 36 864 | 1.0 |
| 512 (OFF) | 0 | 0 | — |
| 4096 (ON) | 294 912 | 294 912 | 1.0 |
| 4096 (OFF) | 0 | 0 | — |

If the counter had been per-invocation instead of per-token, 4K ON would have read ~4 608 (launches) rather than 294 912 — three orders of magnitude low. Observed 294 912 rules that out.

---

## E4M3 grid assertion

**Method:** After a full 4K walk, for every layer’s K and V cache rows `[0, length)`, recompute `scale = absmax(row)/448` and count elements where quantize→dequantize changes the stored BF16 value.

| flag | violations | interpretation |
|------|------------|----------------|
| ON | 0 | Every stored value already sits on the E4M3 lattice for its per-token scale (second pass is a no-op). Absmax maps to 448 and is preserved. |
| OFF | 282 047 598 | Raw BF16-KV values are almost never E4M3-representable under the same scale rule. **Assertion’s own negative control holds** — the check is not a no-op. |

Total elements checked ≈ 36 layers × 4096 tokens × 1024 (`kv_out`) × 2 (K+V) = 301 989 888. OFF violation rate ≈ 93.4% of elements — consistent with dense BF16 not landing on the E4M3 lattice.

---

## 512 self-A/B — anomaly resolution on one axis

Prior numbers mixed axes: 512 was **net-of-oracle** (FP8-attributable divergence somewhere in ~5–25 of 768 steps), 4K was **self-A/B** (FP8-KV vs BF16-KV). This run measures self-A/B at both lengths on the **same prompt prefix**.

| length | match | divergent | divergence rate | agreement rate |
|-------:|------:|----------:|----------------:|---------------:|
| **512** | 510 | **2** | **0.3906%** | 99.609% |
| **4096** | 4093 | **3** | **0.0732%** | **99.927%** (re-confirm of probe) |

> **Cross-axis non-comparability (kills a future phantom):** the 768-token oracle run (748/768, 743/768) uses a *different token set* (3 prompts x 256 from /tmp/prompt{1,2,3}_oracle_out.txt) than the 512-token self-A/B window (a prefix of the 4K probe sequence). The 5-token oracle delta and the 2-token self-A/B divergence are measured on different sets AND different axes (net-of-oracle vs self-A/B), so they are not expected to reconcile and do not contradict each other.

**Ratio:** 4K divergence rate / 512 divergence rate ≈ **0.1875** → 4K still diverges **~5.3× less** than 512 on the same axis.

Mismatch positions (self-A/B):

| pos | BF16-KV argmax | FP8-KV argmax | in 512 window? |
|----:|---------------:|--------------:|:--------------:|
| 71 | 220 | 89673 | yes |
| 149 | 1096 | 576 | yes |
| 3071 | 785 | 279 | no |

The two 512 mismatches are exactly the first two 4K mismatches. Within `[0,512)` density is 2/512; the remaining 3584 positions add only one more mismatch.

### Resolution (plain)

On a **consistent self-A/B axis**, 4K still diverges less than 512. **Error accumulation with context length is not the dominant pattern under this per-token absmax/448 scheme.** The earlier “4K is 9× quieter than 512” anomaly was partly apples-to-oranges (oracle net vs self-A/B); after aligning axes it is still quieter (~5×), not louder.

**Hypothesis (labeled as hypothesis, not assertion):** longer-context text is lower-entropy / argmax margins are wider on average, so fewer near-ties flip under the same KV quantization noise. That is consistent with sparse early mismatches and a near-flat tail, but is not proven here.

**Not required before promotion stands:** redesign of the per-token scaling scheme. Scaling remains adequate for the quality bar already cleared by the probe; this verification only asked whether the 99.93% was real and whether the axis anomaly reverses under self-A/B.

---

## Method notes (reproducibility)

1. Free both GPUs; `HIP_VISIBLE_DEVICES=0`.
2. Temporary host-side counter in `kv_fp8_roundtrip_inplace`: append `n_tokens` per call to `/tmp/fp8_kv_rt_tokens.log` (sum = tokens covered). Reset file between runs.
3. Temporary GPU lattice-check kernel: per-token absmax/448, count non-bit-identical round-trips; atomic sum to a device `int64`.
4. Harness walked 16 → 32 (calibrate) → 512 self-A/B → 4K OFF (counter + grid) → 4K ON (counter + grid + self-A/B).
5. Instrumentation **reverted**; commit tree for engine sources matches `85879df` again.

Raw log: `/tmp/fp8_kv_verification_out/run.log`  
Report: `/tmp/fp8_kv_verification_out/out_report.txt`

---

## Hard-gate summary

| Gate | Result |
|------|--------|
| Counter calibrates linearly in tokens | **PASS** |
| Flag ON fires (4K count = tokens×36×2) | **PASS** |
| Flag OFF is silent (count = 0) | **PASS** |
| E4M3 lattice ON: 0 violations | **PASS** |
| E4M3 lattice OFF: >0 violations | **PASS** |
| 4K self-A/B re-confirm 4093/4096 | **PASS** |
| 512 self-A/B on same axis | **2/512 divergent** (reported) |

**Single verdict line:** **99.93% is a real FP8-KV result, not a flag no-op.**
