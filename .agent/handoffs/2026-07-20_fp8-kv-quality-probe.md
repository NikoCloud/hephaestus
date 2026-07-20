# HANDOFF — FP8 KV-cache quality probe (for GLM -> Grok prompt)

**Date:** 2026-07-20. **Decides:** whether FP8-KV enters Phase 2 scope as architecture.
**Rule:** validate FIRST, scope SECOND. Do not write FP8-KV into the Phase 2 spec as a
load-bearing capacity pillar and then discover it fails its own quality bar.

## 1. Why this probe, and why it is the pivot

Phase 2`s capacity thesis rests almost entirely on FP8-KV, not FP8 weights:

| lever | capacity gain |
|---|---|
| FP8 weights (4.02 vs Q8 4.27 GB) | ~250 MB = **~1.7 extra slots**. Marginal |
| **FP8 KV cache** (144 -> ~72 MiB per 1024-cell slot) | **2x concurrent slots at every context length** |

Measured from the llama baseline: at npl=16 llama already spends **2304 MiB on KV** vs
4.27 GB on weights. Push to longer contexts or more agents and **KV is the binding
constraint, not weights.** So the capacity half of Phase 2 lives or dies here.

## 2. KEY DESIGN INSIGHT — simulate, do not implement

**The quality question is separable from the storage implementation.** Do NOT build the
FP8 KV storage format, the dequant-on-read path, or new memory layout for this probe.

Instead: **quantize->dequantize K and V through E4M3 at cache-write time, keeping BF16
storage.** The values attention subsequently reads are *numerically identical* to what a
real FP8 KV cache would return. Quality answer, near-zero implementation cost.

Only if the probe passes do you build the real storage path (which is where the actual
memory and bandwidth savings come from).

## 3. The gate

**PASS = >= 95%% teacher-forced argmax parity at 4K context.**

Also report, because the number alone is not the decision:
- **Delta vs the current baseline** (FP8 weights + BF16 KV = **97.4%%**, 748/768).
  If FP8-KV costs 97.4 -> 95.1, that is a 2.3-point quality price for 2x capacity --
  a tradeoff for a human to rule on, not an automatic pass.
- **Divergence shape**: uniform quantization noise vs outlier-shaped (a few tokens
  wildly off). Same diagnostic used for the FP8 weight validation. Outlier-shaped
  implies the scaling scheme is wrong, not that FP8-KV is unviable.

**512 is not sufficient.** FP8-KV quantizes what attention *reads*, and error compounds
across keys as context grows. A probe that passes at 512 and is never run at 4K is
exactly the result that looks fine and fails in production. This codebase has hard-won
evidence (`.agent/notes/spike-investigation.md`) that attention is where 1-ULP
perturbations flip tokens.

## 4. The 4K oracle problem -- read before designing the test

The existing teacher-forced oracles are **short** (prompt files are ~10 tokens). There is
**no 4K HF oracle**, and generating one is a separate, slower task. Handle it thus:

- **At 512:** FP8-KV vs the existing HF oracle -> absolute parity, directly comparable
  to the 97.4%% baseline.
- **At 4K:** FP8-KV vs **BF16-KV on our own engine** (self-A/B, same prompt, same seed).
  This isolates the FP8-KV delta, which is the actual question, without needing a 4K
  oracle. Report divergence rate and shape.

If an absolute 4K number is wanted later, generating a 4K HF oracle is its own task.

## 5. Technical specifics

- **What to quantize:** K after RoPE, V directly -- i.e. the values as written to cache.
- **Scaling: per-token, per-layer, separate for K and V** (mirrors the existing per-token
  activation absmax). Per-tensor static scaling will likely fail -- post-RoPE K has
  outliers. Scale storage is negligible: 2 x 4 bytes x 36 layers = 288 B/token against
  ~73.7 KB/token saved (<0.4%%).
- **E4M3 range:** finite max 448. Use the same absmax/448 scheme as the activation quant.
- **Capacity check (cheap, do it):** confirm bytes/token BF16 vs FP8 including scale
  overhead, verify it is ~2x. Do not claim 2x if scales eat into it.
- **Behind a flag.** Do not change the default path.

## 6. Traps

1. **Do not build the storage path.** Simulate via quantize->dequantize (section 2).
2. **Do not run only at 512.** The 4K test is the one that decides.
3. **Outlier-shaped divergence means fix the scaling, not abandon FP8-KV.** Check the
   shape before concluding.
4. Machine drift: co-measure BF16-KV and FP8-KV in the same session.
5. `rocprof` is not installed. `main` is 94db405+, compiles, has all four kernel lines.

## 7. Decision rule

- **PASS (>=95%% at 4K, uniform noise):** FP8-KV enters Phase 2 scope as architecture --
  it changes KV storage format, the dequant-on-read path, and slot accounting, so it must
  be designed in, not bolted on later.
- **FAIL:** Phase 2`s capacity story needs a different lever. Be honest that the
  remaining capacity argument is then **paged attention for variable-length workloads
  only** -- and note that the llama baseline`s uniform 512+128 workload is the *best
  case* for static slot allocation, so it structurally understates paging`s value.

## 8. Phase 2 sizing context (renormalized -- carry into the scope)

The 569 GB/s figure is a **synthetic pure-stream** benchmark. The honest practical
ceiling is llama`s own npl=1 result (**460 GB/s**, where KV is only 1.9%% of bytes -- an
essentially pure weight sweep).

| npl | llama eff BW | %% of 569 (synthetic) | **%% of 460 (practical)** |
|---|---|---|---|
| 1 | 460 | 81%% | **100%%** |
| 8 | 338 | 59%% | **73%%** |
| 16 | 337 | 59%% | **73%%** |

llama`s droop is **100 -> 73%%**, not 81 -> 59%%, and it **flattens** at npl 8->16
(338 -> 337) -- a bounded floor, not a collapse. The opening is real but bounded.

Phase 2 prize: matching llama`s efficiency buys **~5%%** (the byte edge). Beating their
curve buys **~40%%**. **The prize is implementation quality, not the format.**

Sobering anchor: llama serves 16 streams at 969 aggregate tok/s using ~6.5 GB total. Our
8-process probe managed 453 using 32 GB. That is the distance fused batching must cover.
