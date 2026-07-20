# ATTN-STOPGAP — Parallelized Attention (measurement baseline) Implementation Spec

**Status:** RATIFIED 2026-07-13 — parallelize softmax + PV in the current kernel; keep QK. Measurement-first: the deliverable is a *number + a sub-split*, and a validation baseline for the vendored port.
**Owner of spec:** Opus. **Implementer:** Grok. **Target:** `src/hephaestus/kernels.mojo` (`attention_kernel`, new parallel variant alongside the current one).
**Est. effort:** ~2 hours. This is a stopgap, not the Phase-2 kernel.
**Depends on:** current `attention_kernel` (correct, single-warp) + layer-diff harness.

---

## 0. Intent (and what this is NOT)

Profiling put attention at **65% of prefill (412 ms)** despite being ~2% of prefill FLOPs. Cause: the current kernel serializes **softmax onto one lane** (`if warp_id==0 and lane==0: for jj in range(n_keys)`) and **PV onto one warp**. This spec parallelizes both across the block. Three goals, in priority order:

1. **Measure.** Produce the prefill tok/s and a **QK / softmax / PV sub-split**, to answer: does attention parallelism alone reach the G1b-3 gate (~2100), and if not, *which part* needs WMMA?
2. **Baseline.** A trusted, reference-checkable parallel kernel to diff the vendored WMMA port against.
3. **Speed.** Remove the serial bottleneck; capture whatever end-to-end lift falls out.

**NOT in scope:** streaming/online softmax (the current kernel already materializes the full `scores[MAX_KEYS]` in LDS — we parallelize reductions *over* it, no streaming needed; true online softmax belongs to the vendored flash kernel), WMMA, paging/batching, QK changes, decode-path changes. **QK is held constant deliberately** to isolate the softmax+PV variable and let the sub-split attribute the remaining time.

---

## 1. Block structure (unchanged from current)

```
grid  = (n_heads, seq)              # block_idx.x = head h, block_idx.y = query token t
block  = NT threads = WARP * NUM_WARPS   # set NUM_WARPS = 4  -> NT = 128
```
One block per (head, query token). `tid = thread_idx.x` (0..NT-1), `warp_id = tid/WARP`, `lane = tid%WARP`.

Derived per block:
```
kv_head = h // GROUP                # GQA, GROUP = n_heads / n_kv_heads = 32/8 = 4
past    = cache.length
n_keys  = past + t + 1              # causal
scale   = 1 / sqrt(head_dim)        # head_dim = 128
```

LDS:
```
q_sh   : F32[head_dim]              # scaled query for (h,t)
scores : F32[MAX_KEYS]              # QK scores -> exp -> normalized probs (reused in place)
red    : F32[NUM_WARPS]             # block-reduction scratch (max/sum)
```

---

## 2. Phase 0 — load query (fold in scale)

```
for d = tid; d < head_dim; d += NT:
    q_sh[d] = scale * Float32(q_global[t, h, d])
barrier()
```

## 3. Phase 1 — QK  (UNCHANGED from current kernel)

Keep the existing multi-warp warp-reduce dot that fills `scores[0..n_keys)`. Do not touch it. `barrier()` after (as today). *Rationale: isolates the change; §7 sub-timing tells us if QK is worth attacking next.*

## 4. Phase 2 — parallel softmax  (REPLACES the single-lane block)

All NT threads cooperate over the materialized `scores`.

```
# 2a. block max
var lmax = -inf
for j = tid; j < n_keys; j += NT:
    lmax = max(lmax, scores[j])
var m = block_reduce_max(lmax)          # see §6; includes barriers, broadcasts to all threads

# 2b. exp + block sum  (overwrite scores with unnormalized exp)
var lsum = Float32(0)
for j = tid; j < n_keys; j += NT:
    var e = exp(scores[j] - m)
    scores[j] = e
    lsum += e
var s = block_reduce_sum(lsum)          # §6; broadcasts
var inv_s = Float32(1) / s

# 2c. normalize in place  (keeps PV math bit-identical to current)
for j = tid; j < n_keys; j += NT:
    scores[j] = scores[j] * inv_s
barrier()
```

## 5. Phase 3 — parallel PV  (REPLACES the single-warp serial loop)

Each thread owns head-dim column(s) `d`; loops keys in order → **same per-column accumulation order as the current warp-0 loop**, so bit-identical PV.

```
for d = tid; d < head_dim; d += NT:
    var acc = Float32(0)
    for j in range(n_keys):             # serial over keys, SAME order as current
        acc += scores[j] * Float32(v_cache[j, kv_head, d])
    out_global[t, h, d] = acc.cast[BF16]()
# no trailing barrier: each thread writes its own output columns
```

With NT=128 and head_dim=128, PV is one column per thread — 128-way parallel vs the current 32 (warp-0 only, 4 cols/lane).

---

## 6. Block reductions (concrete)

`block_reduce_max` / `block_reduce_sum` over NT=128 (4 warps):

```
# 1. intra-warp reduce via shuffle_down (log2(32)=5 steps)
var v = local                                   # lmax or lsum
for offset in [16, 8, 4, 2, 1]:
    v = op(v, shuffle_down(v, offset))          # op = max or +
# 2. warp leaders publish
if lane == 0: red[warp_id] = v
barrier()
# 3. warp 0 reduces the NUM_WARPS partials, writes final to red[0]
if warp_id == 0:
    var w = red[lane] if lane < NUM_WARPS else identity   # identity = -inf (max) / 0 (sum)
    for offset in [2, 1]:                                  # NUM_WARPS=4 -> 2 steps
        w = op(w, shuffle_down(w, offset))
    if lane == 0: red[0] = w
barrier()
return red[0]                                    # all threads read the broadcast result
```
(`shuffle_down` is already imported in kernels.mojo.)

---

## 7. Correctness gate

Near-bit-exact by construction, so use tight validation:
- **PV: bit-identical** to the current kernel (same normalized `scores`, same per-column serial key order). Any PV difference is a bug.
- **Softmax: differs only in reduction order** — `max` is order-invariant (exact); `sum` differs by F32 associativity (parallel tree vs serial). Expect ~1e-6 relative.
- **Gate:** `1e-5 + 1.6e-2·|ref|` vs the current kernel is generous; realistically expect near-bit-exact. Run: (1) **tiny layer-diff** vs the current attention — expect the same 31/32 class, attention layers within F32 ULP; (2) **teacher-forced decode (4B)** — 255–256/256, no regression.

---

## 8. Measurement gate — THE DELIVERABLE (do not skip; this is the point)

1. **Prefill 512 tok/s, 3 reps**, measured at M=512 on the real forward pass. Report actual. Decision thresholds:
   - **≥ ~1700** → gate reachable via attention parallelism alone; vendored WMMA port becomes a **Phase-2 investment**, not a 1b blocker.
   - **~1000–1300** → WMMA attention is needed for G1b-3; vendored port is the **1b path**.
2. **QK / softmax / PV sub-split** (ms each) — separate timed launches or profiler markers around the three phases. This attributes the *remaining* attention time so the vendored port (or a QK follow-up) targets the right part. Document it — do not assume where the time went.
3. Record VGPR/lane for the new kernel (mojo asm metadata, as in v3a).

Deliverable is `bench/attn-stopgap.md`: the three reps, the sub-split, VGPR, and the reachable/short verdict.

---

## 9. Out of scope / do NOT

Streaming online softmax, WMMA, paging/batching, QK modification, decode-path parallelization, wider blocks. One variable: softmax + PV parallelization, measured.

---

## 10. Gate definition (ATTN-STOPGAP)

**PASS** = §7 correctness (layer-diff + teacher-forced ok) ∧ §8 number + sub-split captured and written to `bench/attn-stopgap.md`.
On pass: the prefill number decides whether the vendored WMMA attention port (paged, batch-aware — the Phase-2 kernel) is 1b-urgent or Phase-2-scheduled. Either way the vendored port spec follows; this stopgap is its validation reference.

**Branch:** `attn-stopgap` from `main`. Isolated env `~/projects/hephaestus-wmma-nightly`. GPU 0, check rocm-smi first.

---
*Spec ends. Ratified 2026-07-13 — measurement-first; vendored-port spec follows the number.*
