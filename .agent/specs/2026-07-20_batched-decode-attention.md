# Batched decode-attention kernel — design brief (for GLM → Grok prompt)

**Status:** RATIFIED 2026-07-20 — Opus (design) + GLM (sharpen). For GLM to expand into a Grok build prompt.
**Owner of design:** Opus. **Sharpened by:** GLM. **Implementer:** Grok. **Target:** `src/hephaestus/kernels.mojo` (new batched variant beside `attention_kernel_parallel`) + the batched-decode call site in `src/hephaestus/forward.mojo`.
**Depends on:** `main` @ `8916006` — `forward_fp8_batched_decode` + `BatchedKVCache` + the M-serial attention loop landed as the probe (`bench/batched-generation-m.md`). That serial path is the correctness oracle.

---

## 0. Intent

Fixed-M batched generation measured a pillar of **2.17×** (`bench/batched-generation-m.md`), gated entirely by decode **attention**: the M per-sequence attentions are **M serial launches** of a kernel that already under-fills the GPU (`grid=(n_heads, 1)` = 32 blocks for one decode query). The GEMMs fuse (NC3 flat at 4.02 GB); attention does the opposite. This brief replaces the serial loop with **one batched-attention launch** to recover the pillar into the weight-amortization class. Nothing else in Phase 2 proceeds until this clears.

---

## 1. The core change

Today (per layer, per decode step):
```
for i in 0..M-1: attention(seq=1, KV=cache[i])     # M serial launches, grid=(n_heads,1)
```

Target — one launch:
```
attention_batched(grid=(n_heads, M))               # block (h,s) attends seq s's query vs cache[s]
```

- The block's **second grid dimension changes meaning**: from *query-token position* (prefill) to *sequence index* (batched decode).
- Block `(h, s)` loads sequence *s*'s single decode query for head *h*, computes QK against **`cache[s]`** (its own KV), softmax, PV — writes seq *s*'s output row.
- **Block internals are unchanged** — reuse the `attention_kernel_parallel` (ATTN-STOPGAP) body verbatim: parallel softmax + PV, one query, n_keys keys. Only the grid's second dim and the KV base pointer change. This is what makes it low-risk.
- 32×M blocks resident (256 at M=8, 512 at M=16) instead of M serial 32-block waves.

**Per-sequence indexing (do this even though context is uniform now):** block `(h, s)` reads `n_keys` from `past_lens[s]`, **not** from a shared scalar. One line in the prologue (`var n_keys = past_lens[block_idx.y]`). Uniform at ctx=640 today, but hardcoding a shared `n_keys` silently breaks on ragged lengths later — a phantom we pay for now for free.

---

## 2. Why this recovers the pillar *regardless of attention efficiency* (the load-bearing argument)

The per-sequence attention is inefficient (~8 ms/seq at ctx=640 vs ~0.3 ms bandwidth bound). **That inefficiency does not matter for the pillar**, because it is paid **equally** at M=1 and M=8:

- M=1 pays ~8 ms of attention.
- Batched M=8, run concurrent (one wave), also pays ~8 ms — the *same* cost, not 8× it.

So the attention term **cancels in the M=1→M=8 ratio**, and the pillar collapses to the pure GEMM-amortization ratio. Both derivations agree:
- Byte-model (attention-free): M=1 4.105 GB → 74 tok/s; M=8 4.70 GB → 513 tok/s ⇒ **6.9×**
- Harness (attention-heavy but batched): M=1 48.3 tok/s (measured); M=8 ≈ 15 ms GEMM + ~8 ms concurrent attention = 23 ms → 348 tok/s ⇒ **7.2×**

They coincide because the ~8 ms attention divides out. **Consequence:** we do **not** need to fix attention efficiency to clear the pillar — only to make it concurrent. Per-sequence efficiency is a separate lever that moves only the absolute / competitive-vs-llama number; it is **out of scope here** (§6).

---

## 3. Gate

Re-run the batched-generation probe (**512 prefill + 128 gen**, same harness, same llama sandwich, GPU 0, ROCm) with the batched attention kernel swapped in.

- **Pillar `decode_agg(M=8)/decode_agg(M=1)`:**
  - **≥ 3× = PASS** (the "does the Multiplier multiply" bar)
  - **5–7× = expected target** (the physics ceiling; §2)
  - **< 3× = stop-and-diagnose** (batching didn't take — check the §5 launch-count NC first)
- Report the full **M ∈ {1,2,4,8,16}** table, pillar, and competitive ratio vs llama npl=8 (**report-only** — competitive is the per-seq-efficiency lever's metric, not this gate's).
- Correctness: **bit-exact vs the serial path, per row** (§4).

### Diagnostic ladder (self-localize the pillar number; harness M=1 = 48.3 tok/s fixed)

| pillar | M=8 agg | reading | action |
|---|---|---|---|
| **~7×** | ≥ ~348 | batched attention one-wave concurrent — ideal | ship |
| **~5×** | ~270 | ~1.5–2× grid overhead | acceptable pass, note it |
| **~3.5×** | ~180 | LDS-capped to 2 waves at M=8 | apply §4.1 scores-sizing fix, re-run |
| **< 3×** | < ~145 | batching didn't happen | check §5 launch-count NC |

---

## 4. Correctness oracle (load-bearing)

The batched kernel must be **bit-exact, per row, against the current M-serial path** — identical math, identical reduction order, only the launch reorganized. Same validation pattern that worked for small-M vs v3a: the serial path already passed Gate 1/2/3, so it is a trusted reference.

- Diff each row of a batched-M run against the same row from the serial path, same inputs: **`max_abs_diff = 0`**, gate `!= 0`.
- Any nonzero diff means the sequence-indexing (KV base or `past_lens`) is wrong. **No new numerical tolerance is introduced.**

### 4.1 Occupancy / LDS scores sizing — try first if the initial run lands under ~5×

Occupancy math: 64 KB LDS/CU ÷ scores buffer. If scores is `MAX_KEYS=4096` F32 = **16 KB/block** → only **4 blocks/CU** → 256-block capacity.
- **M=8 = grid(32,8) = 256 blocks = exactly one wave** at that cap — fits by a hair, so M=8 may hit ~7× even before any fix. If it lands *below* ~5×, register/LDS pressure tipped it past the one-wave cliff.
- **M=16 = 512 blocks = two waves** at 16 KB — this is where the fix is load-bearing.
- **Fix:** size scores to actual `n_keys` (~640 → ~2.5 KB) instead of `MAX_KEYS` → ~25 blocks/CU → 1600-block capacity → M=16 fits in one wave. Likely a comptime constant (`MAX_KEYS`) to template down for decode-batch builds, or a smaller static allocation. Cheap: one constant + rebuild.

---

## 5. Negative control that can fail

**One attention launch per layer per step, not M.** Count attention-kernel launches per decode step: expect **`n_layers` (36)**, not **`M × n_layers` (36M)**. If it scales with M, the batching didn't happen and any speedup is coming from elsewhere — same class as the NC3 weight-bytes control. This is the control that proves the fix is the fix. Revert the instrumentation before commit.

Correctness NC (via §4): a row bit-exact against its serial-path counterpart while M−1 other sequences share the batch also proves KV isolation held under the new grid.

---

## 6. Explicitly NOT in scope

- **Per-sequence attention efficiency** (the ~8 ms/seq vs ~0.3 ms bandwidth bound). Separate lever, targets the **absolute/competitive** number, **not** the pillar (§2). Do not chase it here — it balloons scope and the pillar doesn't need it.
- Ragged sequence lengths / continuous batching (index per-seq `n_keys` anyway — §1 — but don't build ragged scheduling).
- Paged KV, FP8-KV storage, GEMM/dispatch changes, head-fusion changes.

---

## 7. Gate definition

**PASS** = §4 bit-exact per row (all M) ∧ §5 launch-count NC (36/step, can-fail) ∧ pillar ≥ 3× reported (targeting 5–7×). On pass: the Multiplier multiplies under real generation; Phase 2 capacity work (paged → FP8-KV storage) unblocks. On < 3×: stop-and-diagnose via §3 ladder before proceeding.

**Branch:** `probe/batched-decode-attention` from `main`. GPU 0 only, both GPUs freed first, named tmux, deliverable `bench/batched-decode-attention.md` with the table, pillar, competitive ratio, NC evidence, and warrant. Do not commit to `main`.

---
*Design ratified 2026-07-20. The pillar-cancellation argument (§2) is the crux: batching recovers the Multiplier independent of attention efficiency; efficiency is a later, separate lever.*
