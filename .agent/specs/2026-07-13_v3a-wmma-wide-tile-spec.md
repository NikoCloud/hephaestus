# G1b-3a — 64×64 Multi-Wave WMMA GEMM with Reuse + Fused Residual (Implementation Spec)

**Status:** RATIFIED 2026-07-13 — 64×64 / 4-wave / BK=16, N-heavy v3b to follow, o_proj+down_proj folded in. Cleared for Grok after the linear() routing prep lands.
**Owner of spec:** Opus. **Implementer:** Grok. **Target:** `src/hephaestus/wmma_gfx12.mojo` (v3 path, alongside v1/v2).
**Depends on:** v2 (LDS staging proven), G1b-0 mappings (`.agent/specs/2026-07-13_g1b0-wmma-tile-spec.md`).
**Gate:** G1b-3a — bitwise-correct 64×64 GEMM with 4×/4× reuse (intensity 32 vs v2's 8), o_proj/down_proj on the WMMA path via fused residual epilogue.

---

## 0. Intent

v2 was memory-bound at arithmetic intensity 8 (one 16×16 tile/wave, zero LDS reuse). v3a raises intensity to **32** by having a **4-wave workgroup compute one 64×64 output tile**: A and B tiles are staged into LDS once per K-strip and reused — each A row read 4× (across N sub-cols), each B col read 4× (across the 4 waves). Simultaneously it closes the Amdahl hole by routing **o_proj and down_proj** (35% of linear FLOPs, incl. the largest matmul `down_proj` K=9728) through this kernel with a **fused residual-add epilogue**, so the prefill benchmark reflects the real number.

Scope: one 64×64 tile shape, single-buffered LDS, BK=16. No double-buffering (v4), no FP8 (step 8), no 128×256 (v3b), no remainder/edge masking (dims assumed ÷64/÷16).

---

## 1. Configuration

| Param | Value | Note |
|---|---|---|
| BM × BN (output tile) | 64 × 64 | one workgroup |
| Sub-tile | 16 × 16 | one WMMA |
| Sub-tile grid | 4 (sr) × 4 (sc) | 16 sub-tiles |
| Waves / workgroup | 4 | 128 threads |
| Wave → ownership | wave `w` owns **sub-row sr = w** (output rows `w*16..w*16+15`, all 64 cols) | 4 accumulators/wave |
| BK (K-strip) | 16 | K-loop = K/16 iters |
| Accumulators | `acc[sc]`, sc∈0..3, `SIMD[f32,8]` each | persist across K-loop |
| LDS_STRIDE | 16 | named const; pad→24 later if banks hurt (changes strides + sizes) |

Reuse: A row read 4× (within a wave, across its 4 `sc`); B col read 4× (across the 4 waves). Intensity = 64·64/(64+64) = **32** = 4× v2.

---

## 2. Grid / block / tile coordinates

```
Grid  = ( ceildiv(N, 64), ceildiv(M, 64) )     # x = N-tiles, y = M-tiles
Block = 128 threads (4 waves × 32)
NB = blockIdx.x * 64        # tile's global col base
MB = blockIdx.y * 64        # tile's global row base
```

## 3. Thread indexing (fix these names)

```
tid = threadIdx.x           # 0..127  (workgroup thread; used for cooperative load)
w   = tid / 32              # 0..3    wave = owned sub-row sr
l   = tid % 32              # 0..31   lane-in-wave (used for G1b-0 map, WMMA, store)
```

---

## 4. LDS layout + allocation

Both tiles are `[64, 16]` row-major with row stride `LDS_STRIDE`.
- `A_lds[row, col]` = A[MB+row, ks+col]   (row = M-index 0..63, col = K-index 0..15)
- `B_lds[row, col]` = W[NB+row, ks+col]   (row = **N**-index 0..63, col = K-index 0..15) — **n-major**, i.e. the transpose is baked into the layout exactly as v2.

```mojo
comptime LDS_STRIDE = 16
var A_lds = stack_allocation[64 * LDS_STRIDE, BF16, address_space = AddressSpace.SHARED]()
var B_lds = stack_allocation[64 * LDS_STRIDE, BF16, address_space = AddressSpace.SHARED]()
# 4 persistent accumulators, zeroed before the K-loop:
var acc = InlineArray[SIMD[DType.float32, 8], 4](SIMD[DType.float32, 8](0))
```
LDS used: 2×(64×16) BF16 = 4096 B / workgroup (of 64 KB). Not the occupancy limiter.

---

## 5. The K-loop (per workgroup)

```
for ks in range(0, K, 16):        # BK = 16
    # ---- 5a. COOPERATIVE LOAD (all 128 threads) ----
    # A tile [64,16]=1024 elems, 128 threads, 8 passes. Coalesced: 8 contiguous
    # 32-byte runs per pass (16 threads per row).
    for i in range(8):
        row = i*8 + tid//16                 # 0..63
        col = tid % 16                      # 0..15
        A_lds[row*LDS_STRIDE + col] = A_global[(MB + row)*K + ks + col]

    # B tile (from W[N,K], n-major) [64,16], identical structure:
    for i in range(8):
        row = i*8 + tid//16                 # N-index 0..63
        col = tid % 16                      # K-index 0..15
        B_lds[row*LDS_STRIDE + col] = W_global[(NB + row)*K + ks + col]

    barrier()                               # workgroup barrier: B is shared across waves

    # ---- 5b. FRAGMENT LOAD + WMMA (reuse loop, per wave) ----
    # A fragment for THIS wave's sub-row (sr = w). Loaded ONCE, reused across 4 sc.
    # (G1b-0 A-map, row base = w*16)
    var a = SIMD[DType.bfloat16, 8]()
    for j in range(8):
        a[j] = A_lds[(w*16 + l%16)*LDS_STRIDE + (l/16)*8 + j]

    for sc in range(4):                     # 4 N sub-cols
        var b = SIMD[DType.bfloat16, 8]()
        for j in range(8):
            b[j] = B_lds[(sc*16 + l%16)*LDS_STRIDE + (l/16)*8 + j]   # G1b-0 B-map, col base = sc*16
        acc[sc] = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16", SIMD[DType.float32, 8]](
            bitcast[DType.int16, 8](a),
            bitcast[DType.int16, 8](b),
            acc[sc],
        )

    barrier()                               # all reads done before next strip overwrites LDS
```

**Critical:** hoist the `a` fragment load OUT of the `sc` loop — same A for all 4 sub-cols is the register-level reuse. Do not reload it per `sc`.

---

## 6. Store + fused residual epilogue (after the K-loop)

Compile-time `FUSE_RESIDUAL` selects the path. Residual is added in **F32 before the BF16 cast**.

```mojo
for sc in range(4):
    for j in range(8):
        m = MB + w*16 + (l/16)*8 + j        # G1b-0 store row: sub-row base w*16
        n = NB + sc*16 + l%16               # G1b-0 store col: sub-col base sc*16
        var val = acc[sc][j]                # F32
        @parameter
        if FUSE_RESIDUAL:
            val += Float32(residual[m*N + n])   # residual stream x[m,n], BF16 -> F32
        C[m*N + n] = val.cast[out_dtype]()
```

Kernel signature gains `residual: UnsafePointer[Scalar[BF16]]` (ignored when `FUSE_RESIDUAL=False`) and comptime `FUSE_RESIDUAL: Bool`.
- q/k/v/gate/up → `FUSE_RESIDUAL=False`.
- o_proj, down_proj → `FUSE_RESIDUAL=True`, `residual` = the residual-stream buffer (`acts.x`); both output `hidden=2560`, so `N=2560` and residual is `[M, 2560]`.

**Grok prep task (before full impl):** route o_proj/down_proj from `linear_add_residual()` to `linear()` with a residual arg, so this epilogue is the single residual-add site.

---

## 7. Register & LDS budget (why 64×64 / 4-wave)

Per lane: 4 accumulators × 8 F32 = **32 VGPRs**; `a` frag 4 VGPRs; `b` frag 4 VGPRs (transient); addressing/loops ~15 → **~55 VGPRs/lane**. Comfortable on RDNA4 (well under the occupancy cliff). LDS 4 KB/workgroup — non-binding. This headroom is the point: prove the multi-wave orchestration here, then v3b scales BN→256 with 8 waves (~8 acc/wave, ~64 VGPRs) and tunes occupancy.

---

## 8. Correctness gate — read carefully, the reference differs by path

The GEMM math (F32 accumulate, same K order) is identical to v1, so:

- **Plain path (q/k/v/gate/up, FUSE_RESIDUAL=False):** v3a ⊕ `FUSE_RESIDUAL=False` must be **bit-identical to `wmma_gemm_bf16_v1`** (no residual). Exact `==`.
- **Fused path (o/down, FUSE_RESIDUAL=True):** the bitwise reference is **`wmma_gemm_bf16_v1` output + an F32 residual-add-then-cast** (i.e. `cast_bf16(f32(v1_result) + f32(residual))`), **NOT** the naive `linear_add_residual`. Same ops → must be exact `==`.
- **vs the naive `linear_add_residual`:** tolerance only (`1e-5 + 1.6e-2·|ref|`) — the naive matmul accumulates in a different order, so it will *not* be bitwise. Do not gate on bitwise here; use it only as a sanity backstop.

Test shapes (all ÷64 in M,N and ÷16 in K): `64×64×64`, `64×256×64`, `512×4096×2560` (real q_proj prefill), and `512×2560×9728` (real down_proj, fused path).

---

## 9. Verification checklist

1. **v3a vs v1 bitwise** (plain path) — 0 mismatches, all test shapes.
2. **v3a-fused vs (v1 + F32 residual)** bitwise — 0 mismatches, o/down shapes.
3. **Layer diff** (tiny, naive vs WMMA-v3, with o/down now fused) — expect 31/32 bit-exact, lm_head ~1e-7.
4. **Teacher-forced decode (4B)** — 255/256 (decode uses gemv; confirms no integration regression).
5. **Prefill 512 tok/s, 3 reps** — the number. Measure at **M=512** (real forward pass, not the microbench). Report actual + per-matmul share if flat. Expectation: ~4× on the WMMA GEMMs *and* the down_proj hole closed → meaningful end-to-end lift (v2 was ~226). This is a v3a step, not the final gate (that's v3b's N-heavy 128×256).

---

## 10. Preconditions & out of scope

**Preconditions:** M, N divisible by 64; K divisible by 16. (Real shapes satisfy this: M=512, N∈{4096,1024,9728,2560,151936}, K∈{2560,4096,9728} all qualify.) Arbitrary prompt lengths (M not ÷64) need remainder masking — **deferred**.

**Do NOT build here:** double-buffering (v4), FP8 packing/scaling (step 8), 128×256 / 8-wave (v3b), remainder/edge tiles, BK>16, wave-assignment other than sub-row=w.

---

## 11. Gate definition (G1b-3a)

**PASS** = §9.1 ∧ §9.2 bitwise-exact ∧ §9.3 layer-diff clean ∧ §9.4 decode 255/256 ∧ §9.5 prefill measured and reported (any regression investigated). On pass: 64×64 multi-wave reuse machinery + fused residual are proven; v3b scales the tile. Record confirmed VGPR/occupancy numbers in the kernel header for v3b sizing.

---
*Spec ends. Ratified 2026-07-13 — cleared for Grok after the linear() routing prep.*
