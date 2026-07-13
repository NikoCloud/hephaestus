# G1b-0 — Single-Tile WMMA Microkernel + Mapping Oracle (Implementation Spec)

**Status:** RATIFIED 2026-07-13 — T1+T2+T3 test set and target path `experiments/exp4a_wmma_tile/` approved by Niko. Cleared for Grok to implement (§5) and run gate G1b-0 (§10).
**Owner of spec:** Opus. **Implementer:** Grok. **Target:** `experiments/exp4a_wmma_tile/` (new).
**Depends on:** exp3g (intrinsic arity + compile verified, returns 16.0 on gfx1201 R9700).
**Gate it satisfies:** proves the A/B/C-D lane mappings are correct in BF16 — the geometry every Phase-1b GEMM stands on — with zero quantization noise.

---

## 0. Why this exists (intent)

`exp3g` proved the WMMA intrinsic **compiles and has the right arity** on gfx1201, but its all-ones input is **mapping-blind**: every permutation of A, B, and the C/D decode yields 16.0. This microkernel replaces uniform inputs with **distinct, integer-valued** tiles so that any error in fragment placement produces a specific, diagnosable wrong-element pattern. It is the smallest artifact that promotes the A/B mappings below from *hypothesis* to *verified*.

Scope is exactly one 16×16×16 tile, one wave (32 lanes), one WMMA call. No K-loop, no shared memory, no multi-tile, no FP8. Those come after this is green.

---

## 1. Operation

Single WMMA tile, all dims = 16:

```
D[m,n] = sum_{k=0..15} A[m,k] * B[k,n] + C[m,n]      m,n in 0..15
```

- A is **M×K** (16×16), row-major in global memory: `A_global[m*16 + k]`.
- B is **K×N** (16×16), row-major in global memory: `B_global[k*16 + n]`.
- C, D are **M×N** (16×16), row-major: `D_global[m*16 + n]`.
- Accumulation dtype is **F32** (hardware accumulates in f32; C and D are f32).
- Wave32: lanes `l in 0..31`, each lane holds 8 elements per fragment, slot `j in 0..7`.

---

## 2. Fragment mappings (0-based; `/` is integer division)

These are **pre-registered** — the oracle in §5 is what confirms them. A and B are marked HYPOTHESIS (derived from GPUOpen "A and B are K-major, 8 contiguous elements/lane"); C/D is AUTHORITATIVE (reverse-engineered in the rdna4-wmma-guide).

### A fragment (M×K, K-major) — HYPOTHESIS
```
m       = l % 16
k_half  = l / 16            # 0 -> K∈[0,7], 1 -> K∈[8,15]
a[j]    = A[m, k_half*8 + j]                       for j in 0..7
# global offset:
a[j]    = A_global[ (l % 16) * 16 + (l / 16) * 8 + j ]
```

### B fragment (K×N, K-major) — HYPOTHESIS
```
n       = l % 16
k_half  = l / 16
b[j]    = B[k_half*8 + j, n]                       for j in 0..7
# global offset:
b[j]    = B_global[ ((l / 16) * 8 + j) * 16 + (l % 16) ]
```

### C/D fragment (M×N) — AUTHORITATIVE
```
n       = l % 16
m       = (l / 16) * 8 + j
D[m,n]  = r[j]                                     for j in 0..7
# global store:
D_global[ ((l / 16) * 8 + j) * 16 + (l % 16) ] = r[j]
```

**Note the deliberate asymmetry:** A/B use `l%16` as the operand's *outer* index (row of A / col of B) and walk K with `j`; C/D uses `l%16` as **N (column)** and walks **M (row)** with `j`. Do **not** assume C/D mirrors A. This transposed output layout is the single most common WMMA bug — it is why the oracle in §5 includes an asymmetric test.

---

## 3. Packing (operand register types)

Per-lane fragment (8 elements) → intrinsic operand:

| dtype | per-lane SIMD | bitcast to | LLVM operand |
|---|---|---|---|
| BF16 (this spec) | `SIMD[bfloat16, 8]` | `SIMD[int16, 8]` | `v8i16` |
| FP8 E4M3 (later) | `SIMD[float8_e4m3fn, 8]` | `SIMD[int32, 2]` | `v2i32` |
| Accumulator C/D | `SIMD[float32, 8]` | — | `v8f32` |

BF16 and FP8 differ **only** in this row. The mappings in §2 are byte-for-byte identical between them.

---

## 4. Intrinsic calls (verified in exp3g — 3 operands, no i1)

```mojo
# BF16  (this spec)
var r = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16", SIMD[DType.float32, 8]](
    bitcast[DType.int16, 8](a_bf16),   # v8i16
    bitcast[DType.int16, 8](b_bf16),   # v8i16
    c_f32,                             # v8f32, = 0 for the tests below
)

# FP8 E4M3 (later phase, same mappings)
var r = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8", SIMD[DType.float32, 8]](
    bitcast[DType.int32, 2](a_fp8),    # v2i32
    bitcast[DType.int32, 2](b_fp8),    # v2i32
    c_f32,                             # v8f32
)
```

---

## 5. Kernel structure (implementable steps)

Launch **1 block × 32 threads** (one wave). Per lane `l`:

1. Build `a: SIMD[bfloat16,8]`, `b: SIMD[bfloat16,8]` via the §2 global offsets.
2. `c = SIMD[float32,8](0)`  (C tile is zero in all tests).
3. `r = wmma_bf16(bitcast a, bitcast b, c)`  (§4).
4. Store: `for j in 0..7: D_global[((l/16)*8+j)*16 + (l%16)] = r[j]`.

Host side:
1. Fill `A_global`, `B_global` (16×16 bf16) per the test vector.
2. Compute `D_ref` with the CPU reference (§7).
3. Upload A, B; run kernel; download `D_global`.
4. Compare `D_global` to `D_ref` (§6 tolerance).

---

## 6. Exactness

For every test vector below, inputs are integers in **[0,15]** (plus identity 0/1). BF16 represents all integers 0–256 exactly, and F32 accumulation of ≤16 such products (max 3600) is exact. **Use exact equality (`==`), not a tolerance.** Any mismatch is a real mapping/packing bug, not rounding. (Same holds for FP8 E4M3: 0–15 and 0/1 are e4m3-exact.)

---

## 7. CPU reference matmul (the oracle)

```mojo
fn cpu_ref(A: Matrix16x16_bf16, B: Matrix16x16_bf16) -> Matrix16x16_f32:
    var D = Matrix16x16_f32(0)
    for m in range(16):
        for n in range(16):
            var acc: Float32 = 0.0
            for k in range(16):
                acc += Float32(A[m, k]) * Float32(B[k, n])   # cast bf16 -> f32 first
            D[m, n] = acc            # C = 0
    return D
```

---

## 8. Test vectors

### T1 — specified sanity vector (symmetric)
```
A[m,k] = m          (constant along k)
B[k,n] = n          (constant along k)
Expected: D[m,n] = 16 * m * n
```
Sample expected tile values: `D[0,*]=0`, `D[*,0]=0`, `D[1,1]=16`, `D[2,3]=96`, `D[15,15]=3600`.
Catches: K-accumulation, packing, gross mislayout, scaling.
**Blind spot:** `D` is symmetric (`D[m,n]==D[n,m]`), so T1 **cannot** detect an m↔n output transposition. T2/T3 exist to close that.

### T2 — asymmetric, isolates A-load ∘ C/D-store
```
A[m,k] = m          (constant along k)
B      = I_16       (identity: B[k,n] = 1 if k==n else 0)
Expected: D[m,n] = m          (asymmetric: D[1,2]=1, D[2,1]=2)
```

### T3 — asymmetric, isolates B-load ∘ C/D-store
```
A      = I_16
B[k,n] = n          (constant along k)
Expected: D[m,n] = n          (asymmetric: D[1,2]=2, D[2,1]=1)
```

All three use only values in [0,15] and 0/1 → exact in both BF16 and FP8.

---

## 9. Diagnosis table (if a test fails)

| T1 | T2 | T3 | Verdict |
|----|----|----|---------|
| pass | pass | pass | **All three mappings verified. Gate green.** |
| fail | — | — | K-accumulation / packing / bitcast wrong (see if result is scaled or garbage). |
| pass | fail | pass | **A-load mapping wrong** (§2 A). |
| pass | pass | fail | **B-load mapping wrong** (§2 B). |
| pass | fail | fail (both look transposed vs expected) | **C/D-store mapping wrong** (§2 C/D). |
| pass | T2/T3 outputs are each other's transpose | | A and B fragments swapped. |
| per-row-block wrong (rows 0–7 vs 8–15) | | | `k_half = l/16` half-assignment flipped. |

---

## 10. Gate definition (G1b-0)

**PASS** = T1 ∧ T2 ∧ T3 exact-equal on gfx1201 in the isolated nightly env.
On pass: the §2 mappings are locked and become the contract for the FP8 GEMM loader. Record the confirmed formulas in the engine's `wmma_gfx12.mojo` header comment.
On fail: use §9, correct the offending mapping in §2, re-run. Do not proceed to multi-tile or FP8 until green.

---

## 11. Explicitly out of scope (do NOT build here)
- K-loop / K>16, multi-tile, shared-memory staging, prefetch.
- FP8 path (separate follow-up; only the §3 packing row and §4 second call change).
- Wide-K (16×16×32) fusion — that is a later throughput lever, K=16 hardware only.
- The A/B "swap for N-major output" convention — layout/perf concern, not correctness of a single tile.

---
*Spec ends. Ratified 2026-07-13 — cleared for implementation.*
