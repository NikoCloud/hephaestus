# G1b-0 single-tile WMMA results

**Date:** 2026-07-13  
**Gate:** T1 ∧ T2 ∧ T3 exact equality on gfx1201  
**Env:** `~/projects/hephaestus-wmma-nightly` — Mojo `1.0.0b3.dev2026071206`  
**Hardware:** GPU 0 = R9700 gfx1201  

## Result: **PASS**

| Vector | Expected sample | GPU sample | Status |
|---|---|---|---|
| T1 `D=16*m*n` | D[1,1]=16, D[2,3]=96, D[15,15]=3600 | same | **PASS** 256/256 |
| T2 `D=m` (B=I) | D[1,1]=1, D[2,3]=2, D[15,15]=15 | same | **PASS** 256/256 |
| T3 `D=n` (A=I) | D[1,1]=1, D[2,3]=3, D[15,15]=15 | same | **PASS** 256/256 |

Comparison: bitwise `==` on f32 (integer inputs in [0,15], exact BF16 + F32 accum).

## Confirmed mappings (spec §2 — locked)

```
A: m = l%16,  k_half = l/16,  a[j] = A[m, k_half*8 + j]
B: n = l%16,  k_half = l/16,  b[j] = B[k_half*8 + j, n]
D: n = l%16,  m = (l/16)*8 + j,  store D[m,n] = r[j]
```

T2/T3 both pass with asymmetric expected tiles → C/D is **not** transposed relative to A; the column-distributed C/D layout is correct.

## Reproduce

```sh
cd experiments/exp4a_wmma_tile
sh run.sh
```
