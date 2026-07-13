# exp4d — WMMA residual fusion for o_proj / down_proj (2026-07-13)

## Change
- `wmma_gemm_bf16_residual` / `wmma_gemm_kernel_residual`: same LDS-v2 body;
  store is `(F32(residual) + acc).cast[BF16]()` in-place RMW.
- `linear_add_residual`: M=1 still gemv+epilogue; M>1 → WMMA residual when
  N,K % 16 == 0; `use_wmma=False` forces naive (layer-diff).

## Gates
| Check | Result |
|-------|--------|
| fused vs separate (WMMA matmul + add) bitwise | **PASS** 0 mismatches (4 shapes) |
| tiny layer-diff naive vs WMMA | first div `layer0_step0_o_proj_residual` (expected: residual now WMMA) |
| o_proj \|ref\|≥1e-2 exceed | **0** / 417; max_abs=4.88e-4 (BF16 ULP class) |
| teacher-forced decode 4B 256 | **256/256** (M=1 gemv; no regression) |

## Layer-diff note
Previously residual stayed on naive for both paths → 31/32 bitexact.
With residual on WMMA, pure-matmul cut points before o_proj remain bitexact;
o_proj/down_proj show reduction-order BF16 ULP. Cascade fails the loose
1.6% relative gate on near-zero entries only — significant magnitudes agree.
