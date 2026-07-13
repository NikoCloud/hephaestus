# exp4c — WMMA GEMM v2 LDS cooperative load (2026-07-13)

## Design
Per K-strip: coalesced global→LDS (16 lanes/row × 8 iters) → `barrier` →
fragment from LDS → WMMA → `barrier`. `LDS_STRIDE=16` (named constant).
v1 kept as `wmma_gemm_kernel_v1` / `wmma_gemm_bf16_v1` for bitwise gate.

## Gates
| Check | Result |
|-------|--------|
| v2 vs v1 bitwise (16×16×32, 32×32×32, 4×128×128, 32×256×64) | **PASS** 0 mismatches |
| tiny layer-diff naive vs WMMA-v2 | **PASS** 31/32 bitexact; lm_head max_abs=1.19e-7 |
| 4B teacher-forced decode 256 | **255/256** (gemv path; near-tie class) |
| prefill 512 tok/s (3 reps) | **226.6 / 225.5 / 225.9** |

## Prefill vs v1
v1 was ~214–236 tok/s; v2 ~226. **No material gain** from coalescing alone
on this model path. Suspected: residual `linear_add_residual` still naive,
attention/norm cost, and single-use LDS (no reuse) + dual barriers wash the
coalescing win. Expected ceiling was 450–900 if global uncoalescing were the
limiter; it is not the dominant limiter at 16×16×16 without reuse.

## Next (not this change)
v3 wider tiles for LDS reuse; WMMA on residual path; LDS_STRIDE=24 if bank
conflicts show up under profiler.
