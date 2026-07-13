# exp4e — WMMA v3a 64×64 multi-wave reuse (G1b-3a) 2026-07-13

## Kernel
- `wmma_gemm_kernel_v3a[c_type, fuse_residual]`: BM=BN=64, 4 waves, BK=16
- A fragment **hoisted** out of `sc` loop (4× A reuse)
- Dispatch: M,N % 64 and K % 16 → v3a; else v2
- Residual: in-place `val = F32(acc) + F32(C); C = cast_bf16(val)`

## Gates
| Check | Result |
|-------|--------|
| v3a plain vs v1 BF16 bitwise | **PASS** (64×64×64, 64×256×64, 128×128×64) |
| v3a fused vs (v1 **F32** product + residual) bitwise | **PASS** (same + 128×128×128) |
| tiny layer-diff (M=4 → v2 fallback) | first div o_proj_residual (WMMA residual ULP; same as exp4d) |
| teacher-forced decode | **256/256** |
| prefill 512 tok/s (3 reps) | **793.4 / 793.9 / 794.0** |

## Prefill
v2 was ~226 tok/s. v3a **~794 tok/s (~3.5×)**. Down_proj hole closed (residual on WMMA when M,N % 64).

Still short of G1b-3 (~2100 / 1.5× llama.cpp); v3b N-heavy next.

## VGPR note
Header records expected ~55 VGPR/lane design budget; measure with rocprof for v3b.
