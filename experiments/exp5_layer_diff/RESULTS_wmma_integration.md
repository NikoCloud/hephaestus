# BF16 WMMA integration — layer-diff + bench (2026-07-13)

## Layer diff (tiny prompt 1, seq=4)
- naive vs wmma via `linear(..., use_wmma=)`
- **31/32 bitexact**, `final_step0_lm_head` max_abs=1.19e-7 (F32 ULP / reduction order)
- All BF16 activation cut points bitexact on tiny
- PASS under `1e-5 + 1.6e-2*|ref|`

## Teacher-forced decode (4B, 256 steps, M=1 gemv path)
- **255/256** argmax matches vs oracle (same near-tie class as Phase 1a)

## Prefill tok/s (512-token prompt, forward only)
| rep | prefill tok/s | TTFT ms |
|-----|---------------|---------|
| 1 | 213.9 | 2393 |
| 2 | 235.8 | 2171 |
| 3 | 234.9 | 2180 |

vs Phase 1a naive ~121–123 tok/s at 512 → **~1.9×**.  
Still far from G1b-3 (1.5× llama.cpp Q8_0 ~1400+): needs LDS + linear_add_residual WMMA.

## Decode (short prompt 10 + 32 steps)
- forward-only decode ~67 tok/s (M=1, gemv unchanged)
