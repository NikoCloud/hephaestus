# DECISIONS.md — Decision Log (append-only)

| date | decision | why | reversible? |
| 2026-07-11 | Mojo native, no Vulkan for v1 | No Mojo-SPIR-V backend yet | Yes |
| 2026-07-11 | BF16 before FP8 sequencing | Prove correctness first | Yes |
| 2026-07-11 | Qwen3-4B dense as first arch | Small, known, single-GPU fits | Yes |
| 2026-07-11 | Vendor MAX kernels, do not write | Do not reinvent what exists | Yes |
| 2026-07-11 | 90 percent single-stream bar | Bandwidth bound, batching wins | No |
| 2026-07-11 | GGUF permanently out | Thesis is escaping that gravity well | No |
| 2026-07-11 | torch installed system-wide, not in pixi | pixi resolver cannot solve pytorch-triton-rocm dependency; torch only needed for GGUF converter, not Mojo work | Yes |
| 2026-07-11 | Both cards confirmed gfx1201 via rocminfo | R9700 and 9070 XT are both RDNA4 gfx1201; earlier gfx1150 claim was wrong | No |
| 2026-07-11 | Baseline 55.14 tok/s tg128 = ~69% bandwidth utilization | R9700 has ~512 GB/s peak; 4B model at F16 = ~7.5GB per forward pass; 55 t/s × 7.5GB = ~413 GB/s effective | No |
