# DECISIONS.md --- Decision Log (append-only)

| date | decision | why | reversible? |
|---|---|---|---|
| 2026-07-11 | Mojo native, no Vulkan for v1 | No Mojo-SPIR-V backend yet | Yes |
| 2026-07-11 | BF16 before FP8 sequencing | Prove correctness first | Yes |
| 2026-07-11 | Qwen3-4B dense as first arch | Small, known, single-GPU fits | Yes |
| 2026-07-11 | Vendor MAX kernels, do not write | Do not reinvent what exists | Yes |
| 2026-07-11 | 90 percent single-stream bar | Bandwidth bound, batching wins | No |
| 2026-07-11 | GGUF permanently out | Thesis is escaping that gravity well | No |
| 2026-07-11 | torch installed system-wide, not in pixi | pixi resolver cannot solve pytorch-triton-rocm dependency; torch only needed for GGUF converter, not Mojo work | Yes |
| 2026-07-11 | Both cards confirmed gfx1201 via rocminfo | R9700 and 9070 XT are both RDNA4 gfx1201; earlier gfx1150 claim was wrong | No |
| 2026-07-11 | Baseline 55.14 tok/s tg128 = ~69% bandwidth utilization | R9700 has ~512 GB/s peak; 4B model at F16 = ~7.5GB per forward pass; 55 t/s x 7.5GB = ~413 GB/s effective | No |
| 2026-07-11 | Zero-copy weight loading: transpose_b=True + safetensors [out,in] = kernel-ready | safetensors stores weights in PyTorch [out_features, in_features] order; MAX matmul kernel transpose_b=True expects exactly this; no transposition at load time | No |
| 2026-07-11 | Dossier section 6 shapes corrected from GGUF-order to safetensors-order | Original table pulled GGUF shapes (reversed dims); verified against actual safetensors header | No |
| 2026-07-11 | Dossier tensor count corrected: 398 not 434 | 1 + 36x11 + 1 = 398 (11 per layer, not 12); verified from safetensors header | No |
| 2026-07-11 | Scripts executed as files, never piped through interactive shell | cachyos runs fish; base64-smuggling past fish expansion is fragile; committed scripts = debuggable archaeology | Yes |
| 2026-07-11 | Single DeviceBuffer arena for all weights | 398 tensors; avoids allocator fragmentation; one allocation, one lifetime; trivial memory budget calc for Phase 2 | Yes |
| 2026-07-11 | Python pre-pass stages safetensors to flat binary + text offsets | Mojo has no JSON parser; vendoring safetensors reader as Python (scripts/stage_weights.py) rather than writing one | Yes |
| 2026-07-12 | Exp 1 resolved: TileTensor type erasure via Origin-parameterized structs | Struct with [origin: Origin[mut=True]] holds TileTensor fields of differing compile-time layouts using origin=Self.origin; evidence: src/hephaestus/model.mojo compiles with all 11 layouts | Yes |
| 2026-07-12 | Exp 2 resolved: HostBuffer pipeline over mmap for host-to-device copy | enqueue_create_buffer -> enqueue_create_host_buffer -> fill -> enqueue_copy -> synchronize verified working; callers of DeviceContext must be raises; mmap-direct path untested, revisit only if load exceeds 30s gate | Yes |
