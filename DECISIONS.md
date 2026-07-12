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
| 2026-07-12 | Stager drops tied lm_head.weight after byte-identity check vs embed_tokens | tiny_random saves lm_head despite tie_word_embeddings; 4B omits it; dropping at staging gives the Mojo loader one uniform tensor set (2 + 11L) | Yes |
| 2026-07-12 | Model dims are comptime struct params, not hard-coded 4B constants | tiny_random debug loop (mandated) needs different dims; compile-time parameterization, zero runtime config-driven generality; 4B stays the only shipped instantiation | Yes |
| 2026-07-12 | Loader verifies device copy by round-trip compare (full <=16MB, sampled above) | GPU copies fail silently (project has already seen an all-zeros kernel); ~1s overhead on 8GB load, still 6.3s total | Yes |
| 2026-07-12 | G1a-3 PASS: 4B loads in ~6.3s warm, <=8.5s cold bound | 3 reps in bench/1a-load.md; NVMe direct read 4.3 GB/s bounds the cold case | No |
| 2026-07-12 | A12 (metadata format == "pt") enforced in stager, not Mojo loader | staging strips safetensors headers; the stager is the only place the header is visible; verified passing on tiny + 4B | Yes |
| 2026-07-12 | Tiny oracle extended with step logits from EXISTING weights (scripts/build_tiny_logits.py) | tiny had token IDs only; logit-level diffing needs logits; every step token-verified vs reference_outputs.json so transformers drift fails loudly instead of poisoning fixtures | Yes |
| 2026-07-12 | **BLOCKER: no WMMA of any dtype compiles on gfx1201 in Mojo 1.0.0b3.dev2026071006** | BF16 16-elem frags (RDNA3 encoding, what stdlib emits) -> `LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.bf16`; BF16 8-elem frags (RDNA4 encoding) -> stdlib comptime assert "no valid implementation"; FP8 E4M3 8- and 16-elem -> `LLVM ERROR: Do not know how to split this operator's operand!`. Evidence: experiments/exp3c,exp3d,exp3f. stdlib RDNA WMMA is gfx11-only (mma_amd_rdna.mojo:370 hard-asserts `_has_shape[(16,16,8,8)]`; buffers.mojo:39 `RDNA_AB_FRAG_SIZE = 16`) | No (upstream toolchain) |
| 2026-07-12 | Consequence: high-level `matmul()` dispatcher unusable for BF16 on gfx1201 | it instantiates `gemm_kernel_rdna` at comptime regardless of the runtime m/n branch, so even an M=1 call that would route to gemv fails to compile (exp3b) | No |
| 2026-07-12 | Consequence: vendored RDNA attention (AttentionRDNA / mha_prefill / mha_decode) also blocked | mha.mojo:2215 `comptime if _is_amd_rdna()` -> AttentionRDNA -> rdna_mma -> the same 16-elem BF16 mma() call proven failing in exp3c. Chain verified in source; the failing unit is verified empirically | No |
| 2026-07-12 | Exp 1 resolved: call WMMA-free kernels directly, not the matmul() dispatcher | `gemv_gpu` (decode M=1) and `matmul_kernel_naive` (prefill M>1) both compile and are correct vs CPU fp32 ref on gfx1201 (exp3e: 0 errors, max_diff 0.031 at bf16 tol). Decode is bandwidth-bound so gemv is the G1a-2 path anyway; prefill perf is not gated in 1a | Yes |
| 2026-07-12 | Exp 2 resolved: vendor `LayoutTensorMHAOperand`, do not write a KV wrapper | mha_operand.mojo:873 is "an implementation for contiguous tensor arguments to MHA kernels", rank-4 [batch, tok, head, head_dim] — exactly our single-stream contiguous cache. NOTE: moot until attention is unblocked; the RDNA attention kernels it feeds do not compile | Yes |
| 2026-07-12 | Exp 3 resolved: rms_norm_gpu with multiply_before_cast=False, weight_offset=0 | reproduces HF Qwen3RMSNorm exactly (`self.weight * hidden.to(input_dtype)` = cast-then-multiply, verified in transformers 5.13.1 source). Bit-exact (max_diff 0.0) vs host fp32 ref at [4,2560] gamma[2560] and per-head [128,128] gamma[128] (exp4). Per-head QK norm needs no special casing: kernel derives cols from gamma.dim[0] | Yes |
| 2026-07-12 | Exp 4 resolved: `nn.gather[axis=0]` is the embedding lookup; nothing to write | bit-exact vs host ref at tiny shape incl. vocab-edge id 255 (exp5) | Yes |
