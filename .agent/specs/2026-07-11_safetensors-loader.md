# Implementation Spec: Safetensors Loader (Phase 1a)
## Date: 2026-07-11
## Gate: G1a-1 (token-identical output vs HF reference)

## Context

Hephaestus is a Mojo-native LLM inference engine for AMD RDNA4 (gfx1201).
Phase 1a = BF16 correctness baseline. First deliverable = safetensors loader.

Read SPEC.md (project root) for full scope law, phase gates, and non-goals.
Read docs/architecture-dossier.md for every model constant (verified from safetensors headers).
Read docs/environment-notes.md for toolchain, shell quirks, and GPU constraints.
Read docs/kernel-interface-report.md for MAX kernel vendoring contract.

## Staging Approach (Verified)

Mojo has no JSON parser. Python pre-pass (scripts/stage_weights.py) reads safetensors
and produces:
- `.weights` - flat binary blob, all tensor data concatenated
- `.offsets` - text file: `name\toffset\tsize\tshape\tdtype` per line

Mojo loader reads these two files. Run staging with:
  pixi run python scripts/stage_weights.py <model_dir> <output_prefix>

## Experiments Resolved (2026-07-11)

**Exp 1 (TileTensor type erasure): PASSED.**
Structs with `[origin: Origin[mut=True]]` can hold TileTensor fields with different
compile-time layout types. Use `origin=Self.origin` in field types. Construct via
`TileTensor[...](ptr=base_ptr + offset, layout=row_major[ROWS, COLS]())`.

**Exp 2 (DeviceBuffer pipeline): PASSED.**
`DeviceContext()` -> `enqueue_create_buffer` -> `enqueue_create_host_buffer` ->
fill host -> `enqueue_copy` -> `synchronize` -> `TileTensor(dev_buf, layout)`.
Functions calling DeviceContext must be marked `raises`.

## Buffer Strategy: Single Arena

One DeviceBuffer for all 398 weights. Tensors at byte offsets.
Avoids allocator fragmentation. One allocation, one lifetime.

## Weight Mapping

| HF Key | Field | Shape [out,in] |
|---|---|---|
| model.embed_tokens.weight | embed_tokens | [151936, 2560] |
| model.layers.N.input_layernorm.weight | layers[N].attn_norm | [2560] |
| model.layers.N.self_attn.q_proj.weight | layers[N].q_proj | [4096, 2560] |
| model.layers.N.self_attn.k_proj.weight | layers[N].k_proj | [1024, 2560] |
| model.layers.N.self_attn.v_proj.weight | layers[N].v_proj | [1024, 2560] |
| model.layers.N.self_attn.o_proj.weight | layers[N].o_proj | [2560, 4096] |
| model.layers.N.self_attn.q_norm.weight | layers[N].q_norm | [128] |
| model.layers.N.self_attn.k_norm.weight | layers[N].k_norm | [128] |
| model.layers.N.post_attention_layernorm.weight | layers[N].ffn_norm | [2560] |
| model.layers.N.mlp.gate_proj.weight | layers[N].gate_proj | [9728, 2560] |
| model.layers.N.mlp.up_proj.weight | layers[N].up_proj | [9728, 2560] |
| model.layers.N.mlp.down_proj.weight | layers[N].down_proj | [2560, 9728] |
| model.norm.weight | output_norm | [2560] |

Tied embeddings: assert lm_head.weight absent. lm_head = embed_tokens (same pointer).
Total: 1 + 36x11 + 1 = 398 tensors.

## WMMA Divisibility

All K dimensions (2560, 4096, 9728) are divisible by 32.
Every matmul takes the BK=32 fast path. No fallback anywhere.

## Assertions

A1: All dtypes BF16
A2: lm_head.weight absent (tied embeddings)
A3-A9: Shape checks for each tensor type
A10: Total count = 398
A11: data_offsets span = product(shape) x 2
A12: metadata format = "pt"

## Definition of Done

1. Tiny model: loads fixtures/tiny_random, all assertions pass
2. 4B model: loads 3 shards, 398 tensors, under 30s (G1a-3)
3. Tied embedding: lm_head ptr == embed_tokens ptr
4. No GPU execution beyond allocation+copy

## Existing Code

- src/hephaestus/constants.mojo - compile-time constants (compiles)
- src/hephaestus/model.mojo - Qwen3Layer struct with 11 TileTensor fields (compiles)
- scripts/stage_weights.py - Python safetensors stager (tested on both models)
- scripts/prepare_manifest.py - Python binary manifest generator (tested)

## Next Steps

1. Write loader.mojo: read .offsets text file, mmap/read .weights blob, allocate DeviceBuffer, copy, construct Qwen3Weights
2. Write main.mojo: CLI that calls loader and prints verification
3. Test on tiny model first, then 4B
4. For 4B: need bulk copy (not per-byte loop) - use memcpy or bulk read
