# FP8 E4M3 Checkpoint Format — Qwen3-4B-Instruct-2507

## Date: 2026-07-13
## Source: `/mnt/models/models/qwen3-4b-instruct-2507-fp8/`, produced by `scripts/quantize_fp8.py`

Weight-only FP8 E4M3 quantization of the Phase 1a BF16 checkpoint, for the
Phase 1b FP8 WMMA path (SPEC.md §3). Activations stay BF16 — this is a
weight-format change only, not W8A8. Produced with `llm-compressor`
(`llmcompressor==0.9.0.3`, `compressed-tensors==0.13.0`) in a dedicated venv
(`~/.venvs/hephaestus-fp8`, Python 3.11 — the system Python is 3.14, where
`compressed-tensors` crashes at import time because one of its classes is
decorated with `@torch.compile`, unsupported on 3.14).

## 1. Which tensors are FP8 vs BF16 vs F32

| Tensor category | Count | dtype | Notes |
|---|---|---|---|
| `model.embed_tokens.weight` | 1 | **F8_E4M3** | Quantized. Also serves as the LM head (tied). |
| `*.self_attn.{q,k,v,o}_proj.weight` | 4 × 36 = 144 | **F8_E4M3** | Quantized |
| `*.mlp.{gate,up,down}_proj.weight` | 3 × 36 = 108 | **F8_E4M3** | Quantized |
| `*_scale` (one per FP8 tensor above) | 253 | **F32** | Per-output-channel weight scale |
| `*.{input,post_attention}_layernorm.weight` | 2 × 36 = 72 | BF16 | RMSNorm gamma, unquantized |
| `*.self_attn.{q,k}_norm.weight` | 2 × 36 = 72 | BF16 | Per-head RMSNorm gamma, unquantized |
| `model.norm.weight` | 1 | BF16 | Final RMSNorm, unquantized |

Total: 651 tensors (253 FP8 + 253 F32 scales + 145 BF16). Same 253
quantization targets as the architecture-dossier's Linear/Embedding
inventory: 1 embedding + 36 layers × 7 projections (q/k/v/o/gate/up/down).

**No `lm_head.weight` tensor** — same convention as the Phase 1a BF16
checkpoint. `tie_word_embeddings: true` in `config.json`.
`model.embed_tokens.weight` (FP8) is used for both the input embedding
lookup and the output LM head projection; `loader.mojo`'s `verify_manifest`
(A2) already asserts this tensor is absent.

Tensor **names** are unchanged from the BF16 checkpoint (`model.layers.N....`,
HF/safetensors convention) — only the weight tensors' dtype changed, plus one
new `<weight-tensor-name>_scale` tensor per quantized weight.

## 2. Scale tensor naming and shape

Scale tensor name = weight tensor name + `_scale` suffix, e.g.:

```
model.layers.0.self_attn.q_proj.weight        F8_E4M3  [4096, 2560]
model.layers.0.self_attn.q_proj.weight_scale   F32      [4096, 1]
```

Shape is `[out_features, 1]` — **one scale per output channel**, i.e. one
scale per row of the `[out, in]` weight matrix (safetensors/PyTorch
`nn.Linear` convention, per architecture-dossier.md §8). For
`model.embed_tokens.weight` (`[151936, 2560]`), "output channel" means one
scale per vocabulary row (`[151936, 1]`).

This is per-channel (`strategy="channel"` in compressed-tensors terms), not
per-tensor — each output channel of each weight matrix has its own
independently-fit scale, not one scalar for the whole tensor.

## 3. How the loader will use scales

Dequantization is a **post-WMMA, per-output-channel multiply on the F32
accumulator** — not a pre-matmul dequant to BF16/F32 (SPEC.md G1b-4 requires
no FP8→FP32 dequant path in any hot loop; the WMMA units consume FP8 weights
directly, per the already-proven `exp3g` direct-`llvm_intrinsic` path).

For a projection `y = x @ W^T` where `W` is `[out, in]` FP8 and `scale` is
`[out, 1]` F32:

1. WMMA computes `acc[out] = sum_in(x[in] * W[out, in])` directly on the FP8
   `W` values (raw FP8 bit patterns, no dequant), accumulating in F32.
2. After the WMMA pass, multiply each output element by its channel's scale:
   `y[out] = acc[out] * scale[out]`.

Because the scale is per-output-channel (not per-input-channel or
per-tensor), it factors out of the inner accumulation dot product entirely —
it's a single scalar multiply per output element, applied once after
accumulation, not once per multiply-add. This is what makes per-channel
scale cheap to apply post-WMMA rather than needing per-element rescaling
inside the hot loop.

## 4. Verification performed

- Safetensors header inspected directly (not assumed): 253 F8_E4M3 weight
  tensors, 253 F32 scale tensors (shape `[out, 1]` each), 145 BF16 tensors,
  no `lm_head.weight`, `tie_word_embeddings: true`.
- The dropped `lm_head.weight` was verified byte-identical to the source
  checkpoint's `model.embed_tokens.weight` (BF16) before being discarded —
  not assumed redundant, checked (`scripts/quantize_fp8.py::fixup_checkpoint`).
- `scripts/stage_weights.py` run against this checkpoint unmodified: staged
  651 tensors correctly, per-tensor dtype tags intact, byte accounting
  correct (verified against expected per-tensor sizes, e.g.
  `q_proj.weight_scale`: 4096 rows × 1 × 4 bytes F32 = 16384 bytes).
- Checkpoint size: 3.8 GB (`model.safetensors`), vs ~7.6 GB combined for the
  source BF16 shards — consistent with FP8 (1 byte/weight-element) replacing
  BF16 (2 bytes/weight-element) plus a small F32 scale overhead.

## 5. Known gaps in llm-compressor 0.9.0.3, worked around here

Two behaviors of this llm-compressor/compressed-tensors version did not
match the naive expectation and required a post-save fixup pass
(`fixup_checkpoint()` in `scripts/quantize_fp8.py`), rather than an in-model
fix before saving:

- `QuantizationArgs(scale_dtype=torch.float32)` has no effect — scale
  tensors are emitted as BF16 regardless. Confirmed by inspecting
  `embed_tokens.weight_scale.dtype` in memory immediately after `oneshot()`.
- The actual BF16→FP8 packing happens **inside** `model.save_pretrained()`
  (or inside `oneshot()`'s own internal save when `output_dir` is passed),
  not inside `oneshot()` itself — confirmed empirically: right after
  `oneshot(model, recipe)` with no `output_dir`, `embed_tokens.weight.dtype`
  is still `bfloat16`. Any attempt to re-tie `lm_head.weight` to
  `embed_tokens.weight` *before* the save call gets silently undone, because
  compression replaces `embed_tokens.weight` with a brand-new `Parameter`
  object during the save step. Fixing this by operating on the final
  serialized tensors (post-save) sidesteps the internal lifecycle entirely
  and is directly verifiable against the actual bytes on disk.

Both fixups are applied to the saved `model.safetensors` + `config.json`
directly, not to the in-memory model.
