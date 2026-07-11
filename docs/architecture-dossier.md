# Architecture Dossier — Qwen3-4B-Instruct-2507
## Date: 2026-07-11
## Source: config.json + tokenizer_config.json (verbatim from HuggingFace)

Every constant the Hephaestus forward pass will hard-code, sourced.

---

## 1. Model Identity

| Field | Value | Source |
|---|---|---|
| Architecture | `Qwen3ForCausalLM` | config.json `architectures[0]` |
| Model type | `qwen3` | config.json `model_type` |
| Parameters | 4.02B | llama-bench output |
| Torch dtype | `bfloat16` | config.json `torch_dtype` |
| Transformers version | 4.51.0 | config.json `transformers_version` |

## 2. Embedding Layer

| Constant | Value | Source |
|---|---|---|
| vocab_size | 151936 | config.json |
| hidden_size | 2560 | config.json |
| **Embedding tying** | **YES** | `tie_word_embeddings: true` |

Embedding shape: `[151936, 2560]` — the LM head shares this weight matrix (no separate output projection). This means the final matmul is `[hidden=2560, vocab=151936]` using the transposed embedding.

> **⚠ Tied embeddings — loader assertion required:** There is no separate `lm_head.weight` tensor in the safetensors file. The LM head *is* `model.embed_tokens.weight` transposed. The loader must **assert** this at load time: if a file contains both `embed_tokens.weight` and `lm_head.weight`, verify they share the same `data_ptr()`. If only `embed_tokens.weight` exists, use it for both. Do not assume — assert.

## 3. Transformer Blocks (×36)

| Constant | Value | Source |
|---|---|---|
| num_hidden_layers | 36 | config.json |
| max_position_embeddings | 262144 | config.json |
| max_window_layers | 36 | config.json (all layers use full attention) |
| use_sliding_window | false | config.json |
| sliding_window | null | config.json (no sliding window attention) |

### 3a. Attention (GQA)

| Constant | Value | Source |
|---|---|---|
| num_attention_heads | 32 | config.json |
| num_key_value_heads | 8 | config.json |
| head_dim | 128 | config.json |
| attention_bias | false | config.json (no bias terms on Q/K/V/O projections) |
| attention_dropout | 0.0 | config.json |
| **GQA group size** | **4** | Derived: 32 query heads ÷ 8 KV heads |
| Q projection | `[2560, 32×128=4096]` | No bias |
| K projection | `[2560, 8×128=1024]` | No bias |
| V projection | `[2560, 8×128=1024]` | No bias |
| O projection | `[4096, 2560]` | No bias (output, 4096 = 32 heads × 128 head_dim) |
| Scale factor | `1/sqrt(128)` ≈ 0.0884 | Standard 1/sqrt(head_dim) |

> **⚠ Non-square projections:** Qwen3 decouples `head_dim` from `hidden_size / num_heads`. With hidden=2560, 32 heads, and head_dim=128, the Q projection output is `32×128=4096`, **not** `hidden_size=2560`. So `q_proj` is a non-square `2560→4096` matrix and `o_proj` comes back `4096→2560`. K/V projections are `2560→1024` (8×128). Do **not** assume projection output = hidden size — this is the classic autopilot trap that produces silent shape mismatches.

### 3b. QK Normalization

Qwen3 applies RMSNorm to Q and K per-head before attention:
- Q norm: RMSNorm over `head_dim=128` per query head
- K norm: RMSNorm over `head_dim=128` per KV head
- RMSNorm eps: 1e-6 (from config.json `rms_norm_eps`)

### 3c. Feed-Forward Network (SwiGLU)

| Constant | Value | Source |
|---|---|---|
| hidden_act | `silu` | config.json |
| intermediate_size | 9728 | config.json |
| Gate projection | `[2560, 9728]` | No bias |
| Up projection | `[2560, 9728]` | No bias |
| Down projection | `[9728, 2560]` | No bias |
| FFN formula | `down(silu(gate(x)) * up(x))` | Standard SwiGLU |

### 3d. Layer Normalization

| Constant | Value | Source |
|---|---|---|
| Norm type | RMSNorm | Qwen3 standard |
| rms_norm_eps | 1e-6 | config.json |
| Attention norm | RMSNorm, shape [2560] | Applied before attention |
| FFN norm | RMSNorm, shape [2560] | Applied before FFN |
| Output norm | RMSNorm, shape [2560] | Applied after all layers |

### 3e. RoPE

| Constant | Value | Source |
|---|---|---|
| rope_theta | 5,000,000 | config.json |
| rope_scaling | null | config.json (no scaling) |
| RoPE type | Default (standard) | config.json (no scaling → standard) |
| head_dim | 128 | Applied per-head |
| Interleaved | Depends on impl | GGUF warns "Unknown RoPE type: default" |

**Note:** The GGUF converter logged `WARNING:Unknown RoPE type: default`. This means no special rope scaling (no YaRN, no NTK, etc.) — standard rotary embeddings at theta=5M. Mojo implementation must match HF's default RoPE application.

## 4. Special Tokens

| Token | ID | Source |
|---|---|---|
| BOS / PAD | 151643 (`<|endoftext|>`) | config.json `bos_token_id`, `pad_token_id` |
| EOS | 151645 (`<|im_end|>`) | config.json `eos_token_id` |
| `<|im_start|>` | 151644 | tokenizer_config.json |
| add_bos_token | false | GGUF converter output |

**No BOS is prepended.** The tokenizer does not add a BOS token. Prompt starts raw.

## 5. Chat Template

The tokenizer uses the `<|im_start|>` / `<|im_end|>` pattern:
```
<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
```

For Phase 1a, the CLI takes a raw prompt string (no chat template). The oracle fixtures will use raw prompts too — tokenizing directly without chat formatting.

> **⚠ Tokenizer parity — G1a-1 gate caveat:** Token-identical comparison only validates the forward pass if both sides consume *identical input token IDs*. The oracle fixtures saved raw input token IDs (`promptN_input_ids.json`). Hephaestus must feed those exact IDs directly rather than re-tokenizing text through its own tokenizer path. A whitespace-handling or BPE-merge difference in a vendored tokenizer will produce subtly different token IDs, causing a token mismatch that looks like a forward-pass bug but is actually a tokenizer bug. When the time comes, test tokenizer parity first: run the vendored tokenizer on the same 3 prompt strings, diff against the saved IDs, and only then compare generation output.

## 6. Tensor Inventory (per layer, BF16)

| Tensor Name | Shape | Count | Notes |
|---|---|---|---|
| token_embd.weight | [151936, 2560] | 1 | Shared with output |
| blk.{0-35}.attn_norm.weight | [2560] | 36 | RMSNorm gamma |
| blk.{0-35}.attn_q.weight | [2560, 4096] | 36 | Q projection |
| blk.{0-35}.attn_k.weight | [2560, 1024] | 36 | K projection |
| blk.{0-35}.attn_v.weight | [2560, 1024] | 36 | V projection |
| blk.{0-35}.attn_output.weight | [4096, 2560] | 36 | O projection |
| blk.{0-35}.attn_q_norm.weight | [128] | 36 | Q per-head RMSNorm |
| blk.{0-35}.attn_k_norm.weight | [128] | 36 | K per-head RMSNorm |
| blk.{0-35}.ffn_norm.weight | [2560] | 36 | FFN RMSNorm gamma |
| blk.{0-35}.ffn_gate.weight | [2560, 9728] | 36 | SwiGLU gate |
| blk.{0-35}.ffn_down.weight | [9728, 2560] | 36 | SwiGLU down |
| blk.{0-35}.ffn_up.weight | [2560, 9728] | 36 | SwiGLU up |
| output_norm.weight | [2560] | 1 | Final RMSNorm |

**Total tensors:** 1 + 36×12 + 1 = **434 tensors** (but GGUF reported 398 — some may be fused/omitted; verify at load time)

## 7. Hard-Coded Constants Summary

For the Phase 1a Mojo forward pass, these are compile-time constants:

```mojo
comptime VOCAB_SIZE = 151936
comptime HIDDEN_SIZE = 2560
comptime NUM_LAYERS = 36
comptime NUM_HEADS = 32
comptime NUM_KV_HEADS = 8
comptime HEAD_DIM = 128
comptime GQA_GROUP = 4  // NUM_HEADS / NUM_KV_HEADS
comptime INTERMEDIATE_SIZE = 9728
comptime RMS_NORM_EPS = 1e-6
comptime ROPE_THETA = 5_000_000.0
comptime MAX_POSITION = 262144
comptime BOS_TOKEN_ID = 151643
comptime EOS_TOKEN_ID = 151645
comptime TIE_EMBEDDINGS = true
comptime ATTENTION_BIAS = false
comptime SCALE_FACTOR = 1.0 / sqrt(128.0)  // ~0.0884
```

## 8. Weight Layout (safetensors)

Safetensors stores BF16 tensors in row-major (C contiguous) layout.
- Weight matrices are stored as `[in_features, out_features]` (PyTorch nn.Linear convention: weight is `[out, in]` but Qwen3 uses `[in, out]` in the checkpoint)

**Verify at load time:** The GGUF conversion logged shapes like `attn_q.weight [2560, 4096]` — this is `[hidden, num_heads * head_dim]`, meaning the weight is stored transposed from the typical PyTorch `[out_features, in_features]` convention. The safetensors checkpoint likely matches PyTorch's convention `[out_features, in_features]`, which would be `[4096, 2560]` for Q. **This must be verified when the safetensors loader is written.**
