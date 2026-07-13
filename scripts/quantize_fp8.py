#!/usr/bin/env python3
"""Quantize Qwen3-4B-Instruct-2507 to FP8 E4M3 with per-channel weight scales.

Weight-only quantization: activations stay BF16. This matches Hephaestus's
Phase 1b WMMA path (FP8 weights x BF16 activations, proven via the direct
llvm_intrinsic path -- see exp3g). No calibration dataset is needed: per-
channel scales are derived directly from each weight tensor's own min/max,
not from observed activation statistics.

Targets both nn.Linear (all attention/MLP projections) and the tied
nn.Embedding (model.embed_tokens), since embed_tokens.weight doubles as the
LM head (tie_word_embeddings=True in the source checkpoint).

Runs entirely on CPU; must run in the dedicated venv (~/.venvs/hephaestus-fp8)
because llm-compressor's compressed-tensors dependency decorates a class
with @torch.compile at import time, which is unsupported on the system
Python (3.14). The venv pins Python 3.11 + torch 2.9.1+rocm6.3, matching
the system torch build exactly, and is not used for any GPU work.

Two post-hoc fixups run after llm-compressor's own save, verified empirically
(not assumed) against llm-compressor 0.9.0.3 / compressed-tensors 0.13.0:

1. Scale dtype. QuantizationArgs(scale_dtype=torch.float32) has no effect in
   this version -- weight_scale tensors come out of oneshot() as BF16
   regardless. Upcast them to F32 post-save (lossless: BF16->F32 is exact,
   this just changes storage width, not precision already committed).

2. Tied embeddings. embed_tokens is quantized (Embedding is a target), but
   lm_head is `ignore`d, so it keeps its original BF16 Parameter object.
   Compression replaces embed_tokens.weight with a *new* Parameter (the FP8
   tensor) inside model.save_pretrained() itself -- confirmed by inspecting
   embed_tokens.weight.dtype immediately after oneshot() with no output_dir:
   still bfloat16, meaning the actual BF16->FP8 packing happens inside the
   save call, not inside oneshot(). Any re-tie attempted before save_pretrained
   is silently undone when compression swaps in the new embed_tokens.weight
   object. Net effect: config.json's tie_word_embeddings flips to False and a
   redundant full-precision lm_head.weight is written to disk. Qwen3-4B-
   Instruct-2507 ties these architecturally (architecture-dossier.md section
   2); fix it after the fact by dropping the stale lm_head.weight tensor and
   restoring tie_word_embeddings=True, matching the BF16 checkpoint's own
   convention of no separate lm_head.weight tensor (loader.mojo's
   verify_manifest asserts exactly this).

Usage:
  ~/.venvs/hephaestus-fp8/bin/python scripts/quantize_fp8.py
"""

import json
import os

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_DIR = "/mnt/models/models/qwen3-4b-instruct-2507"
OUTPUT_DIR = "/mnt/models/models/qwen3-4b-instruct-2507-fp8"


def quantize():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype="bfloat16", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    weight_args = QuantizationArgs(
        num_bits=8, type="float", strategy="channel", symmetric=True, dynamic=False
    )
    scheme = QuantizationScheme(
        targets=["Linear", "Embedding"], weights=weight_args
    )
    recipe = QuantizationModifier(
        config_groups={"group_0": scheme}, ignore=["lm_head"]
    )

    oneshot(model=model, recipe=recipe, output_dir=OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


def load_original_embed_tokens_bf16():
    """Reads model.embed_tokens.weight straight out of the source checkpoint's
    safetensors shards, to confirm the FP8 checkpoint's stray lm_head.weight
    really is that same (pre-quantization) tensor before we drop it -- not an
    assumption, a byte comparison."""
    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        shard_file = json.load(f)["weight_map"]["model.embed_tokens.weight"]
    tensors = load_file(os.path.join(MODEL_DIR, shard_file))
    return tensors["model.embed_tokens.weight"]


def fixup_checkpoint():
    st_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    tensors = load_file(st_path)

    if "lm_head.weight" in tensors:
        original_embed = load_original_embed_tokens_bf16()
        if tensors["lm_head.weight"].dtype != original_embed.dtype:
            raise ValueError("lm_head.weight dtype changed -- not the stale tie artifact expected")
        if not torch.equal(tensors["lm_head.weight"], original_embed):
            raise ValueError(
                "lm_head.weight differs from the source embed_tokens.weight -- "
                "not safe to drop as a redundant tied-embedding artifact"
            )
        del tensors["lm_head.weight"]
        print("Dropped redundant lm_head.weight (byte-identical to source embed_tokens.weight)")

    n_upcast = 0
    for name in list(tensors.keys()):
        if name.endswith("_scale") and tensors[name].dtype != torch.float32:
            tensors[name] = tensors[name].to(torch.float32)
            n_upcast += 1
    print(f"Upcast {n_upcast} scale tensors to F32")

    save_file(tensors, st_path, metadata={"format": "pt"})

    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["tie_word_embeddings"] = True
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Restored tie_word_embeddings=True in config.json")


def main():
    quantize()
    fixup_checkpoint()
    print(f"Saved FP8 checkpoint to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
