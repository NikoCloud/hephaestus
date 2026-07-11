#!/usr/bin/env python3
"""Build a tiny-random Qwen3 oracle model for fast debug loops.

Creates a 2-layer, tiny-hidden Qwen3 architecture with random weights,
runs 3 fixed prompts (16 tokens greedy each), and saves:
  - model.safetensors (state dict)
  - config.json
  - reference_outputs.json (input_ids, output_ids per prompt)
  - manifest.json (seed, config, output details)
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen3ForCausalLM, Qwen3Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
OUTPUT_DIR = "/home/nikocloud/projects/hephaestus/fixtures/tiny_random"

TINY_CONFIG = Qwen3Config(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=256,
    vocab_size=256,
    rms_norm_eps=1e-06,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    hidden_act="silu",
    attention_bias=False,
    torch_dtype=torch.bfloat16,
)

# 3 fixed prompts — short, deterministic token-id sequences (within tiny vocab)
PROMPTS = [
    [0, 1, 2, 3],
    [10, 20, 30, 40, 50],
    [100, 200, 255, 5, 10],
]

NUM_NEW_TOKENS = 16  # greedy generation length


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Build model --------------------------------------------------------
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = Qwen3ForCausalLM(TINY_CONFIG)
    model.to(torch.bfloat16)
    model.eval()

    # --- Generate reference outputs ----------------------------------------
    reference_outputs = []
    device = next(model.parameters()).device  # CPU

    for idx, prompt_ids in enumerate(PROMPTS):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            # Greedy: use generate with do_sample=False
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=NUM_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_k=1,
                pad_token_id=None,
            )
        out_ids = generated[0].tolist()
        reference_outputs.append(
            {
                "prompt_index": idx,
                "input_ids": prompt_ids,
                "output_ids": out_ids,
                "num_input_tokens": len(prompt_ids),
                "num_output_tokens": len(out_ids),
                "num_new_tokens": len(out_ids) - len(prompt_ids),
            }
        )
        print(f"Prompt {idx}: input={prompt_ids}")
        print(f"  output (full)={out_ids}")
        print(f"  new tokens    ={out_ids[len(prompt_ids):]}")

    # --- Save state dict (safetensors) -------------------------------------
    # Handle tied weights: lm_head.weight shares memory with model.embed_tokens.weight
    state_dict = model.state_dict()
    seen_data_ptrs = {}
    state_dict_dedup = {}
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        if ptr in seen_data_ptrs:
            # Skip duplicate (tied) tensor — record the tie
            print(f"Skipping tied tensor '{k}' (shares memory with '{seen_data_ptrs[ptr]}')")
            continue
        seen_data_ptrs[ptr] = k
        state_dict_dedup[k] = v.contiguous()
    safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    save_file(state_dict_dedup, safetensors_path, metadata={"format": "pt"})
    print(f"Saved state dict ({len(state_dict_dedup)} tensors, {len(state_dict) - len(state_dict_dedup)} tied) -> {safetensors_path}")

    # --- Save config.json ---------------------------------------------------
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(TINY_CONFIG.to_dict(), f, indent=2)
    print(f"Saved config -> {config_path}")

    # --- Save reference_outputs.json ----------------------------------------
    ref_path = os.path.join(OUTPUT_DIR, "reference_outputs.json")
    with open(ref_path, "w") as f:
        json.dump(reference_outputs, f, indent=2)
    print(f"Saved reference outputs -> {ref_path}")

    # --- Save manifest.json -------------------------------------------------
    manifest = {
        "seed": SEED,
        "model_type": "Qwen3ForCausalLM",
        "torch_dtype": "bfloat16",
        "config": TINY_CONFIG.to_dict(),
        "prompts": PROMPTS,
        "num_new_tokens": NUM_NEW_TOKENS,
        "num_prompts": len(PROMPTS),
        "reference_outputs_file": "reference_outputs.json",
        "model_file": "model.safetensors",
        "config_file": "config.json",
        "state_dict_num_tensors": len(state_dict),
        "state_dict_keys": list(state_dict.keys()),
        "generation": {
            "method": "greedy",
            "do_sample": False,
            "top_k": 1,
        },
        "purpose": "Tiny random Qwen3 oracle for millisecond debug loops — verify forward pass end-to-end quickly.",
    }
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {manifest_path}")

    # --- Summary -----------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nDone! Total parameters: {total_params:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
