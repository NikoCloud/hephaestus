#!/usr/bin/env python3
"""Generate step-logit fixtures for the tiny_random oracle model.

The tiny fixture set (fixtures/tiny_random/) saved only token IDs; the 4B
oracle (fixtures/oracle/) also saved per-step logits, which is what makes
logit-level diffing possible. This script fills the gap: it loads the
EXISTING tiny model from disk (never regenerates weights), replays the 3
manifest prompts with a manual greedy loop, and saves logits for every
generation step in the 4B oracle's naming convention:

    fixtures/tiny_random/oracle/prompt{N}_logits_step{K}.npy   (float32 [256])

Every greedy token is cross-checked against reference_outputs.json, so a
transformers-version drift that changes the forward pass fails loudly here
instead of poisoning the fixtures.

Run with system python (torch + transformers are system-wide, not in pixi):
    python3 scripts/build_tiny_logits.py
"""

import json
import os

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

TINY_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_random")
OUT_DIR = os.path.join(TINY_DIR, "oracle")


def main() -> None:
    with open(os.path.join(TINY_DIR, "reference_outputs.json")) as f:
        references = json.load(f)

    model = Qwen3ForCausalLM.from_pretrained(TINY_DIR, dtype=torch.bfloat16)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    steps_saved = None

    for ref in references:
        n = ref["prompt_index"] + 1  # oracle naming is 1-based
        input_ids = ref["input_ids"]
        expected = ref["output_ids"][len(input_ids):]
        steps_saved = len(expected)

        ids = torch.tensor([input_ids], dtype=torch.long)
        for k in range(len(expected)):
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            token = int(logits.argmax())
            if token != expected[k]:
                raise SystemExit(
                    f"prompt{n} step{k}: greedy token {token} != reference "
                    f"{expected[k]} — transformers drift, fixtures NOT written"
                )
            np.save(
                os.path.join(OUT_DIR, f"prompt{n}_logits_step{k}.npy"),
                logits.numpy(),
            )
            ids = torch.cat([ids, torch.tensor([[token]], dtype=torch.long)], dim=1)
        print(f"prompt{n}: {len(expected)} steps verified + saved")

    manifest = {
        "source_model": "fixtures/tiny_random/model.safetensors (existing, not regenerated)",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "num_prompts": len(references),
        "logits_steps_saved": steps_saved,
        "logits_dtype": "float32",
        "logits_shape": [256],
        "verified_against": "reference_outputs.json (token-exact, all steps)",
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {OUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
