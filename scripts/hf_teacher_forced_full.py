#!/usr/bin/env python3
"""HF logits for all 256 teacher-forced steps, ONE forward call per prompt.

Mirrors src/qwen_teacher_forced_full.mojo exactly: input = prompt +
oracle_output[:255], single forward pass (sdpa, the committed oracle's
implementation), rows (prompt_len-1) .. (prompt_len-1+255) are the logits
that predict oracle_output[0..255].

Writes <out_prefix>_logits.f32 : 256 x VOCAB_SIZE float32, row k = step k.

Usage: python3 scripts/hf_teacher_forced_full.py <prompt 1-3> <out_prefix>
"""

import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"


def main() -> None:
    n, out_prefix = sys.argv[1], sys.argv[2]

    with open(f"fixtures/oracle/prompt{n}_input_ids.json") as f:
        prompt = json.load(f)
    with open(f"fixtures/oracle/prompt{n}_output_ids.json") as f:
        oracle = json.load(f)
    prompt_len = len(prompt)

    full = prompt + oracle[:255]

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    model.to("cuda:0")
    assert model.config._attn_implementation == "sdpa"

    with torch.no_grad():
        logits = model(torch.tensor([full], device="cuda:0")).logits[0]

    rows = logits[prompt_len - 1 : prompt_len - 1 + 256].float().cpu().numpy()
    assert rows.shape == (256, logits.shape[-1])
    rows.astype(np.float32).tofile(f"{out_prefix}_logits.f32")

    argmax = rows.argmax(axis=1)
    mismatches = int((argmax != np.array(oracle)).sum())
    print(f"prompt{n}: HF self-check {256-mismatches}/256 (should be 256/256: same model, same tokens)")
    print(f"wrote {out_prefix}_logits.f32  shape={rows.shape}")


if __name__ == "__main__":
    main()
