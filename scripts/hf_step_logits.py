#!/usr/bin/env python3
"""Teacher-forced HF logits at an arbitrary step -- a DIAGNOSTIC, not a fixture.

The committed oracle only saved logits for steps 0-9. When Hephaestus diverges
later than that, this reproduces the reference's logits at the divergent step so
the failure can be classified: a genuine forward-pass bug, or a bf16 near-tie
that the reference itself resolves only by rounding.

Feeds the oracle's own output_ids as history (teacher forcing), so the history
is correct by construction and only the step in question is under test.

Usage: python3 scripts/hf_step_logits.py <prompt 1-3> <step> <out.npy>
"""

import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"


def main() -> None:
    n, step, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    with open(f"fixtures/oracle/prompt{n}_input_ids.json") as f:
        prompt = json.load(f)
    with open(f"fixtures/oracle/prompt{n}_output_ids.json") as f:
        generated = json.load(f)

    # History = prompt + the reference's own first `step` generated tokens.
    ids = prompt + generated[:step]

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    model.to("cuda:0")
    print(f"attn_implementation = {model.config._attn_implementation}")

    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda:0")).logits[0, -1]

    arr = logits.float().cpu().numpy()
    np.save(out, arr)
    want = generated[step]
    print(f"prompt{n} step{step}: HF argmax={int(arr.argmax())}  oracle_token={want}")
    top = np.argsort(-arr)[:4]
    print("  top4:", [(int(i), float(arr[i])) for i in top])


if __name__ == "__main__":
    main()
