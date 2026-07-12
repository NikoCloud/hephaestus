#!/usr/bin/env python3
"""Generate a greedy reference with attn_implementation="eager".

WHY THIS EXISTS (proposal, not yet the gate -- see .agent/notes/oracle-sdpa.md):

The committed oracle (fixtures/oracle/) was generated with transformers'
default attention, which resolves to `sdpa`. SDPA dispatches to whichever
backend torch picks (math / mem-efficient / flash) and each rounds its
intermediates differently -- notably whether softmax probabilities are kept in
fp32 or rounded to bf16 before the PV product. That rounding sequence is not
part of any published contract, so "token-identical to the reference" is
chasing an implementation detail of a kernel we cannot see.

`eager` is the opposite: its arithmetic is fully specified in Python
(modeling_qwen3.eager_attention_forward) -- bf16 QK^T, fp32 softmax cast back
to bf16, bf16 PV. It is matchable by construction and reproducible on any
machine.

Writes fixtures/oracle_eager/promptN_output_ids.json (256 greedy tokens each).

Usage: python3 scripts/build_eager_reference.py
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT_DIR = "fixtures/oracle_eager"
N_NEW = 256


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    model.to("cuda:0")
    assert model.config._attn_implementation == "eager"
    print("attn_implementation = eager")

    manifest = {
        "model": "Qwen3-4B-Instruct-2507",
        "attn_implementation": "eager",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "dtype": "bfloat16",
        "generation": "greedy, no sampling, no EOS early-stop",
        "num_new_tokens": N_NEW,
        "note": "Reference with fully-specified attention arithmetic; see script docstring.",
    }

    for n in (1, 2, 3):
        with open(f"fixtures/oracle/prompt{n}_input_ids.json") as f:
            prompt = json.load(f)

        ids = torch.tensor([prompt], device="cuda:0")
        out = []
        past = None
        cur = ids
        with torch.no_grad():
            for _ in range(N_NEW):
                res = model(cur, past_key_values=past, use_cache=True)
                past = res.past_key_values
                tok = int(res.logits[0, -1].argmax())
                out.append(tok)
                cur = torch.tensor([[tok]], device="cuda:0")

        with open(f"{OUT_DIR}/prompt{n}_output_ids.json", "w") as f:
            json.dump(out, f)
        print(f"prompt{n}: {len(out)} tokens -> {OUT_DIR}/prompt{n}_output_ids.json")
        print(f"  first 8: {out[:8]}")

    with open(f"{OUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
