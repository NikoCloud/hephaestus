#!/usr/bin/env python3
"""PROBE 3 (reference side) -- HF residual stream at the spike row, per layer.

Dumps HF's hidden state at EXACTLY the cut points spike_forward.mojo dumps, so
the two can be diffed layer by layer and the first divergent layer identified.

Slot layout (identical to spike_forward.mojo):
  0              embeddings
  1 + 4i + 0     self_attn output   (o_proj out, pre-residual)
  1 + 4i + 1     x after attention residual add
  1 + 4i + 2     mlp output         (down_proj out, pre-residual)
  1 + 4i + 3     x after FFN residual add  (= layer i output)
  1 + 4*36       final normed hidden (LM-head input)

Runs the exact-prefix input (prompt1 + oracle[:67] = 77 tokens); the target is
the LAST row (index 76), which predicts oracle[67] = 15678 "lazy". Also runs the
full 265-token sequence and checks the two agree, since causal attention says
they must.

Usage: python3 experiments/spike/probe3_hf_hidden.py
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = os.path.join(os.path.dirname(__file__), "out")
STEP = 67
HIDDEN = 2560
NL = 36
NSLOT = 1 + 4 * NL + 1


def run(model, ids, target_row):
    slots = np.zeros((NSLOT, HIDDEN), dtype=np.float32)
    layer_in = {}

    def grab(t):
        return t.detach()[0, target_row].float().cpu().numpy()

    hs = []
    m = model.model

    def pre_hook(i):
        def f(mod, args, kwargs):
            # first positional arg (or kwarg) is hidden_states
            h = args[0] if args else kwargs["hidden_states"]
            layer_in[i] = h.detach()
        return f

    def layer_hook(i):
        def f(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            slots[1 + 4 * i + 3] = grab(o)
        return f

    def attn_hook(i):
        def f(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            slots[1 + 4 * i + 0] = grab(o)
            # x after attention residual = layer input + attn out
            slots[1 + 4 * i + 1] = grab(layer_in[i] + o.detach())
        return f

    def mlp_hook(i):
        def f(mod, args, out):  # registered without with_kwargs
            slots[1 + 4 * i + 2] = grab(out)
        return f

    hs.append(m.embed_tokens.register_forward_hook(
        lambda mod, a, o: slots.__setitem__(0, grab(o))))
    hs.append(m.norm.register_forward_hook(
        lambda mod, a, o: slots.__setitem__(1 + 4 * NL, grab(o))))
    for i, layer in enumerate(m.layers):
        hs.append(layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        hs.append(layer.register_forward_hook(layer_hook(i), with_kwargs=True))
        hs.append(layer.self_attn.register_forward_hook(attn_hook(i), with_kwargs=True))
        hs.append(layer.mlp.register_forward_hook(mlp_hook(i)))

    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda:0")).logits[0, target_row]
    for h in hs:
        h.remove()
    return slots, logits.float().cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    with open("fixtures/oracle/prompt1_input_ids.json") as f:
        prompt = json.load(f)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval().to("cuda:0")
    print(f"attn_implementation = {model.config._attn_implementation}")

    target = len(prompt) - 1 + STEP  # 76
    runs = {
        "prefix": prompt + gen[:STEP],          # 77 tokens
        "full": prompt + gen[:255],             # 265 tokens
    }
    out = {}
    for name, ids in runs.items():
        slots, lg = run(model, ids, target)
        np.save(f"{OUT}/hf_{name}_hidden.npy", slots)
        np.save(f"{OUT}/hf_{name}_logits.npy", lg)
        out[name] = (slots, lg)
        print(f"{name}: seq={len(ids)} target_row={target} "
              f"argmax={int(lg.argmax())} (oracle={gen[STEP]}) "
              f"logit[96874]={lg[96874]:.6f}")

    a, b = out["prefix"][0], out["full"][0]
    la, lb = out["prefix"][1], out["full"][1]
    print(f"\nHF prefix vs full: hidden max|diff| = {np.abs(a - b).max():.3e}, "
          f"logits max|diff| = {np.abs(la - lb).max():.3e}")
    print("  (causal attention says these must agree; 0.0 = bit-identical)")

    fin = a[1 + 4 * NL]
    print(f"\nHF final normed hidden (row {target}): "
          f"norm={np.linalg.norm(fin):.3f}")
    top = np.argsort(-np.abs(fin))[:6]
    print("  top dims:", [(int(i), round(float(fin[i]), 2)) for i in top])
    x = a[1 + 4 * (NL - 1) + 3]  # last layer output, pre final-norm
    print(f"HF pre-final-norm residual: norm={np.linalg.norm(x):.1f} "
          f"rms={float(np.sqrt((x.astype(np.float64)**2).mean())):.3f}")
    top = np.argsort(-np.abs(x))[:6]
    print("  top dims:", [(int(i), round(float(x[i]), 1)) for i in top])
    print(f"\nwrote {OUT}/hf_*_hidden.npy")


if __name__ == "__main__":
    main()
