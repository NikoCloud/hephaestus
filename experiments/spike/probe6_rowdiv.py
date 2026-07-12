#!/usr/bin/env python3
"""PROBE 6 -- per-row hidden divergence for all 256 teacher-forced steps.

Recovers h from each logit row by least squares against the tied embedding E
(the method probe 1 established and probe 3 VALIDATED: the recovered h_hf for
row 67 matched HF's actual model.norm output exactly -- norm 220.495 vs 220.498,
same top dims). CPU only; no GPU needed.

Produces out/heph_rowdiv.npy: ||h_heph - h_hf|| / ||h_hf|| per row, which probe 7
correlates against the intrinsic amplification measured in probe 5.

Caches the Gram to /tmp/spike_gram.npy (52MB, not committed) -- it takes ~4min
to build and is reused.

Usage: python3 experiments/spike/probe6_rowdiv.py
"""

import os

import numpy as np
import torch  # noqa: F401  (safetensors bf16 needs the torch framework)
from safetensors import safe_open

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
HEPH = "/tmp/spike-det-1783875368/rep1_logits.f32"
HF = "/tmp/hftf1_logits.f32"
OUT = os.path.join(os.path.dirname(__file__), "out")
GRAM = "/tmp/spike_gram.npy"
VOCAB, HIDDEN, NSTEPS = 151936, 2560, 256
CH = 16384


def main():
    os.makedirs(OUT, exist_ok=True)
    with safe_open(f"{MODEL}/model-00001-of-00003.safetensors", framework="pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float().numpy()
    assert E.shape == (VOCAB, HIDDEN)

    if os.path.exists(GRAM):
        G = np.load(GRAM)
        print(f"loaded cached Gram {GRAM}")
    else:
        print("building fp64 Gram (~4min) ...")
        G = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
        for i in range(0, VOCAB, CH):
            Ec = E[i:i + CH].astype(np.float64)
            G += Ec.T @ Ec
        np.save(GRAM, G)
        print(f"cached {GRAM}")

    def recover_all(path):
        """Batched: solve E h_r = y_r for all 256 rows at once."""
        Y = np.fromfile(path, dtype=np.float32).reshape(NSTEPS, VOCAB)
        B = np.zeros((HIDDEN, NSTEPS), dtype=np.float64)
        for i in range(0, VOCAB, CH):
            B += E[i:i + CH].astype(np.float64).T @ Y[:, i:i + CH].T.astype(np.float64)
        return np.linalg.solve(G, B).T  # [256, 2560]

    print("recovering hidden states from logit rows ...")
    Hh = recover_all(HEPH)
    Hf = recover_all(HF)

    div = np.linalg.norm(Hh - Hf, axis=1) / np.linalg.norm(Hf, axis=1)
    np.save(f"{OUT}/heph_rowdiv.npy", div)
    print(f"per-row Hephaestus-vs-HF hidden divergence:")
    print(f"  median {np.median(div):.4f}  mean {div.mean():.4f}  max {div.max():.4f}")
    print(f"  row 67  = {div[67]:.4f}   (bisect measured 0.3345 directly -- cross-check)")
    print(f"  row 202 = {div[202]:.4f}")
    for ph in range(10):
        s = np.arange(NSTEPS) % 10 == ph
        print(f"  phase {ph}: mean {div[s].mean():.4f}")
    print(f"wrote {OUT}/heph_rowdiv.npy")


if __name__ == "__main__":
    main()
