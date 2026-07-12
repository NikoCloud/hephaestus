#!/usr/bin/env python3
"""Diff Hephaestus logits against an oracle .npy.

Usage: python3 scripts/diff_logits.py <mojo.f32> <oracle.npy>

The Mojo side writes raw float32 [vocab]. Bit-exactness is NOT expected (BF16
math, different accumulation order); argmax match IS required — that is what
G1a-1 turns on.
"""

import sys

import numpy as np


def main() -> None:
    mine = np.fromfile(sys.argv[1], dtype=np.float32)
    oracle = np.load(sys.argv[2]).astype(np.float32)

    if mine.shape != oracle.shape:
        raise SystemExit(f"shape mismatch: {mine.shape} vs {oracle.shape}")

    a_mine, a_oracle = int(mine.argmax()), int(oracle.argmax())
    max_abs = float(np.abs(mine - oracle).max())
    # Rank correlation of the top-k tells you whether it's "close but noisy"
    # or "structurally wrong" when argmax disagrees.
    top_mine = set(np.argsort(-mine)[:5].tolist())
    top_oracle = set(np.argsort(-oracle)[:5].tolist())

    print(f"argmax  mine={a_mine}  oracle={a_oracle}  {'MATCH' if a_mine == a_oracle else 'MISMATCH'}")
    print(f"max_abs_diff = {max_abs:.6f}")
    print(f"top5 overlap = {len(top_mine & top_oracle)}/5")
    print(f"mine[top]   = {mine[a_oracle]:.4f}   oracle[top] = {oracle[a_oracle]:.4f}")

    if a_mine != a_oracle:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
