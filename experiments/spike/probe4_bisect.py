#!/usr/bin/env python3
"""PROBE 4 -- layerwise bisect: where does the row-67 hidden state diverge?

Diffs Hephaestus's residual stream against HF's at identical cut points, for the
row that produces the 12.06 logit spike (prompt 1, teacher-forced step 67,
sequence row 76).

Cut points per layer i (slot layout shared with spike_forward.mojo):
  1+4i+0  attn contribution (o_proj out, pre-residual)
  1+4i+1  x after attention residual
  1+4i+2  ffn  contribution (down_proj out, pre-residual)
  1+4i+3  x after FFN residual  (= layer output)

Reading it: the residual stream carries error forward, so *everything* after the
first fault looks wrong. What identifies the culprit is where the error JUMPS --
in particular where a per-layer CONTRIBUTION is far more wrong (relatively) than
the residual feeding into it. A smooth exponential ramp from layer 0 means
ordinary bf16 noise being amplified; a step change at one layer means a bug.

Usage: python3 experiments/spike/probe4_bisect.py
"""

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")
HIDDEN, NL = 2560, 36
NSLOT = 1 + 4 * NL + 1
DIM9 = 9


def load_heph(mode):
    a = np.fromfile(f"/tmp/spike_{mode}_hidden.f32", dtype=np.float32)
    assert a.size == NSLOT * HIDDEN, (mode, a.size)
    return a.reshape(NSLOT, HIDDEN)


def rel(a, b):
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def main():
    rep = {}
    hf = {m: np.load(f"{OUT}/hf_{m}_hidden.npy") for m in ("prefix", "full")}
    heph = {m: load_heph(m) for m in ("full", "prefix", "seq")}

    # ---- Hephaestus internal consistency across the three routes -----------
    print("=== Hephaestus route-vs-route (same row, same math, 3 code paths) ===")
    for a, b in (("full", "prefix"), ("full", "seq")):
        d = np.abs(heph[a] - heph[b])
        print(f"  {a:6s} vs {b:6s}: final-hidden rel {rel(heph[a][-1], heph[b][-1]):.3e}"
              f"   max|diff| over all slots {d.max():.4f}")
    print(f"  HF prefix vs full: rel {rel(hf['prefix'][-1], hf['full'][-1]):.3e}")

    # ---- the bisect --------------------------------------------------------
    for mode in ("prefix", "full"):
        H, R = heph[mode], hf[mode]
        print(f"\n=== BISECT ({mode}) -- rel err ||heph-hf||/||hf|| per cut ===")
        print(f"  embeddings: rel {rel(H[0], R[0]):.3e}   "
              f"max|d| {np.abs(H[0]-R[0]).max():.4f}")
        print(f"{'L':>3} {'attn_out':>10} {'x_post_at':>10} {'ffn_out':>10} "
              f"{'x_post_ff':>10} | {'d9_heph':>9} {'d9_hf':>9} {'d9_rel':>8}")
        rows = []
        for i in range(NL):
            s = 1 + 4 * i
            r_a, r_xa = rel(H[s + 0], R[s + 0]), rel(H[s + 1], R[s + 1])
            r_f, r_xf = rel(H[s + 2], R[s + 2]), rel(H[s + 3], R[s + 3])
            h9, f9 = float(H[s + 3][DIM9]), float(R[s + 3][DIM9])
            d9r = (h9 - f9) / f9 if f9 != 0 else float("nan")
            rows.append(dict(layer=i, attn_out=r_a, x_post_attn=r_xa,
                             ffn_out=r_f, x_post_ffn=r_xf,
                             dim9_heph=h9, dim9_hf=f9, dim9_rel=d9r))
            flag = ""
            prev = rows[i - 1]["x_post_ffn"] if i else rel(H[0], R[0])
            if r_xf > max(3 * prev, 1e-3) and r_xf > 0.01:
                flag = "  <<< JUMP"
            print(f"{i:>3} {r_a:>10.3e} {r_xa:>10.3e} {r_f:>10.3e} {r_xf:>10.3e} |"
                  f" {h9:>9.1f} {f9:>9.1f} {d9r:>7.1%}{flag}")
        fin_h, fin_r = H[1 + 4 * NL], R[1 + 4 * NL]
        print(f"\n  final normed hidden: rel {rel(fin_h, fin_r):.4f}   "
              f"||heph|| {np.linalg.norm(fin_h):.2f}  ||hf|| {np.linalg.norm(fin_r):.2f}")
        print(f"    dim9: heph {fin_h[DIM9]:.2f}  hf {fin_r[DIM9]:.2f}")
        rep[mode] = rows

    # ---- what does the last-layer residual actually look like? -------------
    R = hf["prefix"]
    x = R[1 + 4 * (NL - 1) + 3]
    H = heph["prefix"][1 + 4 * (NL - 1) + 3]
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    print(f"\n=== pre-final-norm residual (HF) at row 76 ===")
    print(f"  rms {rms:.3f}   max|x| {np.abs(x).max():.1f}   "
          f"dim9 {x[DIM9]:.1f} ({x[DIM9]/rms:.1f} sigma)")
    top = np.argsort(-np.abs(x))[:8]
    print("  top dims (hf):  ", [(int(i), round(float(x[i]), 1)) for i in top])
    print("  same dims (heph):", [(int(i), round(float(H[i]), 1)) for i in top])

    with open(f"{OUT}/probe4_bisect.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/probe4_bisect.json")


if __name__ == "__main__":
    main()
