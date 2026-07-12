#!/usr/bin/env python3
"""PROBE 7 -- is Hephaestus INSIDE the bf16 implementation ensemble?

Probe 5 showed the phase-7/phase-3 rows amplify tiny perturbations enormously,
and that HF's own eager-vs-sdpa spread is large exactly there. But eager and sdpa
share their GEMM kernels, weights and layout -- they differ only in the attention
inner loop -- so their spread is a LOWER BOUND on how far two genuinely
independent bf16 implementations can sit apart. It is not a fair yardstick for
Hephaestus, which shares no kernel with HF.

So build the yardstick properly. A different-but-correct bf16 engine differs from
HF by re-rounding the residual stream at every layer in a different order. Model
that directly: inject a fresh random relative perturbation of one bf16 rounding
at every decoder layer output, N seeds -> an ENSEMBLE of plausible alternative
bf16 implementations, all equally "correct".

Then ask the falsifiable question:

    Does Hephaestus's per-row divergence fall INSIDE that ensemble's spread?

  inside  -> Hephaestus is numerically indistinguishable from a legitimate bf16
             implementation; the spike is intrinsic model conditioning, not a bug.
  outside -> Hephaestus injects excess error; a real defect remains to be found.

bf16 has 8 bits of precision (eps = 2^-8 = 3.906e-3). Round-to-nearest gives a
relative error uniform in +-eps/2, so RMS relative error = eps/(2*sqrt(3)) =
1.128e-3. That is ONE rounding of the residual stream -- a deliberately
CONSERVATIVE (small) estimate, since a real engine also re-rounds inside every
matmul, attention and norm. We also run 2x and 4x it to bracket.

Usage: sh experiments/spike/run_py_gpu.sh experiments/spike/probe7_ensemble.py
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = os.path.join(os.path.dirname(__file__), "out")
PLEN, NSTEPS, TARGET = 10, 256, 67
BF16_RMS = 2.0 ** -8 / (2.0 * np.sqrt(3.0))   # 1.128e-3, one bf16 rounding
NSEED = 12


def run(model, ids, eps=0.0, seed=0):
    """Final normed hidden for every row, with per-layer bf16-sized noise."""
    got = {}
    hs = [model.model.norm.register_forward_hook(
        lambda m, a, o: got.__setitem__("h", o.detach()[0].float().cpu().numpy()))]

    if eps > 0:
        gen = torch.Generator(device="cuda:0").manual_seed(seed)

        def noisy(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            f = o.float()
            n = torch.randn(f.shape, generator=gen, device=f.device)
            # relative perturbation, per element, of RMS size eps
            f = f * (1.0 + n * eps)
            r = f.to(o.dtype)
            return (r,) + out[1:] if isinstance(out, tuple) else r

        for layer in model.model.layers:
            hs.append(layer.register_forward_hook(noisy, with_kwargs=True))

    with torch.no_grad():
        model(torch.tensor([ids], device="cuda:0"))
    for h in hs:
        h.remove()
    return got["h"][PLEN - 1: PLEN - 1 + NSTEPS]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open("fixtures/oracle/prompt1_input_ids.json") as f:
        prompt = json.load(f)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)
    ids = prompt + gen[:255]

    heph = np.load(f"{OUT}/heph_rowdiv.npy")  # per-row Hephaestus divergence

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m.eval().to("cuda:0")
    print(f"attn_implementation = {m.config._attn_implementation}")
    H0 = run(m, ids)

    rep = {}
    print(f"\none bf16 rounding of the residual stream = {BF16_RMS:.3e} RMS relative\n")
    for mult in (1.0, 2.0, 4.0):
        eps = BF16_RMS * mult
        E = np.stack([
            np.linalg.norm(run(m, ids, eps, s) - H0, axis=1) / np.linalg.norm(H0, axis=1)
            for s in range(NSEED)
        ])  # [NSEED, 256]
        rep[f"{mult}x"] = dict(
            eps=eps,
            median_row=float(np.median(E)),
            target_mean=float(E[:, TARGET].mean()),
            target_p5=float(np.percentile(E[:, TARGET], 5)),
            target_p95=float(np.percentile(E[:, TARGET], 95)),
            target_max=float(E[:, TARGET].max()),
        )
        t = E[:, TARGET]
        # fraction of rows where Hephaestus sits inside the ensemble's [min,max]
        inside = float(((heph >= E.min(0)) & (heph <= E.max(0))).mean())
        below = float((heph < E.min(0)).mean())
        above = float((heph > E.max(0)).mean())
        rep[f"{mult}x"].update(frac_inside=inside, frac_below=below, frac_above=above)
        print(f"=== per-layer bf16 noise x{mult:g}  (eps={eps:.3e}) ===")
        print(f"  row {TARGET}: ensemble mean {t.mean():.4f}  "
              f"p5..p95 [{np.percentile(t,5):.4f}, {np.percentile(t,95):.4f}]  "
              f"max {t.max():.4f}   |   Hephaestus {heph[TARGET]:.4f}")
        print(f"  all rows : Hephaestus inside ensemble [min,max] on "
              f"{inside:.1%} of rows  (below {below:.1%}, above {above:.1%})")
        # per-phase
        ph_line = []
        for p in range(10):
            s = np.arange(NSTEPS) % 10 == p
            ph_line.append(f"p{p} {E[:, s].mean():.3f}/{heph[s].mean():.3f}")
        print("  per-phase ensemble/heph mean: " + "  ".join(ph_line))
        print()

    # the headline comparison at 1x
    E1 = rep["1.0x"]
    print("=== VERDICT INPUT ===")
    print(f"  Hephaestus row {TARGET} divergence : {heph[TARGET]:.4f}")
    print(f"  bf16 ensemble  row {TARGET} (1x)   : mean {E1['target_mean']:.4f}, "
          f"p5..p95 [{E1['target_p5']:.4f}, {E1['target_p95']:.4f}]")
    print(f"  => Hephaestus is {'INSIDE' if E1['target_p5'] <= heph[TARGET] <= E1['target_p95'] else 'OUTSIDE'}"
          f" the 1x ensemble p5-p95 band at row {TARGET}")

    with open(f"{OUT}/probe7_ensemble.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/probe7_ensemble.json")


if __name__ == "__main__":
    main()
