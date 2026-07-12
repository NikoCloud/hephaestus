#!/usr/bin/env python3
"""PROBE 5 -- is row 67 intrinsically ill-conditioned, or is Hephaestus wrong?

The bisect (probe 4) found NO first bad layer. The row-76 error grows smoothly
from 1.7e-3 at layer 0 to ~0.67 by layer 23 -- roughly 1.3x amplification per
layer. A layer-0 attention output that is 0.17% off (inside bf16 noise, eps =
3.9e-3) becomes a 33% wrong final hidden state.

That is the signature of an ill-conditioned computation, not a broken kernel.
But "ill-conditioned" is a claim, and it has a falsifiable prediction:

  If the amplification is a property of the MODEL at these rows, then perturbing
  the REFERENCE by a bf16-ulp-sized amount must blow up at the SAME rows -- and
  HF's own sdpa-vs-eager spread must already be large there.

  If instead Hephaestus injects a systematically larger error, HF will be stable
  at row 67 under equivalent perturbation, and only Hephaestus will diverge.

This measures, per row, over the full 265-token teacher-forced sequence:
  A. amplification: ||dh_final|| / ||h_final|| when the embeddings are perturbed
     by a relative eps comparable to one bf16 rounding.  (2 eps values x 2 seeds,
     to check the response is linear in eps, i.e. that this is amplification and
     not saturation.)
  B. HF sdpa vs HF eager: the reference's own spread between two implementations
     that are both "correct" and differ only in rounding order.

Then correlates both against the per-row Hephaestus divergence and the 10-token
phase. Prediction if ill-conditioning: phase 7 ("lazy") and phase 3 ("fox") rows
top BOTH lists, and HF's own eager-vs-sdpa spread at row 67 is already huge.

Usage: sh experiments/spike/run_py_gpu.sh experiments/spike/probe5_conditioning.py
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = os.path.join(os.path.dirname(__file__), "out")
PLEN, NSTEPS = 10, 256
TARGET_STEP = 67


def final_hidden(model, ids, perturb=None):
    """Returns model.norm output for every row: [seq, 2560] fp32."""
    grabbed = {}
    h1 = model.model.norm.register_forward_hook(
        lambda m, a, o: grabbed.__setitem__("h", o.detach()[0].float().cpu().numpy()))
    hs = [h1]
    if perturb is not None:
        def emb_hook(m, a, o):
            g = torch.Generator(device=o.device).manual_seed(perturb[1])
            noise = torch.randn(o.shape, generator=g, device=o.device,
                                dtype=torch.float32)
            scale = o.float().abs().mean() * perturb[0]
            return (o.float() + noise * scale).to(o.dtype)
        hs.append(model.model.embed_tokens.register_forward_hook(emb_hook))
    with torch.no_grad():
        model(torch.tensor([ids], device="cuda:0"))
    for h in hs:
        h.remove()
    return grabbed["h"]


def relrow(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open("fixtures/oracle/prompt1_input_ids.json") as f:
        prompt = json.load(f)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)
    ids = prompt + gen[:255]  # 265 tokens; row PLEN-1+k predicts gen[k]

    def rows(h):  # [265,2560] -> [256,2560], one per teacher-forced step
        return h[PLEN - 1: PLEN - 1 + NSTEPS]

    print("loading sdpa ...")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m.eval().to("cuda:0")
    print(f"  attn_implementation = {m.config._attn_implementation}")
    H0 = rows(final_hidden(m, ids))

    # --- A. amplification under a bf16-ulp-sized input perturbation ---------
    amp = {}
    for eps in (1e-3, 4e-3):
        accs = []
        for seed in (0, 1):
            Hp = rows(final_hidden(m, ids, perturb=(eps, seed)))
            accs.append(relrow(Hp, H0))
        amp[eps] = np.mean(accs, axis=0)
        print(f"  eps={eps:g}: mean rel response {amp[eps].mean():.4f}  "
              f"max {amp[eps].max():.4f} (row {int(amp[eps].argmax())})")

    # response should be ~linear in eps if this is amplification, not saturation
    ratio = float(np.median(amp[4e-3] / np.maximum(amp[1e-3], 1e-12)))
    print(f"  median response ratio eps 4e-3 / 1e-3 = {ratio:.2f} "
          f"(4.0 == perfectly linear amplification)")
    A = amp[1e-3] / 1e-3  # amplification factor per unit relative input error

    del m
    torch.cuda.empty_cache()

    # --- B. the reference's own spread: sdpa vs eager ------------------------
    print("loading eager ...")
    me = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager")
    me.eval().to("cuda:0")
    print(f"  attn_implementation = {me.config._attn_implementation}")
    HE = rows(final_hidden(me, ids))
    ref_spread = relrow(HE, H0)
    del me
    torch.cuda.empty_cache()

    # --- Hephaestus per-row divergence, from the committed logit dumps -------
    # recovered via the validated least-squares route (probe 1)
    heph_div = None
    p = f"{OUT}/heph_rowdiv.npy"
    if os.path.exists(p):
        heph_div = np.load(p)

    # --- report --------------------------------------------------------------
    phase = np.arange(NSTEPS) % 10
    rep = {"amp_eps1e-3": A.tolist(), "ref_spread_eager_vs_sdpa": ref_spread.tolist(),
           "response_linearity_ratio": ratio}

    print("\n=== per-phase means (row k predicts gen[k], phase = k %% 10) ===")
    print(f"{'phase':>5} {'predicts':>9} {'amplification':>14} {'HF eager-vs-sdpa':>17}")
    for ph in range(10):
        s = phase == ph
        print(f"{ph:>5} {gen[ph]:>9} {A[s].mean():>14.1f} {ref_spread[s].mean():>17.4f}")

    print("\n=== worst 8 rows by intrinsic amplification ===")
    for r in np.argsort(-A)[:8]:
        print(f"  row {r:3d} (phase {r%10}) predicts {gen[r]:6d}  "
              f"amp {A[r]:8.1f}   HF eager-vs-sdpa {ref_spread[r]:.4f}")

    t = TARGET_STEP
    print(f"\n=== TARGET row {t} (phase {t%10}, predicts {gen[t]}) ===")
    print(f"  intrinsic amplification      : {A[t]:.1f}x")
    print(f"  HF eager-vs-sdpa rel spread  : {ref_spread[t]:.4f}")
    print(f"  (Hephaestus-vs-sdpa was      : 0.3345)")
    print(f"  median row amplification     : {np.median(A):.1f}x")
    print(f"  median HF eager-vs-sdpa      : {np.median(ref_spread):.4f}")

    if heph_div is not None:
        c1 = float(np.corrcoef(np.log(A), np.log(heph_div))[0, 1])
        c2 = float(np.corrcoef(np.log(ref_spread), np.log(heph_div))[0, 1])
        print(f"\n  corr(log amplification, log heph divergence) = {c1:.3f}")
        print(f"  corr(log HF-self-spread, log heph divergence) = {c2:.3f}")
        rep["corr_log_amp_vs_heph"] = c1
        rep["corr_log_refspread_vs_heph"] = c2
    c3 = float(np.corrcoef(np.log(A), np.log(ref_spread))[0, 1])
    print(f"  corr(log amplification, log HF-self-spread)   = {c3:.3f}")
    rep["corr_log_amp_vs_refspread"] = c3

    np.save(f"{OUT}/amplification.npy", A)
    np.save(f"{OUT}/ref_spread.npy", ref_spread)
    with open(f"{OUT}/probe5_conditioning.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/probe5_conditioning.json")


if __name__ == "__main__":
    main()
