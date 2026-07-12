#!/usr/bin/env python3
"""PROBE 1 -- Is the Hephaestus logit row even in the column space of E?

Qwen3-4B has tied embeddings and no lm_head bias (verified: the safetensors
index has no lm_head tensor). Therefore EVERY logit row produced by ANY correct
implementation must satisfy

    logits = E @ h        E = model.embed_tokens.weight  [151936, 2560]

for some hidden vector h in R^2560. This is a hard structural constraint, and it
gives a decisive test that needs no GPU run and no instrumentation:

  * Recover h_hf and h_heph from their logit rows by least squares against E.
  * If a logit row is NOT well-approximated by E @ h for any h, the LM head that
    produced it is broken (H1: LM-head corruption).
  * If both rows fit tightly, the LM head is arithmetically consistent and the
    divergence is upstream in the hidden state (H2), and we get delta = h_heph -
    h_hf *exactly*, for free, without touching the GPU.

Also re-derives the per-row phase table so the phase-locking claim is rebuilt
from the raw artifacts rather than inherited.

Method: chunked float64 normal equations (G = E^T E, b = E^T y). E is well
conditioned enough that fp64 normal equations are exact to ~1e-12 here; the
reconstruction residual is reported so this is checked, not assumed.

Usage: python3 experiments/spike/probe1_rowspace.py
"""

import json
import os

import numpy as np
import torch  # noqa: F401  (safetensors bf16 needs the torch framework)
from safetensors import safe_open

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
HEPH = "/tmp/spike-det-1783875368/rep1_logits.f32"
HF = "/tmp/hftf1_logits.f32"
HF_PREFIX_S67 = "/tmp/hf_exact_prefix_p1s67.npy"
OUT = os.path.join(os.path.dirname(__file__), "out")

VOCAB, HIDDEN, NSTEPS = 151936, 2560, 256
TARGET_ROW = 67
TARGET_TOK = 96874


def load_rows(path):
    a = np.fromfile(path, dtype=np.float32)
    assert a.size == NSTEPS * VOCAB, (path, a.size)
    return a.reshape(NSTEPS, VOCAB)


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {}

    heph = load_rows(HEPH)
    hf = load_rows(HF)
    print(f"loaded heph {heph.shape}  hf {hf.shape}")

    # ---- (a) per-row phase table: rebuild the phase-locking evidence --------
    d = np.abs(heph.astype(np.float64) - hf.astype(np.float64))
    med = np.median(d, axis=1)
    mx = d.max(axis=1)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)

    # The 10-token cycle. Row k predicts gen[k]; phase = k % 10.
    phase_med = {p: [] for p in range(10)}
    for k in range(NSTEPS):
        phase_med[k % 10].append(med[k])
    print("\n--- per-phase median-of-median |diff| (row k predicts gen[k]) ---")
    phase_rows = []
    for p in range(10):
        v = np.array(phase_med[p])
        tok = gen[p] if p < len(gen) else -1
        n_bad = int((v > 0.25).sum())
        phase_rows.append(
            dict(phase=p, predicts_token=int(tok), n_rows=len(v),
                 median_of_median=float(np.median(v)), max_median=float(v.max()),
                 n_rows_median_gt_0p25=n_bad)
        )
        print(f"  phase {p}: predicts tok {tok:6d}  median-of-med {np.median(v):.4f}"
              f"  max-med {v.max():.4f}  rows>0.25: {n_bad}/{len(v)}")
    rep["phase_table"] = phase_rows

    order = np.argsort(-med)[:8]
    print("\n--- worst 8 rows by median |diff| ---")
    worst = []
    for r in order:
        worst.append(dict(row=int(r), phase=int(r % 10), predicts=int(gen[r]),
                          median=float(med[r]), max=float(mx[r])))
        print(f"  row {r:3d} (phase {r%10})  predicts {gen[r]:6d}  "
              f"median {med[r]:.4f}  max {mx[r]:.4f}")
    rep["worst_rows"] = worst

    # Pick a clean control row: same phase-family cleanliness, low median.
    ctrl = int(np.argsort(med)[len(med) // 2])  # median-typical row
    print(f"\ncontrol row (typical): {ctrl}  median {med[ctrl]:.4f}  max {mx[ctrl]:.4f}")
    rep["control_row"] = dict(row=ctrl, median=float(med[ctrl]), max=float(mx[ctrl]))

    print(f"\ntarget row {TARGET_ROW}: median {med[TARGET_ROW]:.6f}  "
          f"max {mx[TARGET_ROW]:.6f}")
    print(f"  heph[{TARGET_ROW},{TARGET_TOK}] = {heph[TARGET_ROW, TARGET_TOK]!r}")
    print(f"  hf  [{TARGET_ROW},{TARGET_TOK}] = {hf[TARGET_ROW, TARGET_TOK]!r}")

    # ---- (b) load E and build the fp64 Gram --------------------------------
    print("\nloading embed_tokens ...")
    # safetensors' numpy framework cannot represent bf16; go via torch, which
    # can, then widen to fp32 (bf16 -> fp32 is exact, no rounding).
    with safe_open(f"{MODEL}/model-00001-of-00003.safetensors", framework="pt") as f:
        shape = f.get_slice("model.embed_tokens.weight").get_shape()
        print(f"  model.embed_tokens.weight shape={shape} "
              f"dtype={f.get_slice('model.embed_tokens.weight').get_dtype()}")
        Eb = f.get_tensor("model.embed_tokens.weight")
    assert list(shape) == [VOCAB, HIDDEN]
    assert str(Eb.dtype) == "torch.bfloat16", Eb.dtype
    E = Eb.float().numpy()
    del Eb
    print(f"  E fp32 {E.shape}  |E| rms={float(np.sqrt((E.astype(np.float64)**2).mean())):.5f}")

    print("building fp64 Gram G = E^T E (chunked) ...")
    G = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
    CH = 16384
    for i in range(0, VOCAB, CH):
        Ec = E[i:i + CH].astype(np.float64)
        G += Ec.T @ Ec
    cond = float(np.linalg.cond(G))
    print(f"  cond(G) = {cond:.3e}   (fp64 has ~1e16 headroom)")
    rep["cond_gram"] = cond

    def recover(y):
        """h = argmin ||E h - y||; returns h, relative residual."""
        b = np.zeros(HIDDEN, dtype=np.float64)
        for i in range(0, VOCAB, CH):
            b += E[i:i + CH].astype(np.float64).T @ y[i:i + CH]
        h = np.linalg.solve(G, b)
        # reconstruct and measure residual
        resid = np.zeros(VOCAB, dtype=np.float64)
        for i in range(0, VOCAB, CH):
            resid[i:i + CH] = E[i:i + CH].astype(np.float64) @ h - y[i:i + CH]
        rel = float(np.linalg.norm(resid) / np.linalg.norm(y))
        return h, rel, resid

    # ---- (c) the decisive test on the target row ---------------------------
    results = {}
    for name, row in [("target", TARGET_ROW), ("control", ctrl)]:
        yh = heph[row].astype(np.float64)
        yf = hf[row].astype(np.float64)
        h_heph, rel_heph, res_heph = recover(yh)
        h_hf, rel_hf, res_hf = recover(yf)
        delta = h_heph - h_hf

        # H3: is it a pure scale?  h_heph ~= a * h_hf
        a = float(delta @ h_hf / (h_hf @ h_hf) + 1.0)
        scale_resid = h_heph - a * h_hf
        frac_scale = 1.0 - float(np.linalg.norm(scale_resid) / np.linalg.norm(delta))

        e = results[name] = dict(
            row=row,
            rel_resid_heph=rel_heph,
            rel_resid_hf=rel_hf,
            norm_h_hf=float(np.linalg.norm(h_hf)),
            norm_h_heph=float(np.linalg.norm(h_heph)),
            norm_delta=float(np.linalg.norm(delta)),
            delta_over_h=float(np.linalg.norm(delta) / np.linalg.norm(h_hf)),
            best_scale_a=a,
            frac_of_delta_explained_by_scale=frac_scale,
            delta_top_dims=[[int(i), float(delta[i])]
                            for i in np.argsort(-np.abs(delta))[:10]],
            h_hf_top_dims=[[int(i), float(h_hf[i])]
                           for i in np.argsort(-np.abs(h_hf))[:10]],
        )
        print(f"\n=== {name.upper()} row {row} ===")
        print(f"  rel residual ||E h - y||/||y||:  heph {rel_heph:.3e}   hf {rel_hf:.3e}")
        print(f"  ||h_hf|| {e['norm_h_hf']:.4f}   ||h_heph|| {e['norm_h_heph']:.4f}"
              f"   ||delta|| {e['norm_delta']:.4f}   ratio {e['delta_over_h']:.4%}")
        print(f"  best pure-scale a = {a:.6f}; scale explains "
              f"{frac_scale:.1%} of ||delta||")
        print(f"  h_hf   top dims: {[(i, round(v,2)) for i, v in e['h_hf_top_dims'][:5]]}")
        print(f"  delta  top dims: {[(i, round(v,3)) for i, v in e['delta_top_dims'][:5]]}")

        if name == "target":
            # does E @ delta reproduce the observed 12.06 spike at 96874?
            pred = float(E[TARGET_TOK].astype(np.float64) @ delta)
            obs = float(yh[TARGET_TOK] - yf[TARGET_TOK])
            print(f"  spike check tok {TARGET_TOK}: E@delta = {pred:.6f}  "
                  f"observed diff = {obs:.6f}")
            e["spike_pred"] = pred
            e["spike_obs"] = obs

    rep["rowspace"] = results

    # ---- (d) cross-check the HF exact-prefix row is the same row -----------
    if os.path.exists(HF_PREFIX_S67):
        pre = np.load(HF_PREFIX_S67).astype(np.float64)
        dd = np.abs(pre - hf[TARGET_ROW].astype(np.float64))
        print(f"\nHF exact-prefix vs HF full row {TARGET_ROW}: max {dd.max():.3e} "
              f"(0.0 == bit-identical, confirms row alignment)")
        rep["hf_prefix_vs_full_max"] = float(dd.max())

    with open(f"{OUT}/probe1_rowspace.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/probe1_rowspace.json")


if __name__ == "__main__":
    main()
