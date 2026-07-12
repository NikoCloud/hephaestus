#!/usr/bin/env python3
"""PROBE 15 -- q_proj accumulation: Hephaestus vs HF at layer 0.

Context (post-RoPE-fix): inject_q_proj collapses the spike (16.38→4.71) while
post-q_proj dumps already agree at rel ~1e-4 (only ~110 / 315392 bf16 elements
differ). This probe asks:

  1. Is the prefill path gemv or matmul_kernel_naive? (m=77 → naive, not gemv)
  2. Do the sparse diffs come from f32 vs bf16 accumulation?
  3. Do the sparse differing elements alone cause the spike (hybrid inject)?
  4. Does input_layernorm (xn) already differ?

Prefill uses matmul_kernel_naive with get_accum_type[bf16]=f32 (cast A/B to f32,
sum over K, cast out to bf16). gemv_gpu is m==1 only — same f32 accum pattern.

Usage:
  # after /tmp/spike_p15 dump wrote heph xn/q_proj:
  sh experiments/spike/run_py_gpu.sh experiments/spike/probe15_qproj_accum.py
  # hybrid inject uses probe13 binary:
  #   /tmp/spike_q_cuts_fixed inject_q_proj ... /tmp/p15_hybrid_q_proj.f32
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = Path(__file__).resolve().parent / "out"
TMP = Path("/tmp")
STEP = 67
TARGET = 76  # plen-1+67 with plen=10
HIDDEN = 2560
Q_OUT = 4096
SEQ = 77
TARGET_TOK = 96874


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def to_bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def matmul_f32_accum(x_bf16: np.ndarray, w_bf16: np.ndarray) -> np.ndarray:
    """C[m,n] = x @ W^T with f32 accum — matches matmul_kernel_naive / gemv_gpu.

    x: [m,k] bf16-as-f32, w: [n,k] bf16-as-f32 (safetensors / nn.Linear layout).
    """
    x = to_bf16(x_bf16).astype(np.float32)
    w = to_bf16(w_bf16).astype(np.float32)
    # f32 matmul then cast each output to bf16
    out = x @ w.T
    return to_bf16(out)


def matmul_bf16_step_accum(x_bf16: np.ndarray, w_bf16: np.ndarray) -> np.ndarray:
    """Naive bf16 stepwise accum: round partial sum to bf16 every add (slow, reference).

    For each output element: acc_bf16 = 0; for i: acc = bf16(bf16(acc) + bf16(x_i*w_i))
    """
    x = to_bf16(x_bf16)
    w = to_bf16(w_bf16)
    m, k = x.shape
    n = w.shape[0]
    out = np.zeros((m, n), dtype=np.float32)
    # Vectorized per-row still slow for full; do target row only for diagnosis,
    # full tensor via blocked f32 then not this path for all.
    # Full seq with pure python is too slow; use torch bf16 matmul if available
    # as a proxy for "hardware bf16 pathway" (not exact stepwise).
    xt = torch.from_numpy(x).to(torch.bfloat16)
    wt = torch.from_numpy(w).to(torch.bfloat16)
    # torch bf16 matmul on CPU may promote; force:
    with torch.no_grad():
        # explicit: compute in chunks of k with bf16 round of accumulator periodically
        acc = torch.zeros(m, n, dtype=torch.float32)
        chunk = 64
        for i0 in range(0, k, chunk):
            i1 = min(k, i0 + chunk)
            # f32 sum over chunk then add to acc as bf16
            partial = (
                xt[:, i0:i1].float() @ wt[:, i0:i1].float().T
            )  # f32
            acc = (
                torch.from_numpy(to_bf16(acc.numpy()))
                + torch.from_numpy(to_bf16(partial.numpy()))
            ).float()
            acc = torch.from_numpy(to_bf16(acc.numpy()))
    return acc.numpy()


def matmul_bf16_step_target_row(
    x_row: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """True per-element bf16 stepwise accum for one row [k] @ [n,k]^T → [n]."""
    x = to_bf16(x_row)  # [k]
    w = to_bf16(w)  # [n,k]
    n, k = w.shape
    out = np.zeros(n, dtype=np.float32)
    for j in range(n):
        acc = np.float32(0.0)
        for i in range(k):
            prod = to_bf16(np.array([x[i] * w[j, i]], np.float32))[0]
            acc = to_bf16(np.array([acc + prod], np.float32))[0]
        out[j] = acc
    return out


def load_ids():
    with open("fixtures/oracle/prompt1_input_ids.json") as f:
        prompt = json.load(f)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)
    return prompt + gen[:STEP]


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = load_ids()
    assert len(ids) == SEQ

    heph_qp_path = TMP / "p15_dump_q_proj.f32"
    heph_xn_path = TMP / "p15_dump_xn.f32"
    if not heph_qp_path.exists():
        # fall back to probe13 fixed dump (q_proj only)
        heph_qp_path = TMP / "p15_dump_q_proj.f32"
        if not heph_qp_path.exists():
            heph_qp_path = TMP / "spike13_fixed_dump_q_proj.f32"
    if not heph_qp_path.exists():
        raise SystemExit("missing heph q_proj dump; run probe15_qproj_accum.mojo dump first")

    heph_qp = np.fromfile(heph_qp_path, dtype=np.float32).reshape(SEQ, Q_OUT)
    heph_xn = None
    if heph_xn_path.exists():
        heph_xn = np.fromfile(heph_xn_path, dtype=np.float32).reshape(SEQ, HIDDEN)

    # --- HF dumps ---
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval().to("cuda:0")
    try:
        model.set_attn_implementation("sdpa")
    except Exception:
        pass

    layer0 = model.model.layers[0]
    caps = {}

    def pre_hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        caps["x"] = h.detach()

    def grab_xn_q(mod, args, kwargs, out):
        # after full attention we'd have moved on; use explicit forward below
        pass

    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda:0"))
        xn = layer0.input_layernorm(emb)
        q = layer0.self_attn.q_proj(xn)
        caps["xn"] = xn[0].float().cpu().numpy()  # [seq, hidden]
        caps["q"] = q[0].float().cpu().numpy()  # [seq, q_out]
        # weight
        w = layer0.self_attn.q_proj.weight.detach().float().cpu().numpy()  # [q_out, hidden]

    hf_xn = caps["xn"]
    hf_qp = caps["q"]
    hf_w = w

    # Save HF artifacts
    hf_xn.astype(np.float32).tofile(OUT / "hf_sdpa_l0_xn.f32")
    hf_qp.astype(np.float32).tofile(OUT / "hf_sdpa_q_proj.f32")
    hf_qp.astype(np.float32).tofile(TMP / "spike13_hf_sdpa_q_proj.f32")  # keep inject path warm
    hf_xn.astype(np.float32).tofile(TMP / "p15_hf_xn.f32")
    hf_w.astype(np.float32).tofile(TMP / "p15_hf_q_proj_w.f32")

    report = {
        "path_note": {
            "prefill_m": SEQ,
            "linear_route": "matmul_kernel_naive (m>1)",
            "gemv_route": "gemv_gpu only when m==1 (decode) — NOT this anomaly path",
            "naive_accum": "get_accum_type[bf16] → f32: sum_i f32(a_i)*f32(b_i), cast out to bf16",
        },
        "dumps": {},
        "recompute": {},
        "sparse_diffs": {},
        "predictions": {},
        "verdict": {},
    }

    report["dumps"] = {
        "heph_qp_vs_hf_qp": {
            "all_rel": rel_l2(heph_qp, hf_qp),
            "all_max_abs": max_abs(heph_qp, hf_qp),
            "tgt_rel": rel_l2(heph_qp[TARGET], hf_qp[TARGET]),
            "tgt_max_abs": max_abs(heph_qp[TARGET], hf_qp[TARGET]),
            "n_bf16_diff": int(np.sum(to_bf16(heph_qp) != to_bf16(hf_qp))),
            "n_total": int(heph_qp.size),
        }
    }
    if heph_xn is not None:
        report["dumps"]["heph_xn_vs_hf_xn"] = {
            "all_rel": rel_l2(heph_xn, hf_xn),
            "all_max_abs": max_abs(heph_xn, hf_xn),
            "tgt_rel": rel_l2(heph_xn[TARGET], hf_xn[TARGET]),
            "n_bf16_diff": int(np.sum(to_bf16(heph_xn) != to_bf16(hf_xn))),
        }

    # Recompute f32-accum from HF xn and weights
    ref_f32 = matmul_f32_accum(hf_xn, hf_w)
    report["recompute"]["f32_accum_on_hf_xn_vs_hf_qp"] = {
        "all_rel": rel_l2(ref_f32, hf_qp),
        "all_max_abs": max_abs(ref_f32, hf_qp),
        "tgt_rel": rel_l2(ref_f32[TARGET], hf_qp[TARGET]),
    }
    report["recompute"]["f32_accum_on_hf_xn_vs_heph_qp"] = {
        "all_rel": rel_l2(ref_f32, heph_qp),
        "all_max_abs": max_abs(ref_f32, heph_qp),
        "tgt_rel": rel_l2(ref_f32[TARGET], heph_qp[TARGET]),
    }

    if heph_xn is not None:
        ref_f32_heph_x = matmul_f32_accum(heph_xn, hf_w)
        report["recompute"]["f32_accum_on_heph_xn_vs_heph_qp"] = {
            "all_rel": rel_l2(ref_f32_heph_x, heph_qp),
            "all_max_abs": max_abs(ref_f32_heph_x, heph_qp),
            "tgt_rel": rel_l2(ref_f32_heph_x[TARGET], heph_qp[TARGET]),
        }
        report["recompute"]["f32_accum_on_heph_xn_vs_hf_qp"] = {
            "all_rel": rel_l2(ref_f32_heph_x, hf_qp),
            "tgt_rel": rel_l2(ref_f32_heph_x[TARGET], hf_qp[TARGET]),
        }

    # Chunked bf16-round accum (proxy) on HF xn
    ref_bf16_chunk = matmul_bf16_step_accum(hf_xn, hf_w)
    report["recompute"]["bf16_chunk64_accum_on_hf_xn_vs_hf_qp"] = {
        "all_rel": rel_l2(ref_bf16_chunk, hf_qp),
        "all_max_abs": max_abs(ref_bf16_chunk, hf_qp),
    }
    report["recompute"]["bf16_chunk64_accum_on_hf_xn_vs_heph_qp"] = {
        "all_rel": rel_l2(ref_bf16_chunk, heph_qp),
        "all_max_abs": max_abs(ref_bf16_chunk, heph_qp),
    }

    # True stepwise bf16 on TARGET row only (slow but exact)
    print("computing true bf16 stepwise accum for target row (k=2560, n=4096)...")
    step_tgt = matmul_bf16_step_target_row(hf_xn[TARGET], hf_w)
    report["recompute"]["bf16_stepwise_target_row"] = {
        "vs_hf_tgt_rel": rel_l2(step_tgt, hf_qp[TARGET]),
        "vs_hf_tgt_max_abs": max_abs(step_tgt, hf_qp[TARGET]),
        "vs_heph_tgt_rel": rel_l2(step_tgt, heph_qp[TARGET]),
        "vs_heph_tgt_max_abs": max_abs(step_tgt, heph_qp[TARGET]),
        "vs_f32_tgt_rel": rel_l2(step_tgt, ref_f32[TARGET]),
    }

    # Sparse diff map
    hb = to_bf16(heph_qp)
    fb = to_bf16(hf_qp)
    diff_mask = hb != fb
    n_diff = int(diff_mask.sum())
    flat_idx = np.flatnonzero(diff_mask.ravel())
    report["sparse_diffs"] = {
        "n_diff": n_diff,
        "frac": n_diff / hb.size,
        "max_abs": float(np.max(np.abs(hb - fb))),
        "mean_abs_at_diff": float(np.mean(np.abs(hb - fb)[diff_mask])) if n_diff else 0.0,
        "n_diff_on_target_row": int(diff_mask[TARGET].sum()),
        "target_diff_cols": [int(c) for c in np.flatnonzero(diff_mask[TARGET])[:32]],
    }

    # Hybrid: heph q_proj with differing elements replaced by HF → for inject test
    hybrid = hb.copy()
    hybrid[diff_mask] = fb[diff_mask]
    hybrid.astype(np.float32).tofile(TMP / "p15_hybrid_q_proj.f32")
    # Also pure HF and pure heph for convenience
    fb.astype(np.float32).tofile(TMP / "p15_hf_qp_bf16.f32")
    hb.astype(np.float32).tofile(TMP / "p15_heph_qp_bf16.f32")
    report["sparse_diffs"]["hybrid_path"] = str(TMP / "p15_hybrid_q_proj.f32")
    report["sparse_diffs"]["hybrid_equals_hf"] = bool(np.all(hybrid == fb))
    report["sparse_diffs"]["hybrid_vs_heph_n_diff"] = int(np.sum(hybrid != hb))

    # Predictions
    f32_vs_hf = report["recompute"]["f32_accum_on_hf_xn_vs_hf_qp"]["all_rel"]
    f32_vs_heph = report["recompute"]["f32_accum_on_hf_xn_vs_heph_qp"]["all_rel"]
    step_vs_hf = report["recompute"]["bf16_stepwise_target_row"]["vs_hf_tgt_rel"]
    step_vs_heph = report["recompute"]["bf16_stepwise_target_row"]["vs_heph_tgt_rel"]

    report["predictions"] = {
        "heph_matches_f32_accum_better_than_hf": f32_vs_heph < f32_vs_hf,
        "hf_matches_f32_accum": f32_vs_hf < 1e-4,
        "bf16_stepwise_closer_to_hf_than_heph": step_vs_hf < step_vs_heph,
        "bf16_stepwise_explains_gap": (
            step_vs_heph > 3 * step_vs_hf and step_vs_hf < 1e-3
        ),
    }

    # Verdict classification
    if heph_xn is not None and report["dumps"]["heph_xn_vs_hf_xn"]["all_rel"] > 1e-3:
        root = "input_layernorm_or_upstream"
        hint = "xn already diverges; not q_proj accum"
    elif f32_vs_hf < 1e-5 and f32_vs_heph < 1e-5:
        root = "q_proj_already_matches_f32_ref"
        hint = (
            "Both dumps match f32-accum recompute; residual sparse diffs are "
            "ordinary bf16 noise. Spike sensitivity to inject_q_proj is then "
            "ill-conditioned amplification of ~100 ULP-level elements, OR "
            "inject side-effects — run hybrid inject to discriminate."
        )
    elif step_vs_hf < step_vs_heph and step_vs_heph > 1e-3:
        root = "heph_is_f32_accum_hf_closer_to_bf16_step"
        hint = (
            "Force bf16 accumulation in matmul_kernel_naive / gemv inner loop "
            "(cast partial sum to bf16 each step) to match HF if HF uses that — "
            "but verify HF actually uses bf16 accum (usually f32)."
        )
    elif f32_vs_heph < f32_vs_hf:
        root = "heph_is_f32_accum_hf_differs_elsewhere"
        hint = (
            "Hephaestus matches f32-accum reference; HF dump differs for other "
            "reasons (torch kernel, weight packing, xn). Not a simple "
            "force-bf16-accum fix."
        )
    else:
        root = "compound_or_unexplained"
        hint = "See recompute table."

    report["verdict"] = {
        "root_cause_class": root,
        "one_line_fix_hint": hint,
        "note_on_gemv": (
            "Anomaly path is one-shot prefill m=77 → matmul_kernel_naive, not "
            "gemv_gpu. Any accum fix must target the naive kernel (and gemv for "
            "decode parity)."
        ),
    }

    # Control/inject logits if present
    ctrl = TMP / "p15_ctrl_logits.f32"
    inj = TMP / "p15_inj_qp_logits.f32"
    if ctrl.exists() and inj.exists():
        cl = np.fromfile(ctrl, dtype=np.float32)
        il = np.fromfile(inj, dtype=np.float32)
        report["end_to_end"] = {
            "control_logit_96874": float(cl[TARGET_TOK]),
            "inject_q_proj_logit_96874": float(il[TARGET_TOK]),
            "hf_logit_96874": 4.25,
            "control_abs": abs(float(cl[TARGET_TOK]) - 4.25),
            "inject_abs": abs(float(il[TARGET_TOK]) - 4.25),
        }

    out_path = OUT / "probe15_qproj_accum.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(f"hybrid inject file: {TMP / 'p15_hybrid_q_proj.f32'}")


if __name__ == "__main__":
    main()
