#!/usr/bin/env python3
"""PROBE 13 (HF side) -- layer-0 Q cut points + analyze replace-and-continue.

Dumps Q after q_proj, q_norm, and RoPE for the exact-prefix sequence, under
both SDPA and eager (pre-attention Q path should be bit-identical). Writes:

  out/hf_sdpa_q_proj.f32  etc.   [seq, heads, head_dim] float32
  out/hf_eager_q_*.f32
  out/hf_sdpa_hidden.f32         final normed hidden, target row
  out/probe13_q_cuts.json        comparison + intervention analysis

Also consumes Hephaestus dumps produced by probe13_q_cuts.mojo:
  /tmp/spike13_dump_q_{proj,norm,rope}.f32
  /tmp/spike13_{control,inject_q_proj,inject_q_norm,inject_q_rope}_{hidden,logits}.f32

Usage:
  sh experiments/spike/run_py_gpu.sh experiments/spike/probe13_hf_q_cuts.py
  # or after Mojo: python3 experiments/spike/probe13_hf_q_cuts.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.models.qwen3 import modeling_qwen3 as mq

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = Path(__file__).resolve().parent / "out"
TMP = Path("/tmp")
STEP = 67
TARGET_TOK = 96874
HEAD_DIM = 128
N_HEADS = 32
Q_OUT = N_HEADS * HEAD_DIM  # 4096
HIDDEN = 2560


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def load_ids():
    with open("fixtures/oracle/prompt1_input_ids.json") as f:
        prompt = json.load(f)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)
    return prompt + gen[:STEP]


def dump_hf_cuts(impl: str, ids: list[int], target: int) -> dict:
    """Return and write Q cut tensors + final hidden for one attn implementation."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval().to("cuda:0")
    try:
        model.set_attn_implementation(impl)
    except Exception:
        model.config._attn_implementation = impl

    cuts: dict[str, np.ndarray] = {}
    orig = mq.Qwen3Attention.forward

    def wrapped(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_proj = self.q_proj(hidden_states).view(hidden_shape)  # [bs,seq,h,d]
        k_proj = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)

        q_normed = self.q_norm(q_proj)
        k_normed = self.k_norm(k_proj)

        cos, sin = position_embeddings
        # apply_rotary expects [bs, heads, seq, d]
        q_t = q_normed.transpose(1, 2)
        k_t = k_normed.transpose(1, 2)
        q_rope, k_rope = apply_rotary_pos_emb(q_t, k_t, cos, sin)

        if self.layer_idx == 0:
            # Store as [seq, heads, dim] to match Hephaestus acts.q layout.
            cuts["q_proj"] = (
                q_proj[0].detach().float().cpu().numpy()
            )  # [seq,h,d]
            cuts["q_norm"] = q_normed[0].detach().float().cpu().numpy()
            cuts["q_rope"] = (
                q_rope[0].transpose(0, 1).detach().float().cpu().numpy()
            )  # [seq,h,d] from [h,seq,d]

        # Continue with original path using already-computed tensors would
        # skip recomputation inconsistently; call original for correctness.
        return orig(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    mq.Qwen3Attention.forward = wrapped
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda:0"))
        logits = out.logits[0, target].float().cpu().numpy()
        # final normed hidden via hook on model.model.norm
        hidden_cap = {}

        def nh(mod, args, o):
            hidden_cap["h"] = o.detach()

        handle = model.model.norm.register_forward_hook(nh)
        _ = model(torch.tensor([ids], device="cuda:0"))
        handle.remove()
        h_final = hidden_cap["h"][0, target].float().cpu().numpy()
    mq.Qwen3Attention.forward = orig

    prefix = f"hf_{impl}"
    for name, arr in cuts.items():
        flat = arr.reshape(-1).astype(np.float32)
        path = OUT / f"{prefix}_{name}.f32"
        flat.tofile(path)
        # also under /tmp for Mojo inject convenience
        flat.tofile(TMP / f"spike13_{prefix}_{name}.f32")
        print(f"wrote {path} shape={arr.shape}")

    h_final.astype(np.float32).tofile(OUT / f"{prefix}_hidden.f32")
    h_final.astype(np.float32).tofile(TMP / f"spike13_{prefix}_hidden.f32")
    logits.astype(np.float32).tofile(TMP / f"spike13_{prefix}_logits.f32")

    del model
    torch.cuda.empty_cache()
    return {
        "cuts": {k: v for k, v in cuts.items()},
        "hidden": h_final,
        "logits": logits,
        "argmax": int(np.argmax(logits)),
        "target_logit": float(logits[TARGET_TOK]),
    }


def compare_cuts(heph: dict[str, np.ndarray], hf: dict[str, np.ndarray], tag: str):
    rows = {}
    for cut in ("q_proj", "q_norm", "q_rope"):
        a, b = heph[cut], hf[cut]
        assert a.shape == b.shape, (cut, a.shape, b.shape)
        # full tensor + target-row only
        tgt = a.shape[0] - 1
        rows[cut] = {
            "all_rel": rel_l2(a, b),
            "all_max_abs": max_abs(a, b),
            "tgt_rel": rel_l2(a[tgt], b[tgt]),
            "tgt_max_abs": max_abs(a[tgt], b[tgt]),
            "hf_ref": tag,
        }
    return rows


def load_heph_cut(path: Path, seq: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    assert a.size == seq * Q_OUT, (path, a.size)
    return a.reshape(seq, N_HEADS, HEAD_DIM)


def analyze_interventions(seq: int, hf_hidden: np.ndarray, hf_logits: np.ndarray):
    """Score each replace-and-continue run against HF final hidden/logits."""
    modes = [
        "control",
        "inject_q_proj",
        "inject_q_norm",
        "inject_q_rope",
    ]
    # Also accept dump-mode hidden as alias of control if present
    results = {}
    base_div = None
    for m in modes:
        hp = TMP / f"spike13_{m}_hidden.f32"
        lp = TMP / f"spike13_{m}_logits.f32"
        if not hp.exists() or not lp.exists():
            results[m] = {"status": "missing", "paths": [str(hp), str(lp)]}
            continue
        h = np.fromfile(hp, dtype=np.float32)
        lg = np.fromfile(lp, dtype=np.float32)
        assert h.size == HIDDEN, h.size
        assert lg.size == 151936, lg.size
        div = rel_l2(h, hf_hidden)
        if m == "control":
            base_div = div
        # bf16-style argmax via high-16-bit trunc
        bits = lg.view(np.uint32)
        rb = (bits & np.uint32(0xFFFF0000)).view(np.float32)
        am = int(np.argmax(rb))
        results[m] = {
            "hidden_rel_vs_hf": div,
            "hidden_max_abs_vs_hf": max_abs(h, hf_hidden),
            "logit_target": float(lg[TARGET_TOK]),
            "logit_target_hf": float(hf_logits[TARGET_TOK]),
            "abs_diff_target": float(abs(lg[TARGET_TOK] - hf_logits[TARGET_TOK])),
            "argmax": am,
            "argmax_hf": int(np.argmax(hf_logits)),
            "argmax_match_hf": am == int(np.argmax(hf_logits)),
            "collapse_ratio": None,  # filled below
        }
    if base_div is not None and base_div > 0:
        for m, r in results.items():
            if "hidden_rel_vs_hf" in r:
                r["collapse_ratio"] = float(r["hidden_rel_vs_hf"] / base_div)
                r["collapsed"] = bool(r["hidden_rel_vs_hf"] < 0.25 * base_div)
    return results, base_div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip HF GPU dumps; only compare existing /tmp artifacts",
    )
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = load_ids()
    seq = len(ids)
    target = seq - 1
    assert target == 76, target

    report: dict = {
        "input": {
            "seq_len": seq,
            "target_row": target,
            "step": STEP,
            "token": TARGET_TOK,
        },
        "predictions": {
            "error_appears_at_q_proj": None,
            "error_appears_at_q_norm": None,
            "error_appears_at_rope": None,
            "inject_q_proj_collapses": None,
            "inject_q_norm_collapses": None,
            "inject_q_rope_collapses": None,
        },
    }

    if not args.analyze_only:
        print("=== HF SDPA Q cuts ===")
        sdpa = dump_hf_cuts("sdpa", ids, target)
        print("=== HF eager Q cuts ===")
        eager = dump_hf_cuts("eager", ids, target)

        # SDPA vs eager Q path should match closely (shared pre-attn)
        report["hf_sdpa_vs_eager_q"] = {
            cut: {
                "all_rel": rel_l2(sdpa["cuts"][cut], eager["cuts"][cut]),
                "tgt_rel": rel_l2(
                    sdpa["cuts"][cut][target], eager["cuts"][cut][target]
                ),
            }
            for cut in ("q_proj", "q_norm", "q_rope")
        }
        report["hf_sdpa_hidden_norm"] = float(np.linalg.norm(sdpa["hidden"]))
        report["hf_values"] = {
            "sdpa_target_logit": sdpa["target_logit"],
            "eager_target_logit": eager["target_logit"],
            "sdpa_argmax": sdpa["argmax"],
            "eager_argmax": eager["argmax"],
        }
        # stash for analyze
        np.save(TMP / "spike13_hf_sdpa_hidden.npy", sdpa["hidden"])
        np.save(TMP / "spike13_hf_sdpa_logits.npy", sdpa["logits"])
    else:
        sdpa_h = np.load(TMP / "spike13_hf_sdpa_hidden.npy")
        sdpa_lg = np.load(TMP / "spike13_hf_sdpa_logits.npy")
        sdpa = {"hidden": sdpa_h, "logits": sdpa_lg, "cuts": {}}
        # reload cut files if present
        for cut in ("q_proj", "q_norm", "q_rope"):
            p = OUT / f"hf_sdpa_{cut}.f32"
            if p.exists():
                sdpa["cuts"][cut] = np.fromfile(p, dtype=np.float32).reshape(
                    seq, N_HEADS, HEAD_DIM
                )

    # Hephaestus cut dumps
    heph_cuts = {}
    for cut, fname in (
        ("q_proj", "spike13_dump_q_proj.f32"),
        ("q_norm", "spike13_dump_q_norm.f32"),
        ("q_rope", "spike13_dump_q_rope.f32"),
    ):
        p = TMP / fname
        if p.exists():
            heph_cuts[cut] = load_heph_cut(p, seq)
        else:
            # also accept out/ naming
            p2 = TMP / f"spike13_heph_{cut}.f32"
            if p2.exists():
                heph_cuts[cut] = load_heph_cut(p2, seq)

    if heph_cuts and sdpa.get("cuts"):
        report["heph_vs_hf_sdpa"] = compare_cuts(heph_cuts, sdpa["cuts"], "sdpa")
        # localize first cut where error jumps
        rels = [
            (c, report["heph_vs_hf_sdpa"][c]["all_rel"])
            for c in ("q_proj", "q_norm", "q_rope")
        ]
        report["cut_rel_sequence"] = rels
        # Predictions: error "appears" at first cut with rel > 5x previous
        # (or at q_proj if already large)
        thr = 5e-4
        appears = "none_above_threshold"
        prev = 0.0
        for c, r in rels:
            if r > thr and (prev == 0 or r > 3 * prev):
                appears = c
                break
            prev = r
        if rels[0][1] > thr:
            appears = "q_proj"  # already present at first cut
        report["first_elevated_cut"] = appears
        report["predictions"]["error_appears_at_q_proj"] = appears == "q_proj"
        report["predictions"]["error_appears_at_q_norm"] = appears == "q_norm"
        report["predictions"]["error_appears_at_rope"] = appears == "q_rope"
        # growth factors
        report["growth"] = {
            "q_norm_over_q_proj": rels[1][1] / (rels[0][1] + 1e-30),
            "q_rope_over_q_norm": rels[2][1] / (rels[1][1] + 1e-30),
        }

    # Interventions
    if Path(TMP / "spike13_hf_sdpa_hidden.npy").exists() or not args.analyze_only:
        if not args.analyze_only:
            hf_h, hf_lg = sdpa["hidden"], sdpa["logits"]
        else:
            hf_h = np.load(TMP / "spike13_hf_sdpa_hidden.npy")
            hf_lg = np.load(TMP / "spike13_hf_sdpa_logits.npy")
        interventions, base_div = analyze_interventions(seq, hf_h, hf_lg)
        report["interventions"] = interventions
        report["control_hidden_rel"] = base_div
        for key, mode in (
            ("inject_q_proj_collapses", "inject_q_proj"),
            ("inject_q_norm_collapses", "inject_q_norm"),
            ("inject_q_rope_collapses", "inject_q_rope"),
        ):
            r = interventions.get(mode, {})
            report["predictions"][key] = r.get("collapsed")

        # Causal verdict text
        report["causal_verdict"] = _verdict(report)

    out_path = OUT / "probe13_q_cuts.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_path}")


def _verdict(report: dict) -> str:
    iv = report.get("interventions", {})
    cuts = report.get("heph_vs_hf_sdpa", {})
    parts = []
    if cuts:
        parts.append(
            "cut rels: "
            + ", ".join(
                f"{c}={cuts[c]['all_rel']:.6g}"
                for c in ("q_proj", "q_norm", "q_rope")
                if c in cuts
            )
        )
    for m in ("control", "inject_q_proj", "inject_q_norm", "inject_q_rope"):
        r = iv.get(m)
        if r and "hidden_rel_vs_hf" in r:
            parts.append(
                f"{m}: hidden_rel={r['hidden_rel_vs_hf']:.5f} "
                f"collapse_ratio={r.get('collapse_ratio')} "
                f"logit96874={r['logit_target']:.4f}"
            )
    # interpretation
    collapsed = [
        m
        for m in ("inject_q_proj", "inject_q_norm", "inject_q_rope")
        if iv.get(m, {}).get("collapsed")
    ]
    if collapsed:
        first = collapsed[0]
        parts.append(
            f"INTERPRETATION: earliest collapsing inject is {first} — "
            f"seed is at or before that cut; later-only collapse would mean "
            f"downstream of earlier cuts."
        )
    elif all("hidden_rel_vs_hf" in iv.get(m, {}) for m in ("inject_q_proj", "inject_q_rope")):
        parts.append(
            "INTERPRETATION: no Q inject collapses final hidden divergence "
            "(collapse = <25% of control). Seed is not Q-alone, or "
            "amplification depends on Hephaestus K/V/residuals interacting "
            "with corrected Q."
        )
    return " | ".join(parts)


if __name__ == "__main__":
    main()
