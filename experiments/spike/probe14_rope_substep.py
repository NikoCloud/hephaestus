#!/usr/bin/env python3
"""PROBE 14 -- isolate the RoPE sub-step that creates the q_rope divergence.

Probe 13 localized the seed to post-RoPE Q (×15.7 jump; inject_q_rope collapses
the spike). Raw GPU cos/sin ULP was ruled out (exp6). This probe falsifies:

  A) Hephaestus cos/sin (freq construction + bf16 cast) differ from HF's.
  B) On IDENTICAL cos/sin and pre-RoPE Q, Hephaestus's closed-form rotate
        out_re = re*cos - im*sin
        out_im = im*cos + re*sin
     differs from HF's
        (q * cos) + (rotate_half(q) * sin)
     under bf16 arithmetic.
  C) The observed dump divergence is explained by (A), (B), or both.
  D) Optional: which pair indices / positions carry the error.

Uses committed probe-13 dumps (no GPU required for the main path). Optionally
regenerates HF cos/sin from the model config (CPU) for reference.

Predeclared collapse of hypotheses:
  - cos/sin match, rotate on same inputs matches HF dump → dump/layout bug only
  - cos/sin differ, rotate with HF cos/sin on Heph pre-Q matches HF post-Q
      → root cause is freq/cos/sin construction
  - cos/sin match, rotate formula under bf16 diverges from HF
      → root cause is apply arithmetic (dtype/order)
  - neither alone explains dump → compound / indexing

Usage:
  python3 experiments/spike/probe14_rope_substep.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
TMP = Path("/tmp")
SEQ = 77
N_HEADS = 32
HEAD_DIM = 128
HALF = HEAD_DIM // 2
Q_OUT = N_HEADS * HEAD_DIM
TARGET = 76
THETA = 5_000_000.0
STEP = 67


def rel_l2(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def load_q(path: Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    assert a.size == SEQ * Q_OUT, (path, a.size)
    return a.reshape(SEQ, N_HEADS, HEAD_DIM)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 array to bf16 via high-16-bit truncation (RN not RNE).

    Good enough for localization; for exact RNE we use torch when available.
    """
    try:
        import torch

        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32))
            .to(torch.bfloat16)
            .float()
            .numpy()
        )
    except Exception:
        bits = np.asarray(x, dtype=np.float32).view(np.uint32)
        # round-to-nearest-even approx: add 0x8000 with tie-to-even is hard;
        # truncate for fallback.
        return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def heph_freq_cos_sin(seq: int = SEQ) -> tuple[np.ndarray, np.ndarray]:
    """Hephaestus rope_kernel: freq = pos * theta**(-2p/d); cos/sin cast bf16.

    Returns cos, sin shaped [seq, HALF] (one value per pair).
    """
    cos = np.zeros((seq, HALF), dtype=np.float32)
    sin = np.zeros((seq, HALF), dtype=np.float32)
    for t in range(seq):
        for p in range(HALF):
            freq = float(t) * (THETA ** (-2.0 * p / float(HEAD_DIM)))
            # match kernels.mojo: cos(freq).cast[BF16]() with freq as Float32
            c = to_bf16(np.array([math.cos(freq)], dtype=np.float32))[0]
            s = to_bf16(np.array([math.sin(freq)], dtype=np.float32))[0]
            cos[t, p] = c
            sin[t, p] = s
    return cos, sin


def hf_freq_cos_sin(seq: int = SEQ) -> tuple[np.ndarray, np.ndarray]:
    """HF Qwen3RotaryEmbedding default: inv_freq then outer with positions.

    inv_freq[i] = 1 / (theta ** (arange(0,dim,2)[i] / dim))
    emb = cat(freqs, freqs); cos/sin cast to bf16.
    Returns pair-space [seq, HALF] (first half of the duplicated emb).
    """
    inv = 1.0 / (
        THETA
        ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / float(HEAD_DIM))
    )
    inv = inv.astype(np.float32)
    pos = np.arange(seq, dtype=np.float32)
    freqs = np.outer(pos, inv)  # [seq, HALF]
    cos = to_bf16(np.cos(freqs.astype(np.float32)))
    sin = to_bf16(np.sin(freqs.astype(np.float32)))
    return cos, sin


def expand_pair_to_head(pair: np.ndarray) -> np.ndarray:
    """HF emb = cat(freqs, freqs) → [seq, HEAD_DIM]."""
    return np.concatenate([pair, pair], axis=-1)


def rotate_half_np(q: np.ndarray) -> np.ndarray:
    """q[..., HEAD_DIM] → cat(-q2, q1)."""
    q1 = q[..., :HALF]
    q2 = q[..., HALF:]
    return np.concatenate([-q2, q1], axis=-1)


def apply_hf_rope_bf16(q: np.ndarray, cos_pair: np.ndarray, sin_pair: np.ndarray) -> np.ndarray:
    """HF: (q * cos) + (rotate_half(q) * sin) with cos/sin expanded, bf16 muls.

    q: [seq, heads, dim] or [seq, dim]
    cos_pair/sin_pair: [seq, HALF]
    """
    cos = expand_pair_to_head(cos_pair)  # [seq, dim]
    sin = expand_pair_to_head(sin_pair)
    # broadcast heads
    while cos.ndim < q.ndim:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    q_b = to_bf16(q)
    c_b = to_bf16(cos)
    s_b = to_bf16(sin)
    # bf16 mul/add via round-trip
    t1 = to_bf16(q_b * c_b)
    t2 = to_bf16(rotate_half_np(q_b) * s_b)
    return to_bf16(t1 + t2)


def apply_heph_rope_bf16(q: np.ndarray, cos_pair: np.ndarray, sin_pair: np.ndarray) -> np.ndarray:
    """Hephaestus closed form with bf16 cast after each mul (best-effort match).

    kernels.mojo does:
      x[re] = (re * cos_v) - (im * sin_v)
      x[im] = (im * cos_v) + (re * sin_v)
    with re,im,cos,sin as BF16. Mojo may promote muls to f32; we report both
    pure-bf16 and f32-accum variants.
    """
    q_b = to_bf16(q)
    out = np.empty_like(q_b)
    # cos_pair [seq, HALF]
    if q_b.ndim == 3:
        seq, nh, d = q_b.shape
        for t in range(seq):
            for h in range(nh):
                for p in range(HALF):
                    re = q_b[t, h, p]
                    im = q_b[t, h, p + HALF]
                    c = cos_pair[t, p]
                    s = sin_pair[t, p]
                    # variant: f32 accum then cast (common GPU path)
                    out[t, h, p] = to_bf16(np.array([re * c - im * s], np.float32))[0]
                    out[t, h, p + HALF] = to_bf16(
                        np.array([im * c + re * s], np.float32)
                    )[0]
    else:
        seq, d = q_b.shape
        for t in range(seq):
            for p in range(HALF):
                re = q_b[t, p]
                im = q_b[t, p + HALF]
                c = cos_pair[t, p]
                s = sin_pair[t, p]
                out[t, p] = to_bf16(np.array([re * c - im * s], np.float32))[0]
                out[t, p + HALF] = to_bf16(
                    np.array([im * c + re * s], np.float32)
                )[0]
    return out


def apply_heph_rope_bf16_strict(q: np.ndarray, cos_pair: np.ndarray, sin_pair: np.ndarray) -> np.ndarray:
    """Strict bf16: cast after each mul AND after add/sub (matches eager-style)."""
    q_b = to_bf16(q)
    out = np.empty_like(q_b)
    seq, nh, _ = q_b.shape
    for t in range(seq):
        for h in range(nh):
            for p in range(HALF):
                re = float(q_b[t, h, p])
                im = float(q_b[t, h, p + HALF])
                c = float(cos_pair[t, p])
                s = float(sin_pair[t, p])
                re_c = to_bf16(np.array([re * c], np.float32))[0]
                im_s = to_bf16(np.array([im * s], np.float32))[0]
                im_c = to_bf16(np.array([im * c], np.float32))[0]
                re_s = to_bf16(np.array([re * s], np.float32))[0]
                out[t, h, p] = to_bf16(np.array([re_c - im_s], np.float32))[0]
                out[t, h, p + HALF] = to_bf16(np.array([im_c + re_s], np.float32))[0]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Prefer /tmp heph dumps; fall back to nothing
    heph_norm_p = TMP / "spike13_dump_q_norm.f32"
    heph_rope_p = TMP / "spike13_dump_q_rope.f32"
    hf_norm_p = OUT / "hf_sdpa_q_norm.f32"
    hf_rope_p = OUT / "hf_sdpa_q_rope.f32"
    for p in (heph_norm_p, heph_rope_p, hf_norm_p, hf_rope_p):
        if not p.exists():
            raise SystemExit(f"missing dump: {p} (re-run probe 13 first)")

    heph_pre = load_q(heph_norm_p)
    heph_post = load_q(heph_rope_p)
    hf_pre = load_q(hf_norm_p)
    hf_post = load_q(hf_rope_p)

    report: dict = {
        "input": {
            "seq": SEQ,
            "target_row": TARGET,
            "step": STEP,
            "theta": THETA,
            "head_dim": HEAD_DIM,
        },
        "baselines": {},
        "cos_sin": {},
        "rotate_isolation": {},
        "reconstruction": {},
        "pair_concentration": {},
        "verdict": {},
    }

    # --- baselines from dumps ---
    report["baselines"] = {
        "pre_rope_heph_vs_hf": {
            "all_rel": rel_l2(heph_pre, hf_pre),
            "tgt_rel": rel_l2(heph_pre[TARGET], hf_pre[TARGET]),
            "all_max_abs": max_abs(heph_pre, hf_pre),
        },
        "post_rope_heph_vs_hf": {
            "all_rel": rel_l2(heph_post, hf_post),
            "tgt_rel": rel_l2(heph_post[TARGET], hf_post[TARGET]),
            "all_max_abs": max_abs(heph_post, hf_post),
        },
    }

    # --- cos/sin construction ---
    heph_c, heph_s = heph_freq_cos_sin()
    hf_c, hf_s = hf_freq_cos_sin()
    report["cos_sin"] = {
        "heph_vs_hf_cos_all_rel": rel_l2(heph_c, hf_c),
        "heph_vs_hf_sin_all_rel": rel_l2(heph_s, hf_s),
        "heph_vs_hf_cos_max_abs": max_abs(heph_c, hf_c),
        "heph_vs_hf_sin_max_abs": max_abs(heph_s, hf_s),
        "heph_vs_hf_cos_tgt_rel": rel_l2(heph_c[TARGET], hf_c[TARGET]),
        "heph_vs_hf_sin_tgt_rel": rel_l2(heph_s[TARGET], hf_s[TARGET]),
        "n_cos_differ": int(np.sum(heph_c != hf_c)),
        "n_sin_differ": int(np.sum(heph_s != hf_s)),
        "n_total_pairs": int(SEQ * HALF),
    }
    # per-position max abs cos diff
    cos_pos = np.max(np.abs(heph_c - hf_c), axis=1)
    sin_pos = np.max(np.abs(heph_s - hf_s), axis=1)
    report["cos_sin"]["worst_pos_cos"] = {
        "pos": int(np.argmax(cos_pos)),
        "max_abs": float(cos_pos.max()),
    }
    report["cos_sin"]["worst_pos_sin"] = {
        "pos": int(np.argmax(sin_pos)),
        "max_abs": float(sin_pos.max()),
    }
    # inv_freq scalar compare
    inv_heph = np.array(
        [THETA ** (-2.0 * p / HEAD_DIM) for p in range(HALF)], dtype=np.float64
    )
    inv_hf = 1.0 / (
        THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)
    )
    report["cos_sin"]["inv_freq_max_rel"] = float(
        np.max(np.abs(inv_heph - inv_hf) / (np.abs(inv_hf) + 1e-30))
    )
    report["cos_sin"]["inv_freq_max_abs"] = float(np.max(np.abs(inv_heph - inv_hf)))

    # --- rotate isolation: same pre-Q, swap cos/sin sources, compare to HF post ---
    # Use HF pre-Q (nearly = Heph pre) so we isolate RoPE only.
    cases = {}
    for cos_name, cos_p, sin_p in (
        ("hf_cos", hf_c, hf_s),
        ("heph_cos", heph_c, heph_s),
    ):
        for formula, fn in (
            ("hf_rotate_half", apply_hf_rope_bf16),
            ("heph_closed_f32accum", apply_heph_rope_bf16),
            ("heph_closed_strict_bf16", apply_heph_rope_bf16_strict),
        ):
            out = fn(hf_pre, cos_p, sin_p)
            key = f"{formula}__{cos_name}"
            cases[key] = {
                "vs_hf_post_all_rel": rel_l2(out, hf_post),
                "vs_hf_post_tgt_rel": rel_l2(out[TARGET], hf_post[TARGET]),
                "vs_hf_post_max_abs": max_abs(out, hf_post),
                "vs_heph_post_all_rel": rel_l2(out, heph_post),
                "vs_heph_post_tgt_rel": rel_l2(out[TARGET], heph_post[TARGET]),
                "vs_heph_post_max_abs": max_abs(out, heph_post),
            }
    report["rotate_isolation"] = cases

    # Also: apply each formula to *Hephaestus* pre-Q with each cos source
    cases_heph_pre = {}
    for cos_name, cos_p, sin_p in (
        ("hf_cos", hf_c, hf_s),
        ("heph_cos", heph_c, heph_s),
    ):
        for formula, fn in (
            ("hf_rotate_half", apply_hf_rope_bf16),
            ("heph_closed_f32accum", apply_heph_rope_bf16),
            ("heph_closed_strict_bf16", apply_heph_rope_bf16_strict),
        ):
            out = fn(heph_pre, cos_p, sin_p)
            key = f"{formula}__{cos_name}__heph_pre"
            cases_heph_pre[key] = {
                "vs_hf_post_all_rel": rel_l2(out, hf_post),
                "vs_heph_post_all_rel": rel_l2(out, heph_post),
                "vs_hf_post_tgt_rel": rel_l2(out[TARGET], hf_post[TARGET]),
                "vs_heph_post_tgt_rel": rel_l2(out[TARGET], heph_post[TARGET]),
            }
    report["rotate_isolation_heph_pre"] = cases_heph_pre

    # --- reconstruction: which synthetic post-Q best matches the dump gap? ---
    # Gap explained fraction: 1 - rel(synth, hf_post)/rel(heph_post, hf_post)
    base_rel = report["baselines"]["post_rope_heph_vs_hf"]["all_rel"]
    recon = {}
    for k, v in {**cases, **{kk: vv for kk, vv in cases_heph_pre.items()}}.items():
        r = v["vs_hf_post_all_rel"]
        recon[k] = {
            "vs_hf_post_all_rel": r,
            "fraction_of_dump_gap": float(r / (base_rel + 1e-30)),
            "explains_dump": bool(r < 0.25 * base_rel),
        }
    report["reconstruction"] = recon

    # --- pair concentration on residual heph_post - hf_post ---
    diff = np.abs(heph_post[TARGET] - hf_post[TARGET])  # [heads, dim]
    per_dim = diff.mean(axis=0)
    per_pair = 0.5 * (per_dim[:HALF] + per_dim[HALF:])
    top_pairs = np.argsort(-per_pair)[:8]
    report["pair_concentration"] = {
        "top_pairs_by_mean_abs": [
            {"pair": int(p), "mean_abs": float(per_pair[p]),
             "angle_at_target": float(TARGET * (THETA ** (-2.0 * int(p) / HEAD_DIM)))}
            for p in top_pairs
        ],
        "pair0_mean_abs": float(per_pair[0]),
        "pair_half_mean_abs": float(per_pair[HALF // 2]),
        "pair_last_mean_abs": float(per_pair[HALF - 1]),
        "corr_abs_vs_pair_index": float(
            np.corrcoef(np.arange(HALF), per_pair)[0, 1]
        ),
    }

    # --- element analysis (probe14_roped_element_analysis style) ---
    report["element_target"] = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "argmax_head": int(diff.argmax() // HEAD_DIM),
        "argmax_dim": int(diff.argmax() % HEAD_DIM),
    }

    # --- verdict ---
    cos_mismatch = report["cos_sin"]["heph_vs_hf_cos_max_abs"]
    sin_mismatch = report["cos_sin"]["heph_vs_hf_sin_max_abs"]
    # best reconstruction toward HF post
    best_k = min(recon, key=lambda k: recon[k]["vs_hf_post_all_rel"])
    best_heph_k = min(
        cases_heph_pre,
        key=lambda k: cases_heph_pre[k]["vs_heph_post_all_rel"],
    )

    # Does HF-style rotate + HF cos on HF pre reconstruct HF post?
    hf_self = cases["hf_rotate_half__hf_cos"]["vs_hf_post_all_rel"]
    # Does Heph-style + Heph cos on Heph pre reconstruct Heph post?
    heph_self = cases_heph_pre["heph_closed_f32accum__heph_cos__heph_pre"][
        "vs_heph_post_all_rel"
    ]
    heph_self_strict = cases_heph_pre["heph_closed_strict_bf16__heph_cos__heph_pre"][
        "vs_heph_post_all_rel"
    ]
    heph_self_hf_formula = cases_heph_pre["hf_rotate_half__heph_cos__heph_pre"][
        "vs_heph_post_all_rel"
    ]

    # Key cross tests
    heph_pre_hf_cos_hf_rot = cases_heph_pre["hf_rotate_half__hf_cos__heph_pre"][
        "vs_hf_post_all_rel"
    ]
    hf_pre_heph_cos_hf_rot = cases["hf_rotate_half__heph_cos"]["vs_hf_post_all_rel"]
    hf_pre_hf_cos_heph_rot = cases["heph_closed_f32accum__hf_cos"]["vs_hf_post_all_rel"]

    verdict = {
        "cos_sin_match": bool(cos_mismatch < 1e-6 and sin_mismatch < 1e-6),
        "cos_sin_max_abs": max(cos_mismatch, sin_mismatch),
        "inv_freq_effectively_identical": bool(
            report["cos_sin"]["inv_freq_max_rel"] < 1e-12
        ),
        "hf_self_reconstruction_rel": hf_self,
        "heph_self_reconstruction": {
            "f32accum": heph_self,
            "strict_bf16": heph_self_strict,
            "hf_formula_heph_cos": heph_self_hf_formula,
        },
        "cross": {
            "heph_pre + hf_cos + hf_rotate → vs_hf_post": heph_pre_hf_cos_hf_rot,
            "hf_pre + heph_cos + hf_rotate → vs_hf_post": hf_pre_heph_cos_hf_rot,
            "hf_pre + hf_cos + heph_rotate → vs_hf_post": hf_pre_hf_cos_heph_rot,
        },
        "best_reconstruction_of_hf_post": best_k,
        "best_reconstruction_of_heph_post": best_heph_k,
        "root_cause_class": None,
        "one_line_fix_hint": None,
    }

    # Classification logic (predeclared)
    if cos_mismatch >= 1e-5 or sin_mismatch >= 1e-5:
        # cos/sin construction differs
        if hf_pre_heph_cos_hf_rot > 0.5 * base_rel and heph_pre_hf_cos_hf_rot < 0.25 * base_rel:
            verdict["root_cause_class"] = "freq_cos_sin_construction"
            verdict["one_line_fix_hint"] = (
                "Match HF inv_freq / outer-product cos/sin (including dtype "
                "of arange and the emb=cat(freqs,freqs) broadcast), not the "
                "scalar theta**(-2p/d) path if it diverges under bf16 cast."
            )
        else:
            verdict["root_cause_class"] = "freq_cos_sin_partial"
            verdict["one_line_fix_hint"] = (
                "cos/sin differ; also check rotate arithmetic."
            )
    else:
        # cos/sin match — difference is in apply
        if hf_pre_hf_cos_heph_rot > 3 * hf_self and hf_self < 1e-4:
            verdict["root_cause_class"] = "rotate_apply_arithmetic"
            # distinguish f32 accum vs strict bf16 by which matches heph dump
            if heph_self < heph_self_strict * 0.5:
                verdict["one_line_fix_hint"] = (
                    "RoPE mul/add is accumulating in f32 after bf16 cos/sin; "
                    "HF does bf16 mul/add throughout. Force bf16 after every "
                    "mul and add in rope_kernel (or match HF rotate_half form "
                    "exactly in bf16)."
                )
            elif heph_self_strict < heph_self * 0.5:
                verdict["one_line_fix_hint"] = (
                    "Hephaestus dump matches strict bf16 closed form; HF dump "
                    "matches rotate_half form — check algebraic equivalence "
                    "under bf16 (should be identical; investigate indexing)."
                )
            else:
                verdict["one_line_fix_hint"] = (
                    "Apply arithmetic differs from HF despite matching cos/sin; "
                    "align rope_kernel with (q*cos)+(rotate_half(q)*sin) in bf16."
                )
        elif heph_pre_hf_cos_hf_rot < 0.25 * base_rel:
            verdict["root_cause_class"] = "pre_rope_q_only"
            verdict["one_line_fix_hint"] = (
                "RoPE apply is fine; residual is pre-RoPE Q (contradicts probe 13 cuts)."
            )
        else:
            verdict["root_cause_class"] = "compound_or_unexplained"
            verdict["one_line_fix_hint"] = (
                "See cross table; no single sub-step explains the full dump gap."
            )

    # Stronger: if heph dump ≈ heph formula on heph cos/pre, and that formula
    # with same inputs as HF still ≠ HF, root cause is the formula path.
    report["verdict"] = verdict

    # Human-readable summary lines
    summary = []
    summary.append(
        f"cos max_abs heph vs hf = {cos_mismatch:.6g}; "
        f"sin = {sin_mismatch:.6g}; inv_freq max_rel = "
        f"{report['cos_sin']['inv_freq_max_rel']:.6g}"
    )
    summary.append(
        f"dump post-rope gap rel = {base_rel:.6g}; "
        f"pre-rope gap rel = {report['baselines']['pre_rope_heph_vs_hf']['all_rel']:.6g}"
    )
    summary.append(
        f"hf_rotate+hf_cos on hf_pre vs hf_post rel = {hf_self:.6g} (self-check)"
    )
    summary.append(
        f"heph_closed_f32accum+heph_cos on heph_pre vs heph_post rel = {heph_self:.6g}"
    )
    summary.append(
        f"heph_closed_strict_bf16+heph_cos on heph_pre vs heph_post rel = {heph_self_strict:.6g}"
    )
    summary.append(
        f"hf_rotate+hf_cos on heph_pre vs hf_post rel = {heph_pre_hf_cos_hf_rot:.6g}"
    )
    summary.append(
        f"hf_rotate+heph_cos on hf_pre vs hf_post rel = {hf_pre_heph_cos_hf_rot:.6g}"
    )
    summary.append(
        f"heph_closed_f32accum+hf_cos on hf_pre vs hf_post rel = {hf_pre_hf_cos_heph_rot:.6g}"
    )
    summary.append(f"ROOT_CAUSE_CLASS = {verdict['root_cause_class']}")
    summary.append(f"FIX_HINT = {verdict['one_line_fix_hint']}")
    report["summary_lines"] = summary

    out_path = OUT / "probe14_rope_substep.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    for line in summary:
        print(line)
    print(f"wrote {out_path}")
    return report


if __name__ == "__main__":
    main()
