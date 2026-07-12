#!/usr/bin/env python3
"""PROBE 12 -- FP8-widening bar on the measured anomaly (not a new root-cause claim).

The Phase 1b entry bar for "benign" is NOT "argmax agrees under BF16 today."
It is: the anomaly cannot affect greedy decode even when FP8's coarser mantissa
widens the error.

This probe uses committed logits artifacts only (no engine changes):

  1. At the canonical spike row (prompt1 step 67), measure top1-top2 margin and
     margin of argmax over the worst-diverging tail token (96874).
  2. Across all 768 teacher-forced rows (3 prompts × 256), measure how large a
     relative hidden-state perturbation would need to be to flip argmax, using
     a first-order model: logits shift by ~ ||E_row|| * ||delta_h|| in the worst
     direction. We bound more tightly using the observed full-vocab logit delta
     at each row: if the observed row-wide error were scaled by factor S, when
     does argmax flip?
  3. Compare required scale S_flip to the rough mantissa-coarsening ratio
     between BF16 (7-bit mantissa, ~2^-7 relative) and E4M3 (3-bit mantissa,
     ~2^-3 relative) ≈ 16×. If S_flip >> 16 on every non-tie row, the *existing*
     BF16 anomaly pattern is still decision-safe under pure magnification.
     If any non-tie row has S_flip ≲ 16, the anomaly is NOT FP8-safe under that
     simple widening model.

Also re-states probe8's concrete E4M3 candidate result: that is a stronger
test (actual quantization), and already produced clear flips.

Usage: python3 experiments/spike/probe12_fp8_margin.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
VOCAB = 151936
STEPS = 256

# Committed /tmp artifacts from the investigation (must exist).
HEPH = {
    1: "/tmp/spike-det-1783875368/rep1_logits.f32",
    2: "/tmp/tf2_logits.f32",
    3: "/tmp/tf3_logits.f32",
}
HF = {
    1: "/tmp/hftf1_logits.f32",
    2: "/tmp/hftf2_logits.f32",
    3: "/tmp/hftf3_logits.f32",
}
TARGET = (1, 67, 96874)

# Mantissa relative precision (unit in last place of significand, rough).
BF16_REL = 2 ** -7  # ~0.0078125
E4M3_REL = 2 ** -3  # 0.125
WIDEN = E4M3_REL / BF16_REL  # 16


def bf16_ulp(x: float) -> float:
    if x == 0:
        return 2 ** -133  # subnormal floor; irrelevant here
    e = math.floor(math.log2(abs(x)))
    return 2.0 ** (e - 7)


def load_pair(p: int):
    h = np.fromfile(HEPH[p], dtype="<f4")
    f = np.fromfile(HF[p], dtype="<f4")
    assert h.size == STEPS * VOCAB, (p, h.size)
    assert f.size == STEPS * VOCAB, (p, f.size)
    return h.reshape(STEPS, VOCAB), f.reshape(STEPS, VOCAB)


def to_bf16_via_trunc(row: np.ndarray) -> np.ndarray:
    """Approximate bf16 by zeroing the low 16 bits of IEEE754 float32.

    Matches round-toward-zero bit truncation, not round-to-nearest-even; good
    enough for margin *scale* estimates (ulps of error, not bit-exact ties).
    """
    r = np.asarray(row, dtype=np.float32)
    bits = r.view(np.uint32)
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def row_margin(row: np.ndarray):
    r = np.asarray(row, dtype=np.float32)
    rb = to_bf16_via_trunc(r)
    am = int(np.argmax(rb))
    tmp = rb.copy()
    tmp[am] = -np.inf
    am2 = int(np.argmax(tmp))
    return {
        "argmax": am,
        "top1": float(r[am]),
        "top2_id": am2,
        "top2": float(r[am2]),
        "margin": float(r[am] - r[am2]),
        "margin_bf16": float(rb[am] - rb[am2]),
        "ulp": bf16_ulp(float(r[am])),
    }


def scale_to_flip(heph_row: np.ndarray, hf_row: np.ndarray, n_grid: int = 64):
    """Smallest S>=0 such that argmax(hf + S*(heph-hf)) != argmax(hf), using bf16 argmax.

    S=0 is pure HF; S=1 is pure Hephaestus. If they already share argmax, we
    search S>1 (error magnification) and also S in [0,1] is irrelevant for flip
    from HF. We care about magnification of the *difference*.
    """
    delta = heph_row - hf_row
    base = hf_row

    def argmax_bf16(row):
        return int(np.argmax(to_bf16_via_trunc(row)))

    a0 = argmax_bf16(base)
    a1 = argmax_bf16(base + delta)
    # Search S > 0 for first flip away from HF argmax
    # First check fine grid on [0, WIDEN*2]
    s_max = WIDEN * 4
    flipped_at = None
    prev = a0
    for i in range(0, n_grid + 1):
        s = s_max * i / n_grid
        a = argmax_bf16(base + s * delta)
        if a != a0:
            flipped_at = s
            break
        prev = a
    # binary refine
    if flipped_at is None:
        return {
            "hf_argmax": a0,
            "heph_argmax": a1,
            "already_differs_at_s1": a0 != a1,
            "s_flip": None,
            "s_flip_gt": s_max,
        }
    lo, hi = max(0.0, flipped_at - s_max / n_grid), flipped_at
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if argmax_bf16(base + mid * delta) != a0:
            hi = mid
        else:
            lo = mid
    return {
        "hf_argmax": a0,
        "heph_argmax": a1,
        "already_differs_at_s1": a0 != a1,
        "s_flip": hi,
        "s_flip_gt": None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "widen_ratio_e4m3_over_bf16": WIDEN,
        "bf16_rel": BF16_REL,
        "e4m3_rel": E4M3_REL,
        "target": {},
        "per_prompt_summary": {},
        "worst_rows": [],
        "verdict": {},
    }

    # Target row detail
    h1, f1 = load_pair(1)
    thr = row_margin(h1[67])
    tfr = row_margin(f1[67])
    tok = TARGET[2]
    report["target"] = {
        "heph": thr,
        "hf": tfr,
        "token_96874": {
            "heph": float(h1[67, tok]),
            "hf": float(f1[67, tok]),
            "abs_diff": float(abs(h1[67, tok] - f1[67, tok])),
            "heph_rank": int((h1[67] > h1[67, tok]).sum() + 1),
            "hf_rank": int((f1[67] > f1[67, tok]).sum() + 1),
            "margin_argmax_minus_tok_heph": float(h1[67, thr["argmax"]] - h1[67, tok]),
            "margin_argmax_minus_tok_hf": float(f1[67, tfr["argmax"]] - f1[67, tok]),
        },
        "scale_to_flip": scale_to_flip(h1[67], f1[67], n_grid=128),
        "row_mae": float(np.mean(np.abs(h1[67] - f1[67]))),
        "row_max": float(np.max(np.abs(h1[67] - f1[67]))),
    }

    # All rows across prompts that have artifacts
    worst = []
    for p in (1, 2, 3):
        if not Path(HEPH[p]).exists() or not Path(HF[p]).exists():
            report["per_prompt_summary"][str(p)] = {"status": "missing_artifacts"}
            continue
        heph, hf = load_pair(p)
        s_flips = []
        clear_near = 0
        n_already = 0
        n_safe_at_16 = 0
        n_unsafe_at_16 = 0
        n_no_flip = 0
        for step in range(STEPS):
            info = scale_to_flip(heph[step], hf[step], n_grid=64)
            m = row_margin(hf[step])
            near = m["margin_bf16"] <= m["ulp"] * 1.01
            if info["already_differs_at_s1"]:
                n_already += 1
                if not near:
                    clear_near += 1  # should be 0 per prior work
            s = info["s_flip"]
            if s is None:
                n_no_flip += 1
                n_safe_at_16 += 1
                s_eff = info["s_flip_gt"]
            else:
                s_eff = s
                if s > WIDEN:
                    n_safe_at_16 += 1
                else:
                    n_unsafe_at_16 += 1
            s_flips.append(s_eff)
            worst.append(
                {
                    "prompt": p,
                    "step": step,
                    "s_flip": s,
                    "s_eff": s_eff,
                    "hf_margin": m["margin"],
                    "already_differs": info["already_differs_at_s1"],
                    "near_tie": near,
                }
            )
        s_arr = np.array(s_flips, dtype=np.float64)
        report["per_prompt_summary"][str(p)] = {
            "n_already_differs_at_s1": n_already,
            "n_clear_differs_at_s1": clear_near,
            "n_no_flip_below_4x_widen": n_no_flip,
            "n_safe_s_flip_gt_16": n_safe_at_16,
            "n_unsafe_s_flip_le_16": n_unsafe_at_16,
            "s_flip_min": float(s_arr.min()),
            "s_flip_p10": float(np.percentile(s_arr, 10)),
            "s_flip_median": float(np.median(s_arr)),
        }

    worst_sorted = sorted(
        [w for w in worst if w["s_flip"] is not None],
        key=lambda w: w["s_flip"],
    )[:20]
    report["worst_rows"] = worst_sorted

    # Verdict logic
    unsafe = sum(
        1
        for w in worst
        if (w["s_flip"] is not None and w["s_flip"] <= WIDEN and not w["near_tie"])
    )
    unsafe_including_ties = sum(
        1 for w in worst if (w["s_flip"] is not None and w["s_flip"] <= WIDEN)
    )
    report["verdict"] = {
        "simple_widening_model": {
            "non_tie_rows_with_s_flip_le_16": unsafe,
            "all_rows_with_s_flip_le_16": unsafe_including_ties,
            "interpretation": (
                "If non_tie_rows_with_s_flip_le_16 > 0, magnifying the measured "
                "BF16 Hephaestus-vs-HF delta by the E4M3/BF16 mantissa ratio can "
                "create clear argmax flips — anomaly is NOT benign under this model. "
                "If 0, the measured delta pattern alone is decision-safe under pure "
                "magnification; probe8's actual E4M3 quant still governs FP8 risk."
            ),
        },
        "probe8_concrete_e4m3": {
            "weights_clear_flips": 3,
            "weights_plus_acts_clear_flips": 6,
            "source": "experiments/spike/out/probe8_fp8.json",
            "interpretation": (
                "A concrete per-channel E4M3 candidate already produces clear "
                "non-tie flips. That alone rejects clearing the Phase 1b entry "
                "gate with a benign claim, independent of the widening model."
            ),
        },
        "benign_gate_clearable": False,
        "reason": (
            "probe8 concrete E4M3 candidate yields clear non-tie argmax flips; "
            "benign requires no such flips under FP8-coarsened error."
        ),
    }

    out_path = OUT / "probe12_fp8_margin.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
