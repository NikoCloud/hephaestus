#!/usr/bin/env python3
"""Parse /tmp/ab_post_fix logs into JSON stats for bench/1a-ab.md."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

OUT = Path("/tmp/ab_post_fix")


def parse_heph(log_path: Path) -> list[dict]:
    text = log_path.read_text()
    reps = []
    cur: dict = {}
    for line in text.splitlines():
        if line.startswith("--- rep"):
            if cur:
                reps.append(cur)
            cur = {}
            continue
        m = re.match(r"([a-z0-9_()., +]+):\s*([0-9.eE+-]+)\s*$", line.strip())
        if not m:
            continue
        key = m.group(1).strip()
        val = float(m.group(2))
        # normalize keys
        key = (
            key.replace(" (forward-pass only)", "_fwd")
            .replace(" (incl. GPU argmax)", "_incl")
            .replace(" ", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )
        cur[key] = val
    if cur:
        reps.append(cur)
    return reps


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(xs: list[float]) -> str:
    m, s = mean_std(xs)
    return f"{m:.2f} ± {s:.2f}"


def main():
    short = parse_heph(OUT / "heph_short.log")
    long = parse_heph(OUT / "heph_long.log")
    print("short reps raw keys sample:", list(short[0].keys()) if short else None)
    print("short:", json.dumps(short, indent=2))
    print("long:", json.dumps(long, indent=2))

    def col(reps, *keys):
        for k in keys:
            vals = [r[k] for r in reps if k in r]
            if vals:
                return vals
        return []

    report = {
        "heph_short": {
            "prefill_tok_s": col(short, "prefill_tok_s"),
            "ttft_ms_fwd": col(short, "ttft_ms_fwd", "ttft_ms_forward-pass_only"),
            "ttft_ms_incl": col(short, "ttft_ms_incl", "ttft_ms_incl_GPU_argmax"),
            "decode_fwd": col(
                short, "decode_tok_s_fwd", "decode_tok_s_forward-pass_only"
            ),
            "decode_incl": col(
                short, "decode_tok_s_incl", "decode_tok_s_incl_GPU_argmax"
            ),
            "argmax_ms_per_step": col(short, "argmax_ms_per_step"),
            "total_s": col(short, "total_s"),
        },
        "heph_long": {
            "prefill_tok_s": col(long, "prefill_tok_s"),
            "ttft_ms_fwd": col(long, "ttft_ms_fwd", "ttft_ms_forward-pass_only"),
            "ttft_ms_incl": col(long, "ttft_ms_incl", "ttft_ms_incl_GPU_argmax"),
            "total_s": col(long, "total_s"),
        },
    }

    # VRAM peak from poll
    vram_path = OUT / "heph_short_vram_poll.txt"
    peak_mb = None
    if vram_path.exists():
        bytes_ = []
        for line in vram_path.read_text().splitlines():
            m = re.search(r"vram_B\s+(\d+)", line)
            if m:
                bytes_.append(int(m.group(1)))
        if bytes_:
            peak_mb = max(bytes_) / (1024 * 1024)
    report["peak_vram_mb"] = peak_mb

    # Pretty summary
    print("\n=== Hephaestus short (10 tok prompt, 256 gen) ===")
    for k, vals in report["heph_short"].items():
        if vals:
            print(f"  {k}: {fmt(vals)}  n={len(vals)}")
    print(f"  peak_vram_mb: {peak_mb}")
    print("\n=== Hephaestus long (512 tok prompt, 8 gen) ===")
    for k, vals in report["heph_long"].items():
        if vals:
            print(f"  {k}: {fmt(vals)}  n={len(vals)}")

    out_json = OUT / "summary.json"
    # convert for json
    serial = {
        "heph_short": {k: vals for k, vals in report["heph_short"].items()},
        "heph_long": {k: vals for k, vals in report["heph_long"].items()},
        "peak_vram_mb": peak_mb,
        "heph_short_fmt": {
            k: fmt(v) for k, v in report["heph_short"].items() if v
        },
        "heph_long_fmt": {k: fmt(v) for k, v in report["heph_long"].items() if v},
    }
    out_json.write_text(json.dumps(serial, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
