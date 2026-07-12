#!/usr/bin/env python3
"""Package the five-process determinism and row-shape evidence."""
import glob
import hashlib
import json
import os
import struct

import numpy as np

V, N, STEP, TOK = 151936, 256, 67, 96874
HEPH_GLOB = "/tmp/spike-det-1783875368/rep*_logits.f32"
HF = "/tmp/hftf1_logits.f32"


def bits(x):
    return f"0x{struct.unpack('<I', np.float32(x).tobytes())[0]:08x}"


def rowstats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    idx = np.argsort(-d)[:10]
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p99": float(np.quantile(d, 0.99)),
        "p999": float(np.quantile(d, 0.999)),
        "threshold_counts": {str(t): int((d > t).sum()) for t in (0.1, 0.5, 1.0, 5.0)},
        "max": float(d.max()),
        "top10": [
            {
                "token": int(i),
                "abs_diff": float(d[i]),
                "heph": float(a[i]),
                "hf": float(b[i]),
            }
            for i in idx
        ],
    }


def main():
    files = sorted(glob.glob(HEPH_GLOB))
    assert len(files) >= 5, files
    hf = np.fromfile(HF, np.float32)
    assert hf.size == N * V
    hf = hf.reshape(N, V)

    reps, arrays = [], []
    for path in files:
        a = np.fromfile(path, np.float32)
        assert a.size == N * V
        a = a.reshape(N, V)
        arrays.append(a)
        x, y = np.float32(a[STEP, TOK]), np.float32(hf[STEP, TOK])
        reps.append({
            "file": path,
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "heph_value": float(x),
            "heph_bits": bits(x),
            "hf_value": float(y),
            "hf_bits": bits(y),
            "signed_diff": float(x - y),
        })

    base = arrays[0]
    same = all(np.array_equal(base, a) for a in arrays[1:])
    target = base[STEP, TOK]
    same_tok = np.abs(base[:, TOK].astype(np.float64) - hf[:, TOK].astype(np.float64))
    other = np.delete(same_tok, STEP)
    report = {
        "baseline": {
            "canonical_commit": "d60630d",
            "worktree_commit": "fe3c65a",
            "command": "five independent: pixi run mojo run src/qwen_teacher_forced_full.mojo fixtures/oracle/prompt1_input_ids.json fixtures/oracle/prompt1_output_ids.json staged/qwen3-4b /tmp/.../repN_logits.f32 /tmp/.../repN_argmax.txt",
            "shape": [N, V],
            "dtype": "little-endian float32",
            "row_mapping": "row k predicts oracle output k; step67 is sequence row prompt_len-1+67=76",
        },
        "deterministic": same,
        "repetitions": reps,
        "target": {
            "step": STEP,
            "token": TOK,
            "heph_rank": int((base[STEP] > target).sum() + 1),
            "hf_rank": int((hf[STEP] > hf[STEP, TOK]).sum() + 1),
        },
        "row67": rowstats(base[STEP], hf[STEP]),
        "neighbors": {str(r): rowstats(base[r], hf[r]) for r in (66, 68)},
        "same_token_across_rows": {
            "mean": float(other.mean()),
            "median": float(np.median(other)),
            "max": float(other.max()),
            "target": float(same_tok[STEP]),
            "top10_rows": [
                {"row": int(r), "abs_diff": float(same_tok[r])}
                for r in np.argsort(-same_tok)[:10]
            ],
        },
    }
    out = "experiments/spike/out/probe0_determinism.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
