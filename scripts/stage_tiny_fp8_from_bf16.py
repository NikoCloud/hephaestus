#!/usr/bin/env python3
"""Quantize staged tiny BF16 → FP8 E4M3 + F32 per-row scales (absmax).

Norms (1D BF16) stay BF16. 2D weights become F8_E4M3 + name_scale F32 [out,1].
Used for tiny layer-diff of W8A8 vs BF16.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16, float8_e4m3fn

FP8_MAX = 448.0


def parse_offsets(path: Path):
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, off, size, shape, dtype = line.split("\t")
        shape_t = tuple(int(x) for x in shape.split(",")) if shape else ()
        entries.append(
            {
                "name": name,
                "offset": int(off),
                "size": int(size),
                "shape": shape_t,
                "dtype": dtype,
            }
        )
    return entries


def main():
    src_prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("staged/tiny")
    dst_prefix = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("staged/tiny-fp8")
    )
    blob = (src_prefix.with_suffix(".weights")).read_bytes()
    entries = parse_offsets(src_prefix.with_suffix(".offsets"))

    out_parts: list[tuple[str, tuple[int, ...], str, bytes]] = []
    for e in entries:
        raw = blob[e["offset"] : e["offset"] + e["size"]]
        if e["dtype"] != "BF16":
            raise SystemExit(f"unexpected dtype {e}")
        arr = np.frombuffer(raw, dtype="<u2").view(bfloat16).reshape(e["shape"])
        f32 = arr.astype(np.float32)

        # 1D norms / vectors stay BF16
        if len(e["shape"]) == 1:
            out_parts.append((e["name"], e["shape"], "BF16", raw))
            continue

        # 2D: per-output-channel absmax → FP8 + scale [out, 1]
        out_c = e["shape"][0]
        scales = np.zeros((out_c, 1), dtype=np.float32)
        q = np.zeros(e["shape"], dtype=float8_e4m3fn)
        for i in range(out_c):
            row = f32[i]
            amax = float(np.max(np.abs(row))) if row.size else 0.0
            s = amax / FP8_MAX if amax > 0 else 1.0
            if s < 1e-12:
                s = 1.0
            scales[i, 0] = s
            q[i] = (row / s).astype(float8_e4m3fn)
        out_parts.append((e["name"], e["shape"], "F8_E4M3", q.tobytes()))
        out_parts.append(
            (e["name"] + "_scale", (out_c, 1), "F32", scales.tobytes())
        )

    weights_path = str(dst_prefix) + ".weights"
    offsets_path = str(dst_prefix) + ".offsets"
    off = 0
    with open(weights_path, "wb") as wf, open(offsets_path, "w") as of:
        for name, shape, dtype, data in out_parts:
            wf.write(data)
            shape_str = ",".join(str(s) for s in shape)
            of.write(f"{name}\t{off}\t{len(data)}\t{shape_str}\t{dtype}\n")
            off += len(data)
    print(f"wrote {len(out_parts)} tensors → {weights_path} ({off} bytes)")
    print(f"offsets → {offsets_path}")


if __name__ == "__main__":
    main()
