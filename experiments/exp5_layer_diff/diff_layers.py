#!/usr/bin/env python3
"""Compare two activation-dump directories from dump_activations.mojo.

Subcommands:
  pack   <raw_dir> <npy_dir>   — convert .raw + manifest.tsv → .npy
  diff   <dir_a> <dir_b>       — per-tensor / per-layer report (npy dirs)
  self-test                    — pack+diff helper used by run.sh

Tolerance (same as exp4b GEMM gate):  1e-5 + 1.6e-2 * |reference|

Forward-pass order is used to name the *first* divergent cut point, not
alphabetical file order.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ATOL = 1e-5
RTOL = 1.6e-2

# Op order inside a layer — must match dump_activations.mojo cut points.
LAYER_OPS: list[str] = [
    "attn_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "rope_q",
    "rope_k",
    "attention",
    "o_proj_residual",
    "ffn_norm",
    "gate_proj",
    "up_proj",
    "silu_mul",
    "down_proj_residual",
]
FINAL_OPS: list[str] = ["output_norm", "lm_head"]

_LAYER_RE = re.compile(r"^layer(\d+)_step(\d+)_(.+)$")
_FINAL_RE = re.compile(r"^final_step(\d+)_(.+)$")


def pack_dir(raw_dir: Path, npy_dir: Path) -> int:
    """Convert manifest.tsv + *.raw into ml_dtypes/numpy .npy files."""
    manifest = raw_dir / "manifest.tsv"
    if not manifest.is_file():
        print(f"missing {manifest}", file=sys.stderr)
        return 2
    npy_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            print(f"bad manifest line: {line!r}", file=sys.stderr)
            return 2
        stem, dtype, d0_s, d1_s = parts
        rows, cols = int(d0_s), int(d1_s)
        raw_path = raw_dir / f"{stem}.raw"
        if not raw_path.is_file():
            print(f"missing raw {raw_path}", file=sys.stderr)
            return 2
        payload = raw_path.read_bytes()
        if dtype == "bf16":
            need = rows * cols * 2
            if len(payload) != need:
                print(
                    f"{stem}: raw size {len(payload)} != {need}",
                    file=sys.stderr,
                )
                return 2
            arr = (
                np.frombuffer(payload, dtype="<u2")
                .view(bfloat16)
                .reshape(rows, cols)
                .copy()
            )
        elif dtype == "f32":
            need = rows * cols * 4
            if len(payload) != need:
                print(
                    f"{stem}: raw size {len(payload)} != {need}",
                    file=sys.stderr,
                )
                return 2
            arr = (
                np.frombuffer(payload, dtype="<f4").reshape(rows, cols).copy()
            )
        else:
            print(f"unknown dtype {dtype}", file=sys.stderr)
            return 2
        out = npy_dir / f"{stem}.npy"
        np.save(out, arr)
        n += 1
    # copy manifest for humans
    (npy_dir / "manifest.tsv").write_text(manifest.read_text())
    print(f"packed {n} tensors → {npy_dir}")
    return 0


def parse_stem(stem: str) -> tuple[int, int, int, str]:
    """Return (layer_or_big, step, op_index, op_name) for forward-order sort.

    Final tensors use layer_or_big = 10_000 so they sort after all layers.
    op_index precedes op_name so sort follows LAYER_OPS / FINAL_OPS, not alpha.
    """
    m = _LAYER_RE.match(stem)
    if m:
        layer = int(m.group(1))
        step = int(m.group(2))
        op = m.group(3)
        try:
            op_i = LAYER_OPS.index(op)
        except ValueError:
            op_i = 1000
        return layer, step, op_i, op
    m = _FINAL_RE.match(stem)
    if m:
        step = int(m.group(1))
        op = m.group(2)
        try:
            op_i = FINAL_OPS.index(op)
        except ValueError:
            op_i = 1000
        return 10_000, step, op_i, op
    return 9_999, 0, 0, stem


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    # ml_dtypes bf16 often reloads as void V2
    if arr.dtype == bfloat16:
        return arr
    if arr.dtype.itemsize == 2 and arr.dtype != np.float16:
        return arr.view(bfloat16)
    return arr


def tensor_stats(
    got: np.ndarray,
    ref: np.ndarray,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[float, float, int, int, bool]:
    """max_abs, max_rel, n_exceed, n_elems, bitexact."""
    if got.shape != ref.shape:
        raise ValueError(f"shape mismatch {got.shape} vs {ref.shape}")
    # bitexact on raw bytes
    g_bytes = got.tobytes()
    r_bytes = ref.tobytes()
    bitexact = g_bytes == r_bytes

    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    ae = np.abs(g - r)
    re = ae / np.maximum(np.abs(r), 1e-30)
    tol = atol + rtol * np.abs(r)
    n_exceed = int(np.sum(ae > tol))
    return float(ae.max()), float(re.max()), n_exceed, int(got.size), bitexact


def list_stems(npy_dir: Path) -> list[str]:
    stems = sorted(
        p.stem for p in npy_dir.glob("*.npy") if p.is_file()
    )
    stems.sort(key=lambda s: parse_stem(s))
    return stems


def diff_dirs(
    dir_a: Path,
    dir_b: Path,
    exact: bool,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> int:
    """Compare dir_a (reference) vs dir_b (candidate)."""
    stems_a = set(list_stems(dir_a))
    stems_b = set(list_stems(dir_b))
    only_a = sorted(stems_a - stems_b)
    only_b = sorted(stems_b - stems_a)
    common = [s for s in list_stems(dir_a) if s in stems_b]

    if only_a:
        print(f"only in {dir_a}: {only_a[:8]}{'...' if len(only_a)>8 else ''}")
    if only_b:
        print(f"only in {dir_b}: {only_b[:8]}{'...' if len(only_b)>8 else ''}")
    if not common:
        print("no common tensors", file=sys.stderr)
        return 2

    print(f"comparing {len(common)} tensors  ref={dir_a}  got={dir_b}")
    print(
        f"tolerance: {atol} + {rtol}*|ref|"
        + ("  mode=EXACT" if exact else "")
    )

    first_div: str | None = None
    first_div_layer: int | None = None
    first_div_op: str | None = None
    n_fail = 0
    n_bitexact = 0
    rows: list[tuple[str, float, float, int, int, bool]] = []

    for stem in common:
        ref = load_npy(dir_a / f"{stem}.npy")
        got = load_npy(dir_b / f"{stem}.npy")
        max_abs, max_rel, n_ex, n_el, bitexact = tensor_stats(
            got, ref, atol=atol, rtol=rtol
        )
        rows.append((stem, max_abs, max_rel, n_ex, n_el, bitexact))
        if bitexact:
            n_bitexact += 1
        fail = (not bitexact) if exact else (n_ex > 0)
        if fail:
            n_fail += 1
            if first_div is None:
                first_div = stem
                layer, _step, _oi, op = parse_stem(stem)
                first_div_layer = layer
                first_div_op = op

    # Per-layer rollup
    by_layer: dict[int, list[tuple[str, float, float, int, bool]]] = {}
    for stem, max_abs, max_rel, n_ex, _n_el, bitexact in rows:
        layer, _s, _oi, op = parse_stem(stem)
        by_layer.setdefault(layer, []).append(
            (op if layer < 10_000 else stem, max_abs, max_rel, n_ex, bitexact)
        )

    print("\n=== per-layer summary ===")
    for layer in sorted(by_layer.keys()):
        entries = by_layer[layer]
        label = "final" if layer >= 10_000 else f"layer {layer}"
        worst = max(entries, key=lambda e: e[1])
        if exact:
            n_bad = sum(1 for e in entries if not e[4])
        else:
            n_bad = sum(1 for e in entries if e[3] > 0)
        n_exact = sum(1 for e in entries if e[4])
        print(
            f"{label:10s}  tensors={len(entries):2d}  bitexact={n_exact}/{len(entries)}  "
            f"failing={n_bad}  worst={worst[0]} max_abs={worst[1]:.3e} max_rel={worst[2]:.3e}"
        )

    print("\n=== per-tensor (forward order) ===")
    for stem, max_abs, max_rel, n_ex, n_el, bitexact in rows:
        status = "EXACT" if bitexact else ("FAIL" if (n_ex > 0 or exact) else "ok")
        if bitexact:
            status = "EXACT"
        elif exact or n_ex > 0:
            status = "FAIL"
        else:
            status = "ok"
        print(
            f"  {status:5s}  {stem:40s}  max_abs={max_abs:.3e}  "
            f"max_rel={max_rel:.3e}  exceed={n_ex}/{n_el}"
        )

    print("\n=== first divergence ===")
    if first_div is None:
        if exact:
            print(
                f"NONE — all {len(common)} tensors bit-identical "
                f"({n_bitexact}/{len(common)})"
            )
        else:
            print(
                f"NONE — all {len(common)} tensors within tolerance "
                f"(bitexact {n_bitexact}/{len(common)})"
            )
        print("PASS")
        return 0

    layer_s = (
        "final"
        if first_div_layer is not None and first_div_layer >= 10_000
        else f"layer {first_div_layer}"
    )
    print(f"FIRST: {first_div}")
    print(f"  where: {layer_s}  op={first_div_op}")
    print(f"  → investigate this cut point before later layers")
    print(f"FAIL ({n_fail} tensors differ)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="raw+manifest → npy")
    p.add_argument("raw_dir", type=Path)
    p.add_argument("npy_dir", type=Path)

    d = sub.add_parser("diff", help="compare two npy dump dirs")
    d.add_argument("dir_a", type=Path, help="reference dump dir")
    d.add_argument("dir_b", type=Path, help="candidate dump dir")
    d.add_argument(
        "--exact",
        action="store_true",
        help="require bitwise identity (determinism self-test)",
    )
    d.add_argument(
        "--atol",
        type=float,
        default=None,
        help="absolute tolerance (default 1e-5; use 1e-3 for W8A8)",
    )
    d.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="relative tolerance (default 1.6e-2; use 5e-2 for W8A8)",
    )

    args = ap.parse_args()
    if args.cmd == "pack":
        return pack_dir(args.raw_dir, args.npy_dir)
    if args.cmd == "diff":
        atol = ATOL if args.atol is None else args.atol
        rtol = RTOL if args.rtol is None else args.rtol
        return diff_dirs(
            args.dir_a, args.dir_b, args.exact, atol=atol, rtol=rtol
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
