#!/usr/bin/env python3
"""CPU reference matmul for G1b-0 tile tests T1/T2/T3.

Exact integer expected outputs (BF16-exact inputs, F32 accum).
Compares GPU dump (16×16 f32 LE) against oracle with == (no tolerance).

Usage:
  python3 oracle.py T1 /tmp/d_T1.f32
  python3 oracle.py --write-expected T1 /tmp/exp_T1.f32   # dump only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TILE = 16


def make_ab(test: str) -> tuple[np.ndarray, np.ndarray]:
    """Return A, B as float64 16×16 (values are integers / 0-1)."""
    A = np.zeros((TILE, TILE), dtype=np.float64)
    B = np.zeros((TILE, TILE), dtype=np.float64)
    if test == "T1":
        for m in range(TILE):
            for k in range(TILE):
                A[m, k] = m
        for k in range(TILE):
            for n in range(TILE):
                B[k, n] = n
    elif test == "T2":
        for m in range(TILE):
            for k in range(TILE):
                A[m, k] = m
        B = np.eye(TILE, dtype=np.float64)
    elif test == "T3":
        A = np.eye(TILE, dtype=np.float64)
        for k in range(TILE):
            for n in range(TILE):
                B[k, n] = n
    else:
        raise SystemExit(f"unknown test {test}")
    return A, B


def cpu_ref(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """D[m,n] = sum_k A[m,k]*B[k,n] with f32-style product then sum.

    Inputs are integers in [0,15] / {0,1}; cast through float32 for parity
    with BF16→F32 cast then multiply (still exact for these magnitudes).
    """
    D = np.zeros((TILE, TILE), dtype=np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)
    for m in range(TILE):
        for n in range(TILE):
            acc = np.float32(0.0)
            for k in range(TILE):
                acc = acc + Af[m, k] * Bf[k, n]
            D[m, n] = acc
    return D


def expected(test: str) -> np.ndarray:
    A, B = make_ab(test)
    D = cpu_ref(A, B)
    # closed-form check
    if test == "T1":
        closed = np.array(
            [[16.0 * m * n for n in range(TILE)] for m in range(TILE)],
            dtype=np.float32,
        )
        assert np.array_equal(D, closed), "T1 oracle vs closed-form"
    elif test == "T2":
        closed = np.array(
            [[float(m) for _n in range(TILE)] for m in range(TILE)],
            dtype=np.float32,
        )
        assert np.array_equal(D, closed), "T2 oracle vs closed-form"
    elif test == "T3":
        closed = np.array(
            [[float(n) for n in range(TILE)] for _m in range(TILE)],
            dtype=np.float32,
        )
        assert np.array_equal(D, closed), "T3 oracle vs closed-form"
    return D


def diagnose(test: str, got: np.ndarray, exp: np.ndarray) -> str:
    """Apply §9 diagnosis table based on mismatch patterns."""
    # Check if got looks like transpose of expected
    is_transpose = np.array_equal(got, exp.T)
    # Check if T2 got looks like T3 expected and vice versa
    e2, e3 = expected("T2"), expected("T3")
    t2_looks_like_t3 = test == "T2" and np.array_equal(got, e3)
    t3_looks_like_t2 = test == "T3" and np.array_equal(got, e2)

    # Per half-row blocks
    top = got[:8, :]
    bot = got[8:, :]
    exp_top, exp_bot = exp[:8, :], exp[8:, :]
    half_swap = np.array_equal(top, exp_bot) and np.array_equal(bot, exp_top)

    lines = []
    n_bad = int(np.sum(got != exp))
    lines.append(f"mismatches: {n_bad}/256")
    if is_transpose:
        lines.append("pattern: got == exp.T  → likely C/D-store mapping wrong (§2 C/D)")
    if t2_looks_like_t3 or t3_looks_like_t2:
        lines.append("pattern: T2/T3 swapped  → A and B fragments swapped")
    if half_swap:
        lines.append("pattern: rows 0-7 ↔ 8-15  → k_half = l/16 flipped")
    # sample first few mismatches
    idx = np.argwhere(got != exp)
    for m, n in idx[:8]:
        lines.append(f"  D[{m},{n}] got={got[m,n]} exp={exp[m,n]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["T1", "T2", "T3", "all"])
    ap.add_argument("path", nargs="?", help="GPU dump .f32 (16*16 LE float32)")
    ap.add_argument(
        "--write-expected",
        action="store_true",
        help="write expected tile to path and exit",
    )
    args = ap.parse_args()

    tests = ["T1", "T2", "T3"] if args.test == "all" else [args.test]
    if args.write_expected:
        if not args.path or len(tests) != 1:
            print("--write-expected needs one test and a path", file=sys.stderr)
            return 2
        expected(tests[0]).astype("<f4").tofile(args.path)
        print(f"wrote expected {tests[0]} -> {args.path}")
        return 0

    if not args.path:
        # just print sample expected values
        for t in tests:
            D = expected(t)
            print(f"{t}: D[0,0]={D[0,0]} D[1,1]={D[1,1]} D[2,3]={D[2,3]} D[15,15]={D[15,15]}")
        return 0

    got = np.fromfile(args.path, dtype="<f4")
    if got.size != TILE * TILE:
        print(f"bad dump size {got.size}, want {TILE*TILE}", file=sys.stderr)
        return 2
    got = got.reshape(TILE, TILE)

    # multi-test mode uses path as directory
    if args.test == "all":
        print("use per-test path for comparison", file=sys.stderr)
        return 2

    exp = expected(args.test)
    if np.array_equal(got, exp):
        print(f"{args.test} PASS (exact equality, 256/256)")
        return 0

    print(f"{args.test} FAIL")
    print(diagnose(args.test, got, exp))
    print("\n§9 diagnosis table: use T1/T2/T3 pass pattern across all three runs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
