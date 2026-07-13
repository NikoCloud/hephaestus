#!/usr/bin/env python3
"""CPU reference for multi-tile BF16 WMMA GEMM.

C[M, N] = A[M, K] @ W[N, K]^T
F32 accumulate, BF16 store (ml_dtypes.bfloat16).

Modes:
  structured  — A[m,k]=m*K+k, W[n,k]=n*K+k (cast through BF16)
  random      — load the same .npy files the kernel uses (seed 42 upstream)

Usage:
  # generate shared random inputs once
  python oracle.py gen-random 32 4096 2560 /tmp/exp4b_inputs

  # exact compare (stages 1–2)
  python oracle.py compare structured 16 16 32 /tmp/c.bf16 --exact

  # tolerance compare (stage 3)
  python oracle.py compare random 32 4096 2560 /tmp/c.bf16 \\
      --a /tmp/exp4b_inputs/A.npy --w /tmp/exp4b_inputs/W.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

TILE = 16
SEED = 42
# User-stated pass band (but near-bound clustering is a red flag).
ATOL = 1e-5
RTOL = 1.6e-2


def assert_divisible(M: int, N: int, K: int) -> None:
    if M % TILE or N % TILE or K % TILE:
        raise SystemExit(f"M,N,K must be divisible by {TILE}, got {M},{N},{K}")


def make_structured(M: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer lattice cast to BF16 — matches gemm.mojo fill_structured."""
    A = np.empty((M, K), dtype=bfloat16)
    W = np.empty((N, K), dtype=bfloat16)
    for m in range(M):
        for k in range(K):
            A[m, k] = bfloat16(np.float32(m * K + k))
    for n in range(N):
        for k in range(K):
            W[n, k] = bfloat16(np.float32(n * K + k))
    return A, W


def gen_random(M: int, N: int, K: int, out_dir: Path) -> None:
    """Generate A, W once as BF16; persist .npy (byte-identical for kernel+oracle)."""
    assert_divisible(M, N, K)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Uniform-ish in a moderate range so F32 accum is well-conditioned.
    A_f32 = rng.standard_normal((M, K), dtype=np.float32).astype(np.float32) * 0.5
    W_f32 = rng.standard_normal((N, K), dtype=np.float32).astype(np.float32) * 0.5
    A = A_f32.astype(bfloat16)
    W = W_f32.astype(bfloat16)
    a_path = out_dir / "A.npy"
    w_path = out_dir / "W.npy"
    np.save(a_path, A)
    np.save(w_path, W)
    # meta for humans
    (out_dir / "meta.txt").write_text(
        f"seed={SEED}\nM={M}\nN={N}\nK={K}\ndtype=ml_dtypes.bfloat16\n"
        f"A={a_path}\nW={w_path}\n"
    )
    print(f"wrote {a_path} shape={A.shape} dtype={A.dtype}")
    print(f"wrote {w_path} shape={W.shape} dtype={W.dtype}")


def load_aw(
    mode: str,
    M: int,
    N: int,
    K: int,
    a_path: Path | None,
    w_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "structured":
        return make_structured(M, N, K)
    if mode == "random":
        if not a_path or not w_path:
            raise SystemExit("random mode requires --a and --w")
        A = np.load(a_path)
        W = np.load(w_path)
        if A.shape != (M, K) or W.shape != (N, K):
            raise SystemExit(
                f"shape mismatch: A{A.shape} W{W.shape} vs expected "
                f"A({M},{K}) W({N},{K})"
            )
        # ml_dtypes.bfloat16 often reloads as void('<V2'); re-view as bfloat16.
        if A.dtype != bfloat16:
            if A.dtype.itemsize != 2:
                raise SystemExit(f"A dtype {A.dtype} is not 2-byte")
            A = A.view(bfloat16)
        if W.dtype != bfloat16:
            if W.dtype.itemsize != 2:
                raise SystemExit(f"W dtype {W.dtype} is not 2-byte")
            W = W.view(bfloat16)
        return A, W
    raise SystemExit(f"unknown mode {mode}")


def cpu_ref_f32(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """C = A @ W^T with F32 accumulate (no BF16 store)."""
    Af = A.astype(np.float32)
    Wf = W.astype(np.float32)
    # (M,K) @ (K,N) = (M,N)  where W^T is (K,N)
    return Af @ Wf.T


def cpu_ref(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """C = A @ W^T with F32 accumulate, BF16 store.

    A is [M,K] bf16, W is [N,K] bf16. Cast to f32 before mul/add.
    """
    return cpu_ref_f32(A, W).astype(bfloat16)


def load_gpu_bf16(path: Path, M: int, N: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != M * N:
        raise SystemExit(f"bad dump size {raw.size}, want {M*N} bf16 elems")
    return raw.view(bfloat16).reshape(M, N)


def rel_err(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    denom = np.maximum(np.abs(r), 1e-30)
    return np.abs(g - r) / denom


def abs_err(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(got.astype(np.float64) - ref.astype(np.float64))


def per_tile_stats(
    got: np.ndarray, ref: np.ndarray, M: int, N: int
) -> list[tuple[int, int, float, float, int]]:
    """Return list of (m_tile, n_tile, max_abs, max_rel, n_exceed) per 16×16 tile."""
    rows = []
    tol = ATOL + RTOL * np.abs(ref.astype(np.float64))
    ae = abs_err(got, ref)
    re = rel_err(got, ref)
    exceed = ae > tol
    for mt in range(M // TILE):
        for nt in range(N // TILE):
            sl = (
                slice(mt * TILE, (mt + 1) * TILE),
                slice(nt * TILE, (nt + 1) * TILE),
            )
            rows.append(
                (
                    mt,
                    nt,
                    float(ae[sl].max()) if ae[sl].size else 0.0,
                    float(re[sl].max()) if re[sl].size else 0.0,
                    int(exceed[sl].sum()),
                )
            )
    return rows


def histogram_rel(re: np.ndarray) -> str:
    """Histogram of relative errors (log-ish bins)."""
    bins = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.6e-2, 1e-1, 1.0, np.inf]
    labels = [
        "[0,1e-8)",
        "[1e-8,1e-7)",
        "[1e-7,1e-6)",
        "[1e-6,1e-5)",
        "[1e-5,1e-4)",
        "[1e-4,1e-3)",
        "[1e-3,1e-2)",
        "[1e-2,1.6e-2)",
        "[1.6e-2,1e-1)",
        "[1e-1,1)",
        "[1,inf)",
    ]
    flat = re.ravel()
    counts, _ = np.histogram(flat, bins=bins)
    lines = ["relative error histogram:"]
    for lab, c in zip(labels, counts):
        lines.append(f"  {lab:16s}  {c}")
    return "\n".join(lines)


def compare(
    mode: str,
    M: int,
    N: int,
    K: int,
    gpu_path: Path,
    a_path: Path | None,
    w_path: Path | None,
    exact: bool,
) -> int:
    assert_divisible(M, N, K)
    A, W = load_aw(mode, M, N, K, a_path, w_path)
    ref = cpu_ref(A, W)
    got = load_gpu_bf16(gpu_path, M, N)

    # Bit-exact path for stages 1–2
    if exact:
        # Compare raw uint16 bits
        g_bits = got.view(np.uint16).reshape(M, N)
        r_bits = ref.view(np.uint16).reshape(M, N)
        if np.array_equal(g_bits, r_bits):
            print(
                f"PASS exact bitwise BF16 match "
                f"({M}x{N}, K={K}, mode={mode}, {M*N} elems)"
            )
            # sample
            print(
                f"  sample C[0,0]={float(got[0,0])} "
                f"C[0,1]={float(got[0, min(1,N-1)])} "
                f"C[{min(1,M-1)},{min(1,N-1)}]="
                f"{float(got[min(1,M-1), min(1,N-1)])}"
            )
            return 0
        n_bad = int(np.sum(g_bits != r_bits))
        print(f"FAIL exact: {n_bad}/{M*N} elems differ")
        idx = np.argwhere(g_bits != r_bits)
        for m, n in idx[:16]:
            print(
                f"  C[{m},{n}] got={float(got[m,n])} "
                f"(0x{int(g_bits[m,n]):04x}) "
                f"ref={float(ref[m,n])} (0x{int(r_bits[m,n]):04x})"
            )
        return 1

    # Tolerance path for stage 3
    # Primary: BF16-store vs BF16-store (both sides same quantize).
    g64 = got.astype(np.float64)
    r64 = ref.astype(np.float64)
    ae = np.abs(g64 - r64)
    re = rel_err(got, ref)
    tol = ATOL + RTOL * np.abs(r64)
    exceed = ae > tol
    n_exceed = int(exceed.sum())
    max_abs = float(ae.max())
    max_rel = float(re.max())
    i_abs = np.unravel_index(int(ae.argmax()), ae.shape)
    i_rel = np.unravel_index(int(re.argmax()), re.shape)
    n_bitexact = int(np.sum(got.view(np.uint16).reshape(M, N) ==
                            ref.view(np.uint16).reshape(M, N)))

    # Secondary: GPU BF16 vs ideal F32 accum (quantization + accum order).
    ref_f32 = cpu_ref_f32(A, W).astype(np.float64)
    ae_f = np.abs(g64 - ref_f32)
    re_f = np.abs(g64 - ref_f32) / np.maximum(np.abs(ref_f32), 1e-30)
    max_abs_f = float(ae_f.max())
    max_rel_f = float(re_f.max())
    # median/p99 relative vs F32 — the "near-F32" smell check
    re_f_flat = re_f.ravel()
    med_rel_f = float(np.median(re_f_flat))
    p99_rel_f = float(np.percentile(re_f_flat, 99))
    p99_9_rel_f = float(np.percentile(re_f_flat, 99.9))

    print(f"=== stage3 tolerance report ({M}x{N} K={K}) ===")
    print(f"[BF16 vs BF16 ref]")
    print(f"  bitexact             = {n_bitexact} / {M*N}")
    print(f"  max_abs_diff         = {max_abs:.6e}  at C{i_abs} "
          f"got={g64[i_abs]:.6e} ref={r64[i_abs]:.6e}")
    print(f"  max_relative_error   = {max_rel:.6e}  at C{i_rel} "
          f"got={g64[i_rel]:.6e} ref={r64[i_rel]:.6e}")
    print(f"  count exceeding tol ({ATOL}+{RTOL}*|ref|) = {n_exceed} / {M*N}")
    print(histogram_rel(re))

    print(f"[GPU BF16 vs ideal F32 accum]")
    print(f"  max_abs_diff         = {max_abs_f:.6e}")
    print(f"  max_relative_error   = {max_rel_f:.6e}")
    print(f"  median_rel / p99 / p99.9 = "
          f"{med_rel_f:.3e} / {p99_rel_f:.3e} / {p99_9_rel_f:.3e}")
    print(histogram_rel(re_f))

    # Red-flag only if a non-trivial mass of moderate-magnitude entries is bad.
    # Tiny refs amplify rel error from a single BF16 ULP — not structural.
    sig = np.abs(r64) >= 1e-2
    re_sig = re[sig] if sig.any() else np.array([0.0])
    near_bound_sig = int(np.sum((re_sig >= 1e-2) & (re_sig <= RTOL * 1.01)))
    high_sig = int(np.sum(re_sig > 1e-3))
    print(f"significant |ref|>=1e-2: n={int(sig.sum())}  "
          f"rel>1e-3: {high_sig}  in[1e-2,1.6e-2]: {near_bound_sig}")
    # F32 agreement red flag: p99 rel should be ~BF16 ULP (~1e-3..4e-3), not 1%
    if p99_rel_f > 5e-3 and med_rel_f > 1e-3:
        print(
            "RED FLAG: p99/median rel vs F32 look structural "
            f"(median={med_rel_f:.3e} p99={p99_rel_f:.3e}); "
            "expect ~1e-4..few-e-3 from BF16 store + strip order"
        )
    elif near_bound_sig > 0:
        print(
            "RED FLAG: significant-magnitude entries near 1.6% bound"
        )
    else:
        print(
            "smell OK: disagreements are BF16-ULP / tiny-ref dominated, "
            "not a structural tile bug"
        )

    tiles = per_tile_stats(got, ref, M, N)
    tiles_sorted = sorted(tiles, key=lambda t: t[3], reverse=True)
    print("per-tile error distribution (worst 8 by max_rel, BF16 vs BF16):")
    for mt, nt, ma, mr, ne in tiles_sorted[:8]:
        print(
            f"  tile(m={mt},n={nt}) max_abs={ma:.3e} "
            f"max_rel={mr:.3e} n_exceed={ne}"
        )
    max_rels = [t[3] for t in tiles]
    n_tiles_nonzero = sum(1 for t in max_rels if t > 0)
    print(
        f"tile max_rel: min={min(max_rels):.3e} "
        f"median={float(np.median(max_rels)):.3e} "
        f"max={max(max_rels):.3e}  n_tiles={len(tiles)} "
        f"tiles_with_any_diff={n_tiles_nonzero}"
    )

    if n_exceed == 0:
        print(
            f"PASS tolerance (bitexact={n_bitexact}/{M*N}, "
            f"max_rel_bf16={max_rel:.3e}, p99_rel_vs_f32={p99_rel_f:.3e})"
        )
        return 0

    print("FAIL tolerance")
    idx = np.argwhere(exceed)
    for m, n in idx[:12]:
        print(
            f"  C[{m},{n}] got={g64[m,n]:.6e} ref={r64[m,n]:.6e} "
            f"abs={ae[m,n]:.3e} rel={re[m,n]:.3e} "
            f"tol={tol[m,n]:.3e}"
        )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-random", help="generate A.npy W.npy once")
    g.add_argument("M", type=int)
    g.add_argument("N", type=int)
    g.add_argument("K", type=int)
    g.add_argument("out_dir", type=Path)

    c = sub.add_parser("compare", help="compare GPU dump to CPU reference")
    c.add_argument("mode", choices=["structured", "random"])
    c.add_argument("M", type=int)
    c.add_argument("N", type=int)
    c.add_argument("K", type=int)
    c.add_argument("gpu_path", type=Path)
    c.add_argument("--a", type=Path, default=None)
    c.add_argument("--w", type=Path, default=None)
    c.add_argument(
        "--exact",
        action="store_true",
        help="require bitwise BF16 equality (stages 1–2)",
    )

    # optional: print closed-form samples for structured small cases
    s = sub.add_parser("expected-sample", help="print a few structured C values")
    s.add_argument("M", type=int)
    s.add_argument("N", type=int)
    s.add_argument("K", type=int)

    args = ap.parse_args()

    if args.cmd == "gen-random":
        gen_random(args.M, args.N, args.K, args.out_dir)
        return 0

    if args.cmd == "expected-sample":
        A, W = make_structured(args.M, args.N, args.K)
        C = cpu_ref(A, W)
        print(f"structured {args.M}x{args.N} K={args.K}")
        for m, n in [(0, 0), (0, 1), (1, 0), (1, 1), (15, 15)]:
            if m < args.M and n < args.N:
                print(f"  C[{m},{n}] = {float(C[m,n])}")
        return 0

    if args.cmd == "compare":
        return compare(
            args.mode,
            args.M,
            args.N,
            args.K,
            args.gpu_path,
            args.a,
            args.w,
            args.exact,
        )

    return 2


if __name__ == "__main__":
    sys.exit(main())
