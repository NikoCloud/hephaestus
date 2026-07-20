# G1b-3a correctness gates (spec §8–§9):
#   1) v3a plain  vs v1         — bitwise
#   2) v3a fused  vs (v1 + F32 residual cast) — bitwise
#
# Usage (nightly):
#   mojo build -I $KERNELS -I src experiments/exp4e_wmma_v3a/v3a_gates.mojo \
#       -o /tmp/exp4e_v3a
#   /tmp/exp4e_v3a

from std.gpu.host import DeviceContext
from std.memory import bitcast
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major

from hephaestus.wmma_gfx12 import (
    wmma_gemm_bf16_v1,
    wmma_gemm_bf16_v3a,
    wmma_gemm_bf16_v3a_residual,
)

comptime BF16 = DType.bfloat16
comptime F32 = DType.float32


def fill_aw(
    a_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    for m in range(M):
        for k in range(K):
            a_ptr[m * K + k] = Float32((m * 17 + k * 3) % 251).cast[BF16]()
    for n in range(N):
        for k in range(K):
            w_ptr[n * K + k] = Float32((n * 13 + k * 7) % 241).cast[BF16]()


def fill_residual(
    r_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin], M: Int, N: Int
):
    for m in range(M):
        for n in range(N):
            r_ptr[m * N + n] = Float32(m + 2 * n + 1).cast[BF16]()


def bits_eq_bf16(
    a: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    b: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n: Int,
) -> Int:
    var bad = 0
    for i in range(n):
        var u1 = bitcast[DType.uint16, 1](SIMD[BF16, 1](a[i]))[0]
        var u2 = bitcast[DType.uint16, 1](SIMD[BF16, 1](b[i]))[0]
        if u1 != u2:
            bad += 1
    return bad


def gate_plain(M: Int, N: Int, K: Int) raises -> Bool:
    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[BF16](M * K)
    var w_dev = ctx.enqueue_create_buffer[BF16](N * K)
    var c_v1 = ctx.enqueue_create_buffer[BF16](M * N)
    var c_v3 = ctx.enqueue_create_buffer[BF16](M * N)
    with a_dev.map_to_host() as ha, w_dev.map_to_host() as hw:
        fill_aw(
            ha.unsafe_ptr().as_unsafe_any_origin(),
            hw.unsafe_ptr().as_unsafe_any_origin(),
            M,
            N,
            K,
        )
    ctx.enqueue_memset(c_v1, 0)
    ctx.enqueue_memset(c_v3, 0)
    var a_tt = TileTensor(a_dev, row_major(Coord(Index(M, K))))
    var w_tt = TileTensor(w_dev, row_major(Coord(Index(N, K))))
    var c1 = TileTensor(c_v1, row_major(Coord(Index(M, N))))
    var c3 = TileTensor(c_v3, row_major(Coord(Index(M, N))))
    wmma_gemm_bf16_v1[BF16](c1, a_tt, w_tt, M, N, K, ctx)
    wmma_gemm_bf16_v3a[BF16](c3, a_tt, w_tt, M, N, K, ctx)
    ctx.synchronize()
    var bad = 0
    with c_v1.map_to_host() as h1, c_v3.map_to_host() as h3:
        bad = bits_eq_bf16(
            h1.unsafe_ptr().as_unsafe_any_origin(),
            h3.unsafe_ptr().as_unsafe_any_origin(),
            M * N,
        )
    print("plain", M, "x", N, "x", K, "mismatches", bad, "/", M * N)
    return bad == 0


def gate_fused(M: Int, N: Int, K: Int) raises -> Bool:
    """v3a-fused vs (v1 F32 product + F32 residual → cast BF16).

    Uses v1 with F32 C so the product never rounds through BF16 before the
    residual add — matches §6 (val = F32 acc + residual, then cast).
    """
    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[BF16](M * K)
    var w_dev = ctx.enqueue_create_buffer[BF16](N * K)
    var r_init = ctx.enqueue_create_buffer[BF16](M * N)
    var c_v1 = ctx.enqueue_create_buffer[F32](M * N)
    var r_v3 = ctx.enqueue_create_buffer[BF16](M * N)
    var r_ref = ctx.enqueue_create_buffer[BF16](M * N)

    with a_dev.map_to_host() as ha, w_dev.map_to_host() as hw, r_init.map_to_host() as hr:
        fill_aw(
            ha.unsafe_ptr().as_unsafe_any_origin(),
            hw.unsafe_ptr().as_unsafe_any_origin(),
            M,
            N,
            K,
        )
        fill_residual(hr.unsafe_ptr().as_unsafe_any_origin(), M, N)

    # Copy residual init into r_v3 (fused RMW).
    with r_init.map_to_host() as hi, r_v3.map_to_host() as hv:
        for i in range(M * N):
            hv[i] = hi[i]

    ctx.enqueue_memset(c_v1, 0)
    var a_tt = TileTensor(a_dev, row_major(Coord(Index(M, K))))
    var w_tt = TileTensor(w_dev, row_major(Coord(Index(N, K))))
    var c1 = TileTensor(c_v1, row_major(Coord(Index(M, N))))
    var rv3 = TileTensor(r_v3, row_major(Coord(Index(M, N))))

    # v1 plain product in F32 (full acc, no BF16 store)
    wmma_gemm_bf16_v1[F32](c1, a_tt, w_tt, M, N, K, ctx)
    # v3a fused
    wmma_gemm_bf16_v3a_residual(rv3, a_tt, w_tt, M, N, K, ctx)
    ctx.synchronize()

    # Host: r_ref = cast_bf16(f32(v1) + f32(residual_init))
    with c_v1.map_to_host() as hv1, r_init.map_to_host() as hinit, r_ref.map_to_host() as href, r_v3.map_to_host() as hv3:
        for i in range(M * N):
            var sum = hv1[i] + Float32(hinit[i])
            href[i] = sum.cast[BF16]()
        var bad = bits_eq_bf16(
            href.unsafe_ptr().as_unsafe_any_origin(),
            hv3.unsafe_ptr().as_unsafe_any_origin(),
            M * N,
        )
        print("fused", M, "x", N, "x", K, "mismatches", bad, "/", M * N)
        return bad == 0


def main() raises:
    var ok = True
    # §8 plain shapes
    ok = gate_plain(64, 64, 64) and ok
    ok = gate_plain(64, 256, 64) and ok
    ok = gate_plain(128, 128, 64) and ok
    # §8 fused shapes (o/down-like)
    ok = gate_fused(64, 64, 64) and ok
    ok = gate_fused(64, 256, 64) and ok
    ok = gate_fused(128, 128, 128) and ok
    if ok:
        print("PASS: v3a plain vs v1 and v3a-fused vs (v1+F32 residual)")
    else:
        print("FAIL: v3a gate")
        raise Error("v3a bitwise gate failed")
