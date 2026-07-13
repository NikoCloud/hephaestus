# Bit-identical gate: fused residual WMMA store vs separate linear + residual_add.
#
# residual_f: wmma_gemm_bf16_residual(residual, A, W)   # residual += A@W^T fused
# residual_s: wmma_gemm_bf16(C, A, W); residual += C    # separate (same cast order)
#
# Usage (nightly):
#   mojo build -I $KERNELS -I src experiments/exp4d_wmma_residual/fused_vs_separate.mojo \
#       -o /tmp/exp4d_fused
#   /tmp/exp4d_fused

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major

from hephaestus.wmma_gfx12 import wmma_gemm_bf16, wmma_gemm_bf16_residual

comptime BF16 = DType.bfloat16
comptime F32 = DType.float32
comptime TPB = 256


def fill_structured(
    a_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    r_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    for m in range(M):
        for k in range(K):
            a_ptr[m * K + k] = Float32(m * K + k).cast[BF16]()
    for n in range(N):
        for k in range(K):
            w_ptr[n * K + k] = Float32(n * K + k).cast[BF16]()
    # Non-zero residual so the add is exercised.
    for m in range(M):
        for n in range(N):
            r_ptr[m * N + n] = Float32(m + n).cast[BF16]()


def residual_add_kernel(
    residual_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n_elems: Int,
):
    """residual[i] = (F32(residual[i]) + F32(c[i])).cast[BF16]() — same cast order as fused."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_elems:
        return
    var old = residual_ptr[i].cast[F32]()
    var add = c_ptr[i].cast[F32]()
    residual_ptr[i] = (old + add).cast[BF16]()


def run_case(M: Int, N: Int, K: Int) raises -> Bool:
    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[BF16](M * K)
    var w_dev = ctx.enqueue_create_buffer[BF16](N * K)
    var r_fused = ctx.enqueue_create_buffer[BF16](M * N)
    var r_sep = ctx.enqueue_create_buffer[BF16](M * N)
    var c_tmp = ctx.enqueue_create_buffer[BF16](M * N)

    with a_dev.map_to_host() as ha, w_dev.map_to_host() as hw, r_fused.map_to_host() as hr:
        fill_structured(
            ha.unsafe_ptr().as_unsafe_any_origin(),
            hw.unsafe_ptr().as_unsafe_any_origin(),
            hr.unsafe_ptr().as_unsafe_any_origin(),
            M,
            N,
            K,
        )
    # Copy same residual init to separate path.
    with r_fused.map_to_host() as hf, r_sep.map_to_host() as hs:
        for i in range(M * N):
            hs[i] = hf[i]

    var a_tt = TileTensor(a_dev, row_major(Coord(Index(M, K))))
    var w_tt = TileTensor(w_dev, row_major(Coord(Index(N, K))))
    var rf_tt = TileTensor(r_fused, row_major(Coord(Index(M, N))))
    var rs_tt = TileTensor(r_sep, row_major(Coord(Index(M, N))))
    var c_tt = TileTensor(c_tmp, row_major(Coord(Index(M, N))))

    # Fused
    wmma_gemm_bf16_residual(rf_tt, a_tt, w_tt, M, N, K, ctx)

    # Separate: C = A@W^T then residual += C
    ctx.enqueue_memset(c_tmp, 0)
    wmma_gemm_bf16[BF16](c_tt, a_tt, w_tt, M, N, K, ctx)
    ctx.enqueue_function[residual_add_kernel](
        r_sep.unsafe_ptr().as_unsafe_any_origin(),
        c_tmp.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        M * N,
        grid_dim=(ceildiv(M * N, TPB),),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    var n_bad = 0
    var first = -1
    with r_fused.map_to_host() as hf, r_sep.map_to_host() as hs:
        for i in range(M * N):
            var u1 = bitcast[DType.uint16, 1](SIMD[BF16, 1](hf[i]))[0]
            var u2 = bitcast[DType.uint16, 1](SIMD[BF16, 1](hs[i]))[0]
            if u1 != u2:
                n_bad += 1
                if first < 0:
                    first = i
        print("case", M, "x", N, "x", K, "mismatches", n_bad, "/", M * N)
        if n_bad > 0 and first >= 0:
            print(
                "  first idx",
                first,
                " fused=",
                Float32(hf[first]),
                " separate=",
                Float32(hs[first]),
            )
    return n_bad == 0


def main() raises:
    var ok = True
    ok = run_case(16, 16, 32) and ok
    ok = run_case(32, 32, 32) and ok
    ok = run_case(4, 128, 128) and ok
    ok = run_case(32, 256, 64) and ok
    if ok:
        print("PASS: fused residual bit-identical to separate linear + add")
    else:
        print("FAIL: fused residual differs from separate path")
        raise Error("fused vs separate bitwise mismatch")
