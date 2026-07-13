# v2 LDS vs v1 direct-global bit-identical gate.
# Same structured inputs as exp4b stage1/2; C must match bitwise.
#
# Usage (nightly):
#   mojo build -I $KERNELS -I src experiments/exp4c_wmma_lds/v1_vs_v2.mojo \
#       -o /tmp/exp4c_v1v2
#   /tmp/exp4c_v1v2

from std.gpu.host import DeviceContext
from std.memory import bitcast
from layout import Coord, TileTensor
from layout.tile_layout import row_major
from std.utils.index import Index

from hephaestus.wmma_gfx12 import wmma_gemm_bf16, wmma_gemm_bf16_v1

comptime BF16 = DType.bfloat16


def fill_structured(
    a_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
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


def run_case(M: Int, N: Int, K: Int) raises -> Bool:
    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[BF16](M * K)
    var w_dev = ctx.enqueue_create_buffer[BF16](N * K)
    var c_v1 = ctx.enqueue_create_buffer[BF16](M * N)
    var c_v2 = ctx.enqueue_create_buffer[BF16](M * N)

    with a_dev.map_to_host() as ha, w_dev.map_to_host() as hw:
        fill_structured(
            ha.unsafe_ptr().as_unsafe_any_origin(),
            hw.unsafe_ptr().as_unsafe_any_origin(),
            M,
            N,
            K,
        )

    ctx.enqueue_memset(c_v1, 0)
    ctx.enqueue_memset(c_v2, 0)

    var a_tt = TileTensor(a_dev, row_major(Coord(Index(M, K))))
    var w_tt = TileTensor(w_dev, row_major(Coord(Index(N, K))))
    var c1_tt = TileTensor(c_v1, row_major(Coord(Index(M, N))))
    var c2_tt = TileTensor(c_v2, row_major(Coord(Index(M, N))))

    wmma_gemm_bf16_v1[BF16](c1_tt, a_tt, w_tt, M, N, K, ctx)
    wmma_gemm_bf16[BF16](c2_tt, a_tt, w_tt, M, N, K, ctx)
    ctx.synchronize()

    var n_bad = 0
    var first_m = -1
    var first_n = -1
    with c_v1.map_to_host() as h1, c_v2.map_to_host() as h2:
        for i in range(M * N):
            # Bit compare via float round-trip is wrong; compare raw bits.
            var u1 = bitcast[DType.uint16, 1](SIMD[BF16, 1](h1[i]))[0]
            var u2 = bitcast[DType.uint16, 1](SIMD[BF16, 1](h2[i]))[0]
            if u1 != u2:
                n_bad += 1
                if first_m < 0:
                    first_m = i // N
                    first_n = i % N
        print(
            "case",
            M,
            "x",
            N,
            "x",
            K,
            "mismatches",
            n_bad,
            "/",
            M * N,
        )
        if n_bad > 0:
            var i0 = first_m * N + first_n
            print(
                "  first C[",
                first_m,
                ",",
                first_n,
                "] v1=",
                Float32(h1[i0]),
                " v2=",
                Float32(h2[i0]),
            )
    return n_bad == 0


def main() raises:
    var ok = True
    # Stage shapes from exp4b + a non-multiple-of-16 M (edge mask).
    ok = run_case(16, 16, 32) and ok
    ok = run_case(32, 32, 32) and ok
    ok = run_case(4, 128, 128) and ok  # partial M (tiny-like)
    ok = run_case(32, 256, 64) and ok
    if ok:
        print("PASS: v2 bit-identical to v1 on all cases")
    else:
        print("FAIL: v2 differs from v1")
        raise Error("v2 vs v1 bitwise mismatch")
