# Experiment 3 (spec 2026-07-12 open question 1): kernel launch path.
#
# Question: can the forward pass call the high-level matmul() dispatcher
# (linalg.matmul) rather than launching gemm_kernel_rdna manually via
# enqueue_function?
#
# Dispatch facts read from linalg/matmul/gpu/__init__.mojo (2026-07-12):
#   - m == 1 or n == 1  -> gemv_gpu          (our decode steps)
#   - m > 1, k % 16 == 0 -> RDNA WMMA 64x64  (small prefill)
#   - m >= 128, k % 32 == 0 -> RDNA WMMA 128x128 BK=32 (real prefill)
# All our K dims (2560, 4096, 9728) are % 32 == 0.
#
# Test: q_proj-shaped gemm C[M,4096] = A[M,2560] @ B[4096,2560]^T at M=1
# (decode/gemv), M=4 (tiny prefill/WMMA 64x64), and M=128 (WMMA 128x128),
# BF16 in/out, vs CPU fp32 reference. Tolerances from MAX's own
# test_4wave_matmul.mojo: |diff| <= 1e-5 + 1.6e-2 * |expected|.
#
# Run: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#          experiments/exp3_matmul_dispatch.mojo

from std.gpu.host import DeviceContext
from std.random import random_float64, seed

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul

comptime K = 2560
comptime N = 4096


def run_case[M: Int](ctx: DeviceContext) raises:
    seed(42 + M)
    var dev_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var dev_b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var dev_c = ctx.enqueue_create_buffer[DType.bfloat16](M * N)

    # Keep host copies for the CPU reference.
    var host_a = List[Float32]()
    var host_b = List[Float32]()
    with dev_a.map_to_host() as ha, dev_b.map_to_host() as hb:
        for i in range(M * K):
            var v = random_float64(-0.5, 0.5).cast[DType.bfloat16]()
            ha[i] = v
            host_a.append(v.cast[DType.float32]())
        for i in range(N * K):
            var v = random_float64(-0.5, 0.5).cast[DType.bfloat16]()
            hb[i] = v
            host_b.append(v.cast[DType.float32]())

    var a_tt = TileTensor(dev_a, row_major[M, K]())
    var b_tt = TileTensor(dev_b, row_major[N, K]())  # [out, in] as staged
    var c_tt = TileTensor(dev_c, row_major[M, N]())

    ctx.enqueue_memset(dev_c, 0)
    matmul[transpose_b=True, target="gpu"](c_tt, a_tt, b_tt, ctx)
    ctx.synchronize()

    var errors = 0
    var max_diff = Float32(0)
    with dev_c.map_to_host() as hc:
        for i in range(M):
            for j in range(N):
                var acc = Float32(0)
                for kk in range(K):
                    acc += host_a[i * K + kk] * host_b[j * K + kk]
                var actual = hc[i * N + j].cast[DType.float32]()
                var diff = abs(actual - acc)
                max_diff = max(max_diff, diff)
                if diff > 1e-5 + 1.6e-2 * abs(acc):
                    errors += 1
    print("M =", M, "-> errors:", errors, "/", M * N, " max_diff:", max_diff)
    if errors != 0:
        raise Error("mismatch at M = " + String(M))


def main() raises:
    var ctx = DeviceContext()  # device 0
    run_case[1](ctx)  # decode shape -> gemv_gpu
    run_case[4](ctx)  # tiny prefill -> RDNA WMMA 64x64
    run_case[128](ctx)  # real prefill -> RDNA WMMA 128x128 BK=32
    print("EXP3 PASS: high-level matmul() correct on all three dispatch paths")
