# Nightly-only origin cast fix for Mojo 1.0.0b3.dev2026071206 — not an engine change.
# Experiment 3e: WMMA-free matmul paths on gfx1201.
#
# The high-level matmul() dispatcher cannot compile for BF16 on gfx1201: it
# instantiates gemm_kernel_rdna at comptime regardless of the runtime m/n
# branch, and that kernel emits llvm.amdgcn.wmma.f32.16x16x16.bf16 (RDNA3
# encoding) which LLVM cannot select for gfx12. See exp3c/exp3d.
#
# Workaround under test: call the WMMA-free kernels directly, so the WMMA
# kernel is never instantiated.
#   - decode (M=1): gemv_gpu               <- this is the G1a-2 gated path
#   - prefill (M>1): matmul_kernel_naive   <- correctness only in Phase 1a
#
# Both checked against a CPU fp32 reference at q_proj shape [*, 2560] x
# [4096, 2560]^T. Tolerance from MAX's own matmul test: 1e-5 + 1.6e-2*|exp|.

from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.random import random_float64, seed

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.gemv import gemv_gpu
from linalg.matmul.gpu import matmul_kernel_naive

comptime K = 2560
comptime N = 4096


def check[M: Int](
    hc: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    host_a: List[Float32],
    host_b: List[Float32],
    label: String,
) raises:
    var errors = 0
    var max_diff = Float32(0)
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for kk in range(K):
                acc += host_a[i * K + kk] * host_b[j * K + kk]
            var diff = abs(hc[i * N + j].cast[DType.float32]() - acc)
            max_diff = max(max_diff, diff)
            if diff > 1e-5 + 1.6e-2 * abs(acc):
                errors += 1
    print(label, "-> errors:", errors, "/", M * N, " max_diff:", max_diff)
    if errors != 0:
        raise Error(label + " FAILED")


def main() raises:
    seed(42)
    var ctx = DeviceContext()

    # --- decode shape: M=1 via gemv_gpu ------------------------------------
    var a1 = ctx.enqueue_create_buffer[DType.bfloat16](1 * K)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var c1 = ctx.enqueue_create_buffer[DType.bfloat16](1 * N)
    var host_a1 = List[Float32]()
    var host_b = List[Float32]()
    with a1.map_to_host() as ha, b.map_to_host() as hb:
        for i in range(1 * K):
            var v = random_float64(-0.5, 0.5).cast[DType.bfloat16]()
            ha[i] = v
            host_a1.append(v.cast[DType.float32]())
        for i in range(N * K):
            var v = random_float64(-0.5, 0.5).cast[DType.bfloat16]()
            hb[i] = v
            host_b.append(v.cast[DType.float32]())

    ctx.enqueue_memset(c1, 0)
    gemv_gpu[transpose_b=True](
        TileTensor(c1, row_major[1, N]()),
        TileTensor(a1, row_major[1, K]()),
        TileTensor(b, row_major[N, K]()),
        ctx,
    )
    ctx.synchronize()
    with c1.map_to_host() as hc:
        check[1](hc.unsafe_ptr().as_unsafe_any_origin(), host_a1, host_b, "decode M=1  gemv_gpu")

    # --- prefill shape: M=4 via naive kernel --------------------------------
    comptime M = 4
    comptime BLOCK_DIM = 16
    var a4 = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var c4 = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var host_a4 = List[Float32]()
    with a4.map_to_host() as ha:
        for i in range(M * K):
            var v = random_float64(-0.5, 0.5).cast[DType.bfloat16]()
            ha[i] = v
            host_a4.append(v.cast[DType.float32]())

    var a4_tt = TileTensor(a4, row_major[M, K]())
    var b_tt = TileTensor(b, row_major[N, K]())
    var c4_tt = TileTensor(c4, row_major[M, N]())
    ctx.enqueue_memset(c4, 0)

    comptime naive = matmul_kernel_naive[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        type_of(c4_tt).LayoutType,
        type_of(a4_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        True,  # transpose_b
        c_storage = type_of(c4_tt).Storage,
        a_storage = type_of(a4_tt).Storage,
        b_storage = type_of(b_tt).Storage,
    ]
    ctx.enqueue_function[naive](
        c4_tt,
        a4_tt,
        b_tt,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )
    ctx.synchronize()
    with c4.map_to_host() as hc:
        check[M](hc.unsafe_ptr().as_unsafe_any_origin(), host_a4, host_b, "prefill M=4 naive")

    print("EXP3e PASS: both WMMA-free paths compile and are correct on gfx1201")
