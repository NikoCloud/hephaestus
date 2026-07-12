# Experiment 3b: isolate the M=1 (decode) path, which routes to gemv_gpu and
# never touches WMMA. Confirms whether the LLVM WMMA selection failure seen at
# M=4/M=128 is confined to the WMMA kernels.

from std.gpu.host import DeviceContext
from std.random import random_float64, seed

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul

comptime M = 1
comptime K = 2560
comptime N = 4096


def main() raises:
    seed(42)
    var ctx = DeviceContext()
    var dev_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var dev_b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var dev_c = ctx.enqueue_create_buffer[DType.bfloat16](M * N)

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
    var b_tt = TileTensor(dev_b, row_major[N, K]())
    var c_tt = TileTensor(dev_c, row_major[M, N]())

    ctx.enqueue_memset(dev_c, 0)
    matmul[transpose_b=True, target="gpu"](c_tt, a_tt, b_tt, ctx)
    ctx.synchronize()

    var errors = 0
    var max_diff = Float32(0)
    with dev_c.map_to_host() as hc:
        for j in range(N):
            var acc = Float32(0)
            for kk in range(K):
                acc += host_a[kk] * host_b[j * K + kk]
            var diff = abs(hc[j].cast[DType.float32]() - acc)
            max_diff = max(max_diff, diff)
            if diff > 1e-5 + 1.6e-2 * abs(acc):
                errors += 1
    print("M=1 gemv path -> errors:", errors, "/", N, " max_diff:", max_diff)
    if errors == 0:
        print("EXP3b PASS: decode-shape matmul (gemv_gpu) correct on gfx1201")
