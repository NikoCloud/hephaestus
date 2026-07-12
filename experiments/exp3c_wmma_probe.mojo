# Experiment 3c: does BF16 WMMA work AT ALL on gfx1201 in this Mojo build?
#
# The matmul dispatcher fails to compile with:
#   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.bf16
#
# stdlib emits that intrinsic name for BF16 with A/B fragment size 16
# (mma_amd_rdna.mojo:370-374, _has_shape[(16,16,8,8)]) — the RDNA3 encoding.
# RDNA4/gfx12 uses a different operand form. This probe calls mma() directly
# with the RDNA3-shaped fragments (16,16,8,8) to confirm the failure is in the
# intrinsic itself and not in the surrounding gemm kernel.
#
# This matters far beyond Phase 1a: the FP8 WMMA path is the Phase 1b thesis.

from std.gpu.host import DeviceContext
from std.gpu.compute.mma import mma


def wmma_probe(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    var a = SIMD[DType.bfloat16, 16](1.0)
    var b = SIMD[DType.bfloat16, 16](1.0)
    var c = SIMD[DType.float32, 8](0.0)
    var d = SIMD[DType.float32, 8](0.0)
    mma(d, a, b, c)
    out_ptr[0] = d[0]


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_memset(buf, 0)
    ctx.enqueue_function[wmma_probe](
        buf.unsafe_ptr(), grid_dim=(1,), block_dim=(32,)
    )
    ctx.synchronize()
    with buf.map_to_host() as h:
        print("wmma d[0] =", h[0])
    print("EXP3c: BF16 WMMA 16x16x16 compiled and ran on gfx1201")
