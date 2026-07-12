# Experiment 3f: NORTH STAR DE-RISK (not Phase 1b work — a go/no-go probe).
#
# BF16 WMMA does not compile on gfx1201 (exp3c/exp3d): stdlib emits the RDNA3
# 16-element-fragment encoding and LLVM cannot select it for gfx12.
#
# The FP8 branch of the same stdlib file (mma_amd_rdna.mojo:384) IS gated on
# _is_amd_rdna4() and emits llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8. The entire
# project thesis (Phase 1b: FP8 E4M3 fed natively to RDNA4 WMMA) rests on that
# intrinsic working. If it does not compile either, the north star is blocked
# on the toolchain and Niko needs to know NOW, not in Phase 1b.
#
# This builds nothing for 1b. It is one intrinsic call.

from std.gpu.host import DeviceContext
from std.gpu.compute.mma import mma


def fp8_wmma_probe(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    var a = SIMD[DType.float8_e4m3fn, 8](1.0)
    var b = SIMD[DType.float8_e4m3fn, 8](1.0)
    var c = SIMD[DType.float32, 8](0.0)
    var d = SIMD[DType.float32, 8](0.0)
    mma(d, a, b, c)
    out_ptr[0] = d[0]


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_memset(buf, 0)
    ctx.enqueue_function[fp8_wmma_probe](
        buf.unsafe_ptr(), grid_dim=(1,), block_dim=(32,)
    )
    ctx.synchronize()
    with buf.map_to_host() as h:
        print("fp8 wmma d[0] =", h[0], "(expect 16.0: sum of 16 1.0*1.0)")
    print("EXP3f: FP8 E4M3 WMMA compiled and RAN on gfx1201")
