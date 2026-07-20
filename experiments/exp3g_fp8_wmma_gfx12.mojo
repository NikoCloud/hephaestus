# Experiment 3g: FP8 E4M3 WMMA via DIRECT llvm_intrinsic (bypass Mojo mma() stdlib).
#
# North Star de-risk: does the LLVM AMDGPU backend lower
#   llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8
# on gfx1201 (RDNA4 / wave32)?
#
# Signature from /opt/rocm/.../IntrinsicsAMDGPU.td + BuiltinsAMDGPU.def:
#   AMDGPUWmmaIntrinsic  (NO i1 operands)
#   D = A * B + C
#   A,B: <2 x i32>  — 8 packed fp8 bytes each (LLVM has no native fp8 type)
#   C,D: <8 x f32>  — wave32 C/D fragment
# Clang: __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12("V8fV2iV2iV8f")
# CK: bit_cast to int32x2_t / fp32x8_t, no sign/clamp i1s (those are IU/OPSEL only).
#
# All-ones: E4M3 1.0 is bit pattern 0x38. Eight of them pack as 0x38383838 per i32.
# Expect each C/D lane element = K * 1*1 = 16.0 for 16x16x16 WMMA.
#
# Usage (isolated nightly env recommended):
#   cd ~/projects/hephaestus-wmma-nightly
#   pixi run mojo build ../hephaestus/experiments/exp3g_fp8_wmma_gfx12.mojo -o /tmp/exp3g_fp8
#   HIP_VISIBLE_DEVICES=0 /tmp/exp3g_fp8

from std.gpu.host import DeviceContext
from std.sys.intrinsics import llvm_intrinsic


def fp8_wmma_all_ones(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    # Pack eight E4M3 ones (0x38) into each of two i32s → A/B fragment (8 fp8).
    var a = SIMD[DType.int32, 2](Int32(0x38383838), Int32(0x38383838))
    var b = SIMD[DType.int32, 2](Int32(0x38383838), Int32(0x38383838))
    var c = SIMD[DType.float32, 8](0.0)
    var d = llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8",
        SIMD[DType.float32, 8],
        has_side_effect=False,
    ](a, b, c)
    # Write full fragment so host can inspect all lanes.
    @parameter
    for i in range(8):
        out_ptr[i] = d[i]


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](8)
    ctx.enqueue_memset(buf, 0)
    # One full wave32 — WMMA is wave-cooperative.
    ctx.enqueue_function[fp8_wmma_all_ones](
        buf.unsafe_ptr(), grid_dim=(1,), block_dim=(32,)
    )
    ctx.synchronize()
    var ok = True
    with buf.map_to_host() as h:
        for i in range(8):
            print("d[", i, "] =", h[i])
            if h[i] != Float32(16.0):
                ok = False
    if ok:
        print("EXP3g PASS: FP8 E4M3 WMMA all-ones -> 16.0 on gfx1201")
    else:
        print("EXP3g FAIL: expected all d[i] == 16.0")
        raise Error("FP8 WMMA all-ones mismatch")
