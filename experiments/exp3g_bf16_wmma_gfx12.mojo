# Experiment 3g (BF16 companion): BF16 WMMA via DIRECT llvm_intrinsic for gfx12.
#
# Clang: __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 ("V8fV8sV8sV8f")
# LLVM:  llvm.amdgcn.wmma.f32.16x16x16.bf16 via AMDGPUWmmaIntrinsic
#        A,B: <8 x i16>  (packed bf16 — RDNA4/wave32 uses 8-elem A/B, not RDNA3's 16)
#        C,D: <8 x f32>
#        NO i1 op_sel (that was GFX11 OPSEL; GFX12 f32_bf16 form has none).
#
# All-ones: BF16 1.0 is 0x3f80. Expect d[i] = 16.0 (K=16 ones products).
#
# Usage:
#   cd ~/projects/hephaestus-wmma-nightly
#   pixi run mojo build ../hephaestus/experiments/exp3g_bf16_wmma_gfx12.mojo -o /tmp/exp3g_bf16
#   HIP_VISIBLE_DEVICES=0 /tmp/exp3g_bf16

from std.gpu.host import DeviceContext
from std.sys.intrinsics import llvm_intrinsic


def bf16_wmma_all_ones(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    # 8 packed BF16 ones as i16 (LLVM intrinsic takes integer AB type).
    var one_bf16 = Int16(0x3F80)
    var a = SIMD[DType.int16, 8](one_bf16)
    var b = SIMD[DType.int16, 8](one_bf16)
    var c = SIMD[DType.float32, 8](0.0)
    var d = llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.bf16",
        SIMD[DType.float32, 8],
        has_side_effect=False,
    ](a, b, c)
    @parameter
    for i in range(8):
        out_ptr[i] = d[i]


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](8)
    ctx.enqueue_memset(buf, 0)
    ctx.enqueue_function[bf16_wmma_all_ones](
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
        print("EXP3g PASS: BF16 WMMA all-ones -> 16.0 on gfx1201")
    else:
        print("EXP3g FAIL: expected all d[i] == 16.0")
        raise Error("BF16 WMMA all-ones mismatch")
