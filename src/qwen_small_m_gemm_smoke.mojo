# Small-M FP8 GEMM correctness smoke: compare small-M vs v3a prefill on
# random A/W for M in {2,8,16} and a representative (N,K)=(2560,2560).
#
# Usage (nightly env, GPU 0):
#   mojo build -I $KERNELS -I src src/qwen_small_m_gemm_smoke.mojo \
#       -o /tmp/small_m_smoke
#   HIP_VISIBLE_DEVICES=0 /tmp/small_m_smoke

from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import abs
from std.utils.index import Index

from layout import Coord, TileTensor
from layout.tile_layout import row_major

from hephaestus.wmma_gfx12 import (
    FP8,
    wmma_gemm_fp8_prefill,
    wmma_gemm_fp8_small_m,
)

comptime F32 = DType.float32
comptime BF16 = DType.bfloat16


def fill_fp8(buf: DeviceBuffer[FP8], n: Int, seed: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            # Small integer-ish FP8 values in [-4, 4].
            var v = Float32(((i * 17 + seed * 31) % 17) - 8) * 0.25
            h[i] = v.cast[FP8]()


def fill_f32(buf: DeviceBuffer[F32], n: Int, val: Float32) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = val


def max_abs_diff(
    a: DeviceBuffer[BF16], b: DeviceBuffer[BF16], n: Int, ctx: DeviceContext
) raises -> Float32:
    ctx.synchronize()
    var mx = Float32(0)
    with a.map_to_host() as ha:
        with b.map_to_host() as hb:
            for i in range(n):
                var d = abs(ha[i].cast[F32]() - hb[i].cast[F32]())
                if d > mx:
                    mx = d
    return mx


def run_case(m: Int, n: Int, k: Int, ctx: DeviceContext) raises:
    print("case M=", m, "N=", n, "K=", k)
    var a_fp8 = ctx.enqueue_create_buffer[FP8](m * k)
    var w = ctx.enqueue_create_buffer[FP8](n * k)
    var w_scale = ctx.enqueue_create_buffer[F32](n)
    var act_scale = ctx.enqueue_create_buffer[F32](m)
    var c_sm = ctx.enqueue_create_buffer[BF16](m * n)
    var c_v3 = ctx.enqueue_create_buffer[BF16](m * n)

    fill_fp8(a_fp8, m * k, 1)
    fill_fp8(w, n * k, 2)
    fill_f32(w_scale, n, Float32(0.01))
    fill_f32(act_scale, m, Float32(0.02))
    with c_sm.map_to_host() as h:
        for i in range(m * n):
            h[i] = Scalar[BF16](0)
    with c_v3.map_to_host() as h:
        for i in range(m * n):
            h[i] = Scalar[BF16](0)

    var w_tt = TileTensor(
        ptr=w.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        layout=row_major(Coord(Index(n, k))),
    )
    var ws_tt = TileTensor(
        ptr=w_scale.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        layout=row_major(Coord(Index(n))),
    )
    var c_sm_tt = TileTensor(
        ptr=c_sm.unsafe_ptr().as_unsafe_any_origin(),
        layout=row_major(Coord(Index(m, n))),
    )
    var c_v3_tt = TileTensor(
        ptr=c_v3.unsafe_ptr().as_unsafe_any_origin(),
        layout=row_major(Coord(Index(m, n))),
    )

    wmma_gemm_fp8_small_m[BF16, False](
        c_sm_tt, a_fp8, w_tt, ws_tt, act_scale, m, n, k, ctx
    )
    wmma_gemm_fp8_prefill[BF16, False](
        c_v3_tt, a_fp8, w_tt, ws_tt, act_scale, m, n, k, ctx
    )
    var mad = max_abs_diff(c_sm, c_v3, m * n, ctx)
    print("  max_abs_diff_vs_v3a=", mad)
    if mad > Float32(1e-3):
        raise Error("small-M vs v3a mismatch")
    print("  PASS")


def main() raises:
    var ctx = DeviceContext()
    run_case(2, 2560, 2560, ctx)
    run_case(8, 2560, 2560, ctx)
    run_case(16, 2560, 2560, ctx)
    run_case(8, 4096, 2560, ctx)
    run_case(8, 1024, 2560, ctx)
    run_case(32, 2560, 2560, ctx)
    print("ALL PASS")
