# Experiment 4 (spec open question 3): rms_norm_gpu contract.
#
# Two shapes the forward pass needs:
#   (a) hidden norm:  [seq, 2560] with gamma [2560]
#   (b) QK per-head:  [seq*heads, 128] with gamma [128]
# rms_norm_gpu derives cols from gamma.dim[0] and rows = size // cols, so the
# per-head norm falls out of the shape/gamma pair — no special casing.
#
# Params to pin (both are silent-wrongness traps):
#   weight_offset = 0        (Qwen3 is x*g, not Gemma's x*(1+g))
#   multiply_before_cast=False -> (x*norm).cast[bf16]() * gamma, which is
#     exactly HF Qwen3RMSNorm: `self.weight * hidden_states.to(input_dtype)`
#     (verified from transformers 5.13.1 source). True would multiply in fp32
#     before the cast — different rounding, silently.
#
# Reference computed on host in fp32 mirroring HF's order.

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.random import random_float64, seed

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from nn.normalization import rms_norm_gpu

comptime EPS = Float32(1e-6)


def run_case[ROWS: Int, COLS: Int](ctx: DeviceContext, label: String) raises:
    seed(7 + COLS)
    var dev_x = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * COLS)
    var dev_g = ctx.enqueue_create_buffer[DType.bfloat16](COLS)
    var dev_o = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * COLS)

    var host_x = List[Float32]()
    var host_g = List[Float32]()
    with dev_x.map_to_host() as hx, dev_g.map_to_host() as hg:
        for i in range(ROWS * COLS):
            var v = random_float64(-2.0, 2.0).cast[DType.bfloat16]()
            hx[i] = v
            host_x.append(v.cast[DType.float32]())
        for i in range(COLS):
            var v = random_float64(0.5, 1.5).cast[DType.bfloat16]()
            hg[i] = v
            host_g.append(v.cast[DType.float32]())

    var x_tt = TileTensor(dev_x, row_major[ROWS, COLS]())
    var o_tt = TileTensor(dev_o, row_major[ROWS, COLS]())
    var g_tt = TileTensor(dev_g, row_major[COLS]())

    @always_inline
    @__copy_capture(x_tt)
    @parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[DType.bfloat16, width]:
        return x_tt.raw_load[width=width](x_tt.layout(coords))

    @always_inline
    @__copy_capture(o_tt)
    @parameter
    def output_fn[
        width: SIMDSize, alignment: Int
    ](coords: Coord, val: SIMD[DType.bfloat16, width]) -> None:
        o_tt.raw_store[width=width, alignment=alignment](
            o_tt.layout(coords), val
        )

    ctx.enqueue_memset(dev_o, 0)
    rms_norm_gpu[
        2,
        input_fn,
        output_fn,
        multiply_before_cast=False,
    ](
        Coord(ROWS, COLS),
        g_tt,
        EPS,
        Scalar[DType.bfloat16](0),  # weight_offset: Qwen3 has none
        ctx,
    )
    ctx.synchronize()

    var errors = 0
    var max_diff = Float32(0)
    with dev_o.map_to_host() as ho:
        for r in range(ROWS):
            # HF order: fp32 normalize -> cast to bf16 -> multiply by gamma
            var ss = Float32(0)
            for c in range(COLS):
                var v = host_x[r * COLS + c]
                ss += v * v
            var norm = Float32(1.0) / sqrt(ss / Float32(COLS) + EPS)
            for c in range(COLS):
                var normed = (
                    (host_x[r * COLS + c] * norm)
                    .cast[DType.bfloat16]()
                    .cast[DType.float32]()
                )
                var expected = (
                    (normed * host_g[c])
                    .cast[DType.bfloat16]()
                    .cast[DType.float32]()
                )
                var actual = ho[r * COLS + c].cast[DType.float32]()
                var diff = abs(actual - expected)
                max_diff = max(max_diff, diff)
                if diff > 1e-5 + 1.6e-2 * abs(expected):
                    errors += 1
    print(label, "-> errors:", errors, "/", ROWS * COLS, " max_diff:", max_diff)
    if errors != 0:
        raise Error(label + " FAILED")


def main() raises:
    var ctx = DeviceContext()
    run_case[4, 2560](ctx, "hidden norm  [4,2560] gamma[2560]")
    run_case[128, 128](ctx, "QK per-head  [4x32,128] gamma[128]")
    print("EXP4 PASS: rms_norm_gpu correct for both shapes on gfx1201")
