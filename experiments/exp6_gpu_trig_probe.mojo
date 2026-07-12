# Experiment 6: does GPU cos/sin lose precision at the raw angles RoPE feeds
# it? For head_dim pair 0 (theta^0 = 1), RoPE angle = token position directly
# -- up to ~265 radians in our 768-step check. AMD's v_cos_f32/v_sin_f32
# hardware instructions are documented to require pre-range-reduced input;
# feeding raw large angles risks silent precision loss, which would explain
# the isolated multi-unit logit spikes seen at specific late-sequence steps
# (exp: prompt1 step67 max_abs_diff=12.06) against a background of ~0.03-0.1
# ordinary bf16 drift.

from std.gpu.host import DeviceContext
from std.math import cos, sin

comptime N = 8


def probe(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    var angles = SIMD[DType.float32, N](
        1.0, 10.0, 41.0, 76.0, 100.0, 176.0, 205.0, 265.0
    )
    for i in range(N):
        out_ptr[i * 2] = cos(angles[i])
        out_ptr[i * 2 + 1] = sin(angles[i])


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](N * 2)
    ctx.enqueue_function[probe](buf.unsafe_ptr(), grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()
    var angles = List[Float64]()
    angles.append(1.0)
    angles.append(10.0)
    angles.append(41.0)
    angles.append(76.0)
    angles.append(100.0)
    angles.append(176.0)
    angles.append(205.0)
    angles.append(265.0)
    with buf.map_to_host() as h:
        for i in range(N):
            print(
                "angle", angles[i],
                " gpu_cos", h[i * 2], " gpu_sin", h[i * 2 + 1],
            )
