# Experiment: measure the fixed cost of a single kernel dispatch on gfx1201,
# in this Mojo build, isolated from any real compute. Directly tests the
# G1a-2 profiling hypothesis: "~5us/dispatch accounts for ~2.5ms (12%) of the
# 20.4ms/token decode step." That number was never measured -- this measures
# it.
#
# Method: launch a trivial 1-thread kernel N times, synchronizing after each
# one individually (round-trip: enqueue + wait), and take the average. This
# isolates dispatch+sync overhead from compute time, since the kernel itself
# does effectively nothing (one write to a 1-element buffer).

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns

comptime N = 2000


def noop_kernel(out_ptr: UnsafePointer[Int32, MutAnyOrigin]):
    out_ptr[0] = 1


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(buf, 0)
    ctx.synchronize()

    # Warmup (first few dispatches often include one-time driver/JIT costs).
    for _ in range(50):
        ctx.enqueue_function[noop_kernel](
            buf.unsafe_ptr(), grid_dim=(1,), block_dim=(1,)
        )
        ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N):
        ctx.enqueue_function[noop_kernel](
            buf.unsafe_ptr(), grid_dim=(1,), block_dim=(1,)
        )
        ctx.synchronize()
    var t1 = perf_counter_ns()

    var per_dispatch_us = Float64(t1 - t0) / Float64(N) / 1000.0
    print("dispatch+sync round trips:", N)
    print("total:", Float64(t1 - t0) / 1e6, "ms")
    print("per-dispatch (enqueue + individual sync):", per_dispatch_us, "us")

    # Also measure N back-to-back dispatches with ONE sync at the end -- this
    # is what a real decode step actually does (kernels queued in sequence,
    # one sync at the very end), so it's the fairer estimate of the marginal
    # cost of one MORE dispatch in a queue, without per-call host round trips.
    var t2 = perf_counter_ns()
    for _ in range(N):
        ctx.enqueue_function[noop_kernel](
            buf.unsafe_ptr(), grid_dim=(1,), block_dim=(1,)
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var per_dispatch_queued_us = Float64(t3 - t2) / Float64(N) / 1000.0
    print(
        "per-dispatch (queued, one sync at end):",
        per_dispatch_queued_us,
        "us",
    )
    print(
        "at 504 dispatches/token (14 launches x 36 layers), queued-cost"
        " estimate:",
        per_dispatch_queued_us * 504.0 / 1000.0,
        "ms/token",
    )
