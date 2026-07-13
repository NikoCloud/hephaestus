# Probe: verify hephaestus.kernels.argmax_logits (GPU argmax, replacing the
# 51.6ms/token host-side scan, bench/1a-ab.md Finding 2) against a known
# answer AND against a tie case, before wiring it into production.

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from hephaestus.kernels import argmax_logits, BF16, F32

comptime VOCAB = 151936


def main() raises:
    var ctx = DeviceContext()
    var logits = ctx.enqueue_create_buffer[DType.float32](VOCAB)
    var bf16_scratch = ctx.enqueue_create_buffer[BF16](VOCAB)
    var idx_scratch = ctx.enqueue_create_buffer[DType.int32](1)

    # Case 1: clear winner at index 96874.
    with logits.map_to_host() as h:
        for i in range(VOCAB):
            h[i] = Float32(-10.0)
        h[96874] = Float32(16.3121)
    var best1 = argmax_logits(
        logits.unsafe_ptr(), bf16_scratch, idx_scratch, VOCAB, ctx
    )
    print("case1 (clear winner @ 96874):", best1, "expect 96874")
    if best1 != 96874:
        raise Error("FAIL case1")

    # Case 2: exact tie in bf16 between a lower and a higher index -- must
    # pick the LOWER index (torch.argmax / HF semantics, verified from
    # nn/topk.mojo:736,2168-2170 before relying on it).
    with logits.map_to_host() as h:
        for i in range(VOCAB):
            h[i] = Float32(-10.0)
        h[1632] = Float32(19.25)
        h[11245] = Float32(19.25)  # bit-identical in bf16
    var best2 = argmax_logits(
        logits.unsafe_ptr(), bf16_scratch, idx_scratch, VOCAB, ctx
    )
    print("case2 (exact tie 1632 vs 11245):", best2, "expect 1632 (lower index)")
    if best2 != 1632:
        raise Error("FAIL case2")

    # Case 3: winner only distinguishable AFTER bf16 rounding (fp32 values
    # differ, but round to the same bf16 bit pattern) -- must still pick
    # lower index, matching the established HF-bf16-then-argmax semantics.
    with logits.map_to_host() as h:
        for i in range(VOCAB):
            h[i] = Float32(-10.0)
        h[500] = Float32(17.2510)   # rounds to same bf16 as 17.25
        h[9000] = Float32(17.2495)  # rounds to same bf16 as 17.25
    var best3 = argmax_logits(
        logits.unsafe_ptr(), bf16_scratch, idx_scratch, VOCAB, ctx
    )
    print("case3 (bf16-rounded tie, 500 vs 9000):", best3, "expect 500")
    if best3 != 500:
        raise Error("FAIL case3")

    print("EXP argmax_gpu PASS: clear winner, exact tie, and bf16-rounded tie all correct")

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(100):
        _ = argmax_logits(logits.unsafe_ptr(), bf16_scratch, idx_scratch, VOCAB, ctx)
    var t1 = perf_counter_ns()
    print(
        "100 argmax calls:", Float64(t1 - t0) / 1e6, "ms total,",
        Float64(t1 - t0) / 1e6 / 100.0, "ms/call",
    )
