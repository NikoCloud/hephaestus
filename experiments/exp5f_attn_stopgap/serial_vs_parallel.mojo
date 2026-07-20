# Compare serial attention_kernel vs attention_kernel_parallel.
# Expect near-bit-exact (softmax sum order); max_abs within BF16 ULP class.
#
# Usage (nightly):
#   mojo build -I $KERNELS -I src experiments/exp5f_attn_stopgap/serial_vs_parallel.mojo \
#       -o /tmp/attn_svp
#   /tmp/attn_svp

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import bitcast
from std.utils.index import Index

from hephaestus.kernels import (
    ATTN_NUM_WARPS,
    ATTN_PHASE_ALL,
    BF16,
    WARP,
    attention,
)

comptime HEAD_DIM = 128
comptime N_HEADS = 32
comptime N_KV = 8
comptime GROUP = 4
comptime SEQ = 64
comptime PAST = 0


def fill(
    q: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    k: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    v: UnsafePointer[Scalar[BF16], MutAnyOrigin],
):
    for t in range(SEQ):
        for h in range(N_HEADS):
            for d in range(HEAD_DIM):
                var x = Float32((t * 17 + h * 3 + d) % 127) * 0.01
                q[(t * N_HEADS + h) * HEAD_DIM + d] = x.cast[BF16]()
    for j in range(SEQ):
        for h in range(N_KV):
            for d in range(HEAD_DIM):
                var xk = Float32((j * 11 + h * 5 + d) % 113) * 0.01
                var xv = Float32((j * 7 + h * 9 + d) % 101) * 0.01
                k[(j * N_KV + h) * HEAD_DIM + d] = xk.cast[BF16]()
                v[(j * N_KV + h) * HEAD_DIM + d] = xv.cast[BF16]()


def main() raises:
    var ctx = DeviceContext()
    var nq = SEQ * N_HEADS * HEAD_DIM
    var nkv = SEQ * N_KV * HEAD_DIM
    var q = ctx.enqueue_create_buffer[BF16](nq)
    var k = ctx.enqueue_create_buffer[BF16](nkv)
    var v = ctx.enqueue_create_buffer[BF16](nkv)
    var o_s = ctx.enqueue_create_buffer[BF16](nq)
    var o_p = ctx.enqueue_create_buffer[BF16](nq)
    with q.map_to_host() as hq, k.map_to_host() as hk, v.map_to_host() as hv:
        fill(
            hq.unsafe_ptr().as_unsafe_any_origin(),
            hk.unsafe_ptr().as_unsafe_any_origin(),
            hv.unsafe_ptr().as_unsafe_any_origin(),
        )
    ctx.enqueue_memset(o_s, 0)
    ctx.enqueue_memset(o_p, 0)

    attention[HEAD_DIM, GROUP](
        o_s.unsafe_ptr().as_unsafe_any_origin(),
        q.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        k.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        v.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        N_HEADS,
        N_KV,
        SEQ,
        PAST,
        ctx,
        parallel=False,
    )
    attention[HEAD_DIM, GROUP](
        o_p.unsafe_ptr().as_unsafe_any_origin(),
        q.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        k.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        v.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        N_HEADS,
        N_KV,
        SEQ,
        PAST,
        ctx,
        parallel=True,
        phases=ATTN_PHASE_ALL,
    )
    ctx.synchronize()

    var n_bad = 0
    var n_tol = 0
    var max_abs = Float32(0)
    var ATOL = Float32(1e-5)
    var RTOL = Float32(1.6e-2)
    with o_s.map_to_host() as hs, o_p.map_to_host() as hp:
        for i in range(nq):
            var a = Float32(hs[i])
            var b = Float32(hp[i])
            var ae = abs(a - b)
            if ae > max_abs:
                max_abs = ae
            var u1 = bitcast[DType.uint16, 1](SIMD[BF16, 1](hs[i]))[0]
            var u2 = bitcast[DType.uint16, 1](SIMD[BF16, 1](hp[i]))[0]
            if u1 != u2:
                n_bad += 1
            var tol = ATOL + RTOL * abs(a)
            if ae > tol:
                n_tol += 1
    print(
        "bit_mismatches",
        n_bad,
        "/",
        nq,
        " tol_exceed",
        n_tol,
        " max_abs",
        max_abs,
    )
    if n_tol > 0:
        print("FAIL: exceeds tolerance")
        raise Error("attn serial vs parallel out of tol")
    print("PASS: serial vs parallel within tolerance")
