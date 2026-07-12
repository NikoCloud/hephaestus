# ===----------------------------------------------------------------------=== #
# DIAGNOSTIC ONLY -- attention kernel copied from src/hephaestus/kernels.mojo,
# with the two deliberate rounding choices lifted to compile-time parameters so
# they can be switched OFF as a CAUSAL INTERVENTION.
#
# Production hard-codes both (kernels.mojo:242 and :262):
#   prob_bf16   softmax probabilities are rounded to bf16 before the PV product,
#               mimicking HF *eager* (`softmax(dtype=float32).to(query.dtype)`).
#               The oracle is *sdpa*, which does NOT do this. It is the single
#               largest known arithmetic difference from the reference, and it is
#               applied at every layer.
#   score_bf16  scores rounded to bf16 after the scale (production keeps fp32;
#               the comment there records that rounding them was measured worse).
#
# The hypothesis under test is that row 67 chaotically amplifies ANY bf16-level
# perturbation, so that flipping prob_bf16 -- a bf16-magnitude change -- moves the
# spike a lot WITHOUT moving it toward HF. The competing hypothesis is that
# prob_bf16 IS the defect, in which case turning it off collapses the spike to
# the reference value (4.25).
#
# Nothing else is changed: same indexing, same n_keys, same warp reduction order.
# ===----------------------------------------------------------------------=== #

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_down
from std.math import exp, sqrt
from std.memory import stack_allocation

from std.gpu.host import DeviceContext

from hephaestus.kernels import BF16, F32, MAX_KEYS, WARP


def spike_attention_kernel[
    head_dim: Int, group: Int, prob_bf16: Bool, score_bf16: Bool
](
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n_heads: Int,
    n_kv_heads: Int,
    seq: Int,
    past: Int,
    scale: Float32,
):
    var head = Int(block_idx.x)
    var tok = Int(block_idx.y)
    if head >= n_heads or tok >= seq:
        return
    var kv_head = head // group
    var lane = Int(thread_idx.x)

    var scores = stack_allocation[
        MAX_KEYS, F32, address_space = AddressSpace.SHARED
    ]()
    var q_sh = stack_allocation[
        head_dim, F32, address_space = AddressSpace.SHARED
    ]()

    var q_base = (tok * n_heads + head) * head_dim
    var d = lane
    while d < head_dim:
        q_sh[d] = q_ptr[q_base + d].cast[F32]()
        d += WARP
    barrier()

    var n_keys = past + tok + 1

    var j = 0
    while j < n_keys:
        var k_base = (j * n_kv_heads + kv_head) * head_dim
        var partial = Float32(0)
        var dd = lane
        while dd < head_dim:
            partial += q_sh[dd] * k_ptr[k_base + dd].cast[F32]()
            dd += WARP
        var off = WARP // 2
        while off > 0:
            partial += shuffle_down(partial, UInt32(off))
            off //= 2
        if lane == 0:
            var s = partial * scale
            comptime if score_bf16:
                s = s.cast[BF16]().cast[F32]()
            scores[j] = s
        j += 1
    barrier()

    if lane == 0:
        var m = Float32(-3.4e38)
        for jj in range(n_keys):
            m = max(m, scores[jj])
        var s = Float32(0)
        for jj in range(n_keys):
            var e = exp(scores[jj] - m)
            scores[jj] = e
            s += e
        var inv = Float32(1.0) / s
        for jj in range(n_keys):
            var p = scores[jj] * inv
            comptime if prob_bf16:
                p = p.cast[BF16]().cast[F32]()
            scores[jj] = p
    barrier()

    var o_base = (tok * n_heads + head) * head_dim
    var od = lane
    while od < head_dim:
        var acc = Float32(0)
        for jj in range(n_keys):
            var v_base = (jj * n_kv_heads + kv_head) * head_dim
            acc += scores[jj] * v_ptr[v_base + od].cast[F32]()
        out_ptr[o_base + od] = acc.cast[BF16]()
        od += WARP


def spike_attention[
    head_dim: Int, group: Int, prob_bf16: Bool, score_bf16: Bool
](
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n_heads: Int,
    n_kv_heads: Int,
    seq: Int,
    past: Int,
    ctx: DeviceContext,
) raises:
    if past + seq > MAX_KEYS:
        raise Error("sequence exceeds MAX_KEYS")
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    ctx.enqueue_function[
        spike_attention_kernel[head_dim, group, prob_bf16, score_bf16]
    ](
        out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        n_heads,
        n_kv_heads,
        seq,
        past,
        scale,
        grid_dim=(n_heads, seq),
        block_dim=(WARP,),
    )
