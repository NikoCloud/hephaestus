# ===----------------------------------------------------------------------=== #
# Hephaestus -- hand-written GPU kernels (Phase 1a, BF16)
#
# Everything here is elementwise or reduction work. NOTHING here uses WMMA:
# no WMMA intrinsic of any dtype compiles on gfx1201 in Mojo 1.0.0b3.dev2026071006
# (DECISIONS.md 2026-07-12), which rules out the vendored RDNA matmul and
# attention kernels. Matmul goes through matmul_kernel_naive; attention is
# written here.
#
# Vendored where possible: rms_norm_gpu (normalization), gather (embedding),
# get_safetensors_idx (RoPE index convention).
# ===----------------------------------------------------------------------=== #

from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_down
from std.math import ceildiv, cos, exp, sin, sqrt
from std.memory import stack_allocation

from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.gemv import gemv_gpu
from linalg.matmul.gpu import matmul_kernel_naive

comptime BF16 = DType.bfloat16
comptime F32 = DType.float32


# ===----------------------------------------------------------------------=== #
# Linear projection: C[m, n] = A[m, k] @ B[n, k]^T
# B is the weight straight from the arena in safetensors [out, in] order, so
# transpose_b=True is the zero-copy path (DECISIONS 2026-07-11).
# ===----------------------------------------------------------------------=== #


def linear(
    c: TileTensor[mut=True, ...],
    a: TileTensor[BF16, ...],
    b: TileTensor[BF16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """C = A @ B^T. c may be BF16 (activations) or F32 (logits: the LM head
    output must NOT be rounded to BF16 -- a one-ulp gap between the top two
    tokens is decidable in fp32 and a coin-flip tie in bf16).

    Decode (m=1) routes to the vendored gemv_gpu (G1a-2): the naive kernel's
    16x16 thread block wastes 15/16 of its threads on a single-row output.
    No WMMA involved -- gemv_gpu is one of the two paths proven to compile
    and be correct on gfx1201 (exp3e, DECISIONS.md 2026-07-12).
    """
    if m == 1:
        gemv_gpu[transpose_b=True](c, a, b, ctx)
        return

    comptime BLOCK_DIM = 16
    comptime kernel = matmul_kernel_naive[
        type_of(c).dtype,
        BF16,
        BF16,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        BLOCK_DIM,
        True,  # transpose_b
        c_storage = type_of(c).Storage,
        a_storage = type_of(a).Storage,
        b_storage = type_of(b).Storage,
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )


def linear_add_residual(
    residual: TileTensor[mut=True, BF16, ...],
    a: TileTensor[BF16, ...],
    b: TileTensor[BF16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """residual += A @ B^T, fused via the matmul kernels' elementwise epilogue
    (G1a-2: removes a separate residual_add launch -- used for o_proj and
    down_proj, the two projections immediately followed by a residual add).

    Every elementwise_lambda_fn call site in gemv.mojo and matmul_kernel_naive
    invokes the epilogue with width=1 (one element at a time, no
    vectorization in these paths) -- verified by reading every call site
    before relying on it here.
    """

    @always_inline
    @__copy_capture(residual)
    @parameter
    def epilogue[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var off = residual.layout(Coord(idx))
        var old = residual.raw_load[width=width](off).cast[F32]()
        residual.raw_store[width=width, alignment=alignment](
            off, (old + val.cast[F32]()).cast[BF16]()
        )

    if m == 1:
        gemv_gpu[transpose_b=True, elementwise_lambda_fn=epilogue](
            residual, a, b, ctx
        )
        return

    comptime BLOCK_DIM = 16
    comptime kernel = matmul_kernel_naive[
        BF16,
        BF16,
        BF16,
        type_of(residual).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        BLOCK_DIM,
        True,  # transpose_b
        elementwise_lambda_fn=epilogue,
        c_storage = type_of(residual).Storage,
        a_storage = type_of(a).Storage,
        b_storage = type_of(b).Storage,
    ]
    ctx.enqueue_function[kernel](
        residual,
        a,
        b,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )


# ===----------------------------------------------------------------------=== #
# RoPE (split-half / "rotate_half", the safetensors convention -- NOT
# interleaved; dossier trap #3). For head_dim d, element i pairs with
# i + d/2. get_safetensors_idx is vendored from nn/rope.mojo so the index
# convention comes from MAX, not from memory.
#
# q: [seq, n_heads * head_dim] flattened, k: [seq, n_kv_heads * head_dim]
# Applied in place. pos_offset = index of the first token in the sequence
# (0 for prefill, current length for a decode step).
# ===----------------------------------------------------------------------=== #


def rope_kernel[
    head_dim: Int, theta: Float64
](
    x: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n_heads: Int,
    seq: Int,
    pos_offset: Int,
):
    # One thread per (token, head, half-pair).
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var half = head_dim // 2
    var total = seq * n_heads * half
    if gid >= total:
        return

    var pair = gid % half
    var rest = gid // half
    var head = rest % n_heads
    var tok = rest // n_heads

    # Split-half ("rotate_half") pairing: element p pairs with p + head_dim/2.
    # NOTE: nn.rope.get_safetensors_idx expects an INTERLEAVED-order index and
    # maps it to these same split-half slots (it returns (i//2, i//2 + d/2)).
    # Feeding it a pair index directly halves it -- pairs 0,1 both land on slot
    # 0 -- which rotates half the pairs twice and half not at all. Iterating
    # pairs directly, the mapping is simply (p, p + d/2).
    var i_re = pair
    var i_im = pair + half

    var base = (tok * n_heads + head) * head_dim
    var pos = Float64(tok + pos_offset)
    # HF computes freqs and cos/sin in fp32 (autocast forced off)...
    var freq = Float32(pos * (theta ** (-2.0 * Float64(pair) / Float64(head_dim))))
    # ...but then returns `cos.to(dtype=x.dtype)`, so cos/sin are BF16 and
    # apply_rotary_pos_emb runs `(q*cos) + (rotate_half(q)*sin)` entirely in
    # bf16, rounding after every op. Doing this in fp32 is MORE accurate and
    # therefore wrong: it drifts from the reference we must match token-for-
    # token. Round exactly where torch rounds.
    var cos_v = cos(freq).cast[BF16]()
    var sin_v = sin(freq).cast[BF16]()

    var re = x[base + i_re]
    var im = x[base + i_im]
    # rotate_half: out[i]     = q[i]*cos     - q[i+d/2]*sin
    #              out[i+d/2] = q[i+d/2]*cos + q[i]*sin
    x[base + i_re] = (re * cos_v) - (im * sin_v)
    x[base + i_im] = (im * cos_v) + (re * sin_v)


def rope_kernel_qk[
    head_dim: Int, theta: Float64
](
    q_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n_heads: Int,
    n_kv_heads: Int,
    seq: Int,
    pos_offset: Int,
):
    """Q and K RoPE in one launch (G1a-2: one fewer kernel per layer). Same
    per-element math as rope_kernel; only the source array is picked by
    which half of the combined index range this thread falls in."""
    var half = head_dim // 2
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total_q = seq * n_heads * half

    var x: UnsafePointer[Scalar[BF16], MutAnyOrigin]
    var n: Int
    var local_gid: Int
    if gid < total_q:
        x = q_ptr
        n = n_heads
        local_gid = gid
    else:
        var total_k = seq * n_kv_heads * half
        if gid >= total_q + total_k:
            return
        x = k_ptr
        n = n_kv_heads
        local_gid = gid - total_q

    var pair = local_gid % half
    var rest = local_gid // half
    var head = rest % n
    var tok = rest // n

    var i_re = pair
    var i_im = pair + half
    var base = (tok * n + head) * head_dim
    var pos = Float64(tok + pos_offset)
    var freq = Float32(pos * (theta ** (-2.0 * Float64(pair) / Float64(head_dim))))
    var cos_v = cos(freq).cast[BF16]()
    var sin_v = sin(freq).cast[BF16]()

    var re = x[base + i_re]
    var im = x[base + i_im]
    x[base + i_re] = (re * cos_v) - (im * sin_v)
    x[base + i_im] = (im * cos_v) + (re * sin_v)


def apply_rope_qk_inplace[
    head_dim: Int, theta: Float64
](
    q_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n_heads: Int,
    n_kv_heads: Int,
    seq: Int,
    pos_offset: Int,
    ctx: DeviceContext,
) raises:
    comptime TPB = 256
    var half = head_dim // 2
    var total = seq * (n_heads + n_kv_heads) * half
    ctx.enqueue_function[rope_kernel_qk[head_dim, theta]](
        q_ptr,
        k_ptr,
        n_heads,
        n_kv_heads,
        seq,
        pos_offset,
        grid_dim=(ceildiv(total, TPB),),
        block_dim=(TPB,),
    )


def apply_rope_inplace[
    head_dim: Int, theta: Float64
](
    x: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n_heads: Int,
    seq: Int,
    pos_offset: Int,
    ctx: DeviceContext,
) raises:
    comptime TPB = 256
    var total = seq * n_heads * (head_dim // 2)
    ctx.enqueue_function[rope_kernel[head_dim, theta]](
        x,
        n_heads,
        seq,
        pos_offset,
        grid_dim=(ceildiv(total, TPB),),
        block_dim=(TPB,),
    )


# ===----------------------------------------------------------------------=== #
# Causal GQA attention, one warp per (query position, query head).
#
# No WMMA. Per query row this is: dot products against every key (warp-reduced),
# online-free two-pass softmax over a small score buffer in LDS, then a
# weighted sum of V. Correct by construction; decode (seq=1) is the shape that
# matters for G1a-2 and it is bandwidth-bound anyway.
#
# q:   [seq, n_heads, head_dim]        (post q_norm, post RoPE)
# k/v: [n_keys, n_kv_heads, head_dim]  (the KV cache, contiguous)
# out: [seq, n_heads, head_dim]
# Query token t attends to keys [0, past + t]  (causal).
# ===----------------------------------------------------------------------=== #

comptime WARP = 32
comptime MAX_KEYS = 4096


def attention_kernel[
    head_dim: Int, group: Int
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

    # Causal: this query attends to keys 0 .. past + tok inclusive.
    var n_keys = past + tok + 1

    # --- scores = scale * (q . k_j), warp-reduced dot per key ---------------
    var j = 0
    while j < n_keys:
        var k_base = (j * n_kv_heads + kv_head) * head_dim
        var partial = Float32(0)
        var dd = lane
        while dd < head_dim:
            partial += q_sh[dd] * k_ptr[k_base + dd].cast[F32]()
            dd += WARP
        # warp reduction
        var off = WARP // 2
        while off > 0:
            partial += shuffle_down(partial, UInt32(off))
            off //= 2
        if lane == 0:
            # Scores stay in fp32 through the scale. Rounding them to bf16 here
            # (matching eager's `matmul(q,k^T) * scaling` literally) was MEASURED
            # worse -- prompt2 divergence moved 12 -> 6 against both references.
            # The reference's effective precision is higher than its Python
            # source suggests; do not "match" it past what the evidence supports.
            scores[j] = partial * scale
        j += 1
    barrier()

    # --- softmax over scores[0 .. n_keys) ----------------------------------
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
            # HF: softmax(..., dtype=float32).to(query.dtype) -- the probs are
            # ROUNDED TO BF16 before the PV product. Keeping fp32 here is more
            # accurate but diverges from the reference, and token-exactness is
            # the gate, not accuracy.
            scores[jj] = (
                (scores[jj] * inv).cast[BF16]().cast[F32]()
            )
    barrier()

    # --- out = sum_j p_j * v_j ---------------------------------------------
    var o_base = (tok * n_heads + head) * head_dim
    var od = lane
    while od < head_dim:
        var acc = Float32(0)
        for jj in range(n_keys):
            var v_base = (jj * n_kv_heads + kv_head) * head_dim
            acc += scores[jj] * v_ptr[v_base + od].cast[F32]()
        out_ptr[o_base + od] = acc.cast[BF16]()
        od += WARP


def attention[
    head_dim: Int, group: Int
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
    ctx.enqueue_function[attention_kernel[head_dim, group]](
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


# ===----------------------------------------------------------------------=== #
# SwiGLU: out = silu(gate) * up,  silu(x) = x * sigmoid(x)
# ===----------------------------------------------------------------------=== #


def silu_mul_kernel(
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    gate_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    up_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    var g = gate_ptr[i].cast[F32]()
    # HF: act_fn(gate_proj(x)) is a bf16 tensor before being multiplied by up.
    var silu = (g / (Float32(1.0) + exp(-g))).cast[BF16]().cast[F32]()
    out_ptr[i] = (silu * up_ptr[i].cast[F32]()).cast[BF16]()


def silu_mul(
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    gate_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    up_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime TPB = 256
    ctx.enqueue_function[silu_mul_kernel](
        out_ptr,
        gate_ptr,
        up_ptr,
        n,
        grid_dim=(ceildiv(n, TPB),),
        block_dim=(TPB,),
    )


# ===----------------------------------------------------------------------=== #
# Residual: x += y
# ===----------------------------------------------------------------------=== #


def residual_add_kernel(
    x_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    x_ptr[i] = (x_ptr[i].cast[F32]() + y_ptr[i].cast[F32]()).cast[BF16]()


def residual_add(
    x_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime TPB = 256
    ctx.enqueue_function[residual_add_kernel](
        x_ptr,
        y_ptr,
        n,
        grid_dim=(ceildiv(n, TPB),),
        block_dim=(TPB,),
    )


# ===----------------------------------------------------------------------=== #
# Copy rows of a [seq, n_heads*head_dim] projection into the KV cache at
# position `past`. Cache layout: [MAX_KEYS, n_kv_heads, head_dim].
# ===----------------------------------------------------------------------=== #


def cache_write_kernel(
    cache_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    row_width: Int,
    seq: Int,
    past: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= seq * row_width:
        return
    var tok = i // row_width
    var off = i % row_width
    cache_ptr[(past + tok) * row_width + off] = src_ptr[i]


def cache_write(
    cache_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    row_width: Int,
    seq: Int,
    past: Int,
    ctx: DeviceContext,
) raises:
    comptime TPB = 256
    var n = seq * row_width
    ctx.enqueue_function[cache_write_kernel](
        cache_ptr,
        src_ptr,
        row_width,
        seq,
        past,
        grid_dim=(ceildiv(n, TPB),),
        block_dim=(TPB,),
    )
