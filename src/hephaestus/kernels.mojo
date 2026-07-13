# ===----------------------------------------------------------------------=== #
# Hephaestus -- hand-written GPU kernels (Phase 1a→1b, BF16)
#
# Prefill matmul (m>1): gfx12 BF16 WMMA GEMM (exp4b, G1b-0 mappings) when
# n and k are multiples of 16 (all model dims are). Decode (m=1) stays on
# gemv_gpu — a 16×16 WMMA tile would waste 15/16 rows.
#
# Elementwise / reduction kernels (RoPE, attention, silu, argmax) are unchanged.
# Vendored where possible: rms_norm_gpu, gather, get_safetensors_idx.
#
# Requires Mojo nightly with llvm_intrinsic WMMA (dev2026071206+). The repo
# default pixi pin may be older — build/run prefill-WMMA with the isolated
# hephaestus-wmma-nightly env.
# ===----------------------------------------------------------------------=== #

from std.gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.gpu.memory import AddressSpace
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.gpu.primitives.warp import shuffle_down
from std.math import align_up, ceildiv, cos, exp, sin, sqrt
from std.memory import stack_allocation
from std.sys import simd_width_of
from std.utils.index import Index, IndexList

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.gemv import gemv_gpu, gemv_kernel_vector
from linalg.matmul.gpu import matmul_kernel_naive

from hephaestus.wmma_gfx12 import WMMA_TILE, wmma_gemm_bf16

comptime BF16 = DType.bfloat16
comptime F32 = DType.float32


def _linear_naive(
    c: TileTensor[mut=True, ...],
    a: TileTensor[BF16, ...],
    b: TileTensor[BF16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Fallback: vendored matmul_kernel_naive (transpose_b=True)."""
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
    *,
    use_wmma: Bool = True,
) raises:
    """C = A @ B^T. c may be BF16 (activations) or F32 (logits: the LM head
    output must NOT be rounded to BF16 -- a one-ulp gap between the top two
    tokens is decidable in fp32 and a coin-flip tie in bf16).

    Decode (m=1) → gemv_gpu. Prefill (m>1) → gfx12 BF16 WMMA when n and k
    are divisible by 16 (model dims always are); otherwise naive fallback.
    Pass use_wmma=False to force the naive path (layer-diff harness).
    """
    if m == 1:
        gemv_gpu[transpose_b=True](c, a, b, ctx)
        return

    # WMMA path: N and K must be tile-aligned. M is edge-masked in the kernel.
    if (
        use_wmma
        and n % WMMA_TILE == 0
        and k % WMMA_TILE == 0
    ):
        wmma_gemm_bf16[type_of(c).dtype](c, a, b, m, n, k, ctx)
        return

    _linear_naive(c, a, b, m, n, k, ctx)


# ===----------------------------------------------------------------------=== #
# FP8 E4M3 GEMV (decode M=1): y[n] = scale[n] * sum_k fp8(W[n,k]) * bf16(x[k])
# Weights stay FP8 in VRAM. Pattern follows MAX gemv_split_k for AMD (vectorized
# K-strip, multi-row tile_n, non-temporal weight loads, warp+block reduce) but
# with proper FP8→F32 cast (vendored split_k rebinds weight bits to a_type and
# is unsafe for mixed BF16 acts + FP8 weights).
# ===----------------------------------------------------------------------=== #

comptime FP8 = DType.float8_e4m3fn


def gemv_fp8[
    c_type: DType, add_residual: Bool = False
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a: TileTensor[mut=False, dtype=BF16, ...],
    w: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """C[n] = scale[n] * (W[n,:] · a[0,:])  with W FP8 E4M3, a BF16.
    If add_residual, C += that product (in-place).

    Uses MAX gemv_kernel_vector with FP8 simd_width + scale epilogue + PDL.
    Vendored gemv_gpu split-K rebinds weight bits to a_type (unsafe for mixed
    BF16 acts + FP8 weights). Weights/scales mut=False for exclusivity.
    """
    var c_ptr = c.ptr.as_unsafe_any_origin()
    var s_ptr = scale.ptr.as_unsafe_any_origin()

    @always_inline
    @__copy_capture(c_ptr, s_ptr)
    @parameter
    def epilogue[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var col = idx[1]
        var v = val[0].cast[F32]() * s_ptr[col]
        comptime if add_residual:
            v = v + c_ptr[col].cast[F32]()
        c_ptr[col] = v.cast[c_type]()

    comptime sw = simd_width_of[FP8, target=get_gpu_target()]()
    comptime WARPS_PER_BLOCK = 1024 // WARP_SIZE
    comptime pdl = PDLLevel.ON
    var block_dim = min(
        align_up(k // sw, WARP_SIZE),
        WARP_SIZE * WARPS_PER_BLOCK,
    )
    if block_dim < WARP_SIZE:
        block_dim = WARP_SIZE
    comptime kernel = gemv_kernel_vector[
        c_type,
        FP8,
        BF16,
        type_of(c).LayoutType,
        type_of(w).LayoutType,
        type_of(a).LayoutType,
        type_of(c).Storage,
        type_of(w).Storage,
        type_of(a).Storage,
        simd_width=sw,
        transpose_b=True,
        elementwise_lambda_fn=epilogue,
        check_bounds=True,
        pdl_level=pdl,
    ]
    ctx.enqueue_function[kernel](
        c,
        w,
        a,
        n,
        1,
        k,
        grid_dim=ceildiv(n, block_dim // WARP_SIZE),
        block_dim=block_dim,
        attributes=pdl_launch_attributes(pdl),
    )


def linear_fp8[
    c_type: DType
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a: TileTensor[mut=False, dtype=BF16, ...],
    w: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """C = A @ W^T with FP8 weights + per-channel F32 scale.
    Decode (m=1) uses gemv_fp8. Prefill (m>1) loops gemv over rows (correct,
    not the prefill performance path)."""
    if m == 1:
        gemv_fp8[c_type, False](c, a, w, scale, n, k, ctx)
        return
    # Row-at-a-time gemv for short prefill correctness only (not perf path).
    for row in range(m):
        var a_row = TileTensor(
            ptr=a.ptr + row * k, layout=row_major(Coord(Index(1, k)))
        )
        var c_row = TileTensor(
            ptr=c.ptr + row * n, layout=row_major(Coord(Index(1, n)))
        )
        gemv_fp8[c_type, False](c_row, a_row, w, scale, n, k, ctx)


def linear_add_residual_fp8(
    residual: TileTensor[mut=True, dtype=BF16, ...],
    a: TileTensor[mut=False, dtype=BF16, ...],
    w: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """residual += A @ W^T (FP8 weights)."""
    if m == 1:
        gemv_fp8[BF16, True](residual, a, w, scale, n, k, ctx)
        return
    for row in range(m):
        var a_row = TileTensor(
            ptr=a.ptr + row * k, layout=row_major(Coord(Index(1, k)))
        )
        var r_row = TileTensor(
            ptr=residual.ptr + row * n, layout=row_major(Coord(Index(1, n)))
        )
        gemv_fp8[BF16, True](r_row, a_row, w, scale, n, k, ctx)


def embed_fp8_kernel(
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    emb_ptr: UnsafePointer[Scalar[FP8], ImmutAnyOrigin],
    scale_ptr: UnsafePointer[Scalar[F32], ImmutAnyOrigin],
    ids_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    seq: Int,
    hidden: Int,
):
    """out[t,d] = scale[id[t]] * fp8_to_f32(embed[id[t], d])."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = seq * hidden
    if tid >= total:
        return
    var t = tid // hidden
    var d = tid % hidden
    var tok = Int(ids_ptr[t])
    var s = scale_ptr[tok]
    var v = emb_ptr[tok * hidden + d].cast[F32]()
    out_ptr[t * hidden + d] = (s * v).cast[BF16]()


def embed_lookup_fp8(
    dst: TileTensor[mut=True, dtype=BF16, ...],
    emb: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    ids: DeviceBuffer[DType.int32],
    seq: Int,
    hidden: Int,
    ctx: DeviceContext,
) raises:
    comptime TPB = 256
    ctx.enqueue_function[embed_fp8_kernel](
        dst.ptr.as_unsafe_any_origin(),
        emb.ptr.as_unsafe_any_origin(),
        scale.ptr.as_unsafe_any_origin(),
        ids.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        seq,
        hidden,
        grid_dim=(ceildiv(seq * hidden, TPB),),
        block_dim=(TPB,),
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
    # HF: (q * cos) + (rotate_half(q) * sin) with bf16 after every mul and add.
    # Mojo promotes BF16*BF16 to F32; a fused `(re*cos)-(im*sin)` then one store
    # cast is f32-accum — more accurate than HF and was a real correctness bug
    # (spike investigation probe 14, eabf42c). Round after each mul and after
    # the add/sub to match torch.
    var re_c = (re.cast[F32]() * cos_v.cast[F32]()).cast[BF16]()
    var im_s = (im.cast[F32]() * sin_v.cast[F32]()).cast[BF16]()
    var im_c = (im.cast[F32]() * cos_v.cast[F32]()).cast[BF16]()
    var re_s = (re.cast[F32]() * sin_v.cast[F32]()).cast[BF16]()
    x[base + i_re] = (re_c.cast[F32]() - im_s.cast[F32]()).cast[BF16]()
    x[base + i_im] = (im_c.cast[F32]() + re_s.cast[F32]()).cast[BF16]()


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
    # Same stepwise bf16 as rope_kernel (must stay in lockstep).
    var re_c = (re.cast[F32]() * cos_v.cast[F32]()).cast[BF16]()
    var im_s = (im.cast[F32]() * sin_v.cast[F32]()).cast[BF16]()
    var im_c = (im.cast[F32]() * cos_v.cast[F32]()).cast[BF16]()
    var re_s = (re.cast[F32]() * sin_v.cast[F32]()).cast[BF16]()
    x[base + i_re] = (re_c.cast[F32]() - im_s.cast[F32]()).cast[BF16]()
    x[base + i_im] = (im_c.cast[F32]() + re_s.cast[F32]()).cast[BF16]()


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


comptime ATTN_NUM_WARPS = 4


def attention_kernel[
    head_dim: Int, group: Int, NUM_WARPS: Int = ATTN_NUM_WARPS
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
    var tid = Int(thread_idx.x)
    var lane = tid % WARP
    var warp_id = tid // WARP

    var scores = stack_allocation[
        MAX_KEYS, F32, address_space = AddressSpace.SHARED
    ]()
    var q_sh = stack_allocation[
        head_dim, F32, address_space = AddressSpace.SHARED
    ]()

    var q_base = (tok * n_heads + head) * head_dim
    # Redundant across warps (every warp writes the same values) -- cheap
    # (head_dim=128 elements) and avoids a warp-0-only-then-barrier dance.
    var d = lane
    while d < head_dim:
        q_sh[d] = q_ptr[q_base + d].cast[F32]()
        d += WARP
    barrier()

    # Causal: this query attends to keys 0 .. past + tok inclusive.
    var n_keys = past + tok + 1

    # --- scores = scale * (q . k_j), warp-reduced dot per key ---------------
    # G1a-2: this loop is O(n_keys) and n_keys grows with decode position (up
    # to ~265 here) -- profiling showed attention at 24% of step time when
    # averaged honestly over a full 256-token run (vs 6.9% sampled only from
    # early, short-context steps). Each scores[j] is an INDEPENDENT
    # computation (no cross-j dependency), so striding the j-loop across
    # NUM_WARPS warps -- each still doing the identical per-key warp-reduce
    # over head_dim -- computes every score exactly as before, just more of
    # them at once. This does NOT change any reduction order: softmax and the
    # weighted-V-sum below remain single-warp, unchanged, bit-for-bit.
    var j = warp_id
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
        j += NUM_WARPS
    barrier()

    # --- softmax over scores[0 .. n_keys) -- single warp, UNCHANGED order --
    if warp_id == 0 and lane == 0:
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

    # --- out = sum_j p_j * v_j -- single warp, UNCHANGED accumulation order -
    if warp_id == 0:
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
        block_dim=(WARP * ATTN_NUM_WARPS,),
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


# ===----------------------------------------------------------------------=== #
# GPU argmax sampling: replaces a host-side per-token linear scan over the
# full 151,936-entry vocab (measured at ~51.6ms/token -- more than the whole
# forward pass it follows, ~800x the equivalent llama.cpp cost, bench/1a-ab.md
# Finding 2).
#
# NOT vendored, despite one existing: nn.argmaxmin_gpu.argmax_gpu (wrapping
# topk_gpu, K=1) was tried first. Its own source comments (nn/topk.mojo:736,
# 2168-2170) claim ties resolve to the lowest index -- but that guarantee is
# documented for TopK_2.best() (a single local heap) and for the SEPARATE
# _gumbel_argmax_fused_kernel path, not necessarily for the multi-stage
# reduction plain topk_gpu(sampling=False) actually runs for a
# 151936-element single row. Measured directly (experiments/argmax_gpu_probe.mojo):
# a hand-built exact tie between index 1632 and 11245 returned 11245 (the
# WRONG, higher index) -- oracle over vibes; the comment did not match the
# measured behavior for this call path, so it is not used. Written instead,
# with the identical warp+shared-mem reduction pattern already proven
# throughout this session (attention, QK-norm): one block, strided per-thread
# local reduction with an explicit "value > best OR (value == best AND
# idx < best_idx)" combine rule, then a block-wide tree reduction applying
# the SAME rule. Cast logits F32 -> BF16 first (matching HF's bf16 lm_head,
# the exact rounding already used by the host path this replaces). Only the
# single winning index round-trips to host; no full-vocab host transfer.
# ===----------------------------------------------------------------------=== #


def cast_f32_to_bf16_kernel(
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[F32], ImmutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    out_ptr[i] = in_ptr[i].cast[BF16]()


comptime ARGMAX_TPB = 256


def argmax_kernel(
    out_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    bf16_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    n: Int,
):
    var tid = Int(thread_idx.x)

    var best_val = stack_allocation[
        ARGMAX_TPB, F32, address_space = AddressSpace.SHARED
    ]()
    var best_idx = stack_allocation[
        ARGMAX_TPB, Int32, address_space = AddressSpace.SHARED
    ]()

    var local_val = Float32(-3.4e38)
    var local_idx = Int32(-1)
    var i = tid
    while i < n:
        var v = bf16_ptr[i].cast[F32]()
        if v > local_val:
            local_val = v
            local_idx = Int32(i)
        i += ARGMAX_TPB
    best_val[tid] = local_val
    best_idx[tid] = local_idx
    barrier()

    var stride = ARGMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            var other_val = best_val[tid + stride]
            var other_idx = best_idx[tid + stride]
            # Lowest index wins on an exact tie (matches torch.argmax /
            # the HF-bf16-then-argmax semantics established this session).
            if other_val > best_val[tid] or (
                other_val == best_val[tid] and other_idx < best_idx[tid]
            ):
                best_val[tid] = other_val
                best_idx[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        out_idx_ptr[0] = best_idx[0]


def argmax_logits(
    logits_ptr: UnsafePointer[Scalar[F32], MutAnyOrigin],
    mut bf16_scratch: DeviceBuffer[BF16],
    mut idx_scratch: DeviceBuffer[DType.int32],
    vocab: Int,
    ctx: DeviceContext,
) raises -> Int32:
    comptime TPB = 256
    ctx.enqueue_function[cast_f32_to_bf16_kernel](
        bf16_scratch.unsafe_ptr().as_unsafe_any_origin(),
        logits_ptr,
        vocab,
        grid_dim=(ceildiv(vocab, TPB),),
        block_dim=(TPB,),
    )

    ctx.enqueue_function[argmax_kernel](
        idx_scratch.unsafe_ptr().as_unsafe_any_origin(),
        bf16_scratch.unsafe_ptr().as_unsafe_any_origin(),
        vocab,
        grid_dim=(1,),
        block_dim=(ARGMAX_TPB,),
    )

    var best = Int32(0)
    with idx_scratch.map_to_host() as h:
        best = h[0]
    return best
