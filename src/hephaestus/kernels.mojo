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
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_down
from std.math import ceildiv, cos, exp, sin, sqrt
from std.memory import stack_allocation
from std.utils.index import Index, IndexList

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.gemv import gemv_gpu
from linalg.matmul.gpu import matmul_kernel_naive

from hephaestus.wmma_gfx12 import (
    BN,
    FP8,
    FP8_E4M3_MAX,
    WMMA_TILE,
    wmma_gemm_bf16,
    wmma_gemm_bf16_residual,
    wmma_gemm_fp8_decode,
    wmma_gemm_fp8_prefill,
)

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
# FP8 W8A8 decode (G1b-4): quantize act → FP8 WMMA → dual-scale epilogue
#
# RDNA4 has no scalar FP8 dot. Compute is WMMA-only, both operands 8-bit.
# Weights stay FP8 in the arena (no dequant). Activations: per-token absmax
# quant to E4M3, pad M=1 → 16×K, WMMA f32.16x16x16.fp8.fp8 (exp3g).
# Epilogue: C[n] = act_scale * weight_scale[n] * acc[0, n]  (+ residual).
# ===----------------------------------------------------------------------=== #

comptime QUANT_TPB = 256


def quantize_act_absmax_kernel(
    a_fp8_ptr: UnsafePointer[Scalar[FP8], MutAnyOrigin],
    scale_ptr: UnsafePointer[Scalar[F32], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    k: Int,
):
    """Per-token absmax quant of x[0..k) → a_fp8 row 0; zero-pad rows 1..15.

    scale = max(|x|) / FP8_E4M3_MAX.  One block; shared-mem tree reduce.
    """
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        QUANT_TPB, F32, address_space = AddressSpace.SHARED
    ]()

    # Local absmax.
    var local = Float32(0)
    var i = tid
    while i < k:
        var v = x_ptr[i].cast[F32]()
        if v < Float32(0):
            v = -v
        if v > local:
            local = v
        i += QUANT_TPB
    sh[tid] = local
    barrier()

    var stride = QUANT_TPB // 2
    while stride > 0:
        if tid < stride:
            var o = sh[tid + stride]
            if o > sh[tid]:
                sh[tid] = o
        barrier()
        stride //= 2

    var max_abs = sh[0]
    var scale = max_abs / FP8_E4M3_MAX
    # Avoid /0 and under-scale when activation is all zeros.
    if scale < Float32(1e-12):
        scale = Float32(1.0)
    if tid == 0:
        scale_ptr[0] = scale
    barrier()

    var inv = Float32(1.0) / scale
    # Row 0: quantized activation.
    i = tid
    while i < k:
        a_fp8_ptr[i] = (x_ptr[i].cast[F32]() * inv).cast[FP8]()
        i += QUANT_TPB
    # Rows 1..15: zeros (WMMA pad).
    var pad_elems = 15 * k
    i = tid
    while i < pad_elems:
        a_fp8_ptr[k + i] = Scalar[FP8](0)
        i += QUANT_TPB


def quantize_act_fp8_pad(
    a_fp8: DeviceBuffer[FP8],
    act_scale: DeviceBuffer[F32],
    x: TileTensor[mut=False, dtype=BF16, ...],
    k: Int,
    ctx: DeviceContext,
) raises:
    """Quantize BF16 row x[0,:k] → a_fp8[16,k] pad + act_scale[0]."""
    ctx.enqueue_function[quantize_act_absmax_kernel](
        a_fp8.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        act_scale.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        x.ptr.as_immutable().as_unsafe_any_origin(),
        k,
        grid_dim=(1,),
        block_dim=(QUANT_TPB,),
    )


def quantize_act_rows_kernel(
    a_fp8_ptr: UnsafePointer[Scalar[FP8], MutAnyOrigin],
    scale_ptr: UnsafePointer[Scalar[F32], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[BF16], ImmutAnyOrigin],
    M: Int,
    K: Int,
):
    """Per-row absmax quant: BF16 A[M,K] → FP8 A_fp8[M,K] + act_scale[M].

    One block per row; shared-mem tree reduce over K (no pad-to-16).
    """
    var row = Int(block_idx.x)
    if row >= M:
        return
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        QUANT_TPB, F32, address_space = AddressSpace.SHARED
    ]()

    var local = Float32(0)
    var i = tid
    var base = row * K
    while i < K:
        var v = x_ptr[base + i].cast[F32]()
        if v < Float32(0):
            v = -v
        if v > local:
            local = v
        i += QUANT_TPB
    sh[tid] = local
    barrier()

    var stride = QUANT_TPB // 2
    while stride > 0:
        if tid < stride:
            var o = sh[tid + stride]
            if o > sh[tid]:
                sh[tid] = o
        barrier()
        stride //= 2

    var max_abs = sh[0]
    var scale = max_abs / FP8_E4M3_MAX
    if scale < Float32(1e-12):
        scale = Float32(1.0)
    if tid == 0:
        scale_ptr[row] = scale
    barrier()

    var inv = Float32(1.0) / scale
    i = tid
    while i < K:
        a_fp8_ptr[base + i] = (x_ptr[base + i].cast[F32]() * inv).cast[FP8]()
        i += QUANT_TPB


def quantize_act_rows_fp8(
    a_fp8: DeviceBuffer[FP8],
    act_scale: DeviceBuffer[F32],
    x: TileTensor[mut=False, dtype=BF16, ...],
    m: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Quantize BF16 A[m,k] → FP8 + per-row scales (prefill, no pad)."""
    ctx.enqueue_function[quantize_act_rows_kernel](
        a_fp8.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        act_scale.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        x.ptr.as_immutable().as_unsafe_any_origin(),
        m,
        k,
        grid_dim=(m,),
        block_dim=(QUANT_TPB,),
    )


# ===----------------------------------------------------------------------=== #
# FP8 KV quality probe: quantize→dequantize at cache-write (keep BF16 storage)
#
# Numerically identical to a real FP8 E4M3 KV cache read path. Does NOT change
# storage format — only the values written into the existing BF16 cache.
# Scaling: per-token (row), per-layer (caller), separate K vs V, absmax/448.
# Apply K AFTER RoPE; V after v_proj (V has no RoPE).
# ===----------------------------------------------------------------------=== #

def kv_fp8_roundtrip_kernel(
    data_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n_tokens: Int,
    row_width: Int,
):
    """In-place BF16 → E4M3 → BF16 roundtrip per token row.

    scale = max(|row|) / FP8_E4M3_MAX. One block per token; shared-mem absmax.
    """
    var tok = Int(block_idx.x)
    if tok >= n_tokens:
        return
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        QUANT_TPB, F32, address_space = AddressSpace.SHARED
    ]()

    var base = tok * row_width
    var local = Float32(0)
    var i = tid
    while i < row_width:
        var v = data_ptr[base + i].cast[F32]()
        if v < Float32(0):
            v = -v
        if v > local:
            local = v
        i += QUANT_TPB
    sh[tid] = local
    barrier()

    var stride = QUANT_TPB // 2
    while stride > 0:
        if tid < stride:
            var o = sh[tid + stride]
            if o > sh[tid]:
                sh[tid] = o
        barrier()
        stride //= 2

    var max_abs = sh[0]
    var scale = max_abs / FP8_E4M3_MAX
    if scale < Float32(1e-12):
        scale = Float32(1.0)
    barrier()

    var inv = Float32(1.0) / scale
    i = tid
    while i < row_width:
        # quantize to E4M3 then dequantize back to BF16 (same values a real
        # FP8 cache would return after dequant-on-read).
        var q = (data_ptr[base + i].cast[F32]() * inv).cast[FP8]()
        data_ptr[base + i] = (q.cast[F32]() * scale).cast[BF16]()
        i += QUANT_TPB


def kv_fp8_roundtrip_inplace(
    data_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    n_tokens: Int,
    row_width: Int,
    ctx: DeviceContext,
) raises:
    """Host launch: in-place E4M3 quant→dequant on [n_tokens, row_width] BF16."""
    if n_tokens <= 0:
        return
    ctx.enqueue_function[kv_fp8_roundtrip_kernel](
        data_ptr,
        n_tokens,
        row_width,
        grid_dim=(n_tokens,),
        block_dim=(QUANT_TPB,),
    )


def gemv_fp8[
    c_type: DType,
    add_residual: Bool = False,
    swizzled_b: Bool = False,
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a: TileTensor[mut=False, dtype=BF16, ...],
    w: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    a_fp8_ws: DeviceBuffer[FP8],
    act_scale_ws: DeviceBuffer[F32],
    n: Int,
    k: Int,
    ctx: DeviceContext,
    *,
    do_quantize: Bool = True,
) raises:
    """W8A8 decode: quantize a → FP8 WMMA → C[n] = act_s * w_s[n] * (A@W^T)[0,n].

    No weight dequant. Workspace a_fp8_ws must hold ≥ 16*k FP8 elements.
    Pass do_quantize=False when a_fp8_ws/act_scale_ws already hold the
    quantized activation (shared q/k/v or gate/up).
    swizzled_b=False for embed/lm_head (row-major, not fragment-swizzled).
    """
    if do_quantize:
        quantize_act_fp8_pad(a_fp8_ws, act_scale_ws, a, k, ctx)
    wmma_gemm_fp8_decode[c_type, add_residual, swizzled_b](
        c,
        a_fp8_ws,
        w,
        scale,
        act_scale_ws,
        n,
        k,
        ctx,
    )


def linear_fp8[
    c_type: DType, swizzled_b: Bool = False
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a: TileTensor[mut=False, dtype=BF16, ...],
    w: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    a_fp8_ws: DeviceBuffer[FP8],
    act_scale_ws: DeviceBuffer[F32],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
    *,
    do_quantize: Bool = True,
) raises:
    """C = A @ W^T via W8A8 FP8 WMMA.

    M=1: decode gemv. M>1: v3a prefill when n%64 and k%16; else row-loop.
    """
    if m == 1:
        gemv_fp8[c_type, False, swizzled_b](
            c,
            a,
            w,
            scale,
            a_fp8_ws,
            act_scale_ws,
            n,
            k,
            ctx,
            do_quantize=do_quantize,
        )
        return

    if n % BN == 0 and k % WMMA_TILE == 0:
        if do_quantize:
            quantize_act_rows_fp8(a_fp8_ws, act_scale_ws, a, m, k, ctx)
        wmma_gemm_fp8_prefill[c_type, False](
            c, a_fp8_ws, w, scale, act_scale_ws, m, n, k, ctx
        )
        return

    for row in range(m):
        var a_row = TileTensor(
            ptr=a.ptr + row * k, layout=row_major(Coord(Index(1, k)))
        )
        var c_row = TileTensor(
            ptr=c.ptr + row * n, layout=row_major(Coord(Index(1, n)))
        )
        gemv_fp8[c_type, False, swizzled_b](
            c_row, a_row, w, scale, a_fp8_ws, act_scale_ws, n, k, ctx
        )


def linear_add_residual_fp8[
    swizzled_b: Bool = False
](
    residual: TileTensor[mut=True, dtype=BF16, ...],
    a: TileTensor[mut=False, dtype=BF16, ...],
    w: TileTensor[mut=False, dtype=FP8, ...],
    scale: TileTensor[mut=False, dtype=F32, ...],
    a_fp8_ws: DeviceBuffer[FP8],
    act_scale_ws: DeviceBuffer[F32],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
    *,
    do_quantize: Bool = True,
) raises:
    """residual += A @ W^T (W8A8 FP8 WMMA)."""
    if m == 1:
        gemv_fp8[BF16, True, swizzled_b](
            residual,
            a,
            w,
            scale,
            a_fp8_ws,
            act_scale_ws,
            n,
            k,
            ctx,
            do_quantize=do_quantize,
        )
        return

    if n % BN == 0 and k % WMMA_TILE == 0:
        if do_quantize:
            quantize_act_rows_fp8(a_fp8_ws, act_scale_ws, a, m, k, ctx)
        wmma_gemm_fp8_prefill[BF16, True](
            residual, a_fp8_ws, w, scale, act_scale_ws, m, n, k, ctx
        )
        return

    for row in range(m):
        var a_row = TileTensor(
            ptr=a.ptr + row * k, layout=row_major(Coord(Index(1, k)))
        )
        var r_row = TileTensor(
            ptr=residual.ptr + row * n, layout=row_major(Coord(Index(1, n)))
        )
        gemv_fp8[BF16, True, swizzled_b](
            r_row, a_row, w, scale, a_fp8_ws, act_scale_ws, n, k, ctx
        )


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
    *,
    use_wmma: Bool = True,
) raises:
    """residual += A @ B^T. Used for o_proj and down_proj.

    Decode (m=1) → gemv_gpu with F32 residual epilogue (unchanged).
    Prefill (m>1) → gfx12 BF16 WMMA with fused residual-add store when n,k
    are tile-aligned; otherwise naive matmul + same epilogue.
    Pass use_wmma=False to force the naive path (layer-diff harness).
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

    if use_wmma and n % WMMA_TILE == 0 and k % WMMA_TILE == 0:
        wmma_gemm_bf16_residual(residual, a, b, m, n, k, ctx)
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


# Phase bitmasks for measurement launches (host times each enqueue).
comptime ATTN_PHASE_QK = 1
comptime ATTN_PHASE_SOFTMAX = 2
comptime ATTN_PHASE_PV = 4
comptime ATTN_PHASE_ALL = 7


def attention_kernel_parallel[
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
    phases: Int,
):
    """Parallel softmax + PV stopgap (ATTN-STOPGAP). QK identical to serial.

    Softmax: block max/sum over scores (reduction-order differs on sum only).
    PV: NT-way column-parallel, serial key order → bit-identical to serial PV
    given the same normalized scores (incl. BF16 prob round-trip).
    """
    comptime NT = WARP * NUM_WARPS
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
    var red = stack_allocation[
        NUM_WARPS, F32, address_space = AddressSpace.SHARED
    ]()

    # --- Phase 0: load Q (UNCHANGED from serial — scale applied after QK) ---
    var q_base = (tok * n_heads + head) * head_dim
    var d = tid
    while d < head_dim:
        q_sh[d] = q_ptr[q_base + d].cast[F32]()
        d += NT
    barrier()

    var n_keys = past + tok + 1

    # --- Phase 1: QK (UNCHANGED — same multi-warp warp-reduce as serial) ---
    if (phases & ATTN_PHASE_QK) != 0:
        var j = warp_id
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
                scores[j] = partial * scale
            j += NUM_WARPS
        barrier()

    # --- Phase 2: parallel softmax (block max / exp+sum / normalize) ------
    if (phases & ATTN_PHASE_SOFTMAX) != 0:
        # 2a. block max
        var lmax = Float32(-3.4e38)
        var jm = tid
        while jm < n_keys:
            lmax = max(lmax, scores[jm])
            jm += NT
        var offm = WARP // 2
        while offm > 0:
            lmax = max(lmax, shuffle_down(lmax, UInt32(offm)))
            offm //= 2
        if lane == 0:
            red[warp_id] = lmax
        barrier()
        if warp_id == 0:
            var wmax = Float32(-3.4e38)
            if lane < NUM_WARPS:
                wmax = red[lane]
            var om = 2
            while om > 0:
                wmax = max(wmax, shuffle_down(wmax, UInt32(om)))
                om //= 2
            if lane == 0:
                red[0] = wmax
        barrier()
        var m = red[0]

        # 2b. exp + block sum (overwrite scores with unnormalized exp)
        var lsum = Float32(0)
        var je = tid
        while je < n_keys:
            var e = exp(scores[je] - m)
            scores[je] = e
            lsum += e
            je += NT
        var offs = WARP // 2
        while offs > 0:
            lsum = lsum + shuffle_down(lsum, UInt32(offs))
            offs //= 2
        if lane == 0:
            red[warp_id] = lsum
        barrier()
        if warp_id == 0:
            var wsum = Float32(0)
            if lane < NUM_WARPS:
                wsum = red[lane]
            var os = 2
            while os > 0:
                wsum = wsum + shuffle_down(wsum, UInt32(os))
                os //= 2
            if lane == 0:
                red[0] = wsum
        barrier()
        var inv_s = Float32(1.0) / red[0]

        # 2c. normalize + BF16 round-trip (match serial for PV bit-identity)
        var jn = tid
        while jn < n_keys:
            scores[jn] = (scores[jn] * inv_s).cast[BF16]().cast[F32]()
            jn += NT
        barrier()

    # --- Phase 3: parallel PV (column-parallel, serial key order) ---------
    if (phases & ATTN_PHASE_PV) != 0:
        var o_base = (tok * n_heads + head) * head_dim
        var od = tid
        while od < head_dim:
            var acc = Float32(0)
            for jj in range(n_keys):
                var v_base = (jj * n_kv_heads + kv_head) * head_dim
                acc += scores[jj] * v_ptr[v_base + od].cast[F32]()
            out_ptr[o_base + od] = acc.cast[BF16]()
            od += NT


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
    *,
    parallel: Bool = True,
    phases: Int = ATTN_PHASE_ALL,
) raises:
    """Attention dispatch. Default: parallel softmax+PV stopgap.

    parallel=False uses the original single-lane softmax / single-warp PV
    (validation reference). phases masks QK/SOFTMAX/PV for sub-split timing.
    """
    if past + seq > MAX_KEYS:
        raise Error("sequence exceeds MAX_KEYS")
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    if parallel:
        ctx.enqueue_function[attention_kernel_parallel[head_dim, group]](
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            n_heads,
            n_kv_heads,
            seq,
            past,
            scale,
            phases,
            grid_dim=(n_heads, seq),
            block_dim=(WARP * ATTN_NUM_WARPS,),
        )
    else:
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
