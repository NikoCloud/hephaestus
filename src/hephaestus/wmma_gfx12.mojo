# ===----------------------------------------------------------------------=== #
# gfx12 (RDNA4) BF16 WMMA GEMM — proven in exp3g / exp4a / exp4b
#
# C[M, N] = A[M, K] @ W[N, K]^T
#   A, W: BF16 row-major; C: BF16 or F32 (lm_head keeps F32)
#
# Lane mappings (G1b-0, locked):
#   A-load: a[j] = A[(MB + l%16)*K + ks + (l/16)*8 + j]
#   W-load: b[j] = W[(NB + l%16)*K + ks + (l/16)*8 + j]
#   store:  C[(MB + (l/16)*8 + j)*N + NB + l%16] = acc[j]
#
# v1: direct global loads per lane (correct, uncoalesced).
# v2: cooperative coalesced global→LDS, then fragment from LDS (this file's
#     default). Same accumulation order as v1 → bit-identical C. No reuse;
#     each LDS element is read once. Wider tiles (v3) add reuse later.
#
# Requires N%16==0, K%16==0; M edge-masked (ceildiv grid + zero A / skip C).
# ===----------------------------------------------------------------------=== #

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import bitcast, stack_allocation
from std.sys.intrinsics import llvm_intrinsic

from layout import TileTensor

comptime WMMA_TILE = 16
comptime WMMA_LANES = 32
comptime WMMA_FRAG = 8
# LDS row stride. 16 = packed tile; set to 24 if bank conflicts dominate.
comptime LDS_STRIDE = 16


@always_inline
def wmma_bf16(
    a_i16: SIMD[DType.int16, WMMA_FRAG],
    b_i16: SIMD[DType.int16, WMMA_FRAG],
    c: SIMD[DType.float32, WMMA_FRAG],
) -> SIMD[DType.float32, WMMA_FRAG]:
    """Direct llvm.amdgcn.wmma.f32.16x16x16.bf16 (exp3g arity: 3 operands)."""
    return llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.bf16",
        SIMD[DType.float32, WMMA_FRAG],
        has_side_effect=False,
    ](a_i16, b_i16, c)


# ===----------------------------------------------------------------------=== #
# v1 — direct global loads (reference for bit-identical gate)
# ===----------------------------------------------------------------------=== #


def wmma_gemm_kernel_v1[
    c_type: DType
](
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """v1: one 16×16 tile / block; wave32; K-loop; direct global fragment loads."""
    var l = Int(thread_idx.x)
    if l >= WMMA_LANES:
        return

    var n_tile = Int(block_idx.x)
    var m_tile = Int(block_idx.y)
    var MB = m_tile * WMMA_TILE
    var NB = n_tile * WMMA_TILE

    # Uniform across the block (same block_idx) — safe before any barrier.
    if MB >= M or NB >= N:
        return

    var row_or_col = l % WMMA_TILE
    var half = l // WMMA_TILE

    var acc = SIMD[DType.float32, WMMA_FRAG](0.0)

    var ks = 0
    while ks < K:
        var a_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var b_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var a_row = MB + row_or_col
        if a_row < M:
            var a_row_base = a_row * K + ks + half * WMMA_FRAG
            @parameter
            for j in range(WMMA_FRAG):
                a_bf16[j] = a_ptr[a_row_base + j]
        var w_row_base = (NB + row_or_col) * K + ks + half * WMMA_FRAG
        @parameter
        for j in range(WMMA_FRAG):
            b_bf16[j] = w_ptr[w_row_base + j]

        var a_i16 = bitcast[DType.int16, WMMA_FRAG](a_bf16)
        var b_i16 = bitcast[DType.int16, WMMA_FRAG](b_bf16)
        acc = wmma_bf16(a_i16, b_i16, acc)
        ks += WMMA_TILE

    @parameter
    for j in range(WMMA_FRAG):
        var m = MB + half * WMMA_FRAG + j
        var n = NB + row_or_col
        if m < M:
            c_ptr[m * N + n] = acc[j].cast[c_type]()


# ===----------------------------------------------------------------------=== #
# v2 — LDS cooperative coalesced load + fragment from LDS
# ===----------------------------------------------------------------------=== #


def wmma_gemm_kernel[
    c_type: DType
](
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """v2: coalesced global→LDS per K-strip, then G1b-0 fragment from LDS.

    Per strip:
      1. cooperative load A,B tiles into LDS (16 lanes × row, 8 iters)
      2. barrier()  — LDS writes visible before fragment read
      3. fragment load from LDS → WMMA → accumulate
      4. barrier()  — all LDS reads done before next strip overwrites
    """
    var l = Int(thread_idx.x)
    # block_dim is exactly WMMA_LANES; keep this for safety. No barriers yet.
    if l >= WMMA_LANES:
        return

    var n_tile = Int(block_idx.x)
    var m_tile = Int(block_idx.y)
    var MB = m_tile * WMMA_TILE
    var NB = n_tile * WMMA_TILE

    # Uniform exit (all lanes same block_idx) — no barrier below on this path.
    if MB >= M or NB >= N:
        return

    var row_or_col = l % WMMA_TILE
    var half = l // WMMA_TILE

    # Single-buffered LDS: one 16×LDS_STRIDE A tile + one B tile per strip.
    var A_lds = stack_allocation[
        WMMA_TILE * LDS_STRIDE, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()
    var B_lds = stack_allocation[
        WMMA_TILE * LDS_STRIDE, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()

    var acc = SIMD[DType.float32, WMMA_FRAG](0.0)

    var ks = 0
    while ks < K:
        # --- 1. cooperative coalesced load global → LDS -------------------
        # Iteration i: lanes 0-15 write row 2i (col = l%16); lanes 16-31 write
        # row 2i+1. Each half-wave reads 16 contiguous BF16 (32 B) = one
        # coalesced transaction per row.
        @parameter
        for i in range(8):
            var row = i * 2 + half  # half = l//16 ∈ {0,1}
            var col = row_or_col  # l%16
            var lds_idx = row * LDS_STRIDE + col
            var g_row_a = MB + row
            if g_row_a < M:
                A_lds[lds_idx] = a_ptr[g_row_a * K + ks + col]
            else:
                A_lds[lds_idx] = Scalar[DType.bfloat16](0)
            # N is tile-aligned; NB+row stays inside the launched N tile.
            B_lds[lds_idx] = w_ptr[(NB + row) * K + ks + col]

        # --- 2. barrier: LDS writes visible before any fragment read ------
        barrier()

        # --- 3. fragment load from LDS (same mapping as v1, LDS base) -----
        var a_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var b_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var frag_base = row_or_col * LDS_STRIDE + half * WMMA_FRAG
        @parameter
        for j in range(WMMA_FRAG):
            a_bf16[j] = A_lds[frag_base + j]
            b_bf16[j] = B_lds[frag_base + j]

        var a_i16 = bitcast[DType.int16, WMMA_FRAG](a_bf16)
        var b_i16 = bitcast[DType.int16, WMMA_FRAG](b_bf16)
        acc = wmma_bf16(a_i16, b_i16, acc)

        # --- 4. barrier: all LDS reads done before next strip overwrites --
        barrier()
        ks += WMMA_TILE

    @parameter
    for j in range(WMMA_FRAG):
        var m = MB + half * WMMA_FRAG + j
        var n = NB + row_or_col
        if m < M:
            c_ptr[m * N + n] = acc[j].cast[c_type]()


# ===----------------------------------------------------------------------=== #
# Host launchers
# ===----------------------------------------------------------------------=== #


def wmma_gemm_bf16[
    c_type: DType
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a: TileTensor[DType.bfloat16, ...],
    b: TileTensor[DType.bfloat16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Host launch (v2 LDS default): C[m,n] = A[m,k] @ B[n,k]^T."""
    debug_assert(m > 0, "wmma_gemm_bf16: m must be > 0")
    debug_assert(n % WMMA_TILE == 0, "wmma_gemm_bf16: n must be divisible by 16")
    debug_assert(k % WMMA_TILE == 0, "wmma_gemm_bf16: k must be divisible by 16")

    var n_tiles = n // WMMA_TILE
    var m_tiles = ceildiv(m, WMMA_TILE)
    comptime kernel = wmma_gemm_kernel[c_type]
    ctx.enqueue_function[kernel](
        a.ptr.as_immutable().as_unsafe_any_origin(),
        b.ptr.as_immutable().as_unsafe_any_origin(),
        c.ptr.as_unsafe_any_origin(),
        m,
        n,
        k,
        grid_dim=(n_tiles, m_tiles),
        block_dim=(WMMA_LANES,),
    )


def wmma_gemm_bf16_v1[
    c_type: DType
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a: TileTensor[DType.bfloat16, ...],
    b: TileTensor[DType.bfloat16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Host launch (v1 direct-global): for bit-identical comparison vs v2."""
    debug_assert(m > 0, "wmma_gemm_bf16_v1: m must be > 0")
    debug_assert(n % WMMA_TILE == 0, "wmma_gemm_bf16_v1: n must be divisible by 16")
    debug_assert(k % WMMA_TILE == 0, "wmma_gemm_bf16_v1: k must be divisible by 16")

    var n_tiles = n // WMMA_TILE
    var m_tiles = ceildiv(m, WMMA_TILE)
    comptime kernel = wmma_gemm_kernel_v1[c_type]
    ctx.enqueue_function[kernel](
        a.ptr.as_immutable().as_unsafe_any_origin(),
        b.ptr.as_immutable().as_unsafe_any_origin(),
        c.ptr.as_unsafe_any_origin(),
        m,
        n,
        k,
        grid_dim=(n_tiles, m_tiles),
        block_dim=(WMMA_LANES,),
    )


# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# FP8 E4M3 × FP8 E4M3 → F32 WMMA (W8A8 decode path, G1b-4)
#
# Intrinsic (exp3g PASS on gfx1201):
#   llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8
#   A/B: SIMD[int32, 2]  = 8 packed FP8 bytes (v2i32)
#   C/D: SIMD[float32, 8]
#
# Lane mappings (G1b-0 geometry):
#   A-load (row-major pad): a[j] = A[(l%16)*K + ks + (l/16)*8 + j]
#   B-load (SWIZZLED weights — default):
#     tile_base = n_tile*(K/16)*256 + k_tile*256
#     b[j] = W_swz[tile_base + l*8 + j]   # coalesced: consecutive lanes, +8B
#   B-load (row-major, embed/lm_head only):
#     b[j] = W[(NB + l%16)*K + ks + (l/16)*8 + j]
#   D-row0: half==0 → C[NB + l%16] = scale_act * w_scale[n] * acc[0]
#
# No LDS: M=1 has no reuse; barriers regressed. Swizzle coalesces without LDS.
# ===----------------------------------------------------------------------=== #

comptime FP8 = DType.float8_e4m3fn
# FP8 E4M3 finite max (|max| ≈ 448). Used by activation absmax quant.
comptime FP8_E4M3_MAX = Float32(448.0)


@always_inline
def wmma_fp8(
    a_i32: SIMD[DType.int32, 2],
    b_i32: SIMD[DType.int32, 2],
    c: SIMD[DType.float32, WMMA_FRAG],
) -> SIMD[DType.float32, WMMA_FRAG]:
    """Direct llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8 (exp3g arity: 3 operands)."""
    return llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8",
        SIMD[DType.float32, WMMA_FRAG],
        has_side_effect=False,
    ](a_i32, b_i32, c)


def wmma_fp8_decode_kernel[
    c_type: DType, add_residual: Bool, swizzled_b: Bool
](
    a_ptr: UnsafePointer[Scalar[FP8], ImmutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[FP8], ImmutAnyOrigin],
    w_scale_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    act_scale_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    N: Int,
    K: Int,
):
    """W8A8 decode: C[n] = act_scale * w_scale[n] * (A@W^T)[0,n].

    swizzled_b=True: projection weights in fragment order (coalesced B-load).
    swizzled_b=False: row-major B (embed/lm_head only).
    """
    var l = Int(thread_idx.x)
    if l >= WMMA_LANES:
        return

    var n_tile = Int(block_idx.x)
    var NB = n_tile * WMMA_TILE
    if NB >= N:
        return

    var row_or_col = l % WMMA_TILE
    var half = l // WMMA_TILE
    var k_tiles = K // WMMA_TILE
    var acc = SIMD[DType.float32, WMMA_FRAG](0.0)

    var ks = 0
    var k_tile = 0
    # Precompute n_tile * k_tiles * 256 once (constant across K-loop).
    var n_tile_base = n_tile * k_tiles * 256
    while ks < K:
        # A: row-major padded [16,K] — vector load 8 FP8.
        var a_row_base = row_or_col * K + ks + half * WMMA_FRAG
        var a_fp8 = a_ptr.load[width=WMMA_FRAG](a_row_base)

        var b_fp8: SIMD[FP8, WMMA_FRAG]
        comptime if swizzled_b:
            # Fragment-order: lane l reads contiguous 8 bytes → one coalesced
            # 256B tile per wave (32 lanes × 8 B).
            var b_base = n_tile_base + k_tile * 256 + l * 8
            b_fp8 = w_ptr.load[width=WMMA_FRAG](b_base)
        else:
            # Row-major (embed / unswizzled).
            var w_row_base = (NB + row_or_col) * K + ks + half * WMMA_FRAG
            b_fp8 = w_ptr.load[width=WMMA_FRAG](w_row_base)

        acc = wmma_fp8(
            bitcast[DType.int32, 2](a_fp8),
            bitcast[DType.int32, 2](b_fp8),
            acc,
        )
        ks += WMMA_TILE
        k_tile += 1

    if half == 0:
        var n = NB + row_or_col
        if n < N:
            var sc = act_scale_ptr[0] * w_scale_ptr[n]
            var v = acc[0] * sc
            comptime if add_residual:
                v = v + c_ptr[n].cast[DType.float32]()
            c_ptr[n] = v.cast[c_type]()


def wmma_gemm_fp8_decode[
    c_type: DType, add_residual: Bool = False, swizzled_b: Bool = True
](
    c: TileTensor[mut=True, dtype=c_type, ...],
    a_fp8: DeviceBuffer[FP8],
    w: TileTensor[mut=False, dtype=FP8, ...],
    w_scale: TileTensor[mut=False, dtype = DType.float32, ...],
    act_scale: DeviceBuffer[DType.float32],
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Launch W8A8 decode WMMA. Default: swizzled B (projection weights)."""
    debug_assert(n % WMMA_TILE == 0, "wmma_gemm_fp8_decode: n must be divisible by 16")
    debug_assert(k % WMMA_TILE == 0, "wmma_gemm_fp8_decode: k must be divisible by 16")
    comptime kernel = wmma_fp8_decode_kernel[c_type, add_residual, swizzled_b]
    ctx.enqueue_function[kernel](
        a_fp8.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        w.ptr.as_immutable().as_unsafe_any_origin(),
        w_scale.ptr.as_immutable().as_unsafe_any_origin(),
        act_scale.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        c.ptr.as_unsafe_any_origin(),
        n,
        k,
        grid_dim=(n // WMMA_TILE,),
        block_dim=(WMMA_LANES,),
    )
