# ===----------------------------------------------------------------------=== #
# gfx12 (RDNA4) BF16 WMMA GEMM — proven in exp3g / exp4a / exp4b / exp4c
#
# C[M, N] = A[M, K] @ W[N, K]^T
#   A, W: BF16 row-major; C: BF16 or F32 (lm_head keeps F32)
#
# Lane mappings (G1b-0, locked):
#   A-load: a[j] = A[(MB + l%16)*K + ks + (l/16)*8 + j]
#   W-load: b[j] = W[(NB + l%16)*K + ks + (l/16)*8 + j]
#   store:  C[(MB + (l/16)*8 + j)*N + NB + l%16] = acc[j]
#
# v1: direct global loads (bit-identical reference).
# v2: 16×16 LDS staging, no reuse (coalescing only).
# v3a: 64×64 / 4-wave / BK=16, LDS reuse intensity 32 (spec G1b-3a).
#      A fragment hoisted out of sc loop (4× A reuse). Fused residual epilogue.
#
# VGPR/occupancy (v3a, gfx1201, nightly 2026071206) — from --emit asm metadata:
#   Actual: 92 VGPR/lane, 0 spills; LDS 4096 B/wg; SGPR ~18–20.
#   Design estimate was ~55; compiler uses more temps. Occupancy (theoretical):
#   256 VGPR/SIMD → floor(256/96)≈2 waves/SIMD → ~8 waves/WGP → ~2 WGs of 4 waves.
#   VGPR is the limiter (LDS would allow more). v3b (more acc) must watch this.
#
# Dispatch: M,N % 64 == 0 and K % 16 == 0 → v3a; else N,K % 16 → v2.
# ===----------------------------------------------------------------------=== #

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.host import DeviceContext
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
# v3a workgroup tile
comptime BM = 64
comptime BN = 64
comptime BK = 16
comptime V3A_THREADS = 128  # 4 waves × 32
comptime V3A_WAVES = 4
comptime V3A_SC = 4  # N sub-cols per wave


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
# v3a — 64×64 multi-wave LDS reuse (spec 2026-07-13_v3a-wmma-wide-tile-spec)
# Wave w owns sub-row sr=w (rows w*16..w*16+15), all 4 N-sub-cols sc∈0..3.
# Critical: A fragment loaded ONCE per K-strip, reused across sc (4× A reuse).
# ===----------------------------------------------------------------------=== #


def wmma_gemm_kernel_v3a[
    c_type: DType,
    fuse_residual: Bool,
](
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """v3a: one 64×64 output tile / workgroup (128 threads, 4 waves), BK=16.

    Preconditions (caller asserts): M%64==0, N%64==0, K%16==0.
    fuse_residual: C is residual stream; val = acc + F32(C[m,n]) before BF16 cast.
    """
    var tid = Int(thread_idx.x)
    if tid >= V3A_THREADS:
        return

    var w = tid // WMMA_LANES  # wave 0..3 = owned sub-row sr
    var l = tid % WMMA_LANES  # lane-in-wave 0..31

    var n_tile = Int(block_idx.x)
    var m_tile = Int(block_idx.y)
    var MB = m_tile * BM
    var NB = n_tile * BN

    # Uniform exit (same block_idx).
    if MB >= M or NB >= N:
        return

    # LDS: A[64,16] and B[64,16] (B n-major from W), row stride LDS_STRIDE.
    var A_lds = stack_allocation[
        BM * LDS_STRIDE, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()
    var B_lds = stack_allocation[
        BN * LDS_STRIDE, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()

    # 4 persistent F32 accumulators (one per N sub-col).
    var acc = InlineArray[SIMD[DType.float32, WMMA_FRAG], V3A_SC](
        uninitialized=True
    )
    @parameter
    for sc in range(V3A_SC):
        acc[sc] = SIMD[DType.float32, WMMA_FRAG](0.0)

    var ks = 0
    while ks < K:
        # ---- 5a. cooperative load (all 128 threads) ----
        # A tile [64,16]: 8 passes × 128 → 1024 elems. Coalesced 16-col runs.
        @parameter
        for i in range(8):
            var row = i * 8 + tid // 16  # 0..63
            var col = tid % 16  # 0..15
            A_lds[row * LDS_STRIDE + col] = a_ptr[(MB + row) * K + ks + col]

        # B tile from W[N,K] n-major [64,16], same pattern.
        @parameter
        for i in range(8):
            var row = i * 8 + tid // 16  # N-index 0..63
            var col = tid % 16
            B_lds[row * LDS_STRIDE + col] = w_ptr[(NB + row) * K + ks + col]

        barrier()  # workgroup: LDS visible across waves

        # ---- 5b. fragment + WMMA (per wave); A hoisted out of sc loop ----
        # G1b-0 A-map, row base = w*16 within the 64-row LDS tile.
        var a_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var a_base = (w * WMMA_TILE + l % WMMA_TILE) * LDS_STRIDE + (
            l // WMMA_TILE
        ) * WMMA_FRAG
        @parameter
        for j in range(WMMA_FRAG):
            a_bf16[j] = A_lds[a_base + j]
        var a_i16 = bitcast[DType.int16, WMMA_FRAG](a_bf16)

        @parameter
        for sc in range(V3A_SC):
            var b_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
            # G1b-0 B-map, col base = sc*16 within the 64-col LDS tile.
            var b_base = (sc * WMMA_TILE + l % WMMA_TILE) * LDS_STRIDE + (
                l // WMMA_TILE
            ) * WMMA_FRAG
            @parameter
            for j in range(WMMA_FRAG):
                b_bf16[j] = B_lds[b_base + j]
            var b_i16 = bitcast[DType.int16, WMMA_FRAG](b_bf16)
            acc[sc] = wmma_bf16(a_i16, b_i16, acc[sc])

        barrier()  # all LDS reads done before next strip overwrites
        ks += BK

    # ---- 6. store (+ optional fused residual RMW on c_ptr) ----
    @parameter
    for sc in range(V3A_SC):
        @parameter
        for j in range(WMMA_FRAG):
            var m = MB + w * WMMA_TILE + (l // WMMA_TILE) * WMMA_FRAG + j
            var n = NB + sc * WMMA_TILE + l % WMMA_TILE
            var idx = m * N + n
            var val = acc[sc][j]
            @parameter
            if fuse_residual:
                # residual stream is c_ptr itself (in-place); BF16→F32 then add.
                val = val + c_ptr[idx].cast[DType.float32]()
            c_ptr[idx] = val.cast[c_type]()

# ===----------------------------------------------------------------------=== #
# Residual-fused store (v2 fallback): residual += A @ W^T  (o_proj / down_proj)
# Same LDS v2 body; epilogue is F32(residual) + acc → BF16 (in-place RMW).
# ===----------------------------------------------------------------------=== #


def wmma_gemm_kernel_residual(
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    residual_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """v2 LDS GEMM with fused residual-add store: residual[m,n] += (A@W^T)[m,n].

    Read-modify-write on residual (BF16 → F32 add → BF16), matching the naive
    linear_add_residual epilogue semantics after the full K accumulation.
    """
    var l = Int(thread_idx.x)
    if l >= WMMA_LANES:
        return

    var n_tile = Int(block_idx.x)
    var m_tile = Int(block_idx.y)
    var MB = m_tile * WMMA_TILE
    var NB = n_tile * WMMA_TILE

    if MB >= M or NB >= N:
        return

    var row_or_col = l % WMMA_TILE
    var half = l // WMMA_TILE

    var A_lds = stack_allocation[
        WMMA_TILE * LDS_STRIDE, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()
    var B_lds = stack_allocation[
        WMMA_TILE * LDS_STRIDE, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()

    var acc = SIMD[DType.float32, WMMA_FRAG](0.0)

    var ks = 0
    while ks < K:
        @parameter
        for i in range(8):
            var row = i * 2 + half
            var col = row_or_col
            var lds_idx = row * LDS_STRIDE + col
            var g_row_a = MB + row
            if g_row_a < M:
                A_lds[lds_idx] = a_ptr[g_row_a * K + ks + col]
            else:
                A_lds[lds_idx] = Scalar[DType.bfloat16](0)
            B_lds[lds_idx] = w_ptr[(NB + row) * K + ks + col]

        barrier()

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

        barrier()
        ks += WMMA_TILE

    # Fused residual-add epilogue (same cast order as naive epilogue).
    @parameter
    for j in range(WMMA_FRAG):
        var m = MB + half * WMMA_FRAG + j
        var n = NB + row_or_col
        if m < M:
            var idx = m * N + n
            var old = residual_ptr[idx].cast[DType.float32]()
            residual_ptr[idx] = (old + acc[j]).cast[DType.bfloat16]()


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
    """Host launch: C[m,n] = A[m,k] @ B[n,k]^T.

    Prefers v3a when M,N % 64 == 0 and K % 16 == 0; else v2 (N,K % 16).
    """
    debug_assert(m > 0, "wmma_gemm_bf16: m must be > 0")
    debug_assert(n % WMMA_TILE == 0, "wmma_gemm_bf16: n must be divisible by 16")
    debug_assert(k % WMMA_TILE == 0, "wmma_gemm_bf16: k must be divisible by 16")

    if m % BM == 0 and n % BN == 0 and k % BK == 0:
        comptime kernel_v3 = wmma_gemm_kernel_v3a[c_type, False]
        ctx.enqueue_function[kernel_v3](
            a.ptr.as_immutable().as_unsafe_any_origin(),
            b.ptr.as_immutable().as_unsafe_any_origin(),
            c.ptr.as_unsafe_any_origin(),
            m,
            n,
            k,
            grid_dim=(n // BN, m // BM),
            block_dim=(V3A_THREADS,),
        )
        return

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


def wmma_gemm_bf16_v3a[
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
    """Force v3a plain path (for bit-identical tests). Asserts M,N%64 K%16."""
    debug_assert(m % BM == 0, "v3a: m must be divisible by 64")
    debug_assert(n % BN == 0, "v3a: n must be divisible by 64")
    debug_assert(k % BK == 0, "v3a: k must be divisible by 16")
    comptime kernel_v3 = wmma_gemm_kernel_v3a[c_type, False]
    ctx.enqueue_function[kernel_v3](
        a.ptr.as_immutable().as_unsafe_any_origin(),
        b.ptr.as_immutable().as_unsafe_any_origin(),
        c.ptr.as_unsafe_any_origin(),
        m,
        n,
        k,
        grid_dim=(n // BN, m // BM),
        block_dim=(V3A_THREADS,),
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


def wmma_gemm_bf16_residual(
    residual: TileTensor[mut=True, dtype = DType.bfloat16, ...],
    a: TileTensor[DType.bfloat16, ...],
    b: TileTensor[DType.bfloat16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Host launch: residual += A[m,k] @ B[n,k]^T with fused BF16 residual store.

    Prefers v3a fused path when M,N % 64; else v2 residual kernel.
    residual is both the addend and the destination (in-place RMW).
    """
    debug_assert(m > 0, "wmma_gemm_bf16_residual: m must be > 0")
    debug_assert(
        n % WMMA_TILE == 0, "wmma_gemm_bf16_residual: n must be divisible by 16"
    )
    debug_assert(
        k % WMMA_TILE == 0, "wmma_gemm_bf16_residual: k must be divisible by 16"
    )

    if m % BM == 0 and n % BN == 0 and k % BK == 0:
        comptime kernel_v3r = wmma_gemm_kernel_v3a[DType.bfloat16, True]
        ctx.enqueue_function[kernel_v3r](
            a.ptr.as_immutable().as_unsafe_any_origin(),
            b.ptr.as_immutable().as_unsafe_any_origin(),
            residual.ptr.as_unsafe_any_origin(),
            m,
            n,
            k,
            grid_dim=(n // BN, m // BM),
            block_dim=(V3A_THREADS,),
        )
        return

    var n_tiles = n // WMMA_TILE
    var m_tiles = ceildiv(m, WMMA_TILE)
    ctx.enqueue_function[wmma_gemm_kernel_residual](
        a.ptr.as_immutable().as_unsafe_any_origin(),
        b.ptr.as_immutable().as_unsafe_any_origin(),
        residual.ptr.as_unsafe_any_origin(),
        m,
        n,
        k,
        grid_dim=(n_tiles, m_tiles),
        block_dim=(WMMA_LANES,),
    )


def wmma_gemm_bf16_v3a_residual(
    residual: TileTensor[mut=True, dtype = DType.bfloat16, ...],
    a: TileTensor[DType.bfloat16, ...],
    b: TileTensor[DType.bfloat16, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Force v3a fused residual path (for bit-identical tests)."""
    debug_assert(m % BM == 0, "v3a residual: m must be divisible by 64")
    debug_assert(n % BN == 0, "v3a residual: n must be divisible by 64")
    debug_assert(k % BK == 0, "v3a residual: k must be divisible by 16")
    comptime kernel_v3r = wmma_gemm_kernel_v3a[DType.bfloat16, True]
    ctx.enqueue_function[kernel_v3r](
        a.ptr.as_immutable().as_unsafe_any_origin(),
        b.ptr.as_immutable().as_unsafe_any_origin(),
        residual.ptr.as_unsafe_any_origin(),
        m,
        n,
        k,
        grid_dim=(n // BN, m // BM),
        block_dim=(V3A_THREADS,),
    )
