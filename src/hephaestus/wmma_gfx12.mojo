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
# v1: direct global loads, no LDS. Requires N%16==0, K%16==0; M is padded
# via edge-masking (ceildiv grid + zero A rows / skip C stores past M).
# ===----------------------------------------------------------------------=== #

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.sys.intrinsics import llvm_intrinsic

from layout import TileTensor

comptime WMMA_TILE = 16
comptime WMMA_LANES = 32
comptime WMMA_FRAG = 8


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
    """One 16×16 output tile per block; wave32; K-loop over 16-wide strips.

    M may be any positive size: partial last M-tile zero-fills A rows and
    skips C stores with m >= M. N and K must be multiples of 16 (caller asserts).
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

    var acc = SIMD[DType.float32, WMMA_FRAG](0.0)

    var ks = 0
    while ks < K:
        var a_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var b_bf16 = SIMD[DType.bfloat16, WMMA_FRAG](0)
        var a_row = MB + row_or_col
        # Edge mask: past-M A rows contribute zeros (padding).
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
    """Host launch: C[m,n] = A[m,k] @ B[n,k]^T via gfx12 BF16 WMMA.

    Preconditions (asserted): n % 16 == 0, k % 16 == 0, m > 0.
    m need not be a multiple of 16 (edge tiles are masked).
    """
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
