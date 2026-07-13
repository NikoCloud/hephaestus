# Multi-tile K-looping BF16 WMMA GEMM (v1: no LDS).
# C[M, N] = A[M, K] @ W[N, K]^T
#
# Proven gfx12 lane mappings from experiments/exp4a_wmma_tile/ (G1b-0).
# Spec indices (tile bases MB=m_tile*16, NB=n_tile*16, K-strip ks, lane l, slot j):
#   grid: n_tile = blockIdx.x, m_tile = blockIdx.y
#   A-load: a[j] = A[(MB + l%16)*K + ks + (l/16)*8 + j]
#   B-load: b[j] = W[(NB + l%16)*K + ks + (l/16)*8 + j]   # W is [N,K] row-major
#   accum:  acc = wmma_bf16(bitcast a, bitcast b, acc)     # acc=0 before K-loop
#   D-store: C[(MB + (l/16)*8 + j)*N + NB + l%16] = acc[j].cast[BF16]
#
# Usage:
#   mojo build gemm.mojo -o /tmp/exp4b_gemm
#   /tmp/exp4b_gemm structured <M> <N> <K> <out_c.bf16>
#   /tmp/exp4b_gemm random     <M> <N> <K> <out_c.bf16> <a.npy> <w.npy>
#
# Precondition: M, N, K all divisible by 16. No remainder-tile handling in v1.

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.sys import argv
from std.sys.intrinsics import llvm_intrinsic
from std.memory import bitcast

comptime TILE = 16
comptime N_LANES = 32
comptime FRAG = 8


def wmma_bf16(
    a_i16: SIMD[DType.int16, FRAG],
    b_i16: SIMD[DType.int16, FRAG],
    c: SIMD[DType.float32, FRAG],
) -> SIMD[DType.float32, FRAG]:
    return llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.bf16",
        SIMD[DType.float32, FRAG],
        has_side_effect=False,
    ](a_i16, b_i16, c)


def gemm_kernel(
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """One 16×16 output tile per block; one wave32; K-loop over 16-wide strips.

    Direct global loads (no shared memory). v1 correctness only.
    """
    # Precondition guard (v1 has no remainder tiles).
    if M % TILE != 0 or N % TILE != 0 or K % TILE != 0:
        return

    var l = Int(thread_idx.x)
    if l >= N_LANES:
        return

    var n_tile = Int(block_idx.x)
    var m_tile = Int(block_idx.y)
    var MB = m_tile * TILE
    var NB = n_tile * TILE

    # Bounds: only full tiles are launched; still guard bad grids.
    if MB >= M or NB >= N:
        return

    var row_or_col = l % TILE  # A.m or W.n or C.n
    var half = l // TILE  # 0 or 1 — K-half within a 16-wide strip / M-half for store

    # F32 accumulator persists across K-strips.
    var acc = SIMD[DType.float32, FRAG](0.0)

    var ks = 0
    while ks < K:
        # A-load and W-load are structurally identical (MB vs NB only).
        # X[(base + l%16)*K + ks + (l/16)*8 + j]
        var a_bf16 = SIMD[DType.bfloat16, FRAG](0)
        var b_bf16 = SIMD[DType.bfloat16, FRAG](0)
        var a_row_base = (MB + row_or_col) * K + ks + half * FRAG
        var w_row_base = (NB + row_or_col) * K + ks + half * FRAG
        @parameter
        for j in range(FRAG):
            a_bf16[j] = a_ptr[a_row_base + j]
            b_bf16[j] = w_ptr[w_row_base + j]

        var a_i16 = bitcast[DType.int16, FRAG](a_bf16)
        var b_i16 = bitcast[DType.int16, FRAG](b_bf16)
        acc = wmma_bf16(a_i16, b_i16, acc)
        ks += TILE

    # D-store: column-distributed C/D fragment, BF16.
    @parameter
    for j in range(FRAG):
        var m = MB + half * FRAG + j
        var n = NB + row_or_col
        c_ptr[m * N + n] = acc[j].cast[DType.bfloat16]()


def load_npy_bf16(
    path: String,
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_elems: Int,
) raises:
    """Minimal .npy v1.0 reader for contiguous BF16 (descr '<V2' from ml_dtypes)."""
    var f = open(path, "r")
    var raw = f.read_bytes()
    f.close()
    if len(raw) < 10:
        raise Error("npy too short: " + path)
    # magic \x93NUMPY
    if (
        Int(raw[0]) != 0x93
        or Int(raw[1]) != ord("N")
        or Int(raw[2]) != ord("U")
        or Int(raw[3]) != ord("M")
        or Int(raw[4]) != ord("P")
        or Int(raw[5]) != ord("Y")
    ):
        raise Error("bad npy magic: " + path)
    var major = Int(raw[6])
    if major != 1:
        raise Error("only npy v1.0 supported: " + path)
    var header_len = Int(raw[8]) + Int(raw[9]) * 256
    var data_off = 10 + header_len
    var need = n_elems * 2
    if data_off + need > len(raw):
        raise Error(
            "npy data short: need "
            + String(need)
            + " bytes after header, file="
            + path
        )
    # Pair little-endian bytes into uint16 then bitcast to bf16.
    for i in range(n_elems):
        var b0 = Int(raw[data_off + i * 2])
        var b1 = Int(raw[data_off + i * 2 + 1])
        var u = UInt16(b0 + b1 * 256)
        dst[i] = bitcast[DType.bfloat16, 1](SIMD[DType.uint16, 1](u))[0]


def fill_structured(
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """A[m,k] = m*K+k, W[n,k] = n*K+k as BF16 (integer-valued, may round for large)."""
    for m in range(M):
        for k in range(K):
            var v = Float32(m * K + k)
            a_ptr[m * K + k] = v.cast[DType.bfloat16]()
    for n in range(N):
        for k in range(K):
            var v = Float32(n * K + k)
            w_ptr[n * K + k] = v.cast[DType.bfloat16]()


def write_bf16_raw(
    path: String,
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_elems: Int,
) raises:
    """Dump little-endian BF16 payload (no header) for the oracle."""
    # Copy into a List so write_bytes gets a stable origin (same pattern as exp4a).
    var host = List[Scalar[DType.bfloat16]]()
    for i in range(n_elems):
        host.append(src[i])
    var f = open(path, "w")
    f.write_bytes(
        Span[Byte, origin_of(host)](
            ptr=host.unsafe_ptr().bitcast[Byte](),
            length=n_elems * 2,
        )
    )
    f.close()


def main() raises:
    if len(argv()) < 6:
        print(
            "usage: gemm structured|random <M> <N> <K> <out_c.bf16> [a.npy w.npy]"
        )
        raise Error("bad args")

    var mode = String(argv()[1])
    var M = Int(String(argv()[2]))
    var N = Int(String(argv()[3]))
    var K = Int(String(argv()[4]))
    var out_path = String(argv()[5])

    # Host-side precondition (kernel also guards).
    if M % TILE != 0 or N % TILE != 0 or K % TILE != 0:
        raise Error(
            "M, N, K must be divisible by 16 (got "
            + String(M)
            + ","
            + String(N)
            + ","
            + String(K)
            + ")"
        )
    if M <= 0 or N <= 0 or K <= 0:
        raise Error("M, N, K must be positive")

    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_dev = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * N)

    if mode == "structured":
        with a_dev.map_to_host() as ha, w_dev.map_to_host() as hw:
            fill_structured(
                ha.unsafe_ptr().as_unsafe_any_origin(),
                hw.unsafe_ptr().as_unsafe_any_origin(),
                M,
                N,
                K,
            )
    elif mode == "random":
        if len(argv()) < 8:
            raise Error("random mode needs a.npy w.npy paths")
        var a_path = String(argv()[6])
        var w_path = String(argv()[7])
        with a_dev.map_to_host() as ha, w_dev.map_to_host() as hw:
            load_npy_bf16(
                a_path, ha.unsafe_ptr().as_unsafe_any_origin(), M * K
            )
            load_npy_bf16(
                w_path, hw.unsafe_ptr().as_unsafe_any_origin(), N * K
            )
    else:
        raise Error("mode must be structured or random")

    ctx.enqueue_memset(c_dev, 0)

    var n_tiles = N // TILE
    var m_tiles = M // TILE
    # n_tile = blockIdx.x, m_tile = blockIdx.y
    ctx.enqueue_function[gemm_kernel](
        a_dev.unsafe_ptr(),
        w_dev.unsafe_ptr(),
        c_dev.unsafe_ptr(),
        M,
        N,
        K,
        grid_dim=(n_tiles, m_tiles),
        block_dim=(N_LANES,),
    )
    ctx.synchronize()

    with c_dev.map_to_host() as hc:
        write_bf16_raw(
            out_path, hc.unsafe_ptr().as_unsafe_any_origin(), M * N
        )
        # Sample print for human eyes.
        print(
            "mode",
            mode,
            "M",
            M,
            "N",
            N,
            "K",
            K,
            "C[0,0]=",
            Float32(hc[0]),
            " C[0,1]=",
            Float32(hc[1]),
            " C[min(1,M-1),min(1,N-1)]=",
            Float32(hc[min(1, M - 1) * N + min(1, N - 1)]),
        )
    print("wrote", out_path)
