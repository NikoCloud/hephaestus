# G1b-0 single-tile 16×16×16 BF16 WMMA microkernel.
# Spec: .agent/specs/2026-07-13_g1b0-wmma-tile-spec.md
#
# Usage:
#   mojo build microkernel.mojo -o /tmp/exp4a_mk
#   /tmp/exp4a_mk T1|T2|T3 <out_d.f32>
#
# Fills A,B on host; one wave32 runs WMMA; dumps D (16×16 f32 row-major).

from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.sys import argv
from std.sys.intrinsics import llvm_intrinsic
from std.memory import bitcast

comptime TILE = 16
comptime N_LANES = 32
comptime FRAG = 8


def fill_ab(
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    test_id: Int,
):
    """Host-side fill of 16×16 A,B in row-major. test_id: 1=T1, 2=T2, 3=T3."""
    for m in range(TILE):
        for k in range(TILE):
            var av = Float32(0)
            if test_id == 1 or test_id == 2:
                av = Float32(m)
            elif test_id == 3:
                # identity
                if m == k:
                    av = Float32(1)
                else:
                    av = Float32(0)
            a_ptr[m * TILE + k] = av.cast[DType.bfloat16]()

    for k in range(TILE):
        for n in range(TILE):
            var bv = Float32(0)
            if test_id == 1 or test_id == 3:
                bv = Float32(n)
            elif test_id == 2:
                if k == n:
                    bv = Float32(1)
                else:
                    bv = Float32(0)
            b_ptr[k * TILE + n] = bv.cast[DType.bfloat16]()


def wmma_tile_kernel(
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    d_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """One lane of wave32: load A/B fragments, WMMA, store D fragment.

    Mappings (spec §2):
      A: m=l%16, k_half=l/16, a[j]=A[m, k_half*8+j]
      B: n=l%16, k_half=l/16, b[j]=B[k_half*8+j, n]
      D: n=l%16, m=(l/16)*8+j, D[m,n]=r[j]
    """
    var l = Int(thread_idx.x)
    if l >= N_LANES:
        return

    var row_or_col = l % TILE          # A.m or B.n or D.n
    var half = l // TILE               # 0 or 1

    # --- load A fragment (M×K, K-major) ---
    var a_bf16 = SIMD[DType.bfloat16, FRAG](0)
    var a_base = row_or_col * TILE + half * FRAG
    @parameter
    for j in range(FRAG):
        a_bf16[j] = a_ptr[a_base + j]

    # --- load B fragment (K×N, K-major) ---
    var b_bf16 = SIMD[DType.bfloat16, FRAG](0)
    @parameter
    for j in range(FRAG):
        var k = half * FRAG + j
        b_bf16[j] = b_ptr[k * TILE + row_or_col]

    var c = SIMD[DType.float32, FRAG](0.0)
    var a_i16 = bitcast[DType.int16, FRAG](a_bf16)
    var b_i16 = bitcast[DType.int16, FRAG](b_bf16)
    var r = llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.bf16",
        SIMD[DType.float32, FRAG],
        has_side_effect=False,
    ](a_i16, b_i16, c)

    # --- store D fragment (M×N, column-distributed) ---
    @parameter
    for j in range(FRAG):
        var m = half * FRAG + j
        var n = row_or_col
        d_ptr[m * TILE + n] = r[j]


def main() raises:
    if len(argv()) < 3:
        print("usage: exp4a_mk T1|T2|T3 <out_d.f32>")
        raise Error("bad args")

    var name = String(argv()[1])
    var out_path = String(argv()[2])
    var test_id: Int
    if name == "T1":
        test_id = 1
    elif name == "T2":
        test_id = 2
    elif name == "T3":
        test_id = 3
    else:
        raise Error("test must be T1, T2, or T3")

    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](TILE * TILE)
    var b_dev = ctx.enqueue_create_buffer[DType.bfloat16](TILE * TILE)
    var d_dev = ctx.enqueue_create_buffer[DType.float32](TILE * TILE)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        fill_ab(
            ha.unsafe_ptr().as_unsafe_any_origin(),
            hb.unsafe_ptr().as_unsafe_any_origin(),
            test_id,
        )

    ctx.enqueue_memset(d_dev, 0)
    ctx.enqueue_function[wmma_tile_kernel](
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        d_dev.unsafe_ptr(),
        grid_dim=(1,),
        block_dim=(N_LANES,),
    )
    ctx.synchronize()

    var host_d = List[Float32]()
    with d_dev.map_to_host() as hd:
        for i in range(TILE * TILE):
            host_d.append(hd[i])
            if i < 8 or (i % 16 == 0 and i < 64):
                pass

    # dump little-endian f32
    var f = open(out_path, "w")
    f.write_bytes(
        Span[Byte, origin_of(host_d)](
            ptr=host_d.unsafe_ptr().bitcast[Byte](),
            length=TILE * TILE * 4,
        )
    )
    f.close()

    # sample print for human eyes
    print("test", name, "D samples: D[0,0]=", host_d[0],
          " D[1,1]=", host_d[1 * TILE + 1],
          " D[2,3]=", host_d[2 * TILE + 3],
          " D[15,15]=", host_d[15 * TILE + 15])
    print("wrote", out_path)
