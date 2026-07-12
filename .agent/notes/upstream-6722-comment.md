# Draft comment for modular/modular#6722 (awaiting Niko's go-ahead to post)

Target: https://github.com/modular/modular/issues/6722
Status of issue: OPEN, "Needs Triage", opened 2026-06-26, 1 comment.
Covers f16/bf16 instruction-selection failure on gfx1201.
Does NOT mention FP8 anywhere — that is what this comment adds.

---

Another gfx1201 (R9700, RDNA4) data point on `1.0.0b3.dev2026071006`, plus one
finding I don't think is captured yet: **the FP8 WMMA path fails too, with a
different error** — so fixing f16/bf16 instruction selection won't fix FP8.

### 1. FP8 E4M3 fails with a distinct error (not "cannot select")

```mojo
from std.gpu.host import DeviceContext
from std.gpu.compute.mma import mma

def fp8_wmma_probe(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    var a = SIMD[DType.float8_e4m3fn, 8](1.0)
    var b = SIMD[DType.float8_e4m3fn, 8](1.0)
    var c = SIMD[DType.float32, 8](0.0)
    var d = SIMD[DType.float32, 8](0.0)
    mma(d, a, b, c)
    out_ptr[0] = d[0]

def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_memset(buf, 0)
    ctx.enqueue_function[fp8_wmma_probe](buf.unsafe_ptr(), grid_dim=(1,), block_dim=(32,))
    ctx.synchronize()
    with buf.map_to_host() as h:
        print("fp8 wmma d[0] =", h[0])
```

```
LLVM ERROR: Do not know how to split this operator's operand!
Running pass 'AMDGPU DAG->DAG Pattern Instruction Selection'
```

Same error with 16-element A/B fragments. This is an operand *legalization*
failure, not a selection failure — a separate bug from the f16/bf16 one in the
issue title.

Notable because the FP8 branch in `mma_amd_rdna.mojo` is already explicitly
gated on `_is_amd_rdna4()` and returns `llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8`
— so RDNA4 FP8 scaffolding exists, but the operand packing appears wrong.
There's also an internal inconsistency worth a look: the docstring at
`mma_amd_rdna.mojo:266` describes RDNA4 FP8 as
`llvm.amdgcn.wmma.f32.16x16x32.fp8` (K=32), while `get_intrinsic_name()`
returns the K=16 form.

### 2. Root cause looks like a hard-coded RDNA3 fragment encoding

`mma_amd_rdna.mojo:370` gates the f16/bf16 path on
`_has_shape[(16, 16, 8, 8)]` — 16-element A/B fragments, which is the RDNA3
(gfx11) encoding. gfx12 WMMA takes 8-element A/B fragments.

Passing RDNA4-shaped 8-element fragments doesn't reach LLVM at all; it fails at
comptime:

```
constraint failed: no valid implementation of mma for a=8xbfloat16,
b=8xbfloat16, c=8xfloat32, and d=8xfloat32
```

So on gfx1201 both encodings are dead ends: 16-element → LLVM can't select;
8-element → stdlib has no implementation.

The same 16-element assumption is baked into the kernel library:
`max/kernels/src/nn/attention/gpu/amd_rdna/buffers.mojo:39` sets
`RDNA_AB_FRAG_SIZE = 16`, with a docstring stating "RDNA WMMA always uses
16-element A/B fragments" — true for RDNA3, not for RDNA4.

### 3. Downstream blast radius on gfx1201

- **`linalg.matmul` is uncompilable for bf16 regardless of shape.** The
  dispatcher instantiates `gemm_kernel_rdna` at comptime, so even an `M=1`
  call that would route to `gemv_gpu` at runtime fails to compile. (Matches
  the segfault-during-compilation report in the existing comment.)
- **RDNA attention is blocked by the same root cause.**
  `nn/attention/gpu/mha.mojo:2215` does `comptime if _is_amd_rdna():` →
  `AttentionRDNA` → `rdna_mma` → the same 16-element `mma()` call. There is no
  RDNA3/RDNA4 distinction at that branch.
- Non-WMMA paths are fine: `gemv_gpu` and `matmul_kernel_naive` both compile
  and produce correct results on gfx1201 (verified against a CPU fp32
  reference).

Happy to test a patch on this hardware (R9700 + RX 9070 XT, both gfx1201).
