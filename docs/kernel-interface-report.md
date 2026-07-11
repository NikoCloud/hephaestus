# Kernel Interface Report — Vendorable MAX Kernels for RDNA4
## Date: 2026-07-11
## Source: modular/max `kernels/src/` (cloned at `~/projects/modular/max/`)

This document maps the entry-point signatures, expected dtypes, and memory layouts for each kernel Hephaestus Phase 1a will vendor. This is the vendoring contract the loader must satisfy.

---

## 1. MatMul — `linalg/matmul/gpu/amd_rdna/matmul.mojo`

### Entry Point

```mojo
def gemm_kernel_rdna[
    c_type: DType,        // Output dtype (e.g., bfloat16, float16)
    a_type: DType,        // Input A dtype (e.g., bfloat16)
    b_type: DType,        // Input B dtype (e.g., bfloat16)
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[...] = None,
    s_type: DType = get_accum_type[c_type](),  // Accumulator dtype (float32 for bf16)
    BLOCK_K: Int = 16,
    BLOCK_M: Int = 128,
    BLOCK_N: Int = 128,
    WARPS_M: Int = 8,
    WARPS_N: Int = 2,
    WARP_TILE_M: Int = 1,
    WARP_TILE_N: Int = 4,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],      // Output [M, N]
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],    // Input A [M, K]
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],     // Input B [N, K] if transpose_b, else [K, N]
    m: Int, n: Int, k: Int,                              // Dimensions
)
```

### Key Details

| Aspect | Value |
|---|---|
| WMMA shape | 16×16×16 (hard-coded RDNA constant) |
| Wave size | 32 (RDNA Wave32) |
| Supported dtypes | `bfloat16`, `float16` on RDNA3+ (gfx11xx/gfx12xx) |
| Accumulator | `float32` (via `get_accum_type`) |
| Fallback path | Naive per-thread matmul for RDNA1/2 (gfx10xx) — not our concern |
| Memory layout | `TileTensor` (MAX's tiled tensor abstraction) |
| Double buffering | Yes — compute-before-prefetch ordering |
| Block swizzle | Yes — L2 locality optimization |
| Default tile | 128×128 output, 16 warps (8×2), warp_tile 1×4, BLOCK_K=16 |
| Shared memory | Double-buffered, padded stride (BLOCK_K + 8) for bank conflict avoidance |
| Vectorization | 128-bit loads (8 elements for bf16) on coalesced paths |

### Dispatch Logic
- If `_is_amd_rdna2_or_earlier()` OR dtype not in (float16, bfloat16): naive path
- Otherwise: WMMA path (our case: gfx1201 + bfloat16 → WMMA)

### What the Loader Must Provide
- Weight tensors as `TileTensor[BF16, ..., ImmutAnyOrigin]` — row-major 2D tiles
- Activation tensors similarly tiled
- Dimensions m, n, k as runtime integers
- K must be divisible by BLOCK_K=16 (dispatch guarantee)

---

## 2. MHA Prefill — `nn/attention/gpu/amd_rdna/mha_prefill.mojo`

### Entry Point

```mojo
__extension AttentionRDNA:
    def mha_prefill(mut self)
```

Called on an `AttentionRDNA` struct instance. The struct is constructed via:

```mojo
AttentionRDNA(
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    q: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: MHAOperand,           // Paged KV cache or direct tensor
    v: MHAOperand,
    mask: MHAMask,           // Causal mask
    sink_weights: OptionalReg[...],
    batch_idx: Int,
    scale: Float32,          // 1/sqrt(head_dim)
    seq_len: Int,
    num_keys: Int,
    start_pos: Int,
    cache_start_pos: Int = 0,
)
```

### Key Details

| Aspect | Value |
|---|---|
| MMA shape | 16×16×16 (RDNA_WMMA, same as matmul) |
| BK (K strip) | 32 (hard-coded, `assert Self.BK == 32`) |
| BN (KV tile) | Configurable, from `MHAConfig` |
| Softmax | Online softmax (streaming, no full materialization) |
| V prefetch | DMA'd during second-to-last K strip compute |
| Mask | Causal, applied per-tile with OOB clamping |
| exp2 | Used (`use_exp2 = True`) — log2e scaling for hardware exp |
| K/V layout | `MHAOperand` — paged KV cache or direct tensor |
| Output | Written to `output_ptr` with head-major decode layout |

### QK/PV Loop Structure
1. **QK loop:** K loaded strip-by-strip (BK=32), QK MMA per strip, mask applied, online softmax
2. **PV loop:** P (post-softmax) cast to SMEM, V loaded from LDS, PV MMA per strip
3. V is prefetched during last QK strip to overlap compute

---

## 3. MHA Decode — `nn/attention/gpu/amd_rdna/mha_decode.mojo`

### Entry Point

```mojo
__extension AttentionRDNA:
    def mha_decode(
        mut self,
        exp_sum_ptr: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
        qk_max_ptr: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
        num_partitions: Int,
    )
```

### Key Details

Same struct constructor as prefill. Decode adds:

| Aspect | Value |
|---|---|
| Split-K | Yes — KV span partitioned across blocks for grid-level parallelism |
| Output depth | Must equal depth (`assert output_depth == depth`) — no MLA |
| Partition reduce | exp_sum and qk_max written per-partition for cross-block reduction |
| Empty partitions | Handled — reset to rowsum=0/rowmax=-inf |
| Q rows | Single token (seq_len=1 effectively), GQA group iterated over |
| P SMEM | BM×BN (larger than prefill's BM×BK) |

### What the Loader Must Provide
- Q tensor: `[1, num_heads, head_dim]` (single token decode)
- K/V cache: `MHAOperand` (paged, supports `block_paged_tile`)
- Scale: `1/sqrt(head_dim)` as Float32
- exp_sum and qk_max scratch buffers for partition reduction

---

## 4. Attention Config — `nn/attention/gpu/amd_rdna/config.mojo`

```mojo
comptime RDNA_MMA_M = 16
comptime RDNA_MMA_N = 16
comptime RDNA_MMA_K = 16

struct MHAAttentionConfigRDNA[token_gen: Bool, config: MHAConfig, group: Int]:
    comptime shared_kv = False
    comptime full_kv = False
    comptime depth_padded = True      // V SMEM is depth-padded
    comptime double_buffer = False    // Single-buffered (unlike matmul)
    comptime double_buffer_k_only = False
```

Key: RDNA attention is **single-buffered** (no double-buffer in attention, unlike matmul). V SMEM is depth-padded.

---

## 5. MMA Helper — `nn/attention/gpu/amd_rdna/mma.mojo`

```mojo
def rdna_mma(
    a_reg: TileTensor[..., LOCAL],       // A operand, 16-element fragments
    b_reg: TileTensor[..., LOCAL],       // B operand, 16-element fragments
    c_reg: TileTensor[mut, ..., LOCAL], // Accumulator, 8-element fragments (in-place)
)
```

Wraps `std.gpu.compute.mma.mma` intrinsic. RDNA WMMA:
- A/B fragments: 16 elements per lane
- C/D fragments: 8 elements per lane
- 16×16×16 shape, group_size=1
- Accumulator indexing: col-major over (M, N): `c_idx = m + n * num_m`

---

## 6. RMSNorm — `nn/normalization.mojo`

Read from the module header (truncated, but the key interface is):

```mojo
// Standard RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
// Supports GPU parallelization, FP8 fused variant available
// via rms_norm_fused_fp8 import
```

The file imports `comm.rms_norm_fp8` — indicating a fused FP8 RMSNorm exists (relevant for Phase 1b). For Phase 1a, the BF16 path is standard.

### What the Loader Must Provide
- Input tensor: `[seq_len, hidden_size]` BF16
- Weight: `[hidden_size]` BF16 (gamma)
- Eps: `1e-6` (compile-time constant)
- Output: same shape, BF16

---

## 7. RoPE — `nn/rope.mojo`

### Entry Point

```mojo
def apply_rope[
    dtype: DType,
    freq_dtype: DType,
    rank: Int,
    width: SIMDSize,
    //,
    *,
    interleaved: Bool,        // Whether real/imag are interleaved or split
    alignment: Int,
    output_fn: def[...](idx, val) -> None,
](
    x: TileTensor[dtype, ...],
    idx: IndexList[rank],
    ...
)
```

### Key Details

| Aspect | Value |
|---|---|
| Implementation | Complex multiplication: `(x_re + i*x_im) * (f_re + i*f_im)` |
| Frequencies | Pre-computed, passed as SIMD vector |
| Layout awareness | Handles both GGUF (interleaved) and safetensors (split) layouts |
| safetensors indices | `get_safetensors_idx(head_dim_idx, head_size)` returns `(idx//2, idx//2 + head_size//2)` |
| Identity rope | `get_identity_rope_coeff()` — returns (1, 0) for no-op |

**Critical for Hephaestus:** Since we load from safetensors (not GGUF), we use the split layout: real parts in first half of head_dim, imaginary in second half. The `get_safetensors_idx` function provides the correct indexing.

---

## 8. Softmax — `nn/softmax.mojo` + `nn/attention/gpu/amd_rdna/softmax.mojo`

Two softmax implementations:
1. **Generic** (`nn/softmax.mojo`): Standard softmax for standalone use
2. **RDNA-specific** (`nn/attention/gpu/amd_rdna/softmax.mojo`): Online softmax fused into the attention kernel (max → exp → sum → correction → output update). Uses exp2 with log2e scaling for hardware efficiency.

For Phase 1a, the RDNA online softmax is used inside attention (no standalone softmax needed except for final sampling).

---

## 9. Sampling — `nn/sampling.mojo`

Available for greedy sampling (argmax). The `nn/argmaxmin.mojo` and `nn/argmaxmin_gpu.mojo` files provide GPU argmax.

### What the Loader Must Provide
- Logits tensor: `[vocab_size]` (single token) or `[batch, vocab_size]`
- For greedy: just argmax → token ID
- No top-k, top-p, or temperature needed for Phase 1a

---

## 10. KV Cache — `nn/kv_cache.mojo` + `nn/kv_cache_ragged.mojo`

KV cache management for attention. Supports:
- Paged allocation (block-based)
- Ragged sequences (variable length batching)
- `MHAOperand` interface for attention kernels

For Phase 1a single-stream: a simple contiguous KV cache is sufficient. The `MHAOperand` interface with `block_paged_tile` must be satisfied, but a trivial wrapper around a contiguous buffer should work.

---

## 11. Vendoring Contract Summary

The safetensors loader must produce tensors that satisfy these interfaces:

| Component | Input Format | Notes |
|---|---|---|
| MatMul | `TileTensor[BF16, RowMajor, ImmutAnyOrigin]` | 2D, K divisible by 16 |
| MHA Prefill | `UnsafePointer` for Q, `MHAOperand` for K/V | Q is `[seq, heads, head_dim]` |
| MHA Decode | Same, with `MHAOperand` paged | Q is `[1, heads, head_dim]` |
| RMSNorm | `TileTensor[BF16]` + weight `[hidden]` | Eps = 1e-6 |
| RoPE | `TileTensor[BF16]` with split real/imag | safetensors layout |
| Sampling | Logits `[vocab_size]` → argmax | Greedy only |

**Key dependency:** `TileTensor` and `TensorLayout` from MAX's `layout` module. The loader must produce `TileTensor`-compatible memory, not raw pointers. This means importing MAX's layout system is a prerequisite for vendoring.
