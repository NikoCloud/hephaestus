# Experiment 3g — direct `llvm_intrinsic` WMMA on gfx1201 (go/no-go)

**Date:** 2026-07-12  
**Env:** isolated `~/projects/hephaestus-wmma-nightly` — Mojo `1.0.0b3.dev2026071206`  
**Hardware:** GPU 0 = R9700 gfx1201, wave32

## i1 operand semantics (from `IntrinsicsAMDGPU.td`)

Source: `/opt/rocm/lib/llvm/include/llvm/IR/IntrinsicsAMDGPU.td` (also `/usr/include/llvm/IR/...`).

| Class / intrinsic | i1 operands | Meaning |
|---|---|---|
| **`AMDGPUWmmaIntrinsic`** (used by **`wmma.f32.16x16x16.fp8.fp8`** and **`wmma.f32.16x16x16.bf16`**) | **None** | Args are only `A`, `B`, `C`. `D = A*B+C`. |
| `AMDGPUWmmaIntrinsicOPSEL` (GFX11 f16/bf16 half-register forms) | `%high` (op_sel) | Selects which 16-bit half of the registers is read/written. **On GFX12 this must be 0** (comment in `.td`). Not present on the f32_fp8 / f32_bf16 forms used below. |
| `AMDGPUWmmaIntrinsicIU` (integer iu8/iu4) | `%A_sign`, `%B_sign`, `%clamp` | Sign interpretation of integer A/B and optional clamp. **Not used for FP8/BF16 f32-accum WMMA.** |
| `AMDGPUWmmaIntrinsicModsC` (later gfx1250-scale forms) | matrix reuse / mod bits | Unrelated to 16x16x16 fp8/bf16 on gfx1201. |

**Comment in `.td` for GFX12 FP8 packing:**
> A and B are `<8 x fp8>` or `<8 x bf8>`, but since fp8 and bf8 are not supported by llvm we use `<2 x i32>`.

**Clang builtins (wave32, gfx12):**

| Builtin | Signature string | Meaning |
|---|---|---|
| `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | `"V8fV2iV2iV8f"` | `v8f32 = (v2i32, v2i32, v8f32)` — **no i1** |
| `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` | `"V8fV8sV8sV8f"` | `v8f32 = (v8i16, v8i16, v8f32)` — **no i1** |

CK tile matches this (int32x2_t A/B, fp32x8_t C; no sign/clamp args).

**Bottom line:** for the 16×16×16 FP8 and BF16 f32-accum intrinsics on gfx12 wave32, there are **no i1 operands**. i1s only appear on OPSEL (GFX11 half-select) and IU (integer signedness/clamp) families.

## Corrected Mojo call shapes

| Path | Intrinsic | A | B | C/D |
|---|---|---|---|---|
| FP8 E4M3 | `llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8` | `SIMD[int32, 2]` (8×fp8 packed) | same | `SIMD[float32, 8]` |
| BF16 | `llvm.amdgcn.wmma.f32.16x16x16.bf16` | `SIMD[int16, 8]` (packed bf16) | same | `SIMD[float32, 8]` |

All-ones packing: FP8 `1.0` → byte `0x38` → each i32 `0x38383838`; BF16 `1.0` → `0x3f80`. Expect **16.0** per C/D element (K=16 ones products).

## Results (measured)

| Probe | Build | Run | d[0..7] |
|---|---|---|---|
| `exp3g_fp8_wmma_gfx12.mojo` | **OK** | **PASS** | **all 16.0** |
| `exp3g_bf16_wmma_gfx12.mojo` | **OK** | **PASS** | **all 16.0** |

```
EXP3g PASS: FP8 E4M3 WMMA all-ones -> 16.0 on gfx1201
EXP3g PASS: BF16 WMMA all-ones -> 16.0 on gfx1201
```

Contrast: Mojo `std.gpu.compute.mma` still fails on this nightly (RDNA3 16-elem BF16 / broken FP8 path). **Direct intrinsic + correct fragment widths is the local-shim approach.**

## Go / no-go

| Question | Answer |
|---|---|
| Does FP8 WMMA hardware path work via direct LLVM intrinsic? | **YES — 16.0** |
| Does BF16 WMMA work the same way (8-elem A/B)? | **YES — 16.0** |
| Phase 1b thesis (native FP8 WMMA on RDNA4) de-risked? | **YES** — proceed with local shim + 8-element fragment loaders |
| Mojo stdlib `mma()` fixed? | **No** — still broken; bypass it |

## Reference clones (for fragment loaders next)

| Repo | Path |
|---|---|
| [JohnTDI-cpu/rdna4-wmma-guide](https://github.com/JohnTDI-cpu/rdna4-wmma-guide) | `~/projects/ref/rdna4-wmma-guide` (lane mapping guide) |
| [tlee933/llama.cpp-rdna4-gfx1201](https://github.com/tlee933/llama.cpp-rdna4-gfx1201) | `~/projects/ref/llama.cpp-rdna4-gfx1201` |

Engine code **not** modified. Probes only under `experiments/`.
