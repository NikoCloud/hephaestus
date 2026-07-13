# v3a prefill profiling — VGPR + GEMM/non-GEMM split (2026-07-13)

**Hardware:** GPU 0 Radeon AI PRO R9700 (gfx1201)  
**Env:** hephaestus-wmma-nightly Mojo 1.0.0b3.dev2026071206  
**Branch:** `v3a-profiling` (from `wmma-v3a`)  
**Method:**
- Category split: `src/qwen_profile_prefill.mojo` (synced host timers around each kernel group; 512-token prompt from `bench/ab_prompt_long_ids.txt`)
- VGPR/LDS: `mojo build --emit asm` on a v3a-only probe; AMDGCN metadata (`.vgpr_count`, `.group_segment_fixed_size`)
- `rocprof` **not installed** on this machine (only `rocprofiler-register` helper lib)

Production unsynced reference (same binary family): **~794 tok/s** → wall prefill ≈ **512/794 ≈ 645 ms** (matches prior v3a bench).

Synced profiler wall: **636 ms** (804 tok/s equivalent) — close to production; sync bias small for a single full prefill.

---

## 1. VGPR / LDS / occupancy (v3a kernel)

From AMDGCN metadata on kernels with **LDS=4096** (v3a 64×64 tile):

| Metric | v3a plain | v3a fused residual | v2 (ref, LDS=1024) |
|--------|-----------|--------------------|--------------------|
| **VGPR/lane** | **92** | **92** | 69 |
| **SGPR** | 18–20 | 18–20 | 28 |
| **LDS / workgroup** | **4096 B** | **4096 B** | 1024 B |
| **VGPR spill** | **0** | **0** | 0 |
| **SGPR spill** | **0** | **0** | 0 |
| private/scratch | 0 | 0 | 0 |

**vs design estimate (~55 VGPR):** actual **92** — higher (compiler temps + 4×acc + addressing), still **no spills**.

**Occupancy (theoretical, VGPR-limited):**
- RDNA WGP: 256 VGPRs/SIMD, 4 SIMDs/WGP (standard RDNA3/4 model)
- Alloc grain ~8: 92 → 96 VGPRs → **⌊256/96⌋ = 2 waves/SIMD**
- → **8 waves/WGP** max from VGPR
- v3a workgroup = **4 waves** → **~2 concurrent workgroups/WGP** (VGPR-bound)
- LDS: 4 KB / 64 KB → LDS allows more WGs than VGPR; **VGPR is the limiter**
- Design headroom for v3b (~8 acc ≈ more VGPRs): 92 is already tight vs 55 estimate; v3b must watch occupancy carefully

**Register spilling:** **no** (`.vgpr_spill_count: 0`, `.sgpr_spill_count: 0`)

---

## 2. Prefill 512 — category breakdown (synced)

Full prefill, 36 layers, seq=512. Times are **total prefill**, not per-layer.

| Category | ms (total prefill) | % of total |
|----------|-------------------:|----------:|
| **WMMA GEMM (all projections)** | **209.0** | **32.8%** |
| — q_proj | 13.6 | 2.1% |
| — k_proj | 12.6 | 2.0% |
| — v_proj | 12.5 | 2.0% |
| — o_proj+residual | 19.9 | 3.1% |
| — gate_proj | 34.4 | 5.4% |
| — up_proj | 34.4 | 5.4% |
| — down_proj+residual | 44.0 | 6.9% |
| — lm_head | 37.5 | 5.9% |
| **Attention** | **412.6** | **64.9%** |
| RMSNorm (attn+qk+ffn+out) | 4.4 | 0.7% |
| RoPE | 7.3 | 1.2% |
| silu_mul | 2.9 | 0.5% |
| embed | 0.05 | ~0% |
| **Total (category sum)** | **636.2** | **100%** |
| Barriers (in-kernel LDS) | *inside GEMM* | n/a (not split) |

### Rollup for the decision

| Bucket | ms | % |
|--------|---:|--:|
| **GEMM (WMMA)** | 209 | **33%** |
| **non-GEMM** | 427 | **67%** |
| of which **attention alone** | 413 | **65%** |

**Key answer:** non-GEMM is **~67%** of synced prefill time — far above the 20% threshold. Attention dominates. Wider GEMM tiles / double-buffering can at best move the **~33%** GEMM slice (Amdahl: even infinite GEMM speedup → ~1.5× end-to-end). To approach G1b-3 (~2100 tok/s ≈ 4× from 794), **attention (and not just GEMM) must improve**.

---

## 3. Implications for next lever

| Lever | Hits which slice | Expected ceiling impact |
|-------|------------------|-------------------------|
| v3b wider tiles (128×256) | GEMM 33% | Cap ~1.3–1.5× e2e if GEMM goes free |
| Double-buffering | GEMM (hide LDS barrier) | Marginal on GEMM only |
| **Attention optimization** | **65%** | **Primary path to multi-× e2e** |

Prefill production ~794 tok/s; attention is the hole, not down_proj (down_proj is only 6.9% and already on WMMA residual).

---

## 4. How to reproduce

```bash
export HIP_VISIBLE_DEVICES=0
NIGHTLY=~/projects/hephaestus-wmma-nightly
KERNELS=~/projects/modular/max/kernels/src
REPO=~/projects/hephaestus

# Category profile
(cd "$NIGHTLY" && pixi run mojo build -I "$KERNELS" -I "$REPO/src" \
  "$REPO/src/qwen_profile_prefill.mojo" -o /tmp/qwen_profile_prefill)
/tmp/qwen_profile_prefill "$REPO/bench/ab_prompt_long_ids.txt"

# VGPR metadata (no rocprof required)
(cd "$NIGHTLY" && pixi run mojo build -I "$KERNELS" -I "$REPO/src" --emit asm \
  /tmp/v3a_asm_probe.mojo -o /tmp/v3a_asm/probe)
# inspect *.amdgcn for .vgpr_count / .group_segment_fixed_size
```

Raw log: `/tmp/v3a_prefill_profile.log` (local run).
