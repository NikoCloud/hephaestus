# WMMA nightly check (Tier 1) — 2026-07-12

## Question

Does a newer Mojo nightly than **1.0.0b3.dev2026071006** fix gfx1201 WMMA so BF16 / FP8 `mma` compile and run?  
Bug shape: stdlib emits RDNA3-style 16-element BF16 fragments / `llvm.amdgcn.wmma.f32.16x16x16.bf16`, which LLVM cannot select for gfx12 (needs 8-element fragments for BF16).

## Toolchain under test

| | baseline (working Hephaestus env) | probe env (isolated) |
|---|---|---|
| Mojo | `1.0.0b3.dev2026071006` (d0addce7) | **`1.0.0b3.dev2026071206` (83d74d94)** |
| max / max-core | (project lock) | **`26.5.0.dev2026071206`** |
| Install | `projects/hephaestus/.pixi` (untouched) | `projects/hephaestus-wmma-nightly/.pixi` (throwaway) |

Newest linux-64 nightly available on `https://conda.modular.com/max-nightly` at probe time was **dev2026071206** (two calendar days after baseline).

## Results

| Probe | What | Result on dev2026071206 |
|---|---|---|
| `exp3c_wmma_probe.mojo` | BF16 `mma` with **16-elem** A/B fragments | **BUILD FAIL** |
| `exp3f_fp8_wmma_probe.mojo` | FP8 E4M3 `mma` (8-elem) | **BUILD FAIL** |
| `exp3e_wmma_free_paths.mojo` | gemv + naive matmul (no WMMA) | **PASS** (after origin cast only — see below) |
| extra: BF16 `mma` with **8-elem** fragments | RDNA4-shaped sizes | **BUILD FAIL** (type/constraint) |

### Exact errors

**BF16 WMMA (16-elem, exp3c) — unchanged from baseline:**
```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.bf16
```
Compile aborts in AMDGPU DAG→DAG isel. Same intrinsic name as on dev2026071006.

**FP8 WMMA (exp3f) — still broken (error text differs / still unusable):**
```
LLVM ERROR: Do not know how to split this operator's operand!
```
(LLVM isel crash during `AMDGPU DAG->DAG Pattern Instruction Selection` on the FP8 probe function.)

**BF16 8-elem fragments (extra probe):**
```
constraint failed: no valid implementation of mma for a=8xbfloat16, b=8xbfloat16, c=8xfloat32, and d=8xfloat32
```
(stdlib still has no BF16×BF16 mma overload for 8-wide fragments; only the RDNA3 16-wide path is offered, and that path does not lower.)

**WMMA-free paths (exp3e, adapted copy for new origin rules):**
```
decode M=1  gemv_gpu -> errors: 0 / 4096  max_diff: 0.0312109
prefill M=4 naive -> errors: 0 / 16384  max_diff: 0.031246185
EXP3e PASS: both WMMA-free paths compile and are correct on gfx1201
```
Note: stock `exp3e` fails **typecheck only** on the new nightly (`MutUntrackedOrigin` vs `MutAnyOrigin`); that is unrelated to WMMA. Adapted file: `exp3e_wmma_free_paths_20260712.mojo`.

## Verdict (Tier 1)

| Question | Answer |
|---|---|
| Does BF16 WMMA compile on newest nightly? | **No** |
| Does FP8 WMMA compile on newest nightly? | **No** |
| Is the gfx1201 WMMA blocker retired by Tier 1? | **No** |

**Tier 1 is exhausted for this date.** Proceed to Tier 2 (minimal stdlib patch for 8-elem BF16 / correct gfx12 FP8 lowering) or Tier 3 (upstream issue + decide inline-asm vs stay on WMMA-free BF16 and re-scope 1b).

No Hephaestus engine source was modified for this probe.

## How to reproduce

```sh
# separate env (already created once)
cd ~/projects/hephaestus-wmma-nightly   # pixi.toml pins mojo/max ==dev2026071206
sh run_probes.sh
# logs + this note under hephaestus/experiments/wmma_nightly_check/
```

## Artifacts

| file | content |
|---|---|
| `summary.txt` | tee of toolchain + probe outcomes |
| `exp3c_wmma_bf16.log` | full BF16 fail log |
| `exp3f_wmma_fp8.log` | full FP8 fail log |
| `exp3e_wmma_free.log` | stock exp3e origin-error on new mojo |
| `exp3e_wmma_free_fixed.log` | adapted exp3e PASS |
| `exp3c_bf16_frag8.log` | 8-elem BF16 constraint fail |
| `exp3e_wmma_free_paths_20260712.mojo` | origin-cast copy for new nightly only |
