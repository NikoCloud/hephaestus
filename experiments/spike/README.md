# Logit-spike investigation probes

Focused evidence for prompt 1, teacher-forced step 67, token 96874.

- Canonical anomaly commit: `d60630d`
- Investigation worktree commit: `fe3c65a`
- Full teacher-forced source is byte-identical across them. The intervening numerical source change adds GEMV only for `m == 1`; the anomaly is reproduced on the unchanged one-shot `m > 1` path.
- Raw full-vocabulary and hidden-state dumps live under `/tmp` and are intentionally not committed. Compact JSON is in `out/`.

## Prerequisites

- Staged model: `staged/qwen3-4b`
- Original HF logits: `/tmp/hftf1_logits.f32`
- Five independent Hephaestus logits: `/tmp/spike-det-1783875368/rep{1..5}_logits.f32`
- GPU 0 clear. `gpu_wait.sh` waits for live users; it never kills them.

## Probe order

| probe | purpose | command |
|---|---|---|
| 0 | five-process raw-bit determinism and complete row statistics | `python3 experiments/spike/probe0_determinism.py` |
| 1 | tied-embedding row-space test and hidden recovery | `python3 experiments/spike/probe1_rowspace.py` |
| 2 | full/prefix/sequential execution routes | `sh experiments/spike/run_modes.sh` |
| 3 | HF hidden states at matched layer cut points | `sh experiments/spike/run_py_gpu.sh experiments/spike/probe3_hf_hidden.py` |
| 4 | layerwise Hephaestus/HF bisect | `python3 experiments/spike/probe4_bisect.py` |
| 5 | HF perturbation amplification and eager/SDPA spread | `sh experiments/spike/run_py_gpu.sh experiments/spike/probe5_conditioning.py` |
| 6 | recover all-row hidden divergence from logits | `python3 experiments/spike/probe6_rowdiv.py` |
| 7 | exploratory synthetic BF16 ensemble; not used as a correctness bound | `sh experiments/spike/run_py_gpu.sh experiments/spike/probe7_ensemble.py` |
| 8 | aligned 3-prompt E4M3 candidate scaling test | `sh experiments/spike/run_py_gpu.sh experiments/spike/probe8_fp8.py` |
| 9 | causal attention-rounding interventions | build/run `probe9_intervention.mojo` as documented in source |
| 10 | direct HF eager/SDPA/Hephaestus exact-prefix comparison | `sh experiments/spike/run_py_gpu.sh experiments/spike/probe10_hf_variants.py` |
| 11 | layer-0 Q/K/V seed localization | build `probe11_dump_l0.mojo` then `run_py_gpu.sh probe11_layer0_seed.py`; HF-rope compare writes `out/probe11_qkv_hfrope.json` |
| 12 | FP8 widening bar on measured delta | `python3 experiments/spike/probe12_fp8_margin.py` (needs `/tmp` logit artifacts) |
| 13 | Q cut-points + replace-and-continue | `sh experiments/spike/run_probe13.sh` → `out/probe13_q_cuts.json` |
| 14 | RoPE sub-step root cause | `python3 experiments/spike/probe14_rope_substep.py` (needs probe-13 dumps) |
| 15 | q_proj accum vs HF (prefill naive matmul) | build/run `probe15_qproj_accum.mojo dump` then `run_py_gpu.sh probe15_qproj_accum.py` |

`spike_forward.mojo` adds snapshots to the production operation order. `spike_kernels.mojo` copies production attention and parameterizes only probability/score rounding for probe 9. Its production control must reproduce target value `16.312086` exactly.

## Outputs kept in git

- `out/probe0_determinism.json`
- `out/probe1_rowspace.json`
- `out/probe4_bisect.json`
- `out/probe5_conditioning.json`
- `out/probe8_fp8.json`
- `out/probe10_hf_variants.json`
- `out/probe11_layer0_seed.json`
- `out/probe11_qkv_hfrope.json`
- `out/probe12_fp8_margin.json`
- `out/probe13_q_cuts.json`
- `out/probe14_rope_substep.json`
- `out/p1_prompt.txt`, `out/p1_oracle.txt`

Generated `.npy` matrices and caches are reproducible intermediates and are not repository artifacts.
