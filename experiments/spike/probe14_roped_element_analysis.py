#!/usr/bin/env python3
"""Probe 14: element-level analysis of WHERE in head_dim the q_rope
divergence concentrates, for the target row.

Probe 13 established: q_proj and q_norm show tiny, roughly-equal relative
error (~1e-4, ordinary bf16 noise); q_rope shows a 15.65x jump. Raw GPU
cos/sin precision was independently ruled out (exp6_gpu_trig_probe.mojo:
error ~1e-6 to 1e-8, six orders of magnitude too small). This probe asks:
within the 128 head_dim elements, does the EXTRA q_rope error concentrate at
specific frequency pairs (e.g. the highest-frequency pair, index 0, whose
rotation angle equals raw token position -- the argument range where a
cos/sin approximation would be worst) or is it spread uniformly (pointing at
a systematic composition/cast difference instead)?
"""
import numpy as np

SEQ_LEN = 77
Q_OUT = 4096
HEAD_DIM = 128
N_HEADS = 32
TARGET_ROW = 76

def load(path):
    return np.fromfile(path, dtype=np.float32).reshape(SEQ_LEN, Q_OUT)

heph_proj = load("/tmp/spike13_dump_q_proj.f32")
heph_norm = load("/tmp/spike13_dump_q_norm.f32")
heph_rope = load("/tmp/spike13_dump_q_rope.f32")
hf_proj = load("experiments/spike/out/hf_sdpa_q_proj.f32")
hf_norm = load("experiments/spike/out/hf_sdpa_q_norm.f32")
hf_rope = load("experiments/spike/out/hf_sdpa_q_rope.f32")

def per_head(row, flat):
    return flat[row].reshape(N_HEADS, HEAD_DIM)

for name, h, hf in (("q_proj", heph_proj, hf_proj), ("q_norm", heph_norm, hf_norm), ("q_rope", heph_rope, hf_rope)):
    hh = per_head(TARGET_ROW, h)
    hfhf = per_head(TARGET_ROW, hf)
    diff = np.abs(hh - hfhf)
    print(f"=== {name} (target row {TARGET_ROW}, all 32 heads x 128 head_dim) ===")
    print(f"  max abs diff: {diff.max():.6f} at head={diff.argmax()//HEAD_DIM} dim={diff.argmax()%HEAD_DIM}")
    print(f"  mean abs diff: {diff.mean():.6f}")
    # per-dim-index mean across all heads -- reveals frequency-pair concentration
    per_dim_mean = diff.mean(axis=0)  # [128]
    top5 = np.argsort(-per_dim_mean)[:5]
    print(f"  top-5 head_dim indices by mean abs diff across heads: {[(int(i), float(per_dim_mean[i])) for i in top5]}")
    print(f"  head_dim index 0 (re, highest-freq pair) mean diff: {per_dim_mean[0]:.6f}")
    print(f"  head_dim index 64 (im, highest-freq pair, since split-half pairs i with i+64) mean diff: {per_dim_mean[64]:.6f}")
    print()

print("=== q_rope only: per-head max abs diff (32 heads) ===")
hh = per_head(TARGET_ROW, heph_rope)
hfhf = per_head(TARGET_ROW, hf_rope)
diff = np.abs(hh - hfhf)
per_head_max = diff.max(axis=1)
for hidx in range(N_HEADS):
    print(f"  head {hidx:2d}: max={per_head_max[hidx]:.6f}")
