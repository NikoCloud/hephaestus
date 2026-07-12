#!/bin/sh
# Probe 13: Q cut-point dumps + replace-and-continue.
# GPU 0 only. Never kills foreign jobs.
set -e
cd "$(dirname "$0")/../.."

KERNELS="$HOME/projects/modular/max/kernels/src"
BIN=/tmp/spike_q_cuts
W=staged/qwen3-4b
P=experiments/spike/out/p1_prompt.txt
O=experiments/spike/out/p1_oracle.txt

echo "=== build probe13_q_cuts.mojo ==="
pixi run mojo build \
    -I "$KERNELS" -I src -I experiments/spike \
    experiments/spike/probe13_q_cuts.mojo -o "$BIN"

sh experiments/spike/gpu_wait.sh
export HIP_VISIBLE_DEVICES=0

echo "=== Hephaestus dump (3 Q cuts + control outputs) ==="
"$BIN" dump "$P" "$O" "$W" /tmp/spike13_dump 2>&1 \
    | grep -viE "amdgpu\.ids|deprecated|warning:"
# alias dump outputs as control for analysis if control not run
cp -f /tmp/spike13_dump_hidden.f32 /tmp/spike13_control_hidden.f32
cp -f /tmp/spike13_dump_logits.f32 /tmp/spike13_control_logits.f32

echo "=== HF Q cuts (sdpa + eager) ==="
sh experiments/spike/run_py_gpu.sh experiments/spike/probe13_hf_q_cuts.py

echo "=== replace-and-continue injects (HF SDPA Q at each cut) ==="
for CUT in q_proj q_norm q_rope; do
    HF=/tmp/spike13_hf_sdpa_${CUT}.f32
    if [ ! -f "$HF" ]; then
        echo "MISSING $HF" >&2
        exit 1
    fi
    sh experiments/spike/gpu_wait.sh
    echo "--- inject_$CUT ---"
    "$BIN" "inject_$CUT" "$P" "$O" "$W" "/tmp/spike13_inject_$CUT" "$HF" 2>&1 \
        | grep -viE "amdgpu\.ids|deprecated|warning:"
done

echo "=== analyze ==="
# analyze-only reuses HF npy + all dumps
python3 experiments/spike/probe13_hf_q_cuts.py --analyze-only

echo "=== done ==="
ls -la /tmp/spike13_dump_q_*.f32 /tmp/spike13_*_hidden.f32 \
       experiments/spike/out/probe13_q_cuts.json 2>/dev/null || true
