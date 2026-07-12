#!/bin/sh
# Runs the spike diagnostic at prompt-1 teacher-forced step 67 by three routes.
# GPU 0 only. Build first:
#   pixi run mojo build -I ~/projects/modular/max/kernels/src -I src \
#       -I experiments/spike experiments/spike/probe2_modes.mojo -o /tmp/spike_modes
set -e
cd "$(dirname "$0")/../.."

# GPU guard. NOTE (measured 2026-07-12): `rocm-smi --showpids` reports GHOST
# entries -- dead PIDs whose KFD accounting was never reaped. We saw it report
# `qwen_tfd3` holding 30.9GB when `ps` showed no such PID and real VRAM use was
# 74MB. So: only abort if a reported PID is actually ALIVE. Never kill anything.
# Another agent session shares this box and runs HF reference jobs on GPU 0.
# WAIT for it -- never kill it, never race it.
WAITED=0
while true; do
    LIVE=""
    for P in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1}'); do
        if kill -0 "$P" 2>/dev/null; then
            LIVE="$LIVE $P($(ps -o comm= -p "$P" 2>/dev/null))"
        fi
    done
    [ -z "$LIVE" ] && break
    if [ "$WAITED" -ge 1800 ]; then
        echo "ABORT: GPU still busy after 30min:$LIVE. Not killing."
        exit 1
    fi
    [ "$WAITED" -eq 0 ] && echo "GPU busy (live:$LIVE) -- waiting, not killing."
    sleep 15
    WAITED=$((WAITED + 15))
done
[ "$WAITED" -gt 0 ] && echo "GPU freed after ${WAITED}s."
rocm-smi --showmeminfo vram 2>/dev/null | grep "Used"
echo "GPU clear (no live KFD processes). Using GPU 0 only."

export HIP_VISIBLE_DEVICES=0
W=/home/nikocloud/projects/hephaestus/staged/qwen3-4b
P=experiments/spike/out/p1_prompt.txt
O=experiments/spike/out/p1_oracle.txt

for M in full prefix seq; do
    echo "================ MODE $M ================"
    /tmp/spike_modes "$M" "$P" "$O" "$W" "/tmp/spike_$M" 2>&1 \
        | grep -viE "amdgpu\.ids|deprecated|warning:"
done

echo "=== outputs ==="
ls -la /tmp/spike_full_logits.f32 /tmp/spike_prefix_logits.f32 \
       /tmp/spike_seq_logits.f32 /tmp/spike_full_hidden.f32
