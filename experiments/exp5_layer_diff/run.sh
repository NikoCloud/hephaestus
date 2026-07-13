#!/bin/sh
# Layer-diff harness gate (Phase 1b debug tool).
#
# 1. Dump activations twice on the tiny model (naive BF16 path).
# 2. Pack raw → .npy (ml_dtypes.bfloat16 / float32).
# 3. Diff the two dumps — must be bit-identical (determinism self-test).
#
# GPU 0 only. Does not modify src/hephaestus/forward.mojo.
set -e
cd "$(dirname "$0")/../.."
export HIP_VISIBLE_DEVICES=0
REPO=$(pwd)
KERNELS="${KERNELS:-$HOME/projects/modular/max/kernels/src}"
OUT="${OUT:-/tmp/exp5_layer_diff}"
RAW_A="$OUT/raw_a"
RAW_B="$OUT/raw_b"
NPY_A="$OUT/run_a"
NPY_B="$OUT/run_b"
PROMPT="${PROMPT:-1}"

mkdir -p "$RAW_A" "$RAW_B" "$NPY_A" "$NPY_B"

wait_gpu() {
    WAITED=0
    while true; do
        LIVE=""
        for P in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1}'); do
            if kill -0 "$P" 2>/dev/null; then LIVE="$LIVE $P"; fi
        done
        [ -z "$LIVE" ] && break
        if [ "$WAITED" -ge 600 ]; then
            echo "ABORT: GPU busy:$LIVE" >&2
            exit 1
        fi
        [ "$WAITED" -eq 0 ] && echo "GPU busy ($LIVE) — waiting"
        sleep 5
        WAITED=$((WAITED + 5))
    done
}

echo "=== toolchain ==="
pixi run mojo --version
pixi run python -c "from ml_dtypes import bfloat16; import numpy; print('python ok', numpy.__version__, bfloat16)"

DUMP_SRC="$REPO/experiments/exp5_layer_diff/dump_activations.mojo"
DIFF_PY="$REPO/experiments/exp5_layer_diff/diff_layers.py"

echo "=== dump A (tiny prompt $PROMPT) ==="
wait_gpu
rm -rf "$RAW_A"/* "$NPY_A"/*
pixi run mojo run -I "$KERNELS" -I src \
    "$DUMP_SRC" tiny "$RAW_A" "$PROMPT"

echo "=== pack A ==="
pixi run python "$DIFF_PY" pack "$RAW_A" "$NPY_A"

echo "=== dump B (tiny prompt $PROMPT, second run) ==="
wait_gpu
rm -rf "$RAW_B"/* "$NPY_B"/*
pixi run mojo run -I "$KERNELS" -I src \
    "$DUMP_SRC" tiny "$RAW_B" "$PROMPT"

echo "=== pack B ==="
pixi run python "$DIFF_PY" pack "$RAW_B" "$NPY_B"

echo "=== determinism self-test (exact bitwise) ==="
if pixi run python "$DIFF_PY" diff "$NPY_A" "$NPY_B" --exact; then
    echo ""
    echo "=== gate ==="
    echo "exp5_layer_diff PASS: two tiny dumps bit-identical"
    echo "npy dumps: $NPY_A  $NPY_B"
    exit 0
else
    echo ""
    echo "=== gate ==="
    echo "exp5_layer_diff FAIL: dumps are not bit-identical"
    exit 1
fi
