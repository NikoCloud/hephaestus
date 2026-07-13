#!/bin/sh
# Layer-diff: BF16 naive matmul vs BF16 WMMA path (same forward, toggle linear).
# Requires nightly Mojo (llvm_intrinsic WMMA). GPU 0 only.
set -e
cd "$(dirname "$0")/../.."
export HIP_VISIBLE_DEVICES=0
REPO=$(pwd)
KERNELS="${KERNELS:-$HOME/projects/modular/max/kernels/src}"
NIGHTLY="${NIGHTLY:-$HOME/projects/hephaestus-wmma-nightly}"
OUT="${OUT:-/tmp/exp5_naive_vs_wmma}"
PROMPT="${PROMPT:-1}"

if [ ! -x "$NIGHTLY/.pixi/envs/default/bin/mojo" ]; then
    echo "missing nightly env at $NIGHTLY" >&2
    exit 1
fi

mkdir -p "$OUT/raw_naive" "$OUT/raw_wmma" "$OUT/npy_naive" "$OUT/npy_wmma"
BIN=/tmp/exp5_dump_wmma_integration
DIFF_PY="$REPO/experiments/exp5_layer_diff/diff_layers.py"

echo "=== build dump (nightly) ==="
(cd "$NIGHTLY" && pixi run mojo build -I "$KERNELS" -I "$REPO/src" \
    "$REPO/experiments/exp5_layer_diff/dump_activations.mojo" -o "$BIN") \
    2>&1 | grep -viE 'amdgpu.ids|deprecated|Implicitly converting|@parameter' || true
test -x "$BIN"

echo "=== dump naive ==="
rm -rf "$OUT/raw_naive"/* "$OUT/npy_naive"/*
"$BIN" tiny naive "$OUT/raw_naive" "$PROMPT"
pixi run python "$DIFF_PY" pack "$OUT/raw_naive" "$OUT/npy_naive"

echo "=== dump wmma ==="
rm -rf "$OUT/raw_wmma"/* "$OUT/npy_wmma"/*
"$BIN" tiny wmma "$OUT/raw_wmma" "$PROMPT"
pixi run python "$DIFF_PY" pack "$OUT/raw_wmma" "$OUT/npy_wmma"

echo "=== diff (tolerance) ==="
pixi run python "$DIFF_PY" diff "$OUT/npy_naive" "$OUT/npy_wmma"
