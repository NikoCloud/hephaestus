#!/bin/sh
# Staged bring-up of multi-tile BF16 WMMA GEMM (exp4b).
# Uses isolated Mojo nightly at ~/projects/hephaestus-wmma-nightly.
# GPU 0 only.
#
# Stages:
#   1) M=16 N=16 K=32  — one tile, two K-strips (exact)
#   2) M=32 N=32 K=32  — four tiles, two strips (exact)
#   3) M=32 N=4096 K=2560 — q_proj-shaped random (tolerance)
set -e
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
NIGHTLY="$HOME/projects/hephaestus-wmma-nightly"
BIN=/tmp/exp4b_wmma_gemm
OUT=/tmp/exp4b_out
INP=/tmp/exp4b_inputs
mkdir -p "$OUT"

if [ ! -x "$NIGHTLY/.pixi/envs/default/bin/mojo" ]; then
    echo "missing isolated nightly env at $NIGHTLY" >&2
    exit 1
fi

PY="$NIGHTLY/.pixi/envs/default/bin/python"

# GPU clear (wait, never kill)
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

echo "=== toolchain ==="
(cd "$NIGHTLY" && pixi run mojo --version)
"$PY" -c "from ml_dtypes import bfloat16; import numpy; print('python ok', numpy.__version__, bfloat16)"

HERE=$(pwd)
echo "=== build gemm ==="
(cd "$NIGHTLY" && pixi run mojo build \
    "$HERE/gemm.mojo" -o "$BIN") 2>&1 | grep -viE 'amdgpu.ids|deprecated' || true
test -x "$BIN"

FAIL=0

echo ""
echo "=== stage 1: M=16 N=16 K=32 (one tile, two K-strips) exact ==="
"$PY" oracle.py expected-sample 16 16 32
"$BIN" structured 16 16 32 "$OUT/c_s1.bf16"
if "$PY" oracle.py compare structured 16 16 32 "$OUT/c_s1.bf16" --exact; then
    echo "STAGE1 OK"
else
    echo "STAGE1 FAILED"
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "=== gate: STOP after stage1 failure ==="
    exit 1
fi

echo ""
echo "=== stage 2: M=32 N=32 K=32 (four tiles) exact ==="
"$PY" oracle.py expected-sample 32 32 32
"$BIN" structured 32 32 32 "$OUT/c_s2.bf16"
if "$PY" oracle.py compare structured 32 32 32 "$OUT/c_s2.bf16" --exact; then
    echo "STAGE2 OK"
else
    echo "STAGE2 FAILED"
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "=== gate: STOP after stage2 failure ==="
    exit 1
fi

echo ""
echo "=== stage 3: M=32 N=4096 K=2560 random BF16 (shared .npy) ==="
"$PY" oracle.py gen-random 32 4096 2560 "$INP"
"$BIN" random 32 4096 2560 "$OUT/c_s3.bf16" "$INP/A.npy" "$INP/W.npy"
if "$PY" oracle.py compare random 32 4096 2560 "$OUT/c_s3.bf16" \
    --a "$INP/A.npy" --w "$INP/W.npy"; then
    echo "STAGE3 OK"
else
    echo "STAGE3 FAILED"
    FAIL=1
fi

echo ""
echo "=== gate ==="
if [ "$FAIL" -eq 0 ]; then
    echo "exp4b PASS: stage1 exact ∧ stage2 exact ∧ stage3 tolerance"
    exit 0
else
    echo "exp4b FAIL"
    exit 1
fi
