#!/bin/sh
# G1b-0 gate: build microkernel, run T1/T2/T3, exact-compare to oracle.
# Uses isolated Mojo nightly at ~/projects/hephaestus-wmma-nightly.
set -e
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
NIGHTLY="$HOME/projects/hephaestus-wmma-nightly"
BIN=/tmp/exp4a_wmma_tile
OUT=/tmp/exp4a_out
mkdir -p "$OUT"

if [ ! -x "$NIGHTLY/.pixi/envs/default/bin/mojo" ]; then
    echo "missing isolated nightly env at $NIGHTLY" >&2
    exit 1
fi

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

HERE=$(pwd)
echo "=== build microkernel ==="
(cd "$NIGHTLY" && pixi run mojo build \
    "$HERE/microkernel.mojo" -o "$BIN") 2>&1 | grep -viE 'amdgpu.ids|deprecated' || true
test -x "$BIN"

echo "=== oracle self-check (closed forms) ==="
python3 oracle.py T1
python3 oracle.py T2
python3 oracle.py T3

FAIL=0
for T in T1 T2 T3; do
    echo "=== run $T ==="
    "$BIN" "$T" "$OUT/d_${T}.f32"
    if python3 oracle.py "$T" "$OUT/d_${T}.f32"; then
        echo "$T OK"
    else
        echo "$T FAILED"
        FAIL=1
    fi
done

echo "=== gate ==="
if [ "$FAIL" -eq 0 ]; then
    echo "G1b-0 PASS: T1 ∧ T2 ∧ T3 exact equality on gfx1201"
    exit 0
else
    echo "G1b-0 FAIL: see §9 diagnosis table in the spec"
    exit 1
fi
