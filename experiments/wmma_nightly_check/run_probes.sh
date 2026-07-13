#!/bin/sh
# Run WMMA probes against Mojo nightly 1.0.0b3.dev2026071206 (isolated env).
# Does not modify Hephaestus engine code.
set -e
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
REPO="$HOME/projects/hephaestus"
KERNELS="$HOME/projects/modular/max/kernels/src"
OUT="$REPO/experiments/wmma_nightly_check"
mkdir -p "$OUT"

wait_gpu() {
    WAITED=0
    while true; do
        LIVE=""
        for P in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1}'); do
            if kill -0 "$P" 2>/dev/null; then LIVE="$LIVE $P"; fi
        done
        [ -z "$LIVE" ] && break
        if [ "$WAITED" -ge 600 ]; then
            echo "ABORT: GPU busy" >&2
            exit 1
        fi
        [ "$WAITED" -eq 0 ] && echo "GPU busy ($LIVE) waiting..."
        sleep 10
        WAITED=$((WAITED + 10))
    done
}

echo "=== toolchain ===" | tee "$OUT/summary.txt"
pixi run mojo --version 2>&1 | tee -a "$OUT/summary.txt"
pixi list 2>&1 | grep -E '^(mojo|max|max-core)' | tee -a "$OUT/summary.txt"

run_probe() {
    name="$1"
    src="$2"
    need_kernels="$3"
    echo "" | tee -a "$OUT/summary.txt"
    echo "=== $name ===" | tee -a "$OUT/summary.txt"
    wait_gpu
    LOG="$OUT/${name}.log"
    if [ "$need_kernels" = "1" ]; then
        set +e
        pixi run mojo build -I "$KERNELS" "$src" -o "/tmp/${name}_bin" >"$LOG" 2>&1
        RC=$?
        set -e
        if [ "$RC" -ne 0 ]; then
            echo "BUILD FAIL rc=$RC" | tee -a "$OUT/summary.txt"
            # keep full error (LLVM often dumps a lot)
            tail -80 "$LOG" | tee -a "$OUT/summary.txt"
            return 0
        fi
        set +e
        "/tmp/${name}_bin" >>"$LOG" 2>&1
        RC=$?
        set -e
        if [ "$RC" -ne 0 ]; then
            echo "RUN FAIL rc=$RC" | tee -a "$OUT/summary.txt"
            tail -40 "$LOG" | tee -a "$OUT/summary.txt"
        else
            echo "PASS" | tee -a "$OUT/summary.txt"
            tail -20 "$LOG" | tee -a "$OUT/summary.txt"
        fi
    else
        set +e
        pixi run mojo build "$src" -o "/tmp/${name}_bin" >"$LOG" 2>&1
        RC=$?
        set -e
        if [ "$RC" -ne 0 ]; then
            echo "BUILD FAIL rc=$RC" | tee -a "$OUT/summary.txt"
            tail -80 "$LOG" | tee -a "$OUT/summary.txt"
            return 0
        fi
        set +e
        "/tmp/${name}_bin" >>"$LOG" 2>&1
        RC=$?
        set -e
        if [ "$RC" -ne 0 ]; then
            echo "RUN FAIL rc=$RC" | tee -a "$OUT/summary.txt"
            tail -40 "$LOG" | tee -a "$OUT/summary.txt"
        else
            echo "PASS" | tee -a "$OUT/summary.txt"
            tail -20 "$LOG" | tee -a "$OUT/summary.txt"
        fi
    fi
}

# Baseline: same probes as engine repo experiments/
run_probe exp3c_wmma_bf16 "$REPO/experiments/exp3c_wmma_probe.mojo" 0
run_probe exp3f_wmma_fp8 "$REPO/experiments/exp3f_fp8_wmma_probe.mojo" 0
run_probe exp3e_wmma_free "$REPO/experiments/exp3e_wmma_free_paths.mojo" 1

# Also try RDNA4-shaped 8-element BF16 fragments if API accepts them
cat > /tmp/exp3c_bf16_frag8.mojo <<'EOF'
# RDNA4-shaped BF16 fragments (size 8) — may fail at type-check if API requires 16
from std.gpu.host import DeviceContext
from std.gpu.compute.mma import mma

def wmma_probe(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    var a = SIMD[DType.bfloat16, 8](1.0)
    var b = SIMD[DType.bfloat16, 8](1.0)
    var c = SIMD[DType.float32, 8](0.0)
    var d = SIMD[DType.float32, 8](0.0)
    mma(d, a, b, c)
    out_ptr[0] = d[0]

def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_memset(buf, 0)
    ctx.enqueue_function[wmma_probe](buf.unsafe_ptr(), grid_dim=(1,), block_dim=(32,))
    ctx.synchronize()
    with buf.map_to_host() as h:
        print("wmma frag8 d[0] =", h[0])
    print("EXP3c_frag8: BF16 WMMA with 8-elem fragments compiled and ran")
EOF
run_probe exp3c_bf16_frag8 /tmp/exp3c_bf16_frag8.mojo 0

echo "" | tee -a "$OUT/summary.txt"
echo "=== done $(date -Iseconds) ===" | tee -a "$OUT/summary.txt"
echo "Results in $OUT"
