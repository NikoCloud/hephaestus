#!/bin/sh
# Run a GPU python probe on GPU 0, politely.
#
# This box is shared with another agent session that launches HF/Mojo jobs on
# GPU 0 without warning. gpu_wait alone is not enough: we lost a run to an OOM
# when a foreign job grabbed ~25GB between the wait check and torch's model load.
# So: wait for clear, require headroom, run, and retry the whole thing on failure.
# Never kills anything.
#
# Usage: sh experiments/spike/run_py_gpu.sh <script.py> [args...]
set -e
cd "$(dirname "$0")/../.."
NEED_MB=12000
TRIES=6

i=1
while [ "$i" -le "$TRIES" ]; do
    sh experiments/spike/gpu_wait.sh

    FREE_MB=$(rocm-smi --showmeminfo vram 2>/dev/null \
        | awk '/GPU\[0\].*Total Memory/ {t=$NF} /GPU\[0\].*Total Used Memory/ {u=$NF} END {print int((t-u)/1048576)}')
    if [ "$FREE_MB" -lt "$NEED_MB" ]; then
        echo "attempt $i: only ${FREE_MB}MB free on GPU 0 (need ${NEED_MB}MB) -- waiting."
        sleep 20
        i=$((i + 1))
        continue
    fi

    echo "attempt $i: ${FREE_MB}MB free on GPU 0. Running: $*"
    # NOTE: do NOT pipe python straight into grep -- the pipeline's exit status
    # is grep's, which is 0 whenever it matched *any* line, including a
    # traceback. That silently turned every failed run into a "success" and the
    # retry loop never fired. Capture to a file, then check python's own status.
    LOG=/tmp/spike_gpu_run.$$.log
    if HIP_VISIBLE_DEVICES=0 python3 "$@" >"$LOG" 2>&1; then
        grep -viE "amdgpu\.ids|Loading weights|it/s\]$" "$LOG" || true
        rm -f "$LOG"
        exit 0
    fi
    if grep -qi "out of memory" "$LOG"; then
        echo "attempt $i: lost a race for GPU 0 (OOM -- a foreign job grabbed it). Retrying."
    else
        echo "attempt $i FAILED (not an OOM):"
        tail -15 "$LOG"
        rm -f "$LOG"
        exit 1
    fi
    rm -f "$LOG"
    sleep 20
    i=$((i + 1))
done
echo "ABORT: $TRIES attempts all failed." >&2
exit 1
