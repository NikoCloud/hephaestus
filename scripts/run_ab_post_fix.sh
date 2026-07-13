#!/bin/sh
# Re-run A/B after GPU argmax + chunked loader (bench/1a-ab.md methodology).
# GPU 0 only. Never kills foreign jobs.
set -e
cd "$(dirname "$0")/.."
export HIP_VISIBLE_DEVICES=0
BIN=/tmp/qwen_ab_bench
OUT=/tmp/ab_post_fix
mkdir -p "$OUT"

wait_gpu() {
    WAITED=0
    while true; do
        LIVE=""
        for P in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1}'); do
            if kill -0 "$P" 2>/dev/null; then
                LIVE="$LIVE $P"
            fi
        done
        [ -z "$LIVE" ] && break
        if [ "$WAITED" -ge 1800 ]; then
            echo "ABORT: GPU busy after 30min:$LIVE" >&2
            exit 1
        fi
        [ "$WAITED" -eq 0 ] && echo "GPU busy (live:$LIVE) -- waiting"
        sleep 10
        WAITED=$((WAITED + 10))
    done
}

echo "=== build qwen_ab_bench ==="
pixi run mojo build -I "$HOME/projects/modular/max/kernels/src" -I src \
    src/qwen_ab_bench.mojo -o "$BIN" 2>&1 | grep -viE 'deprecated|warning:|Implicitly|amdgpu' || true

# --- short prompt 10 tok x 256 gen, 3 reps ---
echo "=== Hephaestus short 10x256 x3 ==="
: > "$OUT/heph_short.log"
for r in 1 2 3; do
    wait_gpu
    echo "--- rep $r ---" | tee -a "$OUT/heph_short.log"
    if [ "$r" -eq 1 ]; then
        # VRAM poll every 0.5s during rep 1
        (
            while true; do
                rocm-smi --showmeminfo vram 2>/dev/null \
                    | awk '/GPU\[0\].*Total Used Memory/ {printf "vram_B %s\n", $NF; fflush()}'
                sleep 0.5
            done
        ) > "$OUT/heph_short_vram_poll.txt" &
        POLL=$!
        "$BIN" bench/ab_prompt_short_ids.txt 256 2>&1 \
            | grep -viE 'amdgpu|deprecated|warning:|Implicitly' \
            | tee -a "$OUT/heph_short.log"
        kill "$POLL" 2>/dev/null || true
        wait "$POLL" 2>/dev/null || true
    else
        "$BIN" bench/ab_prompt_short_ids.txt 256 2>&1 \
            | grep -viE 'amdgpu|deprecated|warning:|Implicitly' \
            | tee -a "$OUT/heph_short.log"
    fi
done

# --- long prompt 512 tok x 8 gen, 3 reps ---
echo "=== Hephaestus long 512x8 x3 ==="
: > "$OUT/heph_long.log"
for r in 1 2 3; do
    wait_gpu
    echo "--- rep $r ---" | tee -a "$OUT/heph_long.log"
    "$BIN" bench/ab_prompt_long_ids.txt 8 2>&1 \
        | grep -viE 'amdgpu|deprecated|warning:|Implicitly' \
        | tee -a "$OUT/heph_long.log"
done

# --- llama.cpp short (llama-simple) for total-time / decode-with-sampling ---
echo "=== llama-simple short 3 reps ==="
PROMPT_SHORT=$(cat bench/ab_prompt_short.txt)
: > "$OUT/llama_short.log"
for r in 1 2 3; do
    wait_gpu
    echo "--- rep $r ---" | tee -a "$OUT/llama_short.log"
    HIP_VISIBLE_DEVICES=0 "$HOME/projects/llama.cpp/build/bin/llama-simple" \
        -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf \
        -n 256 -ngl 99 \
        "$PROMPT_SHORT" \
        2>&1 | tee -a "$OUT/llama_short.log" | tail -25
done

# --- llama-bench pp/tg cross-check ---
echo "=== llama-bench pp10,512 / tg256 ==="
wait_gpu
HIP_VISIBLE_DEVICES=0 "$HOME/projects/llama.cpp/build/bin/llama-bench" \
    -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf \
    -p 10,512 -n 256 -r 3 -dev ROCm0 \
    2>&1 | tee "$OUT/llama_bench.log" | tail -30

echo "=== summarize ==="
python3 scripts/summarize_ab_post_fix.py
echo "DONE logs in $OUT"
