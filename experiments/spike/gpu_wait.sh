#!/bin/sh
# Block until no LIVE process holds a GPU, then return. Never kills anything.
#
# Two machine quirks this handles (both measured 2026-07-12):
#  1. `rocm-smi --showpids` reports GHOST entries -- dead PIDs whose KFD
#     accounting was never reaped (we saw `qwen_tfd3` "holding" 30.9GB while
#     `ps` showed no such PID and real VRAM use was 74MB).
#  2. Another agent session shares this box and runs HF reference jobs on GPU 0.
#     We wait for it. We do not kill it and we do not race it.
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
        echo "ABORT: GPU still busy after 30min:$LIVE. Not killing." >&2
        exit 1
    fi
    [ "$WAITED" -eq 0 ] && echo "GPU busy (live:$LIVE) -- waiting, not killing."
    sleep 15
    WAITED=$((WAITED + 15))
done
[ "$WAITED" -gt 0 ] && echo "GPU freed after ${WAITED}s."
echo "GPU clear. Using GPU 0 only."
exit 0
