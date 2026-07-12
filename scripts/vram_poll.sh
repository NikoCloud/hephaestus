#!/bin/bash
# Poll GPU 0 VRAM usage (MB) at a fixed interval while a command runs.
# Usage: scripts/vram_poll.sh <output_csv> -- <command...>
# Writes timestamped MB samples to output_csv; prints peak MB to stdout at end.

out="$1"
shift
if [ "$1" == "--" ]; then shift; fi

: > "$out"
(
  while true; do
    # rocm-smi --showmeminfo vram gives total used across all GPUs in one
    # line per device; grab GPU[0]'s "VRAM Total Used Memory (B)".
    val=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[0\]" | grep -o '[0-9]\+' | tail -1)
    echo "$(date +%s.%N),$val" >> "$out"
    sleep 0.2
  done
) &
poller=$!

"$@"
rc=$?

kill "$poller" 2>/dev/null
wait "$poller" 2>/dev/null

peak=$(awk -F',' 'BEGIN{max=0} {if($2+0>max)max=$2+0} END{print max}' "$out")
echo "PEAK_VRAM_BYTES=$peak"
echo "PEAK_VRAM_MB=$(awk "BEGIN{printf \"%.1f\", $peak/1048576}")"
exit $rc
