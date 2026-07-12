#!/usr/bin/env python3
"""G1a-1 gate check: generated IDs vs fixtures/oracle/promptN_output_ids.json.

Token-exact or it isn't validated. Prints the first divergent step if any --
that step's index is where to point the layer-diff harness.
"""
import json
import sys

n = sys.argv[1]
got_path = sys.argv[2]

with open(f"fixtures/oracle/prompt{n}_output_ids.json") as f:
    want = json.load(f)
with open(got_path) as f:
    got = [int(line) for line in f if line.strip()]

if len(got) != len(want):
    print(f"prompt{n}: LENGTH MISMATCH got={len(got)} want={len(want)}")
    sys.exit(1)

for i, (g, w) in enumerate(zip(got, want)):
    if g != w:
        print(f"prompt{n}: MISMATCH at step {i}: got {g}, want {w}")
        print(f"  matched first {i}/{len(want)} tokens")
        sys.exit(1)

print(f"prompt{n}: TOKEN-EXACT ({len(got)}/{len(want)})")
