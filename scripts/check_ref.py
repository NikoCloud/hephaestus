#!/usr/bin/env python3
"""Compare generated IDs against a named reference dir."""
import json, sys
ref_dir, n, got_path = sys.argv[1], sys.argv[2], sys.argv[3]
want = json.load(open(f"{ref_dir}/prompt{n}_output_ids.json"))
got = [int(l) for l in open(got_path) if l.strip()]
for i, (g, w) in enumerate(zip(got, want)):
    if g != w:
        print(f"prompt{n} vs {ref_dir}: MISMATCH at step {i} (got {g}, want {w}) -- matched {i}/{len(want)}")
        sys.exit(1)
print(f"prompt{n} vs {ref_dir}: TOKEN-EXACT ({len(got)}/{len(want)})")
