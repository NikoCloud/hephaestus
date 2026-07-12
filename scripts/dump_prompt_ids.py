#!/usr/bin/env python3
"""Write an oracle prompt's saved input IDs to a text file, one per line.

The IDs come straight from fixtures/oracle/promptN_input_ids.json -- never
re-tokenized. Tokenizer parity is a separate problem (dossier §5); feeding the
saved IDs is what makes G1a-1 a forward-pass test rather than a tokenizer test.
"""
import json
import sys

n = sys.argv[1]
out = sys.argv[2]
with open(f"fixtures/oracle/prompt{n}_input_ids.json") as f:
    ids = json.load(f)
with open(out, "w") as f:
    for i in ids:
        f.write(f"{i}\n")
print(f"prompt{n}: {len(ids)} tokens -> {out}")
