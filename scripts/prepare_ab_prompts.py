#!/usr/bin/env python3
"""Tokenize the two A/B benchmark prompts with the model's own tokenizer, so
Hephaestus (fed raw IDs, no tokenizer per SPEC.md scope) and llama.cpp (which
tokenizes internally) see the identical token sequence.

Writes:
  bench/ab_prompt_short_ids.txt   -- one ID per line
  bench/ab_prompt_short.txt       -- the raw text (for llama-cli/llama-bench -p)
  bench/ab_prompt_long_ids.txt
  bench/ab_prompt_long.txt
"""
import json
from transformers import AutoTokenizer

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
SHORT_TEXT = "The quick brown fox jumps over the lazy dog."

# Synthetic ~512-token prompt: repeat a standard paragraph until the tokenizer
# count reaches >=512, then trim to exactly 512 tokens (so pp512-comparable).
PARAGRAPH = (
    "The history of computing is a long and winding path shaped by "
    "mathematics, engineering, and the relentless pursuit of automation. "
    "From the abacus to the analytical engine, from vacuum tubes to "
    "integrated circuits, each generation of machines has built upon the "
    "ideas of the last. Modern processors execute billions of instructions "
    "per second, yet the fundamental principles of computation remain "
    "unchanged since Turing's original formulation. "
)

tok = AutoTokenizer.from_pretrained(MODEL)

short_ids = tok.encode(SHORT_TEXT, add_special_tokens=False)
print(f"short prompt: {len(short_ids)} tokens")
with open("bench/ab_prompt_short_ids.txt", "w") as f:
    for i in short_ids:
        f.write(f"{i}\n")
with open("bench/ab_prompt_short.txt", "w") as f:
    f.write(SHORT_TEXT)

text = PARAGRAPH
ids = tok.encode(text, add_special_tokens=False)
while len(ids) < 512:
    text += PARAGRAPH
    ids = tok.encode(text, add_special_tokens=False)
ids = ids[:512]
# Decode back the trimmed ID list to get exactly-matching text for llama.cpp.
long_text = tok.decode(ids)
long_ids_final = tok.encode(long_text, add_special_tokens=False)
print(f"long prompt: requested 512, got {len(long_ids_final)} tokens after decode/re-encode round-trip")

with open("bench/ab_prompt_long_ids.txt", "w") as f:
    for i in long_ids_final:
        f.write(f"{i}\n")
with open("bench/ab_prompt_long.txt", "w") as f:
    f.write(long_text)

print("wrote bench/ab_prompt_{short,long}{_ids,}.txt")
