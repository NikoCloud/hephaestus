# G1a-1 blocked on an under-specified gate, not (only) on our numerics
## 2026-07-12 — needs an architect decision

## The finding

**HF's own two attention implementations of Qwen3-4B disagree with each other,
token-for-token, on our own oracle prompts.**

Same model, same weights, same greedy decode, same machine — only
`attn_implementation` differs:

| prompt | `sdpa` (what the committed oracle used) vs `eager` |
|---|---|
| 1 | diverge at step 4 (`34208` vs `374`) |
| 2 | diverge at step 32 |
| 3 | diverge at step 7 (`1632` vs `11245`) |

So "G1a-1: token-identical output vs HF transformers reference" does not name a
single target past ~4 tokens. Which 256-token sequence is *the* reference?

## Why they disagree

Every divergence we examined is a **one-bf16-ulp near-tie in the logits**:

- prompt1 step4: oracle top-2 = `34208: 17.25`, `374: 17.125` (gap 0.125 = 1 ulp at 17)
- prompt2 step12: HF top-2 = `5562: 12.1875`, `13551: 12.125` (gap 0.0625 = 1 ulp at 12)
- prompt3 step7: oracle top-2 = `11245: 19.25`, `1632: 19.25` — **an exact tie**,
  resolved only by `torch.argmax` returning the first max (lower id)

HF's logits are bf16 (the reference model runs in bf16, `lm_head` emits bf16).
At these ties, an implementation's choice is decided by accumulated rounding
below the bf16 ulp — i.e. by kernel internals, not by the model. sdpa and eager
round differently, so they pick differently. Any third implementation (ours)
lands wherever its own rounding puts it.

## Where Hephaestus actually stands

Against the committed (sdpa) oracle, 3 prompts x 256 greedy tokens:

| prompt | result |
|---|---|
| 1 | **TOKEN-EXACT 256/256** |
| 2 | matches 12/256, diverges at step 12 |
| 3 | matches 106/256, diverges at step 106 |

Plus: **all 20 teacher-forced steps match** (prompts 1 and 3, steps 0-9 each) —
i.e. given correct history, we predict the reference's next token every time
within the range the oracle saved logits for. And tiny is token-exact 3x16.

Being bit-identical to sdpa for a full 256 tokens on prompt1 is strong evidence
the forward pass is structurally correct. The remaining divergences are all
near-ties, not wrong answers.

Honest caveat: on prompt2 we diverge at step 12 while sdpa and eager still agree
with each other until step 32. That gap is *ours* — roughly one ulp of drift we
have not yet eliminated — so this is not purely a reference-ambiguity story.

## What moved the numerics (measured, not guessed)

Each change was tested against the oracle logits, not assumed:

| change | effect |
|---|---|
| RoPE pair-index fix (`get_safetensors_idx` misuse) | tiny max_abs_diff 0.19 -> 0.0059 |
| fp32 logits (don't round the LM head to bf16) | prompt1 step4 fixed |
| bf16 cos/sin + bf16 rotation (HF casts cos/sin to x.dtype) | prompt3 step7 diff 0.256 -> 0.172 |
| argmax over **bf16-rounded** logits, first-index tie-break (torch semantics) | all 20 teacher-forced steps match; prompt3 7 -> 106 |
| bf16 QK^T scores before scaling (literal eager arithmetic) | **WORSE** — prompt2 12 -> 6. Reverted. |

That last row is the useful negative result: the reference's effective precision
is higher than its Python source literally implies (sdpa doesn't execute that
Python), so "match the source" is not a reliable guide — only measurement is.

## Decision needed from Niko

G1a-1 cannot be both "token-identical to HF" and well-defined, as written.
Options:

**(a) Pin the reference to `eager` and re-gate.** Eager's arithmetic is fully
specified in Python and reproducible on any machine; sdpa's is a backend
lottery. Costs: regenerate the oracle (script already written:
`scripts/build_eager_reference.py`, output in `fixtures/oracle_eager/`), and we
would still need to close our ~1-ulp drift to match it. Note this *changes the
artifact we are graded against* — which is exactly why it is your call, not
ours.

**(b) Re-state G1a-1 as a correctness claim that is actually well-posed.** E.g.:
token-exact for the first N tokens (N=32 clears where both references still
agree) + teacher-forced argmax match at every one of 256 steps + logit
max_abs_diff bound. This is a *stronger* statement about our correctness than
"identical to a reference that flips coins", and it is checkable.

**(c) Keep the gate and keep chasing sdpa bit-parity.** Possible but it is
reverse-engineering an unpublished kernel's rounding order; the negative result
above suggests low odds and no principled stopping point.

Recommendation: **(b)**, optionally with (a) as the pinned reference for the
teacher-forced check. The thing we actually care about — "does Hephaestus
compute Qwen3 correctly" — is answered yes by the evidence above. What G1a-1
currently measures on top of that is which side of a coin flip our rounding
lands on.

## Reproduce

```
python3 scripts/dump_prompt_ids.py 1 /tmp/p1_ids.txt
pixi run mojo build -I ~/projects/modular/max/kernels/src -I src src/qwen_generate.mojo -o /tmp/qwen_gen
/tmp/qwen_gen /tmp/p1_ids.txt /tmp/out1.txt 256
python3 scripts/check_g1a1.py 1 /tmp/out1.txt          # -> TOKEN-EXACT 256/256

python3 scripts/build_eager_reference.py               # the disagreeing reference
python3 scripts/hf_step_logits.py 2 12 /tmp/hf.npy     # the 1-ulp gap at prompt2 step12
```
