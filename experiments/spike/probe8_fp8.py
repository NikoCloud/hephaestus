#!/usr/bin/env python3
"""PROBE 8 -- the FP8 bar. Does the artifact stay harmless at E4M3 precision?

The spike is argmax-irrelevant in BF16. That is NOT sufficient to clear Phase 1b:
if the phase-7/phase-3 rows amplify a bf16 rounding (eps = 3.9e-3) into a 30%
hidden-state divergence, then FP8 E4M3 -- which has 4 bits of precision, eps =
2^-4 = 6.25e-2, SIXTEEN TIMES coarser -- will inject a far larger perturbation
into exactly the same amplifier. "It was fine in bf16" predicts nothing there.

So measure it rather than argue it. This does REAL E4M3 weight quantization (not
a noise proxy): every linear weight is cast to torch.float8_e4m3fn with a
per-output-channel absmax scale and dequantized back -- literally what FP8
weight-only serving does -- and then counts ARGMAX FLIPS against the bf16 oracle
over all 256 teacher-forced steps, broken down by phase.

Ladder, so the FP8 number has a scale to be read against:
  bf16 baseline        -> the reference decisions
  bf16-sized noise     -> how many flips ordinary rounding causes (control)
  FP8 E4M3 weights     -> the Phase 1b question

Each flip is classified as a genuine near-tie (|top1-top2| <= 1 bf16 ulp, which
SPEC.md G1a-1(a) already accepts) or a CLEAR flip (which it does not).

Usage: sh experiments/spike/run_py_gpu.sh experiments/spike/probe8_fp8.py
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = os.path.join(os.path.dirname(__file__), "out")
PLEN, NSTEPS, TARGET = 10, 256, 67
E4M3_MAX = 448.0
BF16_RMS = 2.0 ** -8 / (2.0 * np.sqrt(3.0))


def logits_and_hidden(model, ids, prompt_len, noise=None):
    got = {}
    hs = [model.model.norm.register_forward_hook(
        lambda m, a, o: got.__setitem__("h", o.detach()[0].float().cpu().numpy()))]
    if noise is not None:
        eps, seed = noise
        gen = torch.Generator(device="cuda:0").manual_seed(seed)

        def noisy(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            f = o.float()
            f = f * (1.0 + torch.randn(f.shape, generator=gen, device=f.device) * eps)
            r = f.to(o.dtype)
            return (r,) + out[1:] if isinstance(out, tuple) else r
        for layer in model.model.layers:
            hs.append(layer.register_forward_hook(noisy, with_kwargs=True))
    with torch.no_grad():
        lg = model(torch.tensor([ids], device="cuda:0")).logits[0]
    for h in hs:
        h.remove()
    sl = slice(prompt_len - 1, prompt_len - 1 + NSTEPS)
    return lg[sl].float().cpu().numpy(), got["h"][sl]


def quantize_e4m3_(model):
    """In-place per-output-channel E4M3 round-trip of every linear weight."""
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            w = mod.weight.data.float()
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
            q = (w / scale).to(torch.float8_e4m3fn).float() * scale
            mod.weight.data = q.to(mod.weight.dtype)
            n += 1
    return n


def quantize_acts_(model):
    """Also quantize every Linear's INPUT to E4M3 (per-token absmax).

    Phase 1b feeds FP8 to the WMMA units natively, and RDNA4 WMMA needs BOTH
    operands in FP8 -- so activations get quantized too, not just weights. This
    is the bound that actually applies to the 1b plan; weight-only FP8 flatters it.
    """
    hs = []

    def pre(mod, args):
        x = args[0].float()
        s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
        xq = (x / s).to(torch.float8_e4m3fn).float() * s
        return (xq.to(args[0].dtype),) + args[1:]

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            hs.append(mod.register_forward_pre_hook(pre))
    return hs


def flips(base_lg, new_lg):
    """argmax flips + near-tie classification against the bf16 reference."""
    b = base_lg.argmax(1)
    a = new_lg.argmax(1)
    bad = np.nonzero(a != b)[0]
    srt = np.sort(base_lg, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    top1 = srt[:, -1]
    # 1 bf16 ulp at that magnitude, exactly as SPEC.md G1a-1(a) defines it
    ulp = 2.0 ** np.floor(np.log2(np.maximum(np.abs(top1), 1e-30))) * 2.0 ** -7
    tie = gap[bad] <= ulp[bad]
    return bad, int(tie.sum()), int((~tie).sum())


def load_prompt(n):
    with open(f"fixtures/oracle/prompt{n}_input_ids.json") as f:
        p = json.load(f)
    with open(f"fixtures/oracle/prompt{n}_output_ids.json") as f:
        g = json.load(f)
    return p, g


def main():
    os.makedirs(OUT, exist_ok=True)
    # All 3 prompts = 768 teacher-forced steps, the same sample size the existing
    # G1a-1(a) evidence uses. One prompt would not be a defensible 1b bound.
    prompts = {n: load_prompt(n) for n in (1, 2, 3)}
    rep = {}

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m.eval().to("cuda:0")
    base = {}
    for n, (p, g) in prompts.items():
        ids = p + g[:255]
        base[n] = logits_and_hidden(m, ids, len(p)) + (ids, len(p))
        lg = base[n][0]
        print(f"prompt{n}: bf16 baseline argmax == oracle on "
              f"{int((lg.argmax(1) == np.array(g[:NSTEPS])).sum())}/{NSTEPS} steps")

    def report(tag, get):
        """get(n) -> (logits, hidden) for prompt n under the config under test."""
        tot = tie_t = clear_t = 0
        detail = {}
        for n in (1, 2, 3):
            base_lg, base_h, ids, plen = base[n]
            lg, h = get(n)
            bad, tie, clear = flips(base_lg, lg)
            div = np.linalg.norm(h - base_h, axis=1) / np.linalg.norm(base_h, axis=1)
            tot += len(bad); tie_t += tie; clear_t += clear
            d = dict(n_flip=len(bad), near_tie=tie, clear=clear,
                     median_div=float(np.median(div)), max_div=float(div.max()),
                     flip_rows=[int(x) for x in bad])
            if n == 1:  # prompt 1 is the periodic one; phase only means something there
                ph = np.arange(NSTEPS) % 10
                d["flips_by_phase"] = {p: int((ph[bad] == p).sum()) for p in range(10)}
                d["target_div"] = float(div[TARGET])
            detail[n] = d
        rep[tag] = dict(total_flips=tot, near_tie=tie_t, clear=clear_t, per_prompt=detail)
        print(f"\n--- {tag} ---")
        for n in (1, 2, 3):
            d = detail[n]
            print(f"  prompt{n}: {d['n_flip']:2d}/{NSTEPS} flips "
                  f"({d['near_tie']} tie, {d['clear']} CLEAR)  "
                  f"hidden div median {d['median_div']:.4f} max {d['max_div']:.4f}"
                  + (f"  row67 {d['target_div']:.4f}" if n == 1 else ""))
        print(f"  TOTAL: {tot}/768 flips  ({tie_t} near-tie, {clear_t} CLEAR)")
        p1 = detail[1]
        if p1["n_flip"]:
            share = sum(p1["flips_by_phase"][p] for p in (3, 7))
            print(f"  prompt1 phase-3+7 (the amplifier rows) share: {share}/{p1['n_flip']}")
        return tot, clear_t

    for s in (0, 1, 2):
        report(f"bf16-noise seed{s}",
               lambda n, s=s: logits_and_hidden(m, base[n][2], base[n][3], noise=(BF16_RMS, s)))

    del m
    torch.cuda.empty_cache()
    mq = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    mq.eval().to("cuda:0")
    nq = quantize_e4m3_(mq)
    print(f"\nquantized {nq} linear weights to E4M3 (per-output-channel absmax)")
    fw, cw = report("FP8 E4M3 weights", lambda n: logits_and_hidden(mq, base[n][2], base[n][3]))

    hs = quantize_acts_(mq)
    print(f"\nalso quantizing activations: {len(hs)} Linear inputs -> E4M3 "
          f"(per-token absmax) -- what RDNA4 WMMA actually requires")
    fa, ca = report("FP8 E4M3 weights+activations",
                    lambda n: logits_and_hidden(mq, base[n][2], base[n][3]))
    for x in hs:
        x.remove()

    print("\n=== FP8 BAR (768 teacher-forced steps, 3 prompts) ===")
    b = [rep[f"bf16-noise seed{s}"]["total_flips"] for s in (0, 1, 2)]
    bc = [rep[f"bf16-noise seed{s}"]["clear"] for s in (0, 1, 2)]
    print(f"  bf16-sized rounding     : {b} flips, {bc} CLEAR")
    print(f"  FP8 E4M3 weights        : {fw} flips, {cw} CLEAR")
    print(f"  FP8 E4M3 weights + acts : {fa} flips, {ca} CLEAR")

    with open(f"{OUT}/probe8_fp8.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/probe8_fp8.json")


if __name__ == "__main__":
    main()
