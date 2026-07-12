#!/usr/bin/env python3
"""PROBE 11 -- localize the layer-0 seed difference (Q/K/V vs attention).

Falsifiable predictions (stated before running):
  A) If projections/RoPE already diverge at the measured ~1e-3 residual scale,
     then post-RoPE Q/K/V relative error at the target query is comparable to
     (or larger than) the measured layer-0 attn_out relative error (~0.00174).
  B) If Q/K/V match closely and only the attention output diverges, the seed
     lives inside the attention kernel (score/softmax/PV reduction order).
  C) If HF SDPA vs HF eager already produce Q/K/V-identical states (they share
     the pre-attention path) but differ after attention, that bounds how much
     pure attention-impl variation can move the residual at layer 0.

Hephaestus side uses the existing instrumented prefix dump of o_proj /
residual (slots) plus a one-shot dump of Q/K/V written by a small companion
Mojo binary if present; when that dump is missing, this probe still reports
the HF-only bound (C) and residual-side numbers from committed artifacts.

Usage:
  sh experiments/spike/run_py_gpu.sh experiments/spike/probe11_layer0_seed.py
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = Path(__file__).resolve().parent / "out"
STEP = 67
HEAD_DIM = 128
N_HEADS = 32
N_KV = 8
HIDDEN = 2560
THETA = 5_000_000.0
TARGET_TOK = 96874


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
    den = float(np.linalg.norm(b.astype(np.float64))) + 1e-30
    return num / den


def bf16_round_np(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float()
    return t.numpy()


def rope_split_half(x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
    """Match Hephaestus / HF safetensors split-half RoPE on [seq, n_heads, head_dim]."""
    seq, nh, d = x.shape
    half = d // 2
    re = x[..., :half]
    im = x[..., half:]
    device = x.device
    pos = torch.arange(pos_offset, pos_offset + seq, device=device, dtype=torch.float32)
    pair = torch.arange(half, device=device, dtype=torch.float32)
    # freq[t, p] = pos * theta ** (-2p/d)  -- matches kernels.mojo rope_kernel
    inv = THETA ** (-2.0 * pair / float(d))
    freqs = pos[:, None] * inv[None, :]  # [seq, half]
    cos_v = torch.cos(freqs).to(torch.bfloat16).to(torch.float32)
    sin_v = torch.sin(freqs).to(torch.bfloat16).to(torch.float32)
    cos_v = cos_v[:, None, :].to(x.dtype)
    sin_v = sin_v[:, None, :].to(x.dtype)
    out_re = re * cos_v - im * sin_v
    out_im = im * cos_v + re * sin_v
    return torch.cat([out_re, out_im], dim=-1)


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prob_bf16: bool,
    score_bf16: bool,
) -> torch.Tensor:
    """Hephaestus-shaped attention: warp-less reference in fp32 with optional bf16 rounds.

    q: [seq, n_heads, d], k/v: [seq, n_kv, d]
    Runs on CPU for the slow triple loop (seq=77 is fine).
    """
    q = q.detach().float().cpu()
    k = k.detach().float().cpu()
    v = v.detach().float().cpu()
    seq, nh, d = q.shape
    n_kv = k.shape[1]
    group = nh // n_kv
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros_like(q)
    for t in range(seq):
        n_keys = t + 1
        for h in range(nh):
            kvh = h // group
            qh = q[t, h]
            scores = []
            for j in range(n_keys):
                s = float((qh * k[j, kvh]).sum() * scale)
                if score_bf16:
                    s = float(torch.tensor(s, dtype=torch.float32).to(torch.bfloat16).float())
                scores.append(s)
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            inv = 1.0 / sum(exps)
            probs = []
            for e in exps:
                p = e * inv
                if prob_bf16:
                    p = float(torch.tensor(p, dtype=torch.float32).to(torch.bfloat16).float())
                probs.append(p)
            acc = torch.zeros(d, dtype=torch.float32)
            for j in range(n_keys):
                acc += probs[j] * v[j, kvh]
            out[t, h] = acc
    return out


def load_ids():
    with open("fixtures/oracle/prompt1_input_ids.json") as f:
        prompt = json.load(f)
    with open("fixtures/oracle/prompt1_output_ids.json") as f:
        gen = json.load(f)
    ids = prompt + gen[:STEP]
    return ids, len(prompt) - 1 + STEP  # target row in this seq


def main():
    os.makedirs(OUT, exist_ok=True)
    ids, target_row = load_ids()
    assert target_row == len(ids) - 1

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval().to("cuda:0")

    # Force two implementations for pre-attn identity + post-attn spread.
    results = {
        "input": {
            "seq_len": len(ids),
            "target_row": target_row,
            "step": STEP,
            "token": TARGET_TOK,
        },
        "predictions": {
            "A_qkv_diverges_at_seed_scale": None,
            "B_attention_kernel_is_seed": None,
            "C_hf_attn_impl_bound_at_layer0": None,
        },
    }

    layer0 = model.model.layers[0]
    captured = {}

    def capture_pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        captured["x"] = h.detach()

    def capture_attn(mod, args, kwargs, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["attn_out"] = o.detach()

    # Build Q/K/V the same way Qwen3Attention does, from HF weights.
    def qkv_from_x(x_bf16: torch.Tensor):
        # x: [1, seq, hidden]
        xn = layer0.input_layernorm(x_bf16)
        bsz, seq, _ = xn.shape
        q = layer0.self_attn.q_proj(xn).view(bsz, seq, N_HEADS, HEAD_DIM)
        k = layer0.self_attn.k_proj(xn).view(bsz, seq, N_KV, HEAD_DIM)
        v = layer0.self_attn.v_proj(xn).view(bsz, seq, N_KV, HEAD_DIM)
        q = layer0.self_attn.q_norm(q)
        k = layer0.self_attn.k_norm(k)
        # Apply our rope (should match HF); also get HF rope via position_ids path if needed.
        q_r = rope_split_half(q[0], 0)
        k_r = rope_split_half(k[0], 0)
        return q_r, k_r, v[0]

    # --- HF SDPA full forward for residual baseline ---
    model.config._attn_implementation = "sdpa"
    # re-init attn modules? transformers stores impl at load; use model.set if available
    try:
        model.set_attn_implementation("sdpa")
    except Exception:
        pass

    def run_with_impl(impl: str):
        try:
            model.set_attn_implementation(impl)
        except Exception:
            model.config._attn_implementation = impl
        cap = {}

        def ah(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            cap["attn"] = o.detach()

        def lh(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            cap["h0"] = o.detach()

        h_attn = layer0.self_attn.register_forward_hook(ah, with_kwargs=True)
        h_layer = layer0.register_forward_hook(lh, with_kwargs=True)
        with torch.no_grad():
            # Full model forward so rotary / position_embeddings are built correctly.
            # We only need layer-0 outputs via hooks; later layers still run (cheap enough).
            _ = model(torch.tensor([ids], device="cuda:0"))
        h_attn.remove()
        h_layer.remove()
        return cap

    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda:0"))
        results["embed_norm"] = float(emb[0, target_row].float().norm())

        # Prefer HF's own RoPE (via intercepting Q/K after rotary) for QKV ground truth.
        # Also compute our split-half rope path for comparison.
        q_r, k_r, v = qkv_from_x(emb)
        q_tgt = q_r[target_row].float().cpu().numpy()
        k_all = k_r.float().cpu().numpy()
        v_all = v.float().cpu().numpy()

        attn_prod = manual_attention(q_r, k_r, v, prob_bf16=True, score_bf16=False)
        attn_fp32p = manual_attention(q_r, k_r, v, prob_bf16=False, score_bf16=False)
        attn_scorebf = manual_attention(q_r, k_r, v, prob_bf16=True, score_bf16=True)

        def o_proj(a_cpu_f32):
            flat = a_cpu_f32.reshape(a_cpu_f32.shape[0], N_HEADS * HEAD_DIM)
            t = flat.to(device="cuda:0", dtype=torch.bfloat16).unsqueeze(0)
            return layer0.self_attn.o_proj(t)[0]

        o_prod = o_proj(attn_prod)
        o_fp32 = o_proj(attn_fp32p)
        o_sc = o_proj(attn_scorebf)

        cap_sdpa = run_with_impl("sdpa")
        cap_eager = run_with_impl("eager")
        h0_sdpa = cap_sdpa["h0"]
        h0_eager = cap_eager["h0"]
        hf_attn_sdpa = cap_sdpa["attn"][0, target_row].float().cpu().numpy()
        hf_attn_eager = cap_eager["attn"][0, target_row].float().cpu().numpy()

    o_prod_t = o_prod[target_row].float().cpu().numpy()
    o_fp32_t = o_fp32[target_row].float().cpu().numpy()
    o_sc_t = o_sc[target_row].float().cpu().numpy()

    # How close is our manual rope+attention to HF sdpa attn output?
    results["layer0"] = {
        "manual_prod_vs_hf_sdpa_attn": rel_l2(o_prod_t, hf_attn_sdpa),
        "manual_fp32prob_vs_hf_sdpa_attn": rel_l2(o_fp32_t, hf_attn_sdpa),
        "manual_scorebf16_vs_hf_sdpa_attn": rel_l2(o_sc_t, hf_attn_sdpa),
        "manual_prod_vs_hf_eager_attn": rel_l2(o_prod_t, hf_attn_eager),
        "hf_eager_vs_sdpa_attn": rel_l2(hf_attn_eager, hf_attn_sdpa),
        "manual_prod_vs_fp32prob": rel_l2(o_prod_t, o_fp32_t),
        "measured_heph_layer0_attn_out_rel": 0.0017399141797795892,
        "q_tgt_l2": float(np.linalg.norm(q_tgt)),
        "k_all_l2": float(np.linalg.norm(k_all)),
        "v_all_l2": float(np.linalg.norm(v_all)),
    }

    # Residual after layer 0 (attn residual + ffn) — bound from HF self-spread
    h0s = h0_sdpa[0, target_row].float().cpu().numpy()
    h0e = h0_eager[0, target_row].float().cpu().numpy()
    results["layer0"]["hf_eager_vs_sdpa_residual"] = rel_l2(h0e, h0s)

    # Optional Hephaestus QKV dumps (bf16 as f32)
    heph_q = Path("/tmp/spike_l0_q.f32")
    heph_k = Path("/tmp/spike_l0_k.f32")
    heph_v = Path("/tmp/spike_l0_v.f32")
    heph_attn = Path("/tmp/spike_l0_attn.f32")
    if heph_q.exists() and heph_k.exists() and heph_v.exists():
        hq = np.fromfile(heph_q, dtype=np.float32).reshape(len(ids), N_HEADS, HEAD_DIM)
        hk = np.fromfile(heph_k, dtype=np.float32).reshape(len(ids), N_KV, HEAD_DIM)
        hv = np.fromfile(heph_v, dtype=np.float32).reshape(len(ids), N_KV, HEAD_DIM)
        results["heph_vs_hf_qkv"] = {
            "q_tgt_rel": rel_l2(hq[target_row], q_tgt.reshape(N_HEADS, HEAD_DIM)),
            "k_all_rel": rel_l2(hk, k_all),
            "v_all_rel": rel_l2(hv, v_all),
        }
        if heph_attn.exists():
            ha = np.fromfile(heph_attn, dtype=np.float32)
            # o_proj space [hidden] or [heads*dim]
            if ha.size == HIDDEN:
                results["heph_vs_hf_qkv"]["attn_o_proj_rel_sdpa"] = rel_l2(ha, hf_attn_sdpa)
            elif ha.size == N_HEADS * HEAD_DIM:
                results["heph_vs_hf_qkv"]["attn_heads_rel"] = float(
                    np.linalg.norm(ha - o_prod_t.reshape(-1))
                    / (np.linalg.norm(o_prod_t) + 1e-30)
                )
        qkv_seed = max(
            results["heph_vs_hf_qkv"]["q_tgt_rel"],
            results["heph_vs_hf_qkv"]["k_all_rel"],
            results["heph_vs_hf_qkv"]["v_all_rel"],
        )
        results["predictions"]["A_qkv_diverges_at_seed_scale"] = bool(
            qkv_seed >= 0.001
        )
        if "attn_o_proj_rel_sdpa" in results["heph_vs_hf_qkv"]:
            results["predictions"]["B_attention_kernel_is_seed"] = bool(
                results["heph_vs_hf_qkv"]["attn_o_proj_rel_sdpa"] > 3 * qkv_seed
            )
    else:
        results["heph_vs_hf_qkv"] = {
            "status": "missing",
            "note": "Run probe11_dump_l0.mojo to populate /tmp/spike_l0_{q,k,v,attn}.f32",
        }

    # Prediction C: HF attn-impl variation at layer 0 is an upper-ish bound on
    # "attention algorithm alone" when QKV are identical.
    results["predictions"]["C_hf_attn_impl_bound_at_layer0"] = {
        "eager_vs_sdpa_attn_rel": results["layer0"]["hf_eager_vs_sdpa_attn"],
        "eager_vs_sdpa_residual_rel": results["layer0"]["hf_eager_vs_sdpa_residual"],
        "interpretation": (
            "If Hephaestus layer-0 attn_out rel error (~0.00174) is near this "
            "HF self-spread, the seed is consistent with attention-impl variation; "
            "if much larger, something else (matmul/QKV/RoPE) dominates the seed."
        ),
    }

    # Compare measured heph seed to HF self-spread
    heph_seed = results["layer0"]["measured_heph_layer0_attn_out_rel"]
    hf_spread = results["layer0"]["hf_eager_vs_sdpa_attn"]
    results["seed_comparison"] = {
        "heph_layer0_attn_out_rel": heph_seed,
        "hf_eager_sdpa_attn_rel": hf_spread,
        "heph_over_hf_spread": heph_seed / (hf_spread + 1e-30),
        "manual_prod_closer_to_eager_than_sdpa": (
            results["layer0"]["manual_prod_vs_hf_eager_attn"]
            < results["layer0"]["manual_prod_vs_hf_sdpa_attn"]
        ),
    }

    out_path = OUT / "probe11_layer0_seed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
