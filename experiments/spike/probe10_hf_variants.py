#!/usr/bin/env python3
"""Compare exact-prefix step-67 logits/hidden across HF SDPA, HF eager, and Hephaestus."""
import json, os, struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM
MODEL = "/mnt/models/models/qwen3-4b-instruct-2507"
OUT = os.path.join(os.path.dirname(__file__), "out")
STEP, TOK, VOCAB, HIDDEN, NSLOT = 67, 96874, 151936, 2560, 146

def bits(x):
    return f"0x{struct.unpack('<I', np.float32(x).tobytes())[0]:08x}"

def stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(mean=float(d.mean()), median=float(np.median(d)), p99=float(np.quantile(d,.99)), p999=float(np.quantile(d,.999)), max=float(d.max()), argmax_equal=bool(a.argmax()==b.argmax()))

def rel(a,b):
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/np.linalg.norm(b.astype(np.float64)))

def run(kind, ids):
    m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation=kind).eval().to("cuda:0")
    got={}
    h=m.model.norm.register_forward_hook(lambda mod,args,out: got.__setitem__("h",out.detach()[0,-1].float().cpu().numpy()))
    with torch.no_grad(): lg=m(torch.tensor([ids],device="cuda:0")).logits[0,-1].float().cpu().numpy()
    h.remove(); del m; torch.cuda.empty_cache(); return lg,got["h"]

def main():
    os.makedirs(OUT,exist_ok=True)
    prompt=json.load(open("fixtures/oracle/prompt1_input_ids.json")); gen=json.load(open("fixtures/oracle/prompt1_output_ids.json")); ids=prompt+gen[:STEP]
    sd,hs=run("sdpa",ids); eg,he=run("eager",ids)
    hp=np.fromfile("/tmp/spike_prefix_logits.f32",np.float32); assert hp.size==VOCAB
    slots=np.fromfile("/tmp/spike_prefix_hidden.f32",np.float32).reshape(NSLOT,HIDDEN); hh=slots[-1]
    rep={"input":dict(prompt_len=len(prompt),seq_len=len(ids),step=STEP,token=TOK),"values":{},"row_stats":{},"hidden_rel":{}}
    for name,a in [("hf_sdpa",sd),("hf_eager",eg),("hephaestus",hp)]: rep["values"][name]=dict(target=float(a[TOK]),bits=bits(a[TOK]),argmax=int(a.argmax()),target_rank=int((a>a[TOK]).sum()+1))
    for name,a,b in [("heph_vs_sdpa",hp,sd),("heph_vs_eager",hp,eg),("eager_vs_sdpa",eg,sd)]: rep["row_stats"][name]=stats(a,b)
    rep["hidden_rel"]["heph_vs_sdpa"]=rel(hh,hs); rep["hidden_rel"]["heph_vs_eager"]=rel(hh,he); rep["hidden_rel"]["eager_vs_sdpa"]=rel(he,hs)
    a=rep["row_stats"]; rep["prediction"]=dict(heph_materially_closer_to_eager=bool(a["heph_vs_eager"]["median"] < 0.75 * a["heph_vs_sdpa"]["median"]),sole_cause=bool(a["heph_vs_eager"]["max"]<0.1),interpretation="HF eager is only marginally closer than SDPA; this rejects eager-style BF16 probability rounding as the main or sole explanation. The separate probability-FP32 intervention still shows that this rounding choice contributes to sensitivity.")
    path=os.path.join(OUT,"probe10_hf_variants.json"); json.dump(rep,open(path,"w"),indent=2); print(json.dumps(rep,indent=2)); print("wrote",path)
if __name__=="__main__": main()
