# PROBE 9 -- CAUSAL INTERVENTION on Hephaestus's attention rounding.
#
# Production attention makes one arithmetic choice the sdpa oracle does not:
# it rounds the softmax probabilities to bf16 before the PV product
# (kernels.mojo:262), mimicking HF *eager*. That is a bf16-magnitude perturbation
# injected at every layer -- the single largest known systematic difference from
# the reference.
#
# Runs the exact-prefix route (77 tokens; probe 2 proved it is bit-identical to
# the full 265-token prefill at this row) with three attention configs:
#
#   A prob_bf16=T score_bf16=F   PRODUCTION. Must reproduce 16.312086 exactly --
#                                 this is the control that proves spike_kernels
#                                 is a faithful copy and not a rewrite.
#   B prob_bf16=F score_bf16=F   probs kept in fp32 -- CLOSER to sdpa.
#   C prob_bf16=T score_bf16=T   scores also rounded -- FURTHER from sdpa.
#
# Predictions, stated before running:
#   If prob-rounding is the DEFECT: B collapses logit[96874] toward HF's 4.25.
#   If the row is CHAOTIC: B is just a different bf16-sized draw. It moves the
#     logit a lot, does NOT converge on 4.25, and argmax stays 15678. C should
#     scatter it too, in an uncorrelated direction.
#
# Usage: spike_interv <prompt_ids.txt> <oracle_ids.txt> <weights_prefix> <out_prefix>

from std.gpu.host import DeviceContext
from std.sys import argv

from hephaestus.constants import (
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    K_PROJ_OUT,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_PROJ_OUT,
    ROPE_THETA,
    VOCAB_SIZE,
)
from hephaestus.forward import Activations, KVCache
from hephaestus.kernels import BF16
from hephaestus.loader import build_weights, load_arena, verify_manifest
from spike_forward import forward_dump, n_slots

comptime MAX_SEQ = 300
comptime STEP = 67
comptime TARGET_TOK = 96874


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def main() raises:
    var prompt = read_ids(String(argv()[1]))
    var oracle = read_ids(String(argv()[2]))
    var wprefix = String(argv()[3])
    var out = String(argv()[4])

    var plen = len(prompt)
    var target = plen - 1 + STEP
    comptime SLOTS = n_slots(NUM_LAYERS)

    var ctx = DeviceContext()
    var arena = load_arena(ctx, wprefix)
    verify_manifest[
        VOCAB_SIZE, HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, HEAD_DIM,
        INTERMEDIATE_SIZE, NUM_LAYERS,
    ](arena.entries, arena.index)
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
    ](base_ptr, arena)

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, MAX_SEQ)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var dump = ctx.enqueue_create_buffer[BF16](SLOTS * HIDDEN_SIZE)

    # exact-prefix input: prompt + oracle[:STEP]
    var ids = List[Int32]()
    for i in range(plen):
        ids.append(prompt[i])
    for i in range(STEP):
        ids.append(oracle[i])
    var slen = len(ids)
    var dev = ctx.enqueue_create_buffer[DType.int32](slen)
    with dev.map_to_host() as h:
        for i in range(slen):
            h[i] = ids[i]

    print("exact-prefix seq =", slen, " target row =", target)
    print("config                       argmax   logit[96874]   ||h_final||")

    @parameter
    def go[prob_bf16: Bool, score_bf16: Bool](tag: String) raises:
        cache.length = 0  # fresh prefill; cache_write overwrites 0..seq-1
        forward_dump[
            vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
            kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
            n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
            theta=ROPE_THETA, prob_bf16=prob_bf16, score_bf16=score_bf16,
        ](weights, acts, cache, dev, slen, dump, target, ctx)
        ctx.synchronize()

        var best = 0
        var best_val = Float32(-3.4e38)
        var tgt = Float32(0)
        var lg = List[Float32]()
        with acts.logits.map_to_host() as h:
            var base = target * VOCAB_SIZE
            for i in range(VOCAB_SIZE):
                var val = h[base + i]
                lg.append(val)
                var r = val.cast[DType.bfloat16]().cast[DType.float32]()
                if r > best_val:
                    best_val = r
                    best = i
            tgt = h[base + TARGET_TOK]

        var nrm = Float64(0)
        var hd = List[Float32]()
        with dump.map_to_host() as h:
            for i in range(SLOTS * HIDDEN_SIZE):
                hd.append(h[i].cast[DType.float32]())
            for i in range(HIDDEN_SIZE):
                var v = Float64(h[(SLOTS - 1) * HIDDEN_SIZE + i].cast[DType.float32]())
                nrm += v * v
        print(tag, best, "   ", tgt, "  ", nrm ** 0.5)

        var f = open(out + "_" + tag.strip() + "_logits.f32", "w")
        f.write_bytes(
            Span[Byte, origin_of(lg)](
                ptr=lg.unsafe_ptr().bitcast[Byte](), length=VOCAB_SIZE * 4
            )
        )
        f.close()
        var g = open(out + "_" + tag.strip() + "_hidden.f32", "w")
        g.write_bytes(
            Span[Byte, origin_of(hd)](
                ptr=hd.unsafe_ptr().bitcast[Byte](),
                length=SLOTS * HIDDEN_SIZE * 4,
            )
        )
        g.close()

    go[True, False]("A_prod       ")
    go[False, False]("B_probfp32   ")
    go[True, True]("C_scorebf16  ")

    print()
    print("HF sdpa reference: argmax 15678   logit[96874] = 4.25")
    print("A_prod must read 16.312086 exactly, or spike_kernels is not a faithful copy.")
