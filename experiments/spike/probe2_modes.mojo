# DIAGNOSTIC ONLY. Runs Hephaestus at the exact spike target (prompt 1,
# teacher-forced step 67, which predicts token 15678 "lazy") by three different
# routes, and dumps the residual stream at that row after every cut point.
#
#   full    265-token prefill (prompt + oracle[:255]) -- the known-anomalous run.
#           Target row = 76.  Prefill path (matmul_kernel_naive).
#   prefix  77-token prefill  (prompt + oracle[:67])  -- exact prefix, nothing
#           after the target in the sequence. Target row = 76. Prefill path.
#   seq     10-token prompt prefill, then 67 one-token KV-cached decode steps
#           feeding oracle[0..66]. Target = last step. DECODE path (gemv_gpu).
#
# All three must produce the same logits for that row if the model is correct:
# causal attention makes row 76 independent of anything after it, and a decode
# step with a warm cache is the same math as the corresponding prefill row.
#
# Writes <prefix>_logits.f32   [151936] float32  target row logits
#        <prefix>_hidden.f32   [146, 2560] float32  residual-stream snapshots
#
# Usage: spike_modes <mode> <prompt_ids.txt> <oracle_ids.txt> <weights_prefix> <out_prefix>

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
comptime STEP = 67  # the teacher-forced step under investigation


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def write_f32(path: String, buf: List[Float32]) raises:
    var f = open(path, "w")
    f.write_bytes(
        Span[Byte, origin_of(buf)](
            ptr=buf.unsafe_ptr().bitcast[Byte](), length=len(buf) * 4
        )
    )
    f.close()


def main() raises:
    var mode = String(argv()[1])
    var prompt = read_ids(String(argv()[2]))
    var oracle = read_ids(String(argv()[3]))
    var wprefix = String(argv()[4])
    var out = String(argv()[5])

    var plen = len(prompt)
    comptime SLOTS = n_slots(NUM_LAYERS)

    var ctx = DeviceContext()
    var arena = load_arena(ctx, wprefix)
    verify_manifest[
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ](arena.entries, arena.index)
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=VOCAB_SIZE,
        hidden=HIDDEN_SIZE,
        q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT,
        head_dim=HEAD_DIM,
        inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
    ](base_ptr, arena)

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, MAX_SEQ)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var dump = ctx.enqueue_create_buffer[BF16](SLOTS * HIDDEN_SIZE)

    # Row of the *final* forward call whose logits predict oracle[STEP].
    var logit_row: Int

    if mode == "seq":
        # ---- prompt prefill, then STEP one-token decode steps ---------------
        var dev = ctx.enqueue_create_buffer[DType.int32](plen)
        with dev.map_to_host() as h:
            for i in range(plen):
                h[i] = prompt[i]
        forward_dump[
            vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
            kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
            n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
            theta=ROPE_THETA,
        ](weights, acts, cache, dev, plen, dump, -1, ctx)

        var one = ctx.enqueue_create_buffer[DType.int32](1)
        for s in range(STEP):
            with one.map_to_host() as h:
                h[0] = oracle[s]
            # Dump only on the final step -- that is the row under test.
            var drow = 0 if s == STEP - 1 else -1
            forward_dump[
                vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
                kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
                n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
                theta=ROPE_THETA,
            ](weights, acts, cache, one, 1, dump, drow, ctx)
        logit_row = 0
        print("mode seq: cache length =", cache.length, "(expect", plen + STEP, ")")
    else:
        # ---- one prefill of the whole sequence ------------------------------
        var n_gen = 255 if mode == "full" else STEP
        var ids = List[Int32]()
        for i in range(plen):
            ids.append(prompt[i])
        for i in range(n_gen):
            ids.append(oracle[i])
        var slen = len(ids)
        var target = plen - 1 + STEP

        var dev = ctx.enqueue_create_buffer[DType.int32](slen)
        with dev.map_to_host() as h:
            for i in range(slen):
                h[i] = ids[i]
        forward_dump[
            vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, q_out=Q_PROJ_OUT,
            kv_out=K_PROJ_OUT, head_dim=HEAD_DIM, inter=INTERMEDIATE_SIZE,
            n_layers=NUM_LAYERS, n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
            theta=ROPE_THETA,
        ](weights, acts, cache, dev, slen, dump, target, ctx)
        logit_row = target
        print("mode", mode, ": seq =", slen, " target row =", target)

    ctx.synchronize()

    # ---- write the target logit row and the hidden snapshots ---------------
    var lg = List[Float32]()
    var best = 0
    var best_val = Float32(-3.4e38)
    with acts.logits.map_to_host() as h:
        var base = logit_row * VOCAB_SIZE
        for i in range(VOCAB_SIZE):
            var val = h[base + i]
            lg.append(val)
            var r = val.cast[DType.bfloat16]().cast[DType.float32]()
            if r > best_val:
                best_val = r
                best = i
    write_f32(out + "_logits.f32", lg)

    var hd = List[Float32]()
    with dump.map_to_host() as h:
        for i in range(SLOTS * HIDDEN_SIZE):
            hd.append(h[i].cast[DType.float32]())
    write_f32(out + "_hidden.f32", hd)

    print("argmax =", best, " (oracle step", STEP, "=", oracle[STEP], ")")
    print("logit[96874] =", lg[96874])
    print("wrote", out + "_logits.f32", "and", out + "_hidden.f32", "slots =", SLOTS)
