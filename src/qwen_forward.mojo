# Qwen3-4B forward pass: prefill prompt N, dump last-position logits.
#
# Success criterion 3 (.agent/specs/2026-07-12_forward-pass.md): argmax must
# match fixtures/oracle/promptN_logits_step0.npy.
#
# This is the shape that exercises the non-square-projection trap: q_out=4096
# != hidden=2560. Tiny cannot catch it (its dims are square by accident).
#
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/qwen_forward.mojo <prompt 1..3> <out.f32> <ids.txt>
# ids.txt: one token id per line (written by scripts/dump_prompt_ids.py).

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
from hephaestus.forward import Activations, KVCache, forward
from hephaestus.loader import build_weights, load_arena, verify_manifest


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def main() raises:
    var ids_path = String("prompt_ids.txt")
    var out_path = String("qwen_logits.f32")
    if len(argv()) > 1:
        ids_path = String(argv()[1])
    if len(argv()) > 2:
        out_path = String(argv()[2])

    var ids = read_ids(ids_path)
    var seq = len(ids)

    var ctx = DeviceContext()
    var arena = load_arena(ctx, "staged/qwen3-4b")
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
    ](ctx, 32)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    forward[
        vocab=VOCAB_SIZE,
        hidden=HIDDEN_SIZE,
        q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT,
        head_dim=HEAD_DIM,
        inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        theta=ROPE_THETA,
    ](weights, acts, cache, dev_ids, seq, ctx)
    ctx.synchronize()

    var f = open(out_path, "w")
    var best = 0
    var best_val = Float32(-3.4e38)
    with acts.logits.map_to_host() as h:
        var base = (seq - 1) * VOCAB_SIZE
        var buf = List[Float32]()
        for i in range(VOCAB_SIZE):
            var val = h[base + i]
            buf.append(val)
            var rounded = val.cast[DType.bfloat16]().cast[DType.float32]()
            if rounded > best_val:
                best_val = rounded
                best = i
        f.write_bytes(
            Span[Byte, origin_of(buf)](
                ptr=buf.unsafe_ptr().bitcast[Byte](), length=VOCAB_SIZE * 4
            )
        )
    f.close()
    print("seq", seq, "-> argmax", best, "logit", best_val)
    print("wrote", out_path)
