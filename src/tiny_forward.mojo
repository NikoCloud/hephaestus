# Tiny-model forward pass: prefill prompt N, dump last-position logits.
#
# Success criterion 1 from .agent/specs/2026-07-12_forward-pass.md:
#   argmax(logits) must equal fixtures/tiny_random/oracle/promptN_logits_step0.npy
#
# Usage:
#   pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#       src/tiny_forward.mojo <prompt_index 1..3> <out.f32>
# Writes raw float32 logits [vocab] to <out.f32> for scripts/diff_logits.py.

from std.gpu.host import DeviceContext
from std.sys import argv

from hephaestus.forward import Activations, KVCache, forward
from hephaestus.loader import build_weights, load_arena, verify_manifest

comptime VOCAB = 256
comptime HIDDEN = 128
comptime N_HEADS = 4
comptime N_KV_HEADS = 2
comptime HEAD_DIM = 32
comptime Q_OUT = N_HEADS * HEAD_DIM  # 128
comptime KV_OUT = N_KV_HEADS * HEAD_DIM  # 64
comptime INTER = 256
comptime LAYERS = 2
comptime THETA = 10000.0  # tiny config.json; the 4B uses 5e6


def prompt_ids(idx: Int) raises -> List[Int32]:
    var ids = List[Int32]()
    if idx == 1:
        ids.append(0)
        ids.append(1)
        ids.append(2)
        ids.append(3)
    elif idx == 2:
        ids.append(10)
        ids.append(20)
        ids.append(30)
        ids.append(40)
        ids.append(50)
    else:
        ids.append(100)
        ids.append(200)
        ids.append(255)
        ids.append(5)
        ids.append(10)
    return ids^


def main() raises:
    var which = 1
    var out_path = String("tiny_logits.f32")
    if len(argv()) > 1:
        which = Int(String(argv()[1]))
    if len(argv()) > 2:
        out_path = String(argv()[2])

    var ids = prompt_ids(which)
    var seq = len(ids)

    var ctx = DeviceContext()
    var arena = load_arena(ctx, "staged/tiny")
    verify_manifest[VOCAB, HIDDEN, Q_OUT, KV_OUT, HEAD_DIM, INTER, LAYERS](
        arena.entries, arena.index
    )
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights[
        vocab=VOCAB,
        hidden=HIDDEN,
        q_out=Q_OUT,
        kv_out=KV_OUT,
        head_dim=HEAD_DIM,
        inter=INTER,
        n_layers=LAYERS,
    ](base_ptr, arena)

    var acts = Activations[HIDDEN, Q_OUT, KV_OUT, INTER, VOCAB](ctx, 32)
    var cache = KVCache[LAYERS, KV_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](seq)
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    forward[
        vocab=VOCAB,
        hidden=HIDDEN,
        q_out=Q_OUT,
        kv_out=KV_OUT,
        head_dim=HEAD_DIM,
        inter=INTER,
        n_layers=LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        theta=THETA,
    ](weights, acts, cache, dev_ids, seq, ctx)
    ctx.synchronize()

    # Last position's logits -> float32 file.
    var f = open(out_path, "w")
    var best = 0
    var best_val = Float32(-3.4e38)
    with acts.logits.map_to_host() as h:
        var base = (seq - 1) * VOCAB
        var buf = List[Float32]()
        for i in range(VOCAB):
            var val = h[base + i].cast[DType.float32]()
            buf.append(val)
            if val > best_val:
                best_val = val
                best = i
        f.write_bytes(
            Span[Byte, origin_of(buf)](
                ptr=buf.unsafe_ptr().bitcast[Byte](), length=VOCAB * 4
            )
        )
    f.close()
    print("prompt", which, "seq", seq, "-> argmax", best, "logit", best_val)
    print("wrote", out_path)
