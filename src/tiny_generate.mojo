# Tiny-model greedy decode: prefill + 16 decode steps, KV cache reused.
# Success criterion 2: token-exact vs fixtures/tiny_random/reference_outputs.json
#
# Usage: pixi run mojo run -I ~/projects/modular/max/kernels/src -I src \
#            src/tiny_generate.mojo

from std.gpu.host import DeviceContext

from hephaestus.forward import Activations, KVCache, forward
from hephaestus.loader import build_weights, load_arena, verify_manifest
from hephaestus.model import Qwen3Weights

comptime VOCAB = 256
comptime HIDDEN = 128
comptime N_HEADS = 4
comptime N_KV_HEADS = 2
comptime HEAD_DIM = 32
comptime Q_OUT = N_HEADS * HEAD_DIM
comptime KV_OUT = N_KV_HEADS * HEAD_DIM
comptime INTER = 256
comptime LAYERS = 2
comptime THETA = 10000.0
comptime N_NEW = 16


def prompt_ids(idx: Int) -> List[Int32]:
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


def expected_new(idx: Int) -> List[Int32]:
    # From fixtures/tiny_random/reference_outputs.json (new tokens only).
    var e = List[Int32]()
    if idx == 1:
        for _ in range(6):
            e.append(3)
        for _ in range(10):
            e.append(30)
    elif idx == 2:
        for _ in range(16):
            e.append(50)
    else:
        for _ in range(16):
            e.append(10)
    return e^


def generate(
    which: Int,
    ctx: DeviceContext,
    weights: Qwen3Weights[
        _, VOCAB, HIDDEN, Q_OUT, KV_OUT, HEAD_DIM, INTER, LAYERS
    ],
    mut acts: Activations[HIDDEN, Q_OUT, KV_OUT, INTER, VOCAB],
) raises -> List[Int32]:
    var ids = prompt_ids(which)
    var seq = len(ids)
    var cache = KVCache[LAYERS, KV_OUT](ctx)

    var dev_ids = ctx.enqueue_create_buffer[DType.int32](max(seq, 1))
    with dev_ids.map_to_host() as h:
        for i in range(seq):
            h[i] = ids[i]

    var generated = List[Int32]()
    var cur_len = seq

    for step in range(N_NEW):
        var n = seq if step == 0 else 1
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
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()

        # Greedy argmax over the last position's logits.
        var best = Int32(0)
        var best_val = Float32(-3.4e38)
        with acts.logits.map_to_host() as h:
            var base = (n - 1) * VOCAB
            for i in range(VOCAB):
                var val = h[base + i].cast[DType.float32]()
                if val > best_val:
                    best_val = val
                    best = Int32(i)
        generated.append(best)
        cur_len += 1

        # Feed the sampled token back as the next single-token input.
        with dev_ids.map_to_host() as h:
            h[0] = best

    return generated^


def main() raises:
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

    var all_pass = True
    for which in range(1, 4):
        var got = generate(which, ctx, weights, acts)
        var want = expected_new(which)
        var ok = True
        for i in range(N_NEW):
            if got[i] != want[i]:
                ok = False
        print("prompt", which, ":", "PASS" if ok else "FAIL")
        if not ok:
            all_pass = False
            var g = String("")
            var w = String("")
            for i in range(N_NEW):
                g += String(got[i]) + " "
                w += String(want[i]) + " "
            print("  got: ", g)
            print("  want:", w)
    if all_pass:
        print("TINY DECODE: token-exact on 3 prompts x 16 tokens")
    else:
        raise Error("tiny decode mismatch")
