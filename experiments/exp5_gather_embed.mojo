# Experiment 5 (spec open question 4): embedding lookup.
#
# nn/gather_scatter.mojo ships an ONNX-semantics gather. Embedding lookup is
# gather(axis=0) over embed_tokens [vocab, hidden] with int32 token ids
# [seq] -> [seq, hidden]. Test it at the tiny model's shape against a host
# reference. If this works, no embedding kernel needs writing (vendor, don't
# write).

from std.gpu.host import DeviceContext
from std.random import random_float64, seed

from layout import TileTensor
from layout.tile_layout import row_major
from nn.gather_scatter import gather

comptime VOCAB = 256
comptime HIDDEN = 128
comptime SEQ = 5


def main() raises:
    seed(11)
    var ctx = DeviceContext()

    var dev_embed = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB * HIDDEN)
    var dev_ids = ctx.enqueue_create_buffer[DType.int32](SEQ)
    var dev_out = ctx.enqueue_create_buffer[DType.bfloat16](SEQ * HIDDEN)

    # Prompt 3 from the tiny fixture manifest: [100, 200, 255, 5, 10] —
    # includes id 255 (last row) to catch off-by-one at the vocab edge.
    var ids = InlineArray[Int32, SEQ](fill=0)
    ids[0] = 100
    ids[1] = 200
    ids[2] = 255
    ids[3] = 5
    ids[4] = 10
    var host_embed = List[Float32]()
    with dev_embed.map_to_host() as he, dev_ids.map_to_host() as hi:
        for i in range(VOCAB * HIDDEN):
            var v = random_float64(-1.0, 1.0).cast[DType.bfloat16]()
            he[i] = v
            host_embed.append(v.cast[DType.float32]())
        for i in range(SEQ):
            hi[i] = ids[i]

    ctx.enqueue_memset(dev_out, 0)
    gather[axis=0, target="gpu"](
        TileTensor(dev_out, row_major[SEQ, HIDDEN]()),
        TileTensor(dev_embed, row_major[VOCAB, HIDDEN]()),
        TileTensor(dev_ids, row_major[SEQ]()),
        context=ctx,
    )
    ctx.synchronize()

    var errors = 0
    with dev_out.map_to_host() as ho:
        for s in range(SEQ):
            var row = Int(ids[s])
            for h in range(HIDDEN):
                var expected = host_embed[row * HIDDEN + h]
                var actual = ho[s * HIDDEN + h].cast[DType.float32]()
                if actual != expected:
                    errors += 1
    print("gather embed [5,128] from [256,128] -> errors:", errors, "/", SEQ * HIDDEN)
    if errors != 0:
        raise Error("gather mismatch")
    print("EXP5 PASS: nn.gather does embedding lookup on gfx1201 (bit-exact)")
