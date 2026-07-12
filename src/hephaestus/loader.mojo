# ===----------------------------------------------------------------------=== #
# Hephaestus -- Safetensors Loader (staged format)
#
# Loads the output of scripts/stage_weights.py:
#   <prefix>.weights  flat binary blob (all tensors concatenated)
#   <prefix>.offsets  text manifest: name\toffset\tsize\tshape\tdtype
#
# Pipeline (Exp 2): HostBuffer <- file read, DeviceBuffer <- enqueue_copy.
# Single device arena for all weights (DECISIONS.md 2026-07-11).
# ===----------------------------------------------------------------------=== #

from std.collections import Dict
from std.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import Span

from hephaestus.model import LayerOffsets, Qwen3Weights


@fieldwise_init
struct TensorEntry(Copyable, Movable):
    var name: String
    var byte_offset: Int
    var byte_size: Int
    var shape: List[Int]
    var dtype: String


@fieldwise_init
struct WeightArena(Movable):
    """The single device buffer holding every weight tensor, plus the manifest.
    """

    var buf: DeviceBuffer[DType.bfloat16]
    var entries: List[TensorEntry]
    var index: Dict[String, Int]
    var total_bytes: Int


def parse_offsets(path: String) raises -> List[TensorEntry]:
    var entries = List[TensorEntry]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        if len(fields) != 5:
            raise Error("malformed offsets line: " + String(line))
        var shape = List[Int]()
        for dim in fields[3].split(","):
            shape.append(Int(String(dim)))
        entries.append(
            TensorEntry(
                name=String(fields[0]),
                byte_offset=Int(String(fields[1])),
                byte_size=Int(String(fields[2])),
                shape=shape^,
                dtype=String(fields[4]),
            )
        )
    return entries^


def _check_2d(
    entries: List[TensorEntry],
    index: Dict[String, Int],
    name: String,
    rows: Int,
    cols: Int,
) raises:
    if name not in index:
        raise Error("missing tensor: " + name)
    ref e = entries[index[name]]
    if len(e.shape) != 2 or e.shape[0] != rows or e.shape[1] != cols:
        raise Error("shape mismatch for " + name)


def _check_1d(
    entries: List[TensorEntry],
    index: Dict[String, Int],
    name: String,
    dim: Int,
) raises:
    if name not in index:
        raise Error("missing tensor: " + name)
    ref e = entries[index[name]]
    if len(e.shape) != 1 or e.shape[0] != dim:
        raise Error("shape mismatch for " + name)


def verify_manifest[
    vocab: Int,
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
    n_layers: Int,
](entries: List[TensorEntry], index: Dict[String, Int]) raises:
    """Assertions A1-A11 from the loader spec, parameterized by model dims."""
    # A10: tensor count (lm_head dropped at staging; tied embeddings)
    if len(entries) != 2 + 11 * n_layers:
        raise Error(
            "A10 tensor count: expected "
            + String(2 + 11 * n_layers)
            + ", got "
            + String(len(entries))
        )
    # A2: tied embeddings, no lm_head
    if "lm_head.weight" in index:
        raise Error("A2: lm_head.weight present; expected tied embeddings")
    # A1 + A11: dtype and size = product(shape) * 2, contiguous offsets
    var expected_offset = 0
    for e in entries:
        if e.dtype != "BF16":
            raise Error("A1 dtype: " + e.name + " is " + e.dtype)
        var elems = 1
        for d in e.shape:
            elems *= d
        if e.byte_size != elems * 2:
            raise Error("A11 size mismatch: " + e.name)
        if e.byte_offset != expected_offset:
            raise Error("offset not contiguous at " + e.name)
        expected_offset += e.byte_size
    # A3-A9: per-tensor shapes
    _check_2d(entries, index, "model.embed_tokens.weight", vocab, hidden)
    _check_1d(entries, index, "model.norm.weight", hidden)
    for i in range(n_layers):
        var p = "model.layers." + String(i) + "."
        _check_1d(entries, index, p + "input_layernorm.weight", hidden)
        _check_2d(entries, index, p + "self_attn.q_proj.weight", q_out, hidden)
        _check_2d(entries, index, p + "self_attn.k_proj.weight", kv_out, hidden)
        _check_2d(entries, index, p + "self_attn.v_proj.weight", kv_out, hidden)
        _check_2d(entries, index, p + "self_attn.o_proj.weight", hidden, q_out)
        _check_1d(entries, index, p + "self_attn.q_norm.weight", head_dim)
        _check_1d(entries, index, p + "self_attn.k_norm.weight", head_dim)
        _check_1d(entries, index, p + "post_attention_layernorm.weight", hidden)
        _check_2d(entries, index, p + "mlp.gate_proj.weight", inter, hidden)
        _check_2d(entries, index, p + "mlp.up_proj.weight", inter, hidden)
        _check_2d(entries, index, p + "mlp.down_proj.weight", hidden, inter)


def load_arena(ctx: DeviceContext, prefix: String) raises -> WeightArena:
    """Reads the staged blob into a HostBuffer, copies to one DeviceBuffer."""
    var entries = parse_offsets(prefix + ".offsets")
    var index = Dict[String, Int]()
    var total_bytes = 0
    for i in range(len(entries)):
        index[entries[i].name] = i
        total_bytes += entries[i].byte_size
    if total_bytes % 2 != 0:
        raise Error("total byte size not BF16-aligned")
    var total_elems = total_bytes // 2

    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](total_elems)
    ctx.synchronize()

    # Bulk read. POSIX read caps at ~2GB per call, so loop.
    var f = open(prefix + ".weights", "r")
    var byte_ptr = host.unsafe_ptr().bitcast[Scalar[DType.uint8]]()
    var done = 0
    while done < total_bytes:
        var n = f.read(
            Span[Scalar[DType.uint8]](
                ptr=byte_ptr + done, length=total_bytes - done
            )
        )
        if n <= 0:
            raise Error(
                ".weights truncated: expected "
                + String(total_bytes)
                + " bytes, got "
                + String(done)
            )
        done += n
    f.close()

    var dev = ctx.enqueue_create_buffer[DType.bfloat16](total_elems)
    dev.enqueue_copy_from(host)
    ctx.synchronize()

    # Round-trip check: copy device memory back and compare against the host
    # buffer we filled from disk. Full compare for small models, sampled
    # (first/last 64B + 4096 strided bytes) for large ones. GPU copies fail
    # silently; this is the only proof the weights actually landed.
    var back = ctx.enqueue_create_host_buffer[DType.bfloat16](total_elems)
    dev.enqueue_copy_to(back)
    ctx.synchronize()
    var back_ptr = back.unsafe_ptr().bitcast[Scalar[DType.uint8]]()
    if total_bytes <= 16 * 1024 * 1024:
        for i in range(total_bytes):
            if byte_ptr[i] != back_ptr[i]:
                raise Error("round-trip mismatch at byte " + String(i))
    else:
        for i in range(64):
            if byte_ptr[i] != back_ptr[i]:
                raise Error("round-trip mismatch at byte " + String(i))
        for i in range(total_bytes - 64, total_bytes):
            if byte_ptr[i] != back_ptr[i]:
                raise Error("round-trip mismatch at byte " + String(i))
        var stride = total_bytes // 4096
        for k in range(4096):
            var i = k * stride
            if byte_ptr[i] != back_ptr[i]:
                raise Error("round-trip mismatch at byte " + String(i))

    return WeightArena(
        buf=dev^, entries=entries^, index=index^, total_bytes=total_bytes
    )


def _element_offset(arena: WeightArena, name: String) raises -> Int:
    if name not in arena.index:
        raise Error("missing tensor: " + name)
    ref e = arena.entries[arena.index[name]]
    if e.byte_offset % 2 != 0:
        raise Error("unaligned BF16 offset for " + name)
    return e.byte_offset // 2


def build_weights[
    origin: Origin[mut=True],
    vocab: Int,
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
    n_layers: Int,
](
    base_ptr: UnsafePointer[Scalar[DType.bfloat16], origin],
    arena: WeightArena,
) raises -> Qwen3Weights[origin, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers]:
    var layer_offsets = List[LayerOffsets]()
    for i in range(n_layers):
        var p = "model.layers." + String(i) + "."
        layer_offsets.append(
            LayerOffsets(
                attn_norm=_element_offset(arena, p + "input_layernorm.weight"),
                q_proj=_element_offset(arena, p + "self_attn.q_proj.weight"),
                k_proj=_element_offset(arena, p + "self_attn.k_proj.weight"),
                v_proj=_element_offset(arena, p + "self_attn.v_proj.weight"),
                o_proj=_element_offset(arena, p + "self_attn.o_proj.weight"),
                q_norm=_element_offset(arena, p + "self_attn.q_norm.weight"),
                k_norm=_element_offset(arena, p + "self_attn.k_norm.weight"),
                ffn_norm=_element_offset(
                    arena, p + "post_attention_layernorm.weight"
                ),
                gate_proj=_element_offset(arena, p + "mlp.gate_proj.weight"),
                up_proj=_element_offset(arena, p + "mlp.up_proj.weight"),
                down_proj=_element_offset(arena, p + "mlp.down_proj.weight"),
            )
        )
    return Qwen3Weights[origin, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers](
        base_ptr,
        off_embed=_element_offset(arena, "model.embed_tokens.weight"),
        off_output_norm=_element_offset(arena, "model.norm.weight"),
        layer_offsets=layer_offsets,
    )
