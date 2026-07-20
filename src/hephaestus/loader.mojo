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
from hephaestus.model_fp8 import LayerOffsetsFP8, Qwen3WeightsFP8


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


comptime LOAD_CHUNK_ELEMS = 64 * 1024 * 1024  # 128MB (BF16) per chunk


def load_arena(ctx: DeviceContext, prefix: String) raises -> WeightArena:
    """Streams the staged blob to the device arena in fixed-size chunks.

    Previously: one ~8GB HostBuffer staged the whole file, one ~8GB
    DeviceBuffer was the arena, and a SECOND ~8GB HostBuffer held a full
    round-trip copy-back for verification -- three ~8GB buffers alive at
    once, measured peak VRAM ~29.6GB for the 4B model (bench/1a-ab.md
    Finding 3). Chunking means only one small (128MB) host stage buffer and
    one small (128MB) host verify buffer are ever alive, reused across
    chunks -- peak VRAM is now the arena itself plus a few hundred MB, not
    the arena times three. This also makes the verification STRICTLY
    stronger: every chunk is checked byte-for-byte (the old large-model path
    only sampled 4096 strided bytes + the first/last 64), not weaker.
    """
    var entries = parse_offsets(prefix + ".offsets")
    var index = Dict[String, Int]()
    var total_bytes = 0
    for i in range(len(entries)):
        index[entries[i].name] = i
        total_bytes += entries[i].byte_size
    if total_bytes % 2 != 0:
        raise Error("total byte size not BF16-aligned")
    var total_elems = total_bytes // 2

    var dev = ctx.enqueue_create_buffer[DType.bfloat16](total_elems)
    var chunk_elems = min(LOAD_CHUNK_ELEMS, total_elems)
    if chunk_elems == 0:
        chunk_elems = 1
    var host_chunk = ctx.enqueue_create_host_buffer[DType.bfloat16](chunk_elems)
    var back_chunk = ctx.enqueue_create_host_buffer[DType.bfloat16](chunk_elems)
    ctx.synchronize()

    var f = open(prefix + ".weights", "r")
    var dev_ptr = dev.unsafe_ptr()
    var elems_done = 0
    while elems_done < total_elems:
        var this_chunk_elems = min(chunk_elems, total_elems - elems_done)
        var chunk_bytes = this_chunk_elems * 2

        # Bulk read this chunk. POSIX read caps at ~2GB per call, well above
        # any chunk size used here, so one read suffices per chunk -- but
        # loop defensively in case the OS returns a short read.
        var byte_ptr = host_chunk.unsafe_ptr().bitcast[Scalar[DType.uint8]]()
        var chunk_read = 0
        while chunk_read < chunk_bytes:
            var n = f.read(
                Span[Scalar[DType.uint8]](
                    ptr=byte_ptr + chunk_read, length=chunk_bytes - chunk_read
                )
            )
            if n <= 0:
                raise Error(
                    ".weights truncated: expected "
                    + String(total_bytes)
                    + " bytes, got "
                    + String(elems_done * 2 + chunk_read)
                )
            chunk_read += n

        ctx.enqueue_copy(
            dev_ptr + elems_done, host_chunk.unsafe_ptr(), this_chunk_elems
        )
        ctx.synchronize()

        # Round-trip check: copy this chunk straight back and compare every
        # byte against what was just staged. GPU copies fail silently; this
        # is the only proof each chunk actually landed.
        ctx.enqueue_copy(
            back_chunk.unsafe_ptr(), dev_ptr + elems_done, this_chunk_elems
        )
        ctx.synchronize()
        var back_byte_ptr = back_chunk.unsafe_ptr().bitcast[Scalar[DType.uint8]]()
        for i in range(chunk_bytes):
            if byte_ptr[i] != back_byte_ptr[i]:
                raise Error(
                    "round-trip mismatch at byte "
                    + String(elems_done * 2 + i)
                )

        elems_done += this_chunk_elems
    f.close()

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


# ===----------------------------------------------------------------------=== #
# Mixed-dtype arena (FP8 E4M3 weights + F32 scales + BF16 norms)
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct WeightArenaBytes(Movable):
    """Byte-addressed device arena for mixed-dtype staged blobs."""

    var buf: DeviceBuffer[DType.uint8]
    var entries: List[TensorEntry]
    var index: Dict[String, Int]
    var total_bytes: Int


comptime LOAD_CHUNK_BYTES = 128 * 1024 * 1024  # 128MB


def _dtype_width(dtype: String) raises -> Int:
    if dtype == "BF16":
        return 2
    if dtype == "F32":
        return 4
    if dtype == "F8_E4M3" or dtype == "F8E4M3":
        return 1
    raise Error("unknown dtype tag: " + dtype)


def load_arena_bytes(ctx: DeviceContext, prefix: String) raises -> WeightArenaBytes:
    """Load mixed-dtype staged blob into a uint8 device arena.

    After upload, FP8 projection weights (all F8_E4M3 tensors except
    embed_tokens) are rearranged into WMMA-fragment order so the decode
    B-load is a single coalesced 8-byte/lane read. Embed stays row-major
    for gather-based embedding lookup; lm_head uses the unswizzled path.
    """
    var entries = parse_offsets(prefix + ".offsets")
    var index = Dict[String, Int]()
    var total_bytes = 0
    for i in range(len(entries)):
        index[entries[i].name] = i
        total_bytes += entries[i].byte_size

    var dev = ctx.enqueue_create_buffer[DType.uint8](total_bytes)
    var chunk = min(LOAD_CHUNK_BYTES, total_bytes)
    if chunk == 0:
        chunk = 1
    var host_chunk = ctx.enqueue_create_host_buffer[DType.uint8](chunk)
    var back_chunk = ctx.enqueue_create_host_buffer[DType.uint8](chunk)
    ctx.synchronize()

    var f = open(prefix + ".weights", "r")
    var dev_ptr = dev.unsafe_ptr()
    var done = 0
    while done < total_bytes:
        var this_chunk = min(chunk, total_bytes - done)
        var byte_ptr = host_chunk.unsafe_ptr()
        var chunk_read = 0
        while chunk_read < this_chunk:
            var n = f.read(
                Span[Scalar[DType.uint8]](
                    ptr=byte_ptr + chunk_read, length=this_chunk - chunk_read
                )
            )
            if n <= 0:
                raise Error("weights truncated at byte " + String(done + chunk_read))
            chunk_read += n
        ctx.enqueue_copy(dev_ptr + done, host_chunk.unsafe_ptr(), this_chunk)
        ctx.synchronize()
        ctx.enqueue_copy(back_chunk.unsafe_ptr(), dev_ptr + done, this_chunk)
        ctx.synchronize()
        var back = back_chunk.unsafe_ptr()
        for i in range(this_chunk):
            if byte_ptr[i] != back[i]:
                raise Error("round-trip mismatch at byte " + String(done + i))
        done += this_chunk
    f.close()

    var arena = WeightArenaBytes(
        buf=dev^, entries=entries^, index=index^, total_bytes=total_bytes
    )
    # Set False to keep row-major weights (debug / A-B vs swizzled).
    comptime DO_FP8_WMMA_SWIZZLE = False
    comptime if DO_FP8_WMMA_SWIZZLE:
        swizzle_fp8_projection_weights(ctx, arena)
    return arena^


# ===----------------------------------------------------------------------=== #
# WMMA-fragment-order swizzle for FP8 projection weights
#
# B-fragment mapping (G1b-0):
#   b[j] = W[(NB + l%16)*K + ks + (l/16)*8 + j]
# Row-major stores force 16 scattered row reads. Swizzled layout packs each
# 16×16 tile so lane l reads bytes [l*8 .. l*8+7] contiguously:
#   swizzled[n_tile*(K/16)*256 + k_tile*256 + l*8 + j] = that B element
# ===----------------------------------------------------------------------=== #


def _swizzle_fp8_matrix_host(
    dst: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    N: Int,
    K: Int,
) raises:
    """Host: row-major [N,K] FP8 → fragment-order [N/16, K/16, 256]."""
    if N % 16 != 0 or K % 16 != 0:
        raise Error("swizzle_fp8: N and K must be multiples of 16")
    var n_tiles = N // 16
    var k_tiles = K // 16
    for nt in range(n_tiles):
        for kt in range(k_tiles):
            var tile_base = (nt * k_tiles + kt) * 256
            for lane in range(32):
                var row = lane % 16
                var half = lane // 16
                for j in range(8):
                    var col = half * 8 + j
                    var src_idx = (nt * 16 + row) * K + kt * 16 + col
                    dst[tile_base + lane * 8 + j] = src[src_idx]


def swizzle_fp8_projection_weights(
    ctx: DeviceContext, mut arena: WeightArenaBytes
) raises:
    """In-place swizzle of every F8_E4M3 2D weight except embed_tokens.

    embed_tokens stays row-major for gather embedding; lm_head uses the
    unswizzled B-load path. All other FP8 mats are WMMA B operands.
    """
    # Max F8 tensor we touch (exclude embed which can be ~389MB).
    var max_bytes = 0
    for e in arena.entries:
        if e.dtype != "F8_E4M3" and e.dtype != "F8E4M3":
            continue
        if e.name == "model.embed_tokens.weight":
            continue
        if e.byte_size > max_bytes:
            max_bytes = e.byte_size
    if max_bytes == 0:
        return

    var host_src = ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
    var host_dst = ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
    ctx.synchronize()
    var n_swizzled = 0
    for e in arena.entries:
        if e.dtype != "F8_E4M3" and e.dtype != "F8E4M3":
            continue
        if e.name == "model.embed_tokens.weight":
            continue
        if len(e.shape) != 2:
            raise Error("FP8 weight not rank-2: " + e.name)
        var N = e.shape[0]
        var K = e.shape[1]
        if N % 16 != 0 or K % 16 != 0:
            raise Error(
                "FP8 weight dims not tile-aligned: "
                + e.name
                + " "
                + String(N)
                + "x"
                + String(K)
            )
        if e.byte_size != N * K:
            raise Error("FP8 byte_size mismatch: " + e.name)

        # Device → host (row-major)
        ctx.enqueue_copy(
            host_src.unsafe_ptr(),
            arena.buf.unsafe_ptr() + e.byte_offset,
            e.byte_size,
        )
        ctx.synchronize()
        # HostBuffer.ptr is mutable; cast origins for the swizzle helper.
        var dst_ptr = host_dst.unsafe_ptr().as_unsafe_any_origin()
        var src_ptr = host_src.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        _swizzle_fp8_matrix_host(dst_ptr, src_ptr, N, K)
        # Host → device (fragment order)
        ctx.enqueue_copy(
            arena.buf.unsafe_ptr() + e.byte_offset,
            host_dst.unsafe_ptr(),
            e.byte_size,
        )
        ctx.synchronize()
        n_swizzled += 1
    print("swizzled", n_swizzled, "FP8 projection tensors into WMMA-fragment order")


def _byte_offset(arena: WeightArenaBytes, name: String) raises -> Int:
    if name not in arena.index:
        raise Error("missing tensor: " + name)
    return arena.entries[arena.index[name]].byte_offset


def _require(
    arena: WeightArenaBytes,
    name: String,
    dtype: String,
    rank: Int,
    d0: Int,
    d1: Int = -1,
) raises:
    if name not in arena.index:
        raise Error("missing tensor: " + name)
    ref e = arena.entries[arena.index[name]]
    if e.dtype != dtype:
        raise Error(
            "dtype mismatch " + name + ": got " + e.dtype + " want " + dtype
        )
    if len(e.shape) != rank:
        raise Error("rank mismatch " + name)
    if e.shape[0] != d0:
        raise Error("shape[0] mismatch " + name)
    if rank == 2 and e.shape[1] != d1:
        raise Error("shape[1] mismatch " + name)
    var elems = 1
    for d in e.shape:
        elems *= d
    var width = _dtype_width(dtype)
    if e.byte_size != elems * width:
        raise Error("byte_size mismatch " + name)


def verify_manifest_fp8[
    vocab: Int,
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
    n_layers: Int,
](entries: List[TensorEntry], index: Dict[String, Int]) raises:
    """FP8 checkpoint: 253 F8 weights + 253 F32 scales + 145 BF16 norms (4B)."""
    # Expected: 1 embed + 1 embed_scale + 1 norm + n_layers * (7 fp8 + 7 scale + 4 bf16)
    # = 3 + n_layers * 18. For n_layers=36: 3+648=651.
    var expected = 3 + 18 * n_layers
    if len(entries) != expected:
        raise Error(
            "FP8 tensor count: expected "
            + String(expected)
            + ", got "
            + String(len(entries))
        )
    if "lm_head.weight" in index:
        raise Error("A2: lm_head.weight present; expected tied embeddings")
    # Contiguous byte offsets + per-tensor size
    var expected_offset = 0
    for e in entries:
        var width = _dtype_width(e.dtype)
        var elems = 1
        for d in e.shape:
            elems *= d
        if e.byte_size != elems * width:
            raise Error("size mismatch: " + e.name)
        if e.byte_offset != expected_offset:
            raise Error("offset not contiguous at " + e.name)
        expected_offset += e.byte_size
    # Pairing: every F8 weight has matching _scale
    for e in entries:
        if e.dtype == "F8_E4M3" or e.dtype == "F8E4M3":
            var sn = e.name + "_scale"
            if sn not in index:
                raise Error("missing scale for " + e.name)
            ref s = entries[index[sn]]
            if s.dtype != "F32":
                raise Error("scale not F32: " + sn)
            if len(s.shape) != 2 or s.shape[0] != e.shape[0] or s.shape[1] != 1:
                raise Error("scale shape bad for " + sn)


def build_weights_fp8[
    origin: Origin[mut=True],
    vocab: Int,
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
    n_layers: Int,
](
    base_ptr: UnsafePointer[Scalar[DType.uint8], origin],
    arena: WeightArenaBytes,
) raises -> Qwen3WeightsFP8[
    origin, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers
]:
    verify_manifest_fp8[
        vocab, hidden, q_out, kv_out, head_dim, inter, n_layers
    ](arena.entries, arena.index)

    _require(arena, "model.embed_tokens.weight", "F8_E4M3", 2, vocab, hidden)
    _require(arena, "model.embed_tokens.weight_scale", "F32", 2, vocab, 1)
    _require(arena, "model.norm.weight", "BF16", 1, hidden)

    var layer_offsets = List[LayerOffsetsFP8]()
    for i in range(n_layers):
        var p = "model.layers." + String(i) + "."
        _require(arena, p + "input_layernorm.weight", "BF16", 1, hidden)
        _require(arena, p + "self_attn.q_proj.weight", "F8_E4M3", 2, q_out, hidden)
        _require(arena, p + "self_attn.q_proj.weight_scale", "F32", 2, q_out, 1)
        _require(arena, p + "self_attn.k_proj.weight", "F8_E4M3", 2, kv_out, hidden)
        _require(arena, p + "self_attn.k_proj.weight_scale", "F32", 2, kv_out, 1)
        _require(arena, p + "self_attn.v_proj.weight", "F8_E4M3", 2, kv_out, hidden)
        _require(arena, p + "self_attn.v_proj.weight_scale", "F32", 2, kv_out, 1)
        _require(arena, p + "self_attn.o_proj.weight", "F8_E4M3", 2, hidden, q_out)
        _require(arena, p + "self_attn.o_proj.weight_scale", "F32", 2, hidden, 1)
        _require(arena, p + "self_attn.q_norm.weight", "BF16", 1, head_dim)
        _require(arena, p + "self_attn.k_norm.weight", "BF16", 1, head_dim)
        _require(arena, p + "post_attention_layernorm.weight", "BF16", 1, hidden)
        _require(arena, p + "mlp.gate_proj.weight", "F8_E4M3", 2, inter, hidden)
        _require(arena, p + "mlp.gate_proj.weight_scale", "F32", 2, inter, 1)
        _require(arena, p + "mlp.up_proj.weight", "F8_E4M3", 2, inter, hidden)
        _require(arena, p + "mlp.up_proj.weight_scale", "F32", 2, inter, 1)
        _require(arena, p + "mlp.down_proj.weight", "F8_E4M3", 2, hidden, inter)
        _require(arena, p + "mlp.down_proj.weight_scale", "F32", 2, hidden, 1)
        layer_offsets.append(
            LayerOffsetsFP8(
                attn_norm=_byte_offset(arena, p + "input_layernorm.weight"),
                q_proj=_byte_offset(arena, p + "self_attn.q_proj.weight"),
                q_proj_scale=_byte_offset(arena, p + "self_attn.q_proj.weight_scale"),
                k_proj=_byte_offset(arena, p + "self_attn.k_proj.weight"),
                k_proj_scale=_byte_offset(arena, p + "self_attn.k_proj.weight_scale"),
                v_proj=_byte_offset(arena, p + "self_attn.v_proj.weight"),
                v_proj_scale=_byte_offset(arena, p + "self_attn.v_proj.weight_scale"),
                o_proj=_byte_offset(arena, p + "self_attn.o_proj.weight"),
                o_proj_scale=_byte_offset(arena, p + "self_attn.o_proj.weight_scale"),
                q_norm=_byte_offset(arena, p + "self_attn.q_norm.weight"),
                k_norm=_byte_offset(arena, p + "self_attn.k_norm.weight"),
                ffn_norm=_byte_offset(arena, p + "post_attention_layernorm.weight"),
                gate_proj=_byte_offset(arena, p + "mlp.gate_proj.weight"),
                gate_proj_scale=_byte_offset(arena, p + "mlp.gate_proj.weight_scale"),
                up_proj=_byte_offset(arena, p + "mlp.up_proj.weight"),
                up_proj_scale=_byte_offset(arena, p + "mlp.up_proj.weight_scale"),
                down_proj=_byte_offset(arena, p + "mlp.down_proj.weight"),
                down_proj_scale=_byte_offset(arena, p + "mlp.down_proj.weight_scale"),
            )
        )
    return Qwen3WeightsFP8[
        origin, vocab, hidden, q_out, kv_out, head_dim, inter, n_layers
    ](
        base_ptr,
        off_embed=_byte_offset(arena, "model.embed_tokens.weight"),
        off_embed_scale=_byte_offset(arena, "model.embed_tokens.weight_scale"),
        off_output_norm=_byte_offset(arena, "model.norm.weight"),
        layer_offsets=layer_offsets,
    )
