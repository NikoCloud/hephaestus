# ===----------------------------------------------------------------------=== #
# Hephaestus -- Model Structs for Qwen3 (dense)
# TileTensor fields with compile-time layouts, parameterized by Origin and
# model dimensions. Dims are comptime parameters so the tiny_random debug
# model and the 4B model instantiate the same struct (Exp 1 pattern).
# ===----------------------------------------------------------------------=== #

from layout import TileTensor
from layout.tile_layout import row_major


@fieldwise_init
struct LayerOffsets(Copyable, Movable):
    """Element offsets (BF16 elements, not bytes) of one layer's tensors
    within the weight arena. Plain value type: keeps origin-carrying
    references out of constructor argument lists (exclusivity checker)."""

    var attn_norm: Int
    var q_proj: Int
    var k_proj: Int
    var v_proj: Int
    var o_proj: Int
    var q_norm: Int
    var k_norm: Int
    var ffn_norm: Int
    var gate_proj: Int
    var up_proj: Int
    var down_proj: Int


struct Qwen3Layer[
    origin: Origin[mut=True],
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
](Copyable, Movable):
    comptime NormLayout = type_of(row_major[Self.hidden]())
    comptime QProjLayout = type_of(row_major[Self.q_out, Self.hidden]())
    comptime KVProjLayout = type_of(row_major[Self.kv_out, Self.hidden]())
    comptime OProjLayout = type_of(row_major[Self.hidden, Self.q_out]())
    comptime QKNormLayout = type_of(row_major[Self.head_dim]())
    comptime GateUpLayout = type_of(row_major[Self.inter, Self.hidden]())
    comptime DownProjLayout = type_of(row_major[Self.hidden, Self.inter]())

    var attn_norm: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.NormLayout, origin = Self.origin]
    var q_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.QProjLayout, origin = Self.origin]
    var k_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.KVProjLayout, origin = Self.origin]
    var v_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.KVProjLayout, origin = Self.origin]
    var o_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.OProjLayout, origin = Self.origin]
    var q_norm: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.QKNormLayout, origin = Self.origin]
    var k_norm: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.QKNormLayout, origin = Self.origin]
    var ffn_norm: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.NormLayout, origin = Self.origin]
    var gate_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.GateUpLayout, origin = Self.origin]
    var up_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.GateUpLayout, origin = Self.origin]
    var down_proj: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.DownProjLayout, origin = Self.origin]

    def __init__(
        out self,
        base_ptr: UnsafePointer[Scalar[DType.bfloat16], Self.origin],
        offs: LayerOffsets,
    ):
        self.attn_norm = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.NormLayout, origin = Self.origin](
            ptr=base_ptr + offs.attn_norm, layout=row_major[Self.hidden]()
        )
        self.q_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.QProjLayout, origin = Self.origin](
            ptr=base_ptr + offs.q_proj, layout=row_major[Self.q_out, Self.hidden]()
        )
        self.k_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.KVProjLayout, origin = Self.origin](
            ptr=base_ptr + offs.k_proj, layout=row_major[Self.kv_out, Self.hidden]()
        )
        self.v_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.KVProjLayout, origin = Self.origin](
            ptr=base_ptr + offs.v_proj, layout=row_major[Self.kv_out, Self.hidden]()
        )
        self.o_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.OProjLayout, origin = Self.origin](
            ptr=base_ptr + offs.o_proj, layout=row_major[Self.hidden, Self.q_out]()
        )
        self.q_norm = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.QKNormLayout, origin = Self.origin](
            ptr=base_ptr + offs.q_norm, layout=row_major[Self.head_dim]()
        )
        self.k_norm = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.QKNormLayout, origin = Self.origin](
            ptr=base_ptr + offs.k_norm, layout=row_major[Self.head_dim]()
        )
        self.ffn_norm = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.NormLayout, origin = Self.origin](
            ptr=base_ptr + offs.ffn_norm, layout=row_major[Self.hidden]()
        )
        self.gate_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.GateUpLayout, origin = Self.origin](
            ptr=base_ptr + offs.gate_proj, layout=row_major[Self.inter, Self.hidden]()
        )
        self.up_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.GateUpLayout, origin = Self.origin](
            ptr=base_ptr + offs.up_proj, layout=row_major[Self.inter, Self.hidden]()
        )
        self.down_proj = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.DownProjLayout, origin = Self.origin](
            ptr=base_ptr + offs.down_proj, layout=row_major[Self.hidden, Self.inter]()
        )


struct Qwen3Weights[
    origin: Origin[mut=True],
    vocab: Int,
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
    n_layers: Int,
](Movable):
    comptime EmbedLayout = type_of(row_major[Self.vocab, Self.hidden]())
    comptime NormLayout = type_of(row_major[Self.hidden]())
    comptime LayerType = Qwen3Layer[
        Self.origin, Self.hidden, Self.q_out, Self.kv_out, Self.head_dim, Self.inter
    ]

    var embed_tokens: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.EmbedLayout, origin = Self.origin]
    var output_norm: TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.NormLayout, origin = Self.origin]
    var layers: List[Self.LayerType]
    # Tied embeddings (A2): lm_head is embed_tokens; same device pointer, no
    # separate field. The forward pass reuses embed_tokens for the LM head.

    def __init__(
        out self,
        base_ptr: UnsafePointer[Scalar[DType.bfloat16], Self.origin],
        off_embed: Int,
        off_output_norm: Int,
        layer_offsets: List[LayerOffsets],
    ) raises:
        if len(layer_offsets) != Self.n_layers:
            raise Error("layer offset count != n_layers")
        self.embed_tokens = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.EmbedLayout, origin = Self.origin](
            ptr=base_ptr + off_embed, layout=row_major[Self.vocab, Self.hidden]()
        )
        self.output_norm = TileTensor[mut=True, dtype = DType.bfloat16, LayoutType = Self.NormLayout, origin = Self.origin](
            ptr=base_ptr + off_output_norm, layout=row_major[Self.hidden]()
        )
        self.layers = List[Self.LayerType]()
        for i in range(Self.n_layers):
            self.layers.append(Self.LayerType(base_ptr, layer_offsets[i]))
