# ===----------------------------------------------------------------------=== #
# Qwen3 weights — FP8 E4M3 weights + F32 per-channel scales + BF16 norms
# Docs: docs/fp8-checkpoint-format.md
# ===----------------------------------------------------------------------=== #

from layout import TileTensor
from layout.tile_layout import row_major

comptime F8 = DType.float8_e4m3fn
comptime F32 = DType.float32
comptime BF16 = DType.bfloat16


@fieldwise_init
struct LayerOffsetsFP8(Copyable, Movable):
    """Byte offsets into the mixed-dtype arena for one layer."""

    var attn_norm: Int
    var q_proj: Int
    var q_proj_scale: Int
    var k_proj: Int
    var k_proj_scale: Int
    var v_proj: Int
    var v_proj_scale: Int
    var o_proj: Int
    var o_proj_scale: Int
    var q_norm: Int
    var k_norm: Int
    var ffn_norm: Int
    var gate_proj: Int
    var gate_proj_scale: Int
    var up_proj: Int
    var up_proj_scale: Int
    var down_proj: Int
    var down_proj_scale: Int


struct Qwen3LayerFP8[
    origin: Origin[mut=True],
    hidden: Int,
    q_out: Int,
    kv_out: Int,
    head_dim: Int,
    inter: Int,
](Copyable, Movable):
    comptime NormLayout = type_of(row_major[Self.hidden]())
    comptime QKNormLayout = type_of(row_major[Self.head_dim]())
    comptime QProjLayout = type_of(row_major[Self.q_out, Self.hidden]())
    comptime KVProjLayout = type_of(row_major[Self.kv_out, Self.hidden]())
    comptime OProjLayout = type_of(row_major[Self.hidden, Self.q_out]())
    comptime GateUpLayout = type_of(row_major[Self.inter, Self.hidden]())
    comptime DownProjLayout = type_of(row_major[Self.hidden, Self.inter]())
    comptime QScaleLayout = type_of(row_major[Self.q_out, 1]())
    comptime KVScaleLayout = type_of(row_major[Self.kv_out, 1]())
    comptime OScaleLayout = type_of(row_major[Self.hidden, 1]())
    comptime GateScaleLayout = type_of(row_major[Self.inter, 1]())
    comptime DownScaleLayout = type_of(row_major[Self.hidden, 1]())

    var attn_norm: TileTensor[
        mut=True, dtype=BF16, LayoutType = Self.NormLayout, origin = Self.origin
    ]
    var q_norm: TileTensor[
        mut=True, dtype=BF16, LayoutType = Self.QKNormLayout, origin = Self.origin
    ]
    var k_norm: TileTensor[
        mut=True, dtype=BF16, LayoutType = Self.QKNormLayout, origin = Self.origin
    ]
    var ffn_norm: TileTensor[
        mut=True, dtype=BF16, LayoutType = Self.NormLayout, origin = Self.origin
    ]

    var q_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.QProjLayout, origin = Self.origin
    ]
    var q_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.QScaleLayout, origin = Self.origin
    ]
    var k_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.KVProjLayout, origin = Self.origin
    ]
    var k_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.KVScaleLayout, origin = Self.origin
    ]
    var v_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.KVProjLayout, origin = Self.origin
    ]
    var v_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.KVScaleLayout, origin = Self.origin
    ]
    var o_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.OProjLayout, origin = Self.origin
    ]
    var o_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.OScaleLayout, origin = Self.origin
    ]
    var gate_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.GateUpLayout, origin = Self.origin
    ]
    var gate_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.GateScaleLayout, origin = Self.origin
    ]
    var up_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.GateUpLayout, origin = Self.origin
    ]
    var up_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.GateScaleLayout, origin = Self.origin
    ]
    var down_proj: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.DownProjLayout, origin = Self.origin
    ]
    var down_proj_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.DownScaleLayout, origin = Self.origin
    ]

    def __init__(
        out self,
        base: UnsafePointer[Scalar[DType.uint8], Self.origin],
        offs: LayerOffsetsFP8,
    ):
        self.attn_norm = TileTensor[
            mut=True, dtype=BF16, LayoutType = Self.NormLayout, origin = Self.origin
        ](
            ptr=(base + offs.attn_norm).bitcast[Scalar[BF16]](),
            layout=row_major[Self.hidden](),
        )
        self.q_norm = TileTensor[
            mut=True, dtype=BF16, LayoutType = Self.QKNormLayout, origin = Self.origin
        ](
            ptr=(base + offs.q_norm).bitcast[Scalar[BF16]](),
            layout=row_major[Self.head_dim](),
        )
        self.k_norm = TileTensor[
            mut=True, dtype=BF16, LayoutType = Self.QKNormLayout, origin = Self.origin
        ](
            ptr=(base + offs.k_norm).bitcast[Scalar[BF16]](),
            layout=row_major[Self.head_dim](),
        )
        self.ffn_norm = TileTensor[
            mut=True, dtype=BF16, LayoutType = Self.NormLayout, origin = Self.origin
        ](
            ptr=(base + offs.ffn_norm).bitcast[Scalar[BF16]](),
            layout=row_major[Self.hidden](),
        )
        self.q_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.QProjLayout, origin = Self.origin
        ](
            ptr=(base + offs.q_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.q_out, Self.hidden](),
        )
        self.q_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.QScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.q_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.q_out, 1](),
        )
        self.k_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.KVProjLayout, origin = Self.origin
        ](
            ptr=(base + offs.k_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.kv_out, Self.hidden](),
        )
        self.k_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.KVScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.k_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.kv_out, 1](),
        )
        self.v_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.KVProjLayout, origin = Self.origin
        ](
            ptr=(base + offs.v_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.kv_out, Self.hidden](),
        )
        self.v_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.KVScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.v_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.kv_out, 1](),
        )
        self.o_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.OProjLayout, origin = Self.origin
        ](
            ptr=(base + offs.o_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.hidden, Self.q_out](),
        )
        self.o_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.OScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.o_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.hidden, 1](),
        )
        self.gate_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.GateUpLayout, origin = Self.origin
        ](
            ptr=(base + offs.gate_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.inter, Self.hidden](),
        )
        self.gate_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.GateScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.gate_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.inter, 1](),
        )
        self.up_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.GateUpLayout, origin = Self.origin
        ](
            ptr=(base + offs.up_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.inter, Self.hidden](),
        )
        self.up_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.GateScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.up_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.inter, 1](),
        )
        self.down_proj = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.DownProjLayout, origin = Self.origin
        ](
            ptr=(base + offs.down_proj).bitcast[Scalar[F8]](),
            layout=row_major[Self.hidden, Self.inter](),
        )
        self.down_proj_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.DownScaleLayout, origin = Self.origin
        ](
            ptr=(base + offs.down_proj_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.hidden, 1](),
        )


struct Qwen3WeightsFP8[
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
    comptime EmbedScaleLayout = type_of(row_major[Self.vocab, 1]())
    comptime NormLayout = type_of(row_major[Self.hidden]())
    comptime LayerType = Qwen3LayerFP8[
        Self.origin, Self.hidden, Self.q_out, Self.kv_out, Self.head_dim, Self.inter
    ]

    var embed_tokens: TileTensor[
        mut=True, dtype=F8, LayoutType = Self.EmbedLayout, origin = Self.origin
    ]
    var embed_scale: TileTensor[
        mut=True, dtype=F32, LayoutType = Self.EmbedScaleLayout, origin = Self.origin
    ]
    var output_norm: TileTensor[
        mut=True, dtype=BF16, LayoutType = Self.NormLayout, origin = Self.origin
    ]
    var layers: List[Self.LayerType]

    def __init__(
        out self,
        base: UnsafePointer[Scalar[DType.uint8], Self.origin],
        off_embed: Int,
        off_embed_scale: Int,
        off_output_norm: Int,
        layer_offsets: List[LayerOffsetsFP8],
    ) raises:
        if len(layer_offsets) != Self.n_layers:
            raise Error("layer offset count != n_layers")
        self.embed_tokens = TileTensor[
            mut=True, dtype=F8, LayoutType = Self.EmbedLayout, origin = Self.origin
        ](
            ptr=(base + off_embed).bitcast[Scalar[F8]](),
            layout=row_major[Self.vocab, Self.hidden](),
        )
        self.embed_scale = TileTensor[
            mut=True, dtype=F32, LayoutType = Self.EmbedScaleLayout, origin = Self.origin
        ](
            ptr=(base + off_embed_scale).bitcast[Scalar[F32]](),
            layout=row_major[Self.vocab, 1](),
        )
        self.output_norm = TileTensor[
            mut=True, dtype=BF16, LayoutType = Self.NormLayout, origin = Self.origin
        ](
            ptr=(base + off_output_norm).bitcast[Scalar[BF16]](),
            layout=row_major[Self.hidden](),
        )
        self.layers = List[Self.LayerType]()
        for i in range(Self.n_layers):
            self.layers.append(Self.LayerType(base, layer_offsets[i]))
