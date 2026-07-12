# ===----------------------------------------------------------------------=== #
# Hephaestus -- Model Structs for Qwen3-4B
# TileTensor fields with compile-time layouts, parameterized by Origin.
# Pattern proven by Experiment 1 (type_erasure test, 2026-07-11).
# ===----------------------------------------------------------------------=== #

from layout.tile_layout import row_major
from layout import TileTensor

# Type aliases for each weight layout
alias EmbedLayout = type_of(row_major[VOCAB_SIZE, HIDDEN_SIZE]())
alias NormLayout = type_of(row_major[HIDDEN_SIZE]())
alias QProjLayout = type_of(row_major[Q_PROJ_OUT, HIDDEN_SIZE]())
alias KVProjLayout = type_of(row_major[K_PROJ_OUT, HIDDEN_SIZE]())
alias OProjLayout = type_of(row_major[HIDDEN_SIZE, Q_PROJ_OUT]())
alias QKNormLayout = type_of(row_major[HEAD_DIM]())
alias GateUpLayout = type_of(row_major[INTERMEDIATE_SIZE, HIDDEN_SIZE]())
alias DownProjLayout = type_of(row_major[HIDDEN_SIZE, INTERMEDIATE_SIZE]())

struct Qwen3Layer[origin: Origin[mut=True]]:
    var attn_norm: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=NormLayout, origin=Self.origin]
    var q_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=QProjLayout, origin=Self.origin]
    var k_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=KVProjLayout, origin=Self.origin]
    var v_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=KVProjLayout, origin=Self.origin]
    var o_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=OProjLayout, origin=Self.origin]
    var q_norm: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=QKNormLayout, origin=Self.origin]
    var k_norm: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=QKNormLayout, origin=Self.origin]
    var ffn_norm: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=NormLayout, origin=Self.origin]
    var gate_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=GateUpLayout, origin=Self.origin]
    var up_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=GateUpLayout, origin=Self.origin]
    var down_proj: TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=DownProjLayout, origin=Self.origin]

    def __init__(
        out self,
        base_ptr: UnsafePointer[Scalar[DType.bfloat16], Self.origin],
        off_attn_norm: Int,
        off_q_proj: Int,
        off_k_proj: Int,
        off_v_proj: Int,
        off_o_proj: Int,
        off_q_norm: Int,
        off_k_norm: Int,
        off_ffn_norm: Int,
        off_gate_proj: Int,
        off_up_proj: Int,
        off_down_proj: Int,
    ):
        self.attn_norm = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=NormLayout, origin=Self.origin](
            ptr=base_ptr + off_attn_norm, layout=row_major[HIDDEN_SIZE]()
        )
        self.q_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=QProjLayout, origin=Self.origin](
            ptr=base_ptr + off_q_proj, layout=row_major[Q_PROJ_OUT, HIDDEN_SIZE]()
        )
        self.k_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=KVProjLayout, origin=Self.origin](
            ptr=base_ptr + off_k_proj, layout=row_major[K_PROJ_OUT, HIDDEN_SIZE]()
        )
        self.v_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=KVProjLayout, origin=Self.origin](
            ptr=base_ptr + off_v_proj, layout=row_major[V_PROJ_OUT, HIDDEN_SIZE]()
        )
        self.o_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=OProjLayout, origin=Self.origin](
            ptr=base_ptr + off_o_proj, layout=row_major[HIDDEN_SIZE, Q_PROJ_OUT]()
        )
        self.q_norm = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=QKNormLayout, origin=Self.origin](
            ptr=base_ptr + off_q_norm, layout=row_major[HEAD_DIM]()
        )
        self.k_norm = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=QKNormLayout, origin=Self.origin](
            ptr=base_ptr + off_k_norm, layout=row_major[HEAD_DIM]()
        )
        self.ffn_norm = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=NormLayout, origin=Self.origin](
            ptr=base_ptr + off_ffn_norm, layout=row_major[HIDDEN_SIZE]()
        )
        self.gate_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=GateUpLayout, origin=Self.origin](
            ptr=base_ptr + off_gate_proj, layout=row_major[INTERMEDIATE_SIZE, HIDDEN_SIZE]()
        )
        self.up_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=GateUpLayout, origin=Self.origin](
            ptr=base_ptr + off_up_proj, layout=row_major[INTERMEDIATE_SIZE, HIDDEN_SIZE]()
        )
        self.down_proj = TileTensor[mut=True, dtype=DType.bfloat16, LayoutType=DownProjLayout, origin=Self.origin](
            ptr=base_ptr + off_down_proj, layout=row_major[HIDDEN_SIZE, INTERMEDIATE_SIZE]()
        )
