# ===----------------------------------------------------------------------=== #
# Hephaestus — Constants for Qwen3-4B-Instruct-2507
# All values sourced from docs/architecture-dossier.md §7
# ===----------------------------------------------------------------------=== #

from std.math import sqrt

comptime VOCAB_SIZE = 151936
comptime HIDDEN_SIZE = 2560
comptime NUM_LAYERS = 36
comptime NUM_HEADS = 32
comptime NUM_KV_HEADS = 8
comptime HEAD_DIM = 128
comptime GQA_GROUP = 4  # NUM_HEADS / NUM_KV_HEADS
comptime INTERMEDIATE_SIZE = 9728
comptime RMS_NORM_EPS = 1e-6
comptime ROPE_THETA = 5_000_000.0
comptime MAX_POSITION = 262144
comptime BOS_TOKEN_ID = 151643
comptime EOS_TOKEN_ID = 151645
comptime TIE_EMBEDDINGS = True
comptime ATTENTION_BIAS = False

# Derived projection sizes
comptime Q_PROJ_OUT = NUM_HEADS * HEAD_DIM       # 4096
comptime K_PROJ_OUT = NUM_KV_HEADS * HEAD_DIM    # 1024
comptime V_PROJ_OUT = NUM_KV_HEADS * HEAD_DIM    # 1024
comptime O_PROJ_OUT = HIDDEN_SIZE                 # 2560 (input is Q_PROJ_OUT)

# Scale factor for attention
comptime SCALE_FACTOR = 1.0 / sqrt(Float64(HEAD_DIM))

# BF16 element size
comptime BF16_BYTES = 2
