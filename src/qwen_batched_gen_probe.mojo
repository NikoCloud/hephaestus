# Fixed-M batched greedy generation — first end-to-end Multiplier numbers.
#
# Architecture (NOT M × forward_fp8):
#   - Prefill + decode: fused M-row GEMM (linear_fp8 / small-M or gemv)
#   - Attention: per-sequence KV, no cross-row (forward_fp8_batched_decode)
#   - One shared FP8 weight arena; per-seq KV slabs in BatchedKVCache
#   - Greedy GPU argmax; KV grows each step
#
# Usage (nightly WMMA env, GPU 0 only):
#   export CONDA_PREFIX=$HOME/projects/hephaestus-wmma-nightly/.pixi/envs/default
#   export PATH=$CONDA_PREFIX/bin:$PATH
#   KERNELS=$HOME/projects/modular/max/kernels/src
#   mojo build -I $KERNELS -I src src/qwen_batched_gen_probe.mojo -o /tmp/batched_gen
#   HIP_VISIBLE_DEVICES=0 /tmp/batched_gen

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

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
from hephaestus.forward import (
    Activations,
    BatchedKVCache,
    KVCache,
    forward_fp8,
    forward_fp8_batched_decode,
)
from hephaestus.kernels import argmax_logits
from hephaestus.loader import build_weights_fp8, load_arena_bytes
from hephaestus.model_fp8 import Qwen3WeightsFP8

comptime PROMPT_LEN = 512
comptime GEN_LEN = 128
comptime EXPECT_WEIGHT_BYTES = 4022272000
comptime WEIGHT_LOG = "/tmp/fp8_weight_bytes.log"
comptime WEIGHT_EN = "/tmp/fp8_weight_bytes_enable"
comptime MAX_BATCH = 16
comptime GATE2_STEPS = 128
# Short prefill for Gate 1 isolation (bit-exact), full length for thruput.
comptime G1_PREFILL = 8
comptime G1_DECODE = 4


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


def disable_weight_log() raises:
    var f = open(WEIGHT_EN, "w")
    f.write("0\n")
    f.close()


def enable_weight_log() raises:
    var f = open(WEIGHT_LOG, "w")
    f.close()
    var f2 = open(WEIGHT_EN, "w")
    f2.write("1\n")
    f2.close()


def sum_weight_bytes() raises -> Int:
    var total = 0
    var n = 0
    var text = open(WEIGHT_LOG, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var parts = String(line).split(" ")
        if len(parts) < 2:
            continue
        total += Int(parts[1])
        n += 1
    print("weight_launches=", n)
    return total


def make_prompt_row(
    base: List[Int32], stream: Int, length: Int
) raises -> List[Int32]:
    """Distinct prompt per stream: base with stream-dependent perturbation."""
    var out = List[Int32]()
    var n = length
    if n > len(base):
        n = len(base)
    for i in range(n):
        # Stream 0 = unmodified base. Others offset by stream*97, wrap vocab.
        if stream == 0:
            out.append(base[i])
        else:
            var t = (Int(base[i]) + stream * 97 + i * 13) % VOCAB_SIZE
            if t < 0:
                t = -t
            out.append(Int32(t))
    return out^


def copy_ids_to_dev(
    mut dev: DeviceBuffer[DType.int32],
    ids: List[Int32],
    n: Int,
) raises:
    with dev.map_to_host() as h:
        for i in range(n):
            h[i] = ids[i]


def set_batch_tokens(
    mut dev: DeviceBuffer[DType.int32],
    tokens: List[Int32],
    batch: Int,
) raises:
    with dev.map_to_host() as h:
        for i in range(batch):
            h[i] = tokens[i]


def prefill_batched(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    mut acts: Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ],
    mut caches: BatchedKVCache[NUM_LAYERS, K_PROJ_OUT],
    prompts: List[List[Int32]],
    batch: Int,
    prompt_len: Int,
    mut dev_ids: DeviceBuffer[DType.int32],
    ctx: DeviceContext,
) raises:
    """Fused M-row prefill: one position at a time, all sequences together."""
    caches.reset()
    caches.set_batch(batch)
    for t in range(prompt_len):
        with dev_ids.map_to_host() as h:
            for s in range(batch):
                h[s] = prompts[s][t]
        forward_fp8_batched_decode[
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
        ](weights, acts, caches, dev_ids, batch, ctx)
    ctx.synchronize()


def decode_step_batched(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    mut acts: Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ],
    mut caches: BatchedKVCache[NUM_LAYERS, K_PROJ_OUT],
    mut dev_ids: DeviceBuffer[DType.int32],
    batch: Int,
    ctx: DeviceContext,
) raises:
    forward_fp8_batched_decode[
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
    ](weights, acts, caches, dev_ids, batch, ctx)


def argmax_row(
    mut acts: Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ],
    row: Int,
    mut argmax_bf16: DeviceBuffer[DType.bfloat16],
    mut argmax_idx: DeviceBuffer[DType.int32],
    ctx: DeviceContext,
) raises -> Int32:
    var logits_base = (
        acts.logits.unsafe_ptr() + row * VOCAB_SIZE
    ).as_unsafe_any_origin()
    return argmax_logits(logits_base, argmax_bf16, argmax_idx, VOCAB_SIZE, ctx)


def logits_max_abs_diff_row0(
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    ctx: DeviceContext,
) raises -> Float32:
    ctx.synchronize()
    var mad = Float32(0)
    with a.map_to_host() as ha:
        with b.map_to_host() as hb:
            for i in range(VOCAB_SIZE):
                var d = ha[i] - hb[i]
                if d < 0:
                    d = -d
                if d > mad:
                    mad = d
    return mad


def run_gate1_isolation(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    base_prompt: List[Int32],
    ctx: DeviceContext,
) raises -> Bool:
    """Gate 1: row0 in M=8 vs row0 in M=2, both small-M, bit-exact logits."""
    print("=== Gate 1: isolation (bit-exact, same small-M kernel) ===")
    var acts8 = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(8, 16))
    var acts2 = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(2, 16))
    var c8 = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, 8)
    var c2 = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, 2)
    var dev8 = ctx.enqueue_create_buffer[DType.int32](8)
    var dev2 = ctx.enqueue_create_buffer[DType.int32](2)
    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    # Build prompts: row0 identical; others distinct.
    var p8 = List[List[Int32]]()
    var p2 = List[List[Int32]]()
    for s in range(8):
        p8.append(make_prompt_row(base_prompt, s, G1_PREFILL))
    for s in range(2):
        p2.append(make_prompt_row(base_prompt, s, G1_PREFILL))

    prefill_batched(weights, acts8, c8, p8, 8, G1_PREFILL, dev8, ctx)
    prefill_batched(weights, acts2, c2, p2, 2, G1_PREFILL, dev2, ctx)

    # Same next token for row0 (and distinct for others) — one decode step.
    var next8 = List[Int32]()
    var next2 = List[Int32]()
    for s in range(8):
        next8.append(Int32(1000 + s * 97))
    for s in range(2):
        next2.append(Int32(1000 + s * 97))
    # Force row0 identical.
    next8[0] = Int32(4242)
    next2[0] = Int32(4242)

    set_batch_tokens(dev8, next8, 8)
    set_batch_tokens(dev2, next2, 2)
    decode_step_batched(weights, acts8, c8, dev8, 8, ctx)
    decode_step_batched(weights, acts2, c2, dev2, 2, ctx)
    ctx.synchronize()

    # Compare row0 logits bit-exact.
    var mad = Float32(0)
    with acts8.logits.map_to_host() as h8:
        with acts2.logits.map_to_host() as h2:
            for i in range(VOCAB_SIZE):
                var d = h8[i] - h2[i]
                if d < 0:
                    d = -d
                if d > mad:
                    mad = d
    var a8 = argmax_row(acts8, 0, argmax_bf16, argmax_idx, ctx)
    var a2 = argmax_row(acts2, 0, argmax_bf16, argmax_idx, ctx)
    print("Gate1 max_abs_diff row0 logits (M=8 vs M=2) =", mad)
    print("Gate1 argmax row0 M8=", a8, " M2=", a2)
    if mad != 0:
        print("Gate1 FAIL: expected max_abs_diff=0")
        return False
    print("Gate1 PASS")
    return True


def run_gate2_teacher_forced(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    base_prompt: List[Int32],
    ctx: DeviceContext,
) raises -> Bool:
    """Gate 2: M=1 gemv vs batched M=2 small-M row0, teacher-forced, ≥95%."""
    print("=== Gate 2: kernel-switch teacher-forced (gemv vs small-M) ===")
    var prompt_len = PROMPT_LEN
    if len(base_prompt) < prompt_len:
        prompt_len = len(base_prompt)

    # Activations must cover full prefill width for single-seq forward_fp8.
    var max_seq_m1 = max(prompt_len, 16)
    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    # --- Reference trajectory: M=1 free-running greedy (TF anchor only) ---
    var acts_ref = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max_seq_m1)
    var cache_ref = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var dev_ref = ctx.enqueue_create_buffer[DType.int32](prompt_len)

    with dev_ref.map_to_host() as h:
        for i in range(prompt_len):
            h[i] = base_prompt[i]

    forward_fp8[
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
    ](weights, acts_ref, cache_ref, dev_ref, prompt_len, ctx)
    ctx.synchronize()

    var ref_tokens = List[Int32]()
    var cur = argmax_row(acts_ref, prompt_len - 1, argmax_bf16, argmax_idx, ctx)
    for _step in range(GATE2_STEPS):
        ref_tokens.append(cur)
        with dev_ref.map_to_host() as h:
            h[0] = cur
        forward_fp8[
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
        ](weights, acts_ref, cache_ref, dev_ref, 1, ctx)
        ctx.synchronize()
        cur = argmax_row(acts_ref, 0, argmax_bf16, argmax_idx, ctx)

    print("Gate2 reference trajectory built:", len(ref_tokens), "tokens")

    # --- Teacher-forced M=1 gemv path ---
    var acts1 = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max_seq_m1)
    var cache1 = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var dev1 = ctx.enqueue_create_buffer[DType.int32](prompt_len)
    with dev1.map_to_host() as h:
        for i in range(prompt_len):
            h[i] = base_prompt[i]
    forward_fp8[
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
    ](weights, acts1, cache1, dev1, prompt_len, ctx)
    ctx.synchronize()

    var argmax1 = List[Int32]()
    for step in range(GATE2_STEPS):
        var a: Int32
        if step == 0:
            a = argmax_row(acts1, prompt_len - 1, argmax_bf16, argmax_idx, ctx)
        else:
            a = argmax_row(acts1, 0, argmax_bf16, argmax_idx, ctx)
        argmax1.append(a)
        with dev1.map_to_host() as h:
            h[0] = ref_tokens[step]
        forward_fp8[
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
        ](weights, acts1, cache1, dev1, 1, ctx)
        ctx.synchronize()

    # --- Teacher-forced batched M=2 small-M, compare row0 ---
    # Batched prefill is token-at-a-time → acts only need width=batch.
    var acts_b = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(2, 16))
    var caches_b = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, 2)
    var dev_b = ctx.enqueue_create_buffer[DType.int32](2)
    var p2 = List[List[Int32]]()
    p2.append(make_prompt_row(base_prompt, 0, prompt_len))
    p2.append(make_prompt_row(base_prompt, 1, prompt_len))
    prefill_batched(weights, acts_b, caches_b, p2, 2, prompt_len, dev_b, ctx)

    var argmax_b = List[Int32]()
    # Post-prefill argmax is first prediction (same as M=1 step 0).
    var a0 = argmax_row(acts_b, 0, argmax_bf16, argmax_idx, ctx)
    argmax_b.append(a0)
    for step in range(GATE2_STEPS):
        with dev_b.map_to_host() as h:
            h[0] = ref_tokens[step]
            h[1] = Int32(1000 + step * 97)
        decode_step_batched(weights, acts_b, caches_b, dev_b, 2, ctx)
        ctx.synchronize()
        if step + 1 < GATE2_STEPS:
            var ab = argmax_row(acts_b, 0, argmax_bf16, argmax_idx, ctx)
            argmax_b.append(ab)

    # Compare per-step argmax (teacher-forced).
    # M=1: argmax1[step] is argmax before feeding ref_tokens[step]
    # Batched: argmax_b[0] after prefill; argmax_b[k] after feeding ref[k-1]
    # Align: both should predict token at each TF step against same history.
    var mismatches = 0
    var compared = 0
    # Compare step 0 (post-prefill) through GATE2_STEPS-1.
    var n_cmp = GATE2_STEPS
    if len(argmax_b) < n_cmp:
        n_cmp = len(argmax_b)
    if len(argmax1) < n_cmp:
        n_cmp = len(argmax1)
    for step in range(n_cmp):
        compared += 1
        if Int(argmax1[step]) != Int(argmax_b[step]):
            mismatches += 1
            if mismatches <= 16:
                print(
                    "Gate2 diverge step",
                    step,
                    "gemv=",
                    argmax1[step],
                    "smallm_row0=",
                    argmax_b[step],
                )
    var matches = compared - mismatches
    var rate = Float64(matches) / Float64(compared)
    print(
        "Gate2 teacher-forced:",
        matches,
        "/",
        compared,
        " match_rate=",
        rate,
        " mismatches=",
        mismatches,
    )
    if rate < 0.95:
        print("Gate2 FAIL: match_rate < 95%")
        return False
    print("Gate2 PASS (≥95%; aim 97.4% class)")
    return True


def run_gate3_tf_smoke(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    base_prompt: List[Int32],
    ctx: DeviceContext,
) raises -> Bool:
    """Gate 3: short teacher-forced smoke on batched path alone (self-consistent)."""
    print("=== Gate 3: TF smoke on batched path ===")
    # Self-consistency: two runs of M=2 TF with same tokens → bit-exact argmax.
    # Also reuses Gate2 rate as the real cross-kernel bar; here we check
    # forward/KV write path didn't explode (match free-running vs itself).
    var prompt_len = 32
    if len(base_prompt) < prompt_len:
        prompt_len = len(base_prompt)
    var gen = 32

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(2, 16))
    var caches = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, 2)
    var dev = ctx.enqueue_create_buffer[DType.int32](2)
    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    var p2 = List[List[Int32]]()
    p2.append(make_prompt_row(base_prompt, 0, prompt_len))
    p2.append(make_prompt_row(base_prompt, 1, prompt_len))
    prefill_batched(weights, acts, caches, p2, 2, prompt_len, dev, ctx)

    var tokens0 = List[Int32]()
    var tokens1 = List[Int32]()
    for step in range(gen):
        var a0 = argmax_row(acts, 0, argmax_bf16, argmax_idx, ctx)
        var a1 = argmax_row(acts, 1, argmax_bf16, argmax_idx, ctx)
        tokens0.append(a0)
        tokens1.append(a1)
        with dev.map_to_host() as h:
            h[0] = a0
            h[1] = a1
        decode_step_batched(weights, acts, caches, dev, 2, ctx)
        ctx.synchronize()

    # Replay TF with stored tokens — must match free-running argmax bit-exact
    # (same kernel, same path).
    var caches2 = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, 2)
    prefill_batched(weights, acts, caches2, p2, 2, prompt_len, dev, ctx)
    var mism = 0
    for step in range(gen):
        var a0 = argmax_row(acts, 0, argmax_bf16, argmax_idx, ctx)
        var a1 = argmax_row(acts, 1, argmax_bf16, argmax_idx, ctx)
        if Int(a0) != Int(tokens0[step]) or Int(a1) != Int(tokens1[step]):
            mism += 1
            if mism <= 8:
                print("Gate3 diverge step", step, a0, tokens0[step], a1, tokens1[step])
        with dev.map_to_host() as h:
            h[0] = tokens0[step]
            h[1] = tokens1[step]
        decode_step_batched(weights, acts, caches2, dev, 2, ctx)
        ctx.synchronize()

    var matches = gen - mism
    var rate = Float64(matches) / Float64(gen)
    print("Gate3 TF self-replay:", matches, "/", gen, " rate=", rate)
    # Free-run vs TF of own tokens must be exact (no kernel switch).
    if mism != 0:
        print("Gate3 FAIL: free-run vs own-token TF not bit-exact")
        return False
    print("Gate3 PASS")
    return True


def run_nc1_distinct(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    base_prompt: List[Int32],
    ctx: DeviceContext,
) raises -> Bool:
    print("=== NC1: M distinct prompts → distinct streams ===")
    var m = 4
    var plen = 64
    var gen = 16
    if len(base_prompt) < plen:
        plen = len(base_prompt)
    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(m, 16))
    var caches = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, m)
    var dev = ctx.enqueue_create_buffer[DType.int32](m)
    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    var prompts = List[List[Int32]]()
    for s in range(m):
        prompts.append(make_prompt_row(base_prompt, s, plen))
    prefill_batched(weights, acts, caches, prompts, m, plen, dev, ctx)

    var streams = List[List[Int32]]()
    for s in range(m):
        streams.append(List[Int32]())

    for step in range(gen):
        with dev.map_to_host() as h:
            for s in range(m):
                var a = argmax_row(acts, s, argmax_bf16, argmax_idx, ctx)
                streams[s].append(a)
                h[s] = a
        decode_step_batched(weights, acts, caches, dev, m, ctx)
        ctx.synchronize()

    var all_same = True
    for s in range(1, m):
        var same = True
        for t in range(gen):
            if Int(streams[s][t]) != Int(streams[0][t]):
                same = False
                break
        if same:
            print("NC1 FAIL: stream", s, "identical to stream 0")
            all_same = True
            return False
        else:
            all_same = False
            print("NC1 stream", s, "differs from stream 0 (good)")
    if all_same and m > 1:
        print("NC1 FAIL: all streams identical")
        return False
    print("NC1 PASS: streams produce distinct tokens")
    return True


def run_nc3_weight_bytes(
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    base_prompt: List[Int32],
    ctx: DeviceContext,
) raises -> Bool:
    print("=== NC3: weight bytes once per step (fused control) ===")
    disable_weight_log()
    var ok = True
    var ms = List[Int]()
    ms.append(1)
    ms.append(2)
    ms.append(4)
    ms.append(8)
    for mi in range(len(ms)):
        var m = ms[mi]
        var acts = Activations[
            HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
        ](ctx, max(m, 16))
        var caches = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, m)
        var dev = ctx.enqueue_create_buffer[DType.int32](m)
        # Tiny prefill so caches.length > 0.
        var plen = 4
        var prompts = List[List[Int32]]()
        for s in range(m):
            prompts.append(make_prompt_row(base_prompt, s, plen))
        prefill_batched(weights, acts, caches, prompts, m, plen, dev, ctx)

        with dev.map_to_host() as h:
            for s in range(m):
                h[s] = Int32(1000 + s * 97)

        enable_weight_log()
        decode_step_batched(weights, acts, caches, dev, m, ctx)
        ctx.synchronize()
        disable_weight_log()

        var wbytes = sum_weight_bytes()
        var ratio = Float64(wbytes) / Float64(EXPECT_WEIGHT_BYTES)
        print(
            "NC3 M=",
            m,
            " weight_bytes=",
            wbytes,
            " ratio_to_fused=",
            ratio,
            " expect=",
            EXPECT_WEIGHT_BYTES,
        )
        if wbytes == 0:
            print("NC3 FAIL: no weight log (instrument dead)")
            ok = False
        elif ratio < 0.90 or ratio > 1.10:
            print("NC3 FAIL: weight bytes not ≈ fused model")
            ok = False
        elif m > 1 and wbytes > EXPECT_WEIGHT_BYTES * m // 2:
            print("NC3 FAIL: weight bytes scale toward ×M")
            ok = False
        else:
            print("NC3 M=", m, " PASS (once, not ×M)")
    if ok:
        print("NC3 PASS overall")
    return ok


def run_throughput_m(
    m: Int,
    weights: Qwen3WeightsFP8[
        _,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        Q_PROJ_OUT,
        K_PROJ_OUT,
        HEAD_DIM,
        INTERMEDIATE_SIZE,
        NUM_LAYERS,
    ],
    base_prompt: List[Int32],
    ctx: DeviceContext,
) raises:
    print("=== THRUPUT M=", m, " prefill=", PROMPT_LEN, " gen=", GEN_LEN, " ===")
    var prompt_len = PROMPT_LEN
    if len(base_prompt) < prompt_len:
        prompt_len = len(base_prompt)

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(m, 16))
    var caches = BatchedKVCache[NUM_LAYERS, K_PROJ_OUT](ctx, m)
    var dev = ctx.enqueue_create_buffer[DType.int32](m)
    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    var prompts = List[List[Int32]]()
    for s in range(m):
        prompts.append(make_prompt_row(base_prompt, s, prompt_len))

    # Prefill timed separately.
    var t0 = perf_counter_ns()
    prefill_batched(weights, acts, caches, prompts, m, prompt_len, dev, ctx)
    var t1 = perf_counter_ns()
    var prefill_s = Float64(t1 - t0) / 1e9

    # Decode-only aggregate over GEN_LEN tokens.
    # Timing matches product qwen_generate: forward+sync only; argmax is
    # outside the timed window (GPU sample still runs for real tokens).
    # Prefill excluded. Report is generation with growing KV, not past=0 floor.
    var decode_ns = Int(0)
    var late_ns = Int(0)
    var late_steps = 0
    for step in range(GEN_LEN):
        with dev.map_to_host() as h:
            for s in range(m):
                var a = argmax_row(acts, s, argmax_bf16, argmax_idx, ctx)
                h[s] = a
        var s0 = perf_counter_ns()
        decode_step_batched(weights, acts, caches, dev, m, ctx)
        ctx.synchronize()
        var s1 = perf_counter_ns()
        decode_ns += s1 - s0
        if step >= GEN_LEN - 16:
            late_ns += s1 - s0
            late_steps += 1

    var decode_s = Float64(decode_ns) / 1e9
    var total_tokens = m * GEN_LEN
    var agg = Float64(total_tokens) / decode_s
    var per = agg / Float64(m)
    var step_ms_late = Float64(0)
    if late_steps > 0:
        step_ms_late = Float64(late_ns) / 1e9 / Float64(late_steps) * 1000.0
    var final_ctx = caches.length

    print("prefill_s=", prefill_s)
    print("decode_s=", decode_s)
    print("decode_agg_tok_s=", agg)
    print("per_stream_tok_s=", per)
    print("total_tokens=", total_tokens)
    print("final_ctx=", final_ctx)
    print("step_ms_late=", step_ms_late)
    print(
        "RESULT M=",
        m,
        " prefill_s=",
        prefill_s,
        " decode_agg=",
        agg,
        " per=",
        per,
        " total_tok=",
        total_tokens,
        " final_ctx=",
        final_ctx,
        " step_ms_late=",
        step_ms_late,
    )


def main() raises:
    disable_weight_log()
    var ctx = DeviceContext()
    print("loading FP8 staged/qwen3-4b-fp8 ONCE ...")
    var arena = load_arena_bytes(ctx, "staged/qwen3-4b-fp8")
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var weights = build_weights_fp8[
        vocab=VOCAB_SIZE,
        hidden=HIDDEN_SIZE,
        q_out=Q_PROJ_OUT,
        kv_out=K_PROJ_OUT,
        head_dim=HEAD_DIM,
        inter=INTERMEDIATE_SIZE,
        n_layers=NUM_LAYERS,
    ](base_ptr, arena)
    print("FP8 weights ready, tensors=", len(arena.entries))

    var base_prompt = read_ids("bench/ab_prompt_long_ids.txt")
    print("base_prompt_len=", len(base_prompt))

    var mode = String("all")
    if len(argv()) > 1:
        mode = String(argv()[1])

    var g1 = True
    var g2 = True
    var g3 = True
    var nc1 = True
    var nc3 = True

    if mode == "all" or mode == "gates":
        g1 = run_gate1_isolation(weights, base_prompt, ctx)
        if not g1:
            print("STOP: Gate1 failed")
            return
        g2 = run_gate2_teacher_forced(weights, base_prompt, ctx)
        if not g2:
            print("STOP: Gate2 failed")
            return
        g3 = run_gate3_tf_smoke(weights, base_prompt, ctx)
        if not g3:
            print("STOP: Gate3 failed")
            return
        nc1 = run_nc1_distinct(weights, base_prompt, ctx)
        if not nc1:
            print("STOP: NC1 failed")
            return
        print("NC2: confirmatory — Gate1 under distinct companions = KV not aliased")
        print("NC2 PASS (via Gate1)")
        nc3 = run_nc3_weight_bytes(weights, base_prompt, ctx)
        if not nc3:
            print("STOP: NC3 failed")
            return
        print("CORRECTNESS+NC: ALL PASS")

    if mode == "all" or mode == "bench":
        var ms = List[Int]()
        ms.append(1)
        ms.append(2)
        ms.append(4)
        ms.append(8)
        ms.append(16)
        for i in range(len(ms)):
            run_throughput_m(ms[i], weights, base_prompt, ctx)

    print("DONE")
    print(
        "GATES g1=",
        g1,
        " g2=",
        g2,
        " g3=",
        g3,
        " nc1=",
        nc1,
        " nc3=",
        nc3,
    )
