# FP8 KV-cache quality probe (quantize→dequantize at cache-write, BF16 storage).
#
# Co-measures BF16-KV vs FP8-KV in one session:
#   1) 512-context teacher-forced vs HF oracle (3 prompts × 256 steps)
#   2) 4K-context self-A/B: same prompt, argmax parity FP8-KV vs BF16-KV
#
# Usage (nightly env, GPU 0):
#   mojo build -I $KERNELS -I src src/qwen_fp8_kv_quality_probe.mojo -o /tmp/fp8_kv_probe
#   /tmp/fp8_kv_probe \
#       /tmp/fp8_kv_probe/prompt1_input_ids.txt /tmp/fp8_kv_probe/prompt1_oracle_out.txt \
#       /tmp/fp8_kv_probe/prompt2_input_ids.txt /tmp/fp8_kv_probe/prompt2_oracle_out.txt \
#       /tmp/fp8_kv_probe/prompt3_input_ids.txt /tmp/fp8_kv_probe/prompt3_oracle_out.txt \
#       /tmp/fp8_kv_probe/prompt_4k_ids.txt \
#       /tmp/fp8_kv_probe_out

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv

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
from hephaestus.forward import Activations, KVCache, forward_fp8
from hephaestus.kernels import MAX_KEYS, argmax_logits
from hephaestus.loader import build_weights_fp8, load_arena_bytes

comptime N_STEPS_512 = 256
comptime CHUNK = 64
# 4K self-A/B: process this many tokens (must be ≤ MAX_KEYS).
comptime N_4K = 4096


def read_ids(path: String) raises -> List[Int32]:
    var ids = List[Int32]()
    var text = open(path, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        ids.append(Int32(Int(String(line))))
    return ids^


from hephaestus.model_fp8 import Qwen3WeightsFP8


def run_tf_oracle_w[
    fp8_kv: Bool
](
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
    mut cache: KVCache[NUM_LAYERS, K_PROJ_OUT],
    prompt: List[Int32],
    oracle: List[Int32],
    n_steps: Int,
    ctx: DeviceContext,
    mut argmax_bf16: DeviceBuffer[DType.bfloat16],
    mut argmax_idx: DeviceBuffer[DType.int32],
    mut mismatch_steps: List[Int],
) raises -> Int:
    cache.length = 0
    var seq0 = len(prompt)
    var dev_ids = ctx.enqueue_create_buffer[DType.int32](max(seq0, 1))
    with dev_ids.map_to_host() as h:
        for i in range(seq0):
            h[i] = prompt[i]

    var matches = 0
    for step in range(n_steps):
        var n = seq0 if step == 0 else 1
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
            fp8_kv_cache=fp8_kv,
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()

        var logits_base = (
            acts.logits.unsafe_ptr() + (n - 1) * VOCAB_SIZE
        ).as_unsafe_any_origin()
        var best = argmax_logits(
            logits_base, argmax_bf16, argmax_idx, VOCAB_SIZE, ctx
        )
        if Int(best) == Int(oracle[step]):
            matches += 1
        else:
            mismatch_steps.append(step)

        with dev_ids.map_to_host() as h:
            h[0] = oracle[step]
    return matches


def collect_argmax_stream[
    fp8_kv: Bool
](
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
    mut cache: KVCache[NUM_LAYERS, K_PROJ_OUT],
    tokens: List[Int32],
    n_tok: Int,
    ctx: DeviceContext,
    mut argmax_bf16: DeviceBuffer[DType.bfloat16],
    mut argmax_idx: DeviceBuffer[DType.int32],
    mut out_argmax: List[Int32],
) raises:
    """Chunked teacher-forced walk over `tokens`; append argmax at each pos."""
    cache.length = 0
    # Fresh list (List has no clear() in this Mojo pin).
    out_argmax = List[Int32]()
    var dev_ids = ctx.enqueue_create_buffer[DType.int32](CHUNK)
    var pos = 0
    while pos < n_tok:
        var n = CHUNK
        if pos + n > n_tok:
            n = n_tok - pos
        with dev_ids.map_to_host() as h:
            for i in range(n):
                h[i] = tokens[pos + i]
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
            fp8_kv_cache=fp8_kv,
        ](weights, acts, cache, dev_ids, n, ctx)
        ctx.synchronize()
        for i in range(n):
            var logits_base = (
                acts.logits.unsafe_ptr() + i * VOCAB_SIZE
            ).as_unsafe_any_origin()
            var best = argmax_logits(
                logits_base, argmax_bf16, argmax_idx, VOCAB_SIZE, ctx
            )
            out_argmax.append(best)
        pos += n
        if pos % 512 == 0:
            print("  ... pos", pos, "/", n_tok, "fp8_kv=", fp8_kv)


def main() raises:
    if len(argv()) < 9:
        print(
            "usage: fp8_kv_probe p1_in p1_ora p2_in p2_ora p3_in p3_ora p4k out_prefix"
        )
        return

    var p1_in = String(argv()[1])
    var p1_ora = String(argv()[2])
    var p2_in = String(argv()[3])
    var p2_ora = String(argv()[4])
    var p3_in = String(argv()[5])
    var p3_ora = String(argv()[6])
    var p4k_path = String(argv()[7])
    var out_prefix = String(argv()[8])

    if N_4K > MAX_KEYS:
        raise Error("N_4K exceeds MAX_KEYS")

    var prompts = List[List[Int32]]()
    var oracles = List[List[Int32]]()
    prompts.append(read_ids(p1_in))
    oracles.append(read_ids(p1_ora))
    prompts.append(read_ids(p2_in))
    oracles.append(read_ids(p2_ora))
    prompts.append(read_ids(p3_in))
    oracles.append(read_ids(p3_ora))
    var tokens_4k = read_ids(p4k_path)
    if len(tokens_4k) < N_4K:
        raise Error("4k prompt shorter than N_4K")

    for i in range(3):
        if len(oracles[i]) < N_STEPS_512:
            raise Error("oracle shorter than N_STEPS_512")

    var ctx = DeviceContext()
    print("loading FP8 staged/qwen3-4b-fp8 ...")
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
    print("MAX_KEYS=", MAX_KEYS, "N_4K=", N_4K, "CHUNK=", CHUNK)

    # Capacity check (theory): BF16 KV vs FP8+scale per token.
    var bf16_b = 8 * 128 * 2 * 36 * 2
    var fp8_b = 8 * 128 * 2 * 36 * 1 + 2 * 4 * 36
    print(
        "capacity_bytes_per_token bf16=",
        bf16_b,
        "fp8+scale=",
        fp8_b,
        "ratio=",
        Float64(bf16_b) / Float64(fp8_b),
    )

    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, CHUNK)
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var argmax_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](VOCAB_SIZE)
    var argmax_idx = ctx.enqueue_create_buffer[DType.int32](1)

    var report = open(out_prefix + "_report.txt", "w")

    # ---- 512: oracle TF, co-measure BF16-KV then FP8-KV --------------------
    print("=== 512 teacher-forced vs HF oracle (co-measure) ===")
    report.write("=== 512 teacher-forced vs HF oracle ===\n")

    var total_bf16 = 0
    var total_fp8kv = 0
    var total_steps = 0
    for pi in range(3):
        var mm_b = List[Int]()
        var mm_f = List[Int]()
        var m_b = run_tf_oracle_w[False](
            weights,
            acts,
            cache,
            prompts[pi],
            oracles[pi],
            N_STEPS_512,
            ctx,
            argmax_bf16,
            argmax_idx,
            mm_b,
        )
        var m_f = run_tf_oracle_w[True](
            weights,
            acts,
            cache,
            prompts[pi],
            oracles[pi],
            N_STEPS_512,
            ctx,
            argmax_bf16,
            argmax_idx,
            mm_f,
        )
        total_bf16 += m_b
        total_fp8kv += m_f
        total_steps += N_STEPS_512
        print(
            "prompt",
            pi + 1,
            "BF16-KV",
            m_b,
            "/",
            N_STEPS_512,
            "FP8-KV",
            m_f,
            "/",
            N_STEPS_512,
        )
        report.write(
            "prompt"
            + String(pi + 1)
            + " BF16-KV "
            + String(m_b)
            + "/"
            + String(N_STEPS_512)
            + " FP8-KV "
            + String(m_f)
            + "/"
            + String(N_STEPS_512)
            + "\n"
        )
        # Log first few mismatch steps for shape diagnosis.
        report.write("  BF16-KV mismatch steps (first 32):")
        var nb = min(32, len(mm_b))
        for j in range(nb):
            report.write(" " + String(mm_b[j]))
        report.write("\n")
        report.write("  FP8-KV mismatch steps (first 32):")
        var nf = min(32, len(mm_f))
        for j in range(nf):
            report.write(" " + String(mm_f[j]))
        report.write("\n")

    print(
        "512 TOTAL BF16-KV",
        total_bf16,
        "/",
        total_steps,
        "rate",
        Float64(total_bf16) / Float64(total_steps),
    )
    print(
        "512 TOTAL FP8-KV",
        total_fp8kv,
        "/",
        total_steps,
        "rate",
        Float64(total_fp8kv) / Float64(total_steps),
    )
    report.write(
        "TOTAL BF16-KV "
        + String(total_bf16)
        + "/"
        + String(total_steps)
        + "\n"
    )
    report.write(
        "TOTAL FP8-KV "
        + String(total_fp8kv)
        + "/"
        + String(total_steps)
        + "\n"
    )

    # ---- 4K self-A/B -------------------------------------------------------
    print("=== 4K self-A/B (FP8-KV vs BF16-KV, same tokens) ===")
    report.write("=== 4K self-A/B ===\n")

    var argmax_bf16kv = List[Int32]()
    var argmax_fp8kv = List[Int32]()

    print("collecting BF16-KV argmax stream @ 4K ...")
    collect_argmax_stream[False](
        weights,
        acts,
        cache,
        tokens_4k,
        N_4K,
        ctx,
        argmax_bf16,
        argmax_idx,
        argmax_bf16kv,
    )
    print("collecting FP8-KV argmax stream @ 4K ...")
    collect_argmax_stream[True](
        weights,
        acts,
        cache,
        tokens_4k,
        N_4K,
        ctx,
        argmax_bf16,
        argmax_idx,
        argmax_fp8kv,
    )

    var match_4k = 0
    var fa_b = open(out_prefix + "_4k_bf16kv_argmax.txt", "w")
    var fa_f = open(out_prefix + "_4k_fp8kv_argmax.txt", "w")
    var fa_mm = open(out_prefix + "_4k_mismatches.txt", "w")
    fa_mm.write("pos bf16_argmax fp8_argmax\n")
    # Track mismatch density in 512-token bins for shape diagnosis.
    var bin_size = 512
    var n_bins = (N_4K + bin_size - 1) // bin_size
    var bin_mm = List[Int]()
    for _ in range(n_bins):
        bin_mm.append(0)

    for t in range(N_4K):
        fa_b.write(String(argmax_bf16kv[t]) + "\n")
        fa_f.write(String(argmax_fp8kv[t]) + "\n")
        if Int(argmax_bf16kv[t]) == Int(argmax_fp8kv[t]):
            match_4k += 1
        else:
            fa_mm.write(
                String(t)
                + " "
                + String(argmax_bf16kv[t])
                + " "
                + String(argmax_fp8kv[t])
                + "\n"
            )
            var b = t // bin_size
            bin_mm[b] = bin_mm[b] + 1

    fa_b.close()
    fa_f.close()
    fa_mm.close()

    var rate_4k = Float64(match_4k) / Float64(N_4K)
    print("4K match", match_4k, "/", N_4K, "rate", rate_4k)
    report.write(
        "4K match " + String(match_4k) + "/" + String(N_4K) + "\n"
    )
    report.write("4K mismatch bins (size " + String(bin_size) + "):\n")
    for b in range(n_bins):
        var lo = b * bin_size
        var hi = lo + bin_size
        if hi > N_4K:
            hi = N_4K
        report.write(
            "  ["
            + String(lo)
            + ","
            + String(hi)
            + ") mismatches="
            + String(bin_mm[b])
            + "\n"
        )
        print(
            "  bin",
            lo,
            "-",
            hi,
            "mismatches",
            bin_mm[b],
        )

    report.close()
    print("wrote", out_prefix + "_report.txt")
    print("DONE")
