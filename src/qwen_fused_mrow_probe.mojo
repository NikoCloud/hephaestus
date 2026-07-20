# Fused M-row decode probe — does weight amortization work?
#
# One forward_fp8 call with seq=M (M activation rows). M=1 uses decode GEMV;
# M>1 uses FP8 v3a prefill GEMM (weights staged once across rows). Not N
# serial single-row forwards.
#
# Usage (nightly env, GPU 0):
#   mojo build -I $KERNELS -I src src/qwen_fused_mrow_probe.mojo \
#       -o /tmp/fused_mrow_probe
#   HIP_VISIBLE_DEVICES=0 /tmp/fused_mrow_probe [n_steps=32]

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
from hephaestus.forward import Activations, KVCache, forward_fp8
from hephaestus.loader import build_weights_fp8, load_arena_bytes
from hephaestus.model_fp8 import Qwen3WeightsFP8

comptime WEIGHT_GB = 4.02
comptime KV_BYTES_PER_TOKEN = 147456
# Serving-shaped avg depth for product bytes/step model (spec framing).
comptime AVG_CTX = 576
comptime ROOFLINE = 569.0
# Expected projection weight bytes (FP8 bodies only, no scales/norms/embed full):
# measured instrument should be ~this and flat vs M if fused.
comptime EXPECT_PROJ_WEIGHT_BYTES = 4022272000  # ~4.022 GB (36 layers + lm_head)
comptime WEIGHT_LOG = "/tmp/fp8_weight_bytes.log"
comptime WARMUP = 3
comptime DEFAULT_STEPS = 32


def reset_weight_log() raises:
    var f = open(WEIGHT_LOG, "w")
    f.close()


def enable_weight_log() raises:
    var f = open("/tmp/fp8_weight_bytes_enable", "w")
    f.write("1\n")
    f.close()


def disable_weight_log() raises:
    var f = open("/tmp/fp8_weight_bytes_enable", "w")
    f.write("0\n")
    f.close()


def sum_weight_bytes() raises -> Int:
    """Sum first field after path tag on each line: 'prefill <bytes> ...'."""
    var total = 0
    var text = open(WEIGHT_LOG, "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        # Format: "gemv 12345 n=..." or "prefill 12345 m=..."
        var parts = String(line).split(" ")
        if len(parts) < 2:
            continue
        total += Int(parts[1])
    return total


def count_launches() raises -> Int:
    var n = 0
    var text = open(WEIGHT_LOG, "r").read()
    for line in text.split("\n"):
        if line.byte_length() > 0:
            n += 1
    return n


def bytes_step_fused_model(m: Int) -> Float64:
    """Fused model: weights once + M sequences' KV (serving-shaped depth)."""
    var kv = Float64(m) * Float64(AVG_CTX) * Float64(KV_BYTES_PER_TOKEN) / 1073741824.0
    return WEIGHT_GB + kv


def bytes_step_unfused_model(m: Int) -> Float64:
    """Unfused model: M × (weights + KV)."""
    var one = WEIGHT_GB + (
        Float64(AVG_CTX) * Float64(KV_BYTES_PER_TOKEN) / 1073741824.0
    )
    return Float64(m) * one


def rows_distinct_logits(
    mut logits: DeviceBuffer[DType.float32], m: Int, ctx: DeviceContext
) raises -> Bool:
    """Secondary NC: with distinct inputs, output rows must not all be identical."""
    if m <= 1:
        return True
    ctx.synchronize()
    var ok = True
    with logits.map_to_host() as h:
        # Compare row 0 vs each other row on a stride of vocabulary elements.
        var stride = 1024
        for r in range(1, m):
            var same = 0
            var checked = 0
            var i = 0
            while i < VOCAB_SIZE:
                var a = h[i]
                var b = h[r * VOCAB_SIZE + i]
                checked += 1
                if a == b:
                    same += 1
                i += stride
            # If every sampled element matches, rows are identical → FAIL.
            if same == checked:
                print(
                    "NC FAIL distinct-rows: row",
                    r,
                    "identical to row0 on",
                    checked,
                    "samples",
                )
                ok = False
            else:
                print(
                    "  row",
                    r,
                    "vs row0: same=",
                    same,
                    "/",
                    checked,
                    "samples (expect some diffs)",
                )
    return ok


def run_m(
    m: Int,
    n_steps: Int,
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
    ctx: DeviceContext,
) raises:
    print("=== M=", m, "===")
    var acts = Activations[
        HIDDEN_SIZE, Q_PROJ_OUT, K_PROJ_OUT, INTERMEDIATE_SIZE, VOCAB_SIZE
    ](ctx, max(m, 16))
    var cache = KVCache[NUM_LAYERS, K_PROJ_OUT](ctx)
    var dev_ids = ctx.enqueue_create_buffer[DType.int32](m)

    # Distinct token ids per row (not copies of one id).
    with dev_ids.map_to_host() as h:
        for i in range(m):
            h[i] = Int32(1000 + i * 97)

    # Warmup (not timed; also settles JIT).
    for _ in range(WARMUP):
        cache.length = 0
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
        ](weights, acts, cache, dev_ids, m, ctx)
        ctx.synchronize()

    # --- Weight-byte NC: one forward, count attributed weight traffic ---
    reset_weight_log()
    enable_weight_log()
    cache.length = 0
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
    ](weights, acts, cache, dev_ids, m, ctx)
    ctx.synchronize()
    disable_weight_log()
    var wbytes = sum_weight_bytes()
    var nlaunch = count_launches()
    var w_gb = Float64(wbytes) / 1073741824.0
    var fused_expect = Float64(EXPECT_PROJ_WEIGHT_BYTES) / 1073741824.0
    var unfused_expect = fused_expect * Float64(m)
    print("weight_bytes_attributed=", wbytes, "GB=", w_gb)
    print("weight_launches=", nlaunch)
    print("fused_model_weight_GB≈", fused_expect, "unfused_M×≈", unfused_expect)
    # Pass if measured ≈ fused (within 5%) and NOT scaling toward unfused.
    var ratio_to_fused = w_gb / fused_expect
    var ratio_to_unfused = Float64(0)
    if m > 1:
        ratio_to_unfused = w_gb / unfused_expect
    else:
        ratio_to_unfused = Float64(1)
    print("ratio_to_fused=", ratio_to_fused, "ratio_to_unfused=", ratio_to_unfused)
    var weight_ok = True
    if wbytes == 0:
        print(
            "NC SKIP amortization: no weight log (instrumentation not active — "
            "timing still valid; re-apply TEMP hooks for NC re-check)"
        )
        weight_ok = True  # do not block timing-only runs after revert
    else:
        if ratio_to_fused < 0.90 or ratio_to_fused > 1.10:
            print("NC WARN: weight bytes not near fused model (~4.02 GB proj)")
        if m > 1:
            if ratio_to_unfused > 0.85:
                print("NC FAIL amortization: weight bytes scale with M (≈ unfused)")
                weight_ok = False
            elif ratio_to_fused > 1.5:
                print(
                    "NC FAIL amortization: weight bytes grow with M (ratio_to_fused=",
                    ratio_to_fused,
                    ")",
                )
                weight_ok = False
            else:
                print("NC PASS amortization: weight bytes ≈ once, not ×M")
        else:
            print("NC amortization: M=1 baseline (n/a scale check)")

    # --- Distinct rows NC ---
    var distinct_ok = rows_distinct_logits(acts.logits, m, ctx)
    if m > 1 and distinct_ok:
        print("NC PASS distinct-rows: outputs differ across rows")
    elif m == 1:
        print("NC distinct-rows: n/a at M=1")
    if m > 1 and not distinct_ok:
        raise Error("negative control failed: rows not distinct")

    if not weight_ok:
        raise Error("negative control failed: weight amortization")

    # --- Timed fused steps (fresh past=0 each step; constant work) ---
    # Each step: one forward_fp8(seq=M) — fused M-row GEMMs + attention over M.
    var total_ns = Int(0)
    for _ in range(n_steps):
        cache.length = 0
        # Re-assert distinct ids each step (not aliased to one token).
        with dev_ids.map_to_host() as h:
            for i in range(m):
                h[i] = Int32(1000 + i * 97)
        var t0 = perf_counter_ns()
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
        ](weights, acts, cache, dev_ids, m, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        total_ns += t1 - t0

    var mean_s = Float64(total_ns) / 1e9 / Float64(n_steps)
    var agg = Float64(m) / mean_s
    var per = agg / Float64(m)
    var b_fused = bytes_step_fused_model(m)
    var b_unfused = bytes_step_unfused_model(m)
    # Eff BW uses fused bytes/step model (weights once).
    var eff_bw = (agg * b_fused) / Float64(m)
    var pct = eff_bw / ROOFLINE * 100.0
    # Per-token effective weight traffic from instrument (GB / token).
    var wt_per_tok = w_gb / Float64(m)

    print("mean_step_s=", mean_s)
    print("mean_step_ms=", mean_s * 1000.0)
    print("aggregate_tok_s=", agg)
    print("per_row_tok_s=", per)
    print("bytes_step_fused_GB=", b_fused)
    print("bytes_step_unfused_GB=", b_unfused)
    print("eff_bw_GB_s=", eff_bw)
    print("pct_roofline=", pct)
    print("weight_GB_per_token=", wt_per_tok)
    print(
        "RESULT M=",
        m,
        " step_ms=",
        mean_s * 1000.0,
        " agg=",
        agg,
        " per=",
        per,
        " roof%=",
        pct,
        " wGB=",
        w_gb,
        " w_ok=",
        weight_ok,
        " rows_ok=",
        distinct_ok,
    )


def main() raises:
    var n_steps = DEFAULT_STEPS
    if len(argv()) > 1:
        n_steps = Int(String(argv()[1]))

    var ctx = DeviceContext()
    print("loading FP8 staged/qwen3-4b-fp8 ONCE ...")
    var arena = load_arena_bytes(ctx, "staged/qwen3-4b-fp8")
    var base_ptr = (
        arena.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    print(
        "weight_arena_ptr=",
        Int(arena.buf.unsafe_ptr()),
        "total_bytes=",
        arena.total_bytes,
        "GB=",
        Float64(arena.total_bytes) / 1073741824.0,
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
    print(
        "protocol: fused forward_fp8(seq=M); M-sweep; steps=",
        n_steps,
        " warmup=",
        WARMUP,
    )
    print(
        "path: M=1 → gemv_fp8 decode; M>1 → wmma_gemm_fp8_prefill (v3a)"
    )
    print("roof=569; weight instrument: n*k FP8 once per linear launch")
    print(
        "anchors: M=1 ref ~66.1 tok/s; perfect M=8 ~529; llama ROCm npl8~550 Vulkan~678"
    )

    var ms = List[Int]()
    ms.append(1)
    ms.append(2)
    ms.append(4)
    ms.append(8)
    ms.append(16)

    for i in range(len(ms)):
        run_m(ms[i], n_steps, weights, ctx)

    # Optional re-check M=1 and M=8 at end (thermal sandwich).
    print("=== recheck M=1 ===")
    run_m(1, n_steps, weights, ctx)
    print("=== recheck M=8 ===")
    run_m(8, n_steps, weights, ctx)

    print("DONE")
