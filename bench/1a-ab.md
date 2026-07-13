# Phase 1a Closeout — A/B Benchmark: Hephaestus (BF16) vs llama.cpp (F16)
## Date: 2026-07-13 (post-fix re-measure same day)
## Hardware: GPU 0 = AMD Radeon AI PRO R9700 32GB (gfx1201), see `bench/hardware.md`
## Model: Qwen3-4B-Instruct-2507 (Hephaestus: BF16 safetensors; llama.cpp: F16 GGUF, 7.49 GiB)
## llama.cpp build: 33ca0dcb9 (9906), ROCm/HIP backend — same build as `bench/0-baseline-llamacpp-f16.md`

GPU 0 checked idle before every run. GPU 1 never touched
(`HIP_VISIBLE_DEVICES=0` / `-dev ROCm0`).

**Post-fix drivers (this re-measure):** GPU `argmax_logits` (Finding 2
RESOLVED) + chunked `load_arena` (cleaner host path; VRAM unchanged —
Finding 3 CHARACTERIZED as MAX runtime pool). Commit `0b9ee04`.

---

## Result table — POST-FIX (GPU argmax + chunked loader)

| Metric | Hephaestus BF16 | llama.cpp F16 | Ratio (Heph/llama) |
|---|---|---|---|
| Prefill tok/s (10-token prompt) | 94.78 ± 0.93 | 147.61 ± 1.71 | 0.64× |
| Prefill tok/s (512-token prompt) | 121.20 ± 0.29 | 1429.81 ± 11.00 (`llama-bench` pp512) | 0.08× |
| TTFT ms (10-token prompt, forward only) | 105.52 ± 1.05 | 67.75 ± 0.78 | 1.56× (slower) |
| TTFT ms (512-token prompt, forward only) | 4224.43 ± 10.18 | (not re-measured; was ~395) | — |
| Decode tok/s (forward-pass only — G1a-2 metric) | 53.67 ± 0.18 | 61.96 ± 0.09 (`llama-bench` tg256) | 0.87× |
| **Decode tok/s (incl. sampling)** | **52.80 ± 0.19** | **61.82 ± 0.08** (`llama-simple` eval) | **0.85×** |
| Argmax / sample ms per decode step | **0.30 ± 0.01** | ~0.064 | ~4.7× (still, but was 800×) |
| **Total time s (256 tokens, 10-token prompt)** | **4.94 ± 0.02** | **5.17 ± 0.01** | **0.96×** |
| Peak VRAM MB | **~29,562** (runtime pool) | ~8,142 (model-sized) | 3.63× |
| VRAM after exit MB | ~74.7 (baseline) | ~74.7 (baseline) | 1.0× (both clean) |

**Headline:** total wall time **18.00s → 4.94s** (GPU argmax). End-to-end
decode with sampling **14.3 → 52.8 tok/s**. Peak VRAM still ~29.6GB —
expected: MAX/AsyncRT reserves ~90% of card on first `createBuffer` (see
Finding 3).

---

## Result table — PRE-FIX (CPU argmax, full-blob loader) — 2026-07-13 earlier

Kept for comparison. Do not use as current status.

| Metric | Hephaestus BF16 | llama.cpp F16 | Ratio (Heph/llama) |
|---|---|---|---|
| Prefill tok/s (10-token prompt) | 99.06 ± 2.26 | 148.57 ± 1.68 | 0.67× |
| Prefill tok/s (512-token prompt) | 123.34 ± 2.68 | 1297.43 ± 3.06 | 0.10× |
| TTFT ms (10-token prompt) | 152.61 ± 2.32 | 67.32 ± 0.77 | 2.27× (slower) |
| TTFT ms (512-token prompt) | 4204.31 ± 88.86 | 394.63 ± 0.93 | 10.66× (slower) |
| Decode tok/s (forward-pass only — the G1a-2 metric) | 54.39 ± 0.02 | 61.82 ± 0.11 | 0.88× |
| Total time s (256 tokens, 10-token prompt) | **18.00 ± 0.00** | 5.19 ± 0.01 | 3.47× (slower) |
| Peak VRAM MB | ~29,590 | ~8,142 | 3.63× |
| VRAM after exit MB | ~74.7 (baseline) | ~74.7 (baseline) | 1.0× (both clean) |

---

## Finding 1: llama.cpp's own decode speed changed since the G1a-2 baseline was set

G1a-2's target (49.63 tok/s, 90% of 55.14) was set from `bench/0-baseline-llamacpp-f16.md`,
measured 2026-07-11. Re-measuring **today, same build, same model file, same
GPU, GPU 0 confirmed idle**:

| | 2026-07-11 (original) | 2026-07-13 (this session) |
|---|---|---|
| pp512 | 1418.39 ± 29.10 | 1456.38–1462.63 |
| tg128 | 55.14 ± 0.13 | 61.90 ± 0.13 |
| tg256 | (not measured) | 61.96–61.97 ± ~0.1 |

pp512 is close (within ~3%, ordinary run-to-run variance). **tg128 is not** —
55.14 → 61.90, a **+12.3% change**, with nothing on the Hephaestus side and
nothing in llama.cpp's own binary or model file touched. The most likely
explanation is an environmental change on the machine between the two dates
(driver/firmware/clock-state) — not investigated further here since it's
outside Hephaestus's own code, but it is exactly the kind of fact that
changes a gate's meaning depending on which llama.cpp number you compare
against:

| comparison basis | Hephaestus 54.35 tok/s is... |
|---|---|
| original citation (55.14, 2026-07-11) | 98.6% — matches the 98.3%/109.2%-of-target figures in `bench/1a.md` |
| freshly measured today (61.82–61.97) | **87.7–87.9%** — would NOT clear a 90% bar if the gate were recomputed against today's number |

**This is not a call I'm making unilaterally.** G1a-2's PASS in `SPEC.md` is
left as-is (that was an explicit instruction this session, and the original
measurement was real and reproducible against its own stated baseline). But
the fact that llama.cpp's own speed moved this much on the identical
hardware between two measurement dates is worth Niko's awareness — it may
warrant re-baselining the target, or at minimum re-measuring the 55.14
citation before it's cited again without a date-scoped caveat.

## Finding 2: Sampling argmax cost — **RESOLVED** (GPU `argmax_logits`)

### Pre-fix (discovery)

Every Phase 1a "decode tok/s" figure measured **forward only**, excluding
host-side argmax. Instrumented A/B found:

| | Hephaestus (CPU argmax) | llama.cpp |
|---|---|---|
| forward-pass-only decode | 54.39 tok/s (18.4 ms/token) | 61.82 tok/s |
| **sampling cost per token** | **51.6 ms** | **0.065 ms** |
| decode incl. sampling | **14.29 tok/s** | ~61.8 tok/s |
| Total time (256 gen, short prompt) | **18.00 s** | 5.19 s |

### Fix

Custom GPU `argmax_logits` in `src/hephaestus/kernels.mojo` (not vendored
`nn.argmaxmin_gpu` — probe showed it returns the **higher** index on exact
ties; torch/HF need **lowest**). Wired into generate / teacher-forced-decode /
ab_bench / tiny_generate. Correctness: tiny token-exact; decode-path
teacher-forced **255/254/252** vs oracle (unchanged).

### Post-fix

| | value |
|---|---|
| argmax ms/step | **0.30 ± 0.01** (~**170×** vs 51.6 ms) |
| decode incl. sampling | **52.80 ± 0.19** tok/s |
| Total time (256 gen, short) | **4.94 ± 0.02 s** (was 18.00) |

## Finding 3: Peak VRAM ~29.6GB — **CHARACTERIZED** (MAX runtime pool, not loader)

### Pre-fix hypothesis (wrong)

Attributed ~29.6GB peak to the loader holding three ~8GB host/device
buffers for full-blob round-trip verify.

### What was actually measured

`experiments/vram_isolate_probe{,2}.mojo`:

- `DeviceContext()` alone ≈ **270 MB**
- First `enqueue_create_buffer` of **any** size (including tiny) triggers
  ≈ **29.56 GB** reservation (~90% of R9700’s 32.6 GB)
- Happens inside closed `AsyncRT_DeviceContext_createBuffer_async` — no
  app-visible knob / env var found

Chunked loader rewrite (128 MB stream + per-chunk byte-for-byte device
verify) is a **better design** (stronger check, less host RAM) but **does
not change peak VRAM** — the pool is independent of loader staging.

**Not a leak:** VRAM returns to ~75 MB baseline after process exit.
**Not app-fixable** from Hephaestus today. Logged to `IDEAS.md` as Phase
2/3 (upstream pool-size control).

---

## Methodology and exact commands

### Prompts

Both engines fed the **identical tokenized sequence** (Hephaestus has no
tokenizer per SPEC.md scope; llama.cpp tokenizes internally — verified both
converge on the same IDs via the model's own HF tokenizer):

```
python3 scripts/prepare_ab_prompts.py
```

- Short: `fixtures/oracle/prompt1.txt` ("The quick brown fox jumps over the
  lazy dog.") → 10 tokens → `bench/ab_prompt_short_ids.txt` /
  `bench/ab_prompt_short.txt`
- Long: a repeated standard paragraph, tokenized and trimmed to exactly
  512 tokens, decode/re-encode round-tripped for exact text-ID
  correspondence → `bench/ab_prompt_long_ids.txt` / `bench/ab_prompt_long.txt`

### Hephaestus

```
# Post-fix re-measure (or manual):
sh scripts/run_ab_post_fix.sh
# equivalent:
pixi run mojo build -I ~/projects/modular/max/kernels/src -I src src/qwen_ab_bench.mojo -o /tmp/qwen_ab_bench
/tmp/qwen_ab_bench bench/ab_prompt_short_ids.txt 256   # 3 reps
/tmp/qwen_ab_bench bench/ab_prompt_long_ids.txt 8      # 3 reps
```

`src/qwen_ab_bench.mojo` mirrors production `forward()` with separate
timers for (a) forward pass and (b) GPU argmax (was host scan pre-fix).

### llama.cpp

Two binaries were needed beyond the existing `llama-bench`:

- **`llama-cli` did not behave as a scriptable single-shot tool in this
  build.** With `-p`/`-f`, `--no-conversation`/`-no-cnv`, and stdin closed
  (`< /dev/null`), it still entered an interactive `>` prompt loop pinning
  one CPU core at ~100% without producing generation output (confirmed via
  `ps` + VRAM climbing to the full model — it *was* computing, just not
  exiting). Not investigated further as a llama.cpp bug; the project scope
  is Hephaestus, not fixing llama.cpp's CLI.
- **`llama-simple`** (a minimal example binary shipped with llama.cpp,
  built fresh from the same source tree/commit as the existing `llama-bench`:
  `cmake --build . --target llama-simple`) uses real
  `llama_sampler_init_greedy()` and prints authoritative internal timers
  (`llama_perf_context_print`: prompt eval time, eval time, total time) —
  exactly the metrics needed, with **real sampling included**, matching
  Hephaestus's own measurement scope. Verified via source read
  (`examples/simple/simple.cpp`) that it does genuine greedy sampling, not
  a skipped/dummy step.
- Confirmed separately (source read, `tools/llama-bench/llama-bench.cpp`):
  **`llama-bench`'s tg/pp tests do not sample at all** (`llama_decode` only,
  no `llama_sampler` calls) — i.e. `llama-bench` measures the same thing
  Hephaestus's "forward-pass only" number measures. This is why the
  "Decode tok/s" row uses `llama-simple`'s eval-time figure (which does
  include real sampling, like Hephaestus's forward-pass timing bracket
  does *not*) for measurement-scope symmetry with the row's own label.

```
# Standardized pp/tg (forward-pass only, cross-check against Finding 1's table)
~/projects/llama.cpp/build/bin/llama-bench -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf \
    -p 10,512 -n 256 -r 3 -dev ROCm0

# End-to-end with real sampling (built once: cmake --build build --target llama-simple)
HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build/bin/llama-simple \
    -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf -ngl 999 -n 256 \
    "The quick brown fox jumps over the lazy dog."   # 3 reps
HIP_VISIBLE_DEVICES=0 ~/projects/llama.cpp/build/bin/llama-simple \
    -m /mnt/models/models/qwen3-4b-instruct-2507-f16.gguf -ngl 999 -n 8 \
    "$(cat bench/ab_prompt_long.txt)"                 # 3 reps
```

### VRAM

`scripts/vram_poll.sh <csv> -- <command...>` polls
`rocm-smi --showmeminfo vram` for GPU 0 every 0.2s in the background while
the command runs; peak is the max sample. "VRAM after exit" checked via a
direct `rocm-smi` read ~2s after process exit (see Finding 3's note on
transient readings immediately at exit).

---

## Raw per-rep data

### Hephaestus, short prompt (10 tokens), 256 tokens generated

| rep | prefill tok/s | TTFT ms (incl. argmax) | decode tok/s (fwd only) | total s |
|---|---|---|---|---|
| 1 | 96.70 | 155.06 | 54.36 | 18.007 |
| 2 | 99.27 | 152.32 | 54.40 | 17.999 |
| 3 | 101.20 | 150.45 | 54.40 | 17.998 |

### Hephaestus, long prompt (512 tokens), 8 tokens generated

| rep | prefill tok/s | TTFT ms (incl. argmax) |
|---|---|---|
| 1 | 126.43 | 4101.70 |
| 2 | 121.79 | 4255.63 |
| 3 | 121.80 | 4255.60 |

### llama.cpp (`llama-simple`), short prompt

| rep | prefill tok/s | TTFT ms | decode tok/s (eval) | total ms (265 tok) |
|---|---|---|---|---|
| 1 | 146.63 | 68.20 | 61.72 | 5196.64 |
| 2 | 149.51 | 66.89 | 61.80 | 5190.64 |
| 3 | 149.57 | 66.86 | 61.94 | 5170.34 |

### llama.cpp (`llama-simple`), long prompt

| rep | prefill tok/s | TTFT ms |
|---|---|---|
| 1 | 1294.52 | 395.51 |
| 2 | 1300.62 | 393.66 |
| 3 | 1297.14 | 394.72 |

### llama-bench (forward-pass only, cross-check)

| test | t/s |
|---|---|
| pp10 | 335.69 ± 51.78 (this session); 326.59 ± 67.00 (first run) |
| pp512 | 1462.63 ± 11.99 (this session); 1456.38 ± 10.61 (first run) |
| tg128 | 61.90 ± 0.13 |
| tg256 | 61.96 ± 0.09, 61.97 ± 0.08 (two runs) |

---

## SPEC.md

G1a-2 checkbox confirmed correct (see `SPEC.md` §3):
```
- [x] **G1a-2: PASS** (2026-07-13) — 54.22 tok/s (98.3% of llama.cpp, 109.2% of target). Attention score loop parallelized across 4 warps. `bench/1a.md`
```
