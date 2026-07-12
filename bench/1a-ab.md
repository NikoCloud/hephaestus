# Phase 1a Closeout — A/B Benchmark: Hephaestus (BF16) vs llama.cpp (F16)
## Date: 2026-07-13
## Hardware: GPU 0 = AMD Radeon AI PRO R9700 32GB (gfx1201), see `bench/hardware.md`
## Model: Qwen3-4B-Instruct-2507 (Hephaestus: BF16 safetensors; llama.cpp: F16 GGUF, 7.49 GiB)
## llama.cpp build: 33ca0dcb9 (9906), ROCm/HIP backend — same build as `bench/0-baseline-llamacpp-f16.md`

GPU 0 checked idle (`rocm-smi`, 0% util) before every run below. GPU 1 never
touched — llama.cpp binaries pinned to GPU 0 only via `-dev ROCm0` /
`HIP_VISIBLE_DEVICES=0` (verified: GPU 1 stayed at 0% util throughout).

---

## Result table

| Metric | Hephaestus BF16 | llama.cpp F16 | Ratio (Heph/llama) |
|---|---|---|---|
| Prefill tok/s (10-token prompt) | 99.06 ± 2.26 | 148.57 ± 1.68 | 0.67× |
| Prefill tok/s (512-token prompt) | 123.34 ± 2.68 | 1297.43 ± 3.06 | 0.10× |
| TTFT ms (10-token prompt) | 152.61 ± 2.32 | 67.32 ± 0.77 | 2.27× (slower) |
| TTFT ms (512-token prompt) | 4204.31 ± 88.86 | 394.63 ± 0.93 | 10.66× (slower) |
| Decode tok/s (forward-pass only — the G1a-2 metric) | 54.39 ± 0.02 | 61.82 ± 0.11 | 0.88× |
| Total time s (256 tokens, 10-token prompt) | 18.00 ± 0.00 | 5.19 ± 0.01 | 3.47× (slower) |
| Peak VRAM MB | ~29,590 | ~8,142 | 3.63× |
| VRAM after exit MB | ~74.7 (baseline) | ~74.7 (baseline) | 1.0× (both clean) |

**Three findings below change how this table should be read. Do not take the
"Decode tok/s" ratio (0.88×) as contradicting G1a-2's PASS — it doesn't, but
it isn't the same number as the 98.3%/109.2% cited there either. All three
are measured, not estimated, and none of them were touched to make a number
look better.**

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

## Finding 2: Hephaestus's own sampling step costs ~51.6ms/token — 3× its
## own forward pass, and ~800× llama.cpp's equivalent cost

Every "decode tok/s" figure quoted for Hephaestus throughout Phase 1a
(11.60 → 47.98 → 49.13 → 54.22 → 54.35) measured **only the GPU forward
pass** (`ctx.synchronize()` bracketing `forward()`), explicitly excluding
the host-side argmax scan that picks the next token. That was the right
methodology for isolating and fixing the model-compute bottleneck (G1a-2's
actual target), but it means "decode tok/s" was never a real end-to-end
number, and the gap shows up starkly in **Total time**, which this A/B
exercise asked for and which does include it.

Instrumented directly (`src/qwen_ab_bench.mojo`, timing bracketed
separately around the forward pass and around the host-side
`map_to_host()` + linear 151,936-element argmax scan):

| | Hephaestus | llama.cpp (`llama-simple`, real `llama_sampler_init_greedy()`) |
|---|---|---|
| forward-pass-only decode | 54.39 tok/s (18.4 ms/token) | 61.82 tok/s (16.2 ms/token) |
| **sampling cost per token** | **51.6 ms** | **0.065 ms** (`llama_perf_sampler_print`) |
| decode incl. real sampling | **14.29 tok/s** | **61.72–61.94 tok/s** (sampling is noise) |

llama.cpp's sampler adds essentially nothing to its own decode time (0.065ms
vs 16.2ms forward pass — 0.4%). Hephaestus's naive CPU argmax scan over the
full BF16→F32-cast 151,936-element vocab, once per token, costs **more than
the entire forward pass it follows**. This is why Total time (256 tokens,
10-token prompt) is 18.00s for Hephaestus vs 5.19s for llama.cpp — a 3.47×
gap that has nothing to do with the model-compute work G1a-2 measured, and
everything to do with an unoptimized sampling step that was never in scope
for G1a-2 (SPEC.md: "Greedy sampling only" was a scope statement about
*what* sampling to do, not a performance target for it).

**Not fixed here** — this is a discovery, not a task I was asked to do, and
SPEC.md's scope law says work should move the current gate, not expand
sideways into a newly-found one. Flagging for a decision: this is a real,
large, fixable inefficiency (the interface report already names a vendored
`nn.argmaxmin_gpu` kernel that would move the argmax onto the GPU and
plausibly collapse most of this 51.6ms), but it's outside what was asked
this session.

## Finding 3: Hephaestus's peak VRAM (~29.6GB) is far above the model's
## actual size — attributable to the loader's own verification design, not
## a leak

Model is 8.04GB in BF16. Hephaestus's peak VRAM during a run is **~29.6GB**,
vs llama.cpp's **~8.1GB** for the identical model (matching its 7.49GB F16
size plus a small compute buffer, `~301MB` for the 512-token case per
`llama_context`'s own reported buffer sizes). The gap traces to
`hephaestus.loader.load_arena`'s round-trip correctness check (`DECISIONS.md`
2026-07-12): it stages the 8GB weight blob into a pinned HostBuffer, copies
to an 8GB DeviceBuffer arena, then copies **back** to a second HostBuffer to
byte-compare — three ~8GB buffers alive at once by design, since "GPU copies
fail silently" was the reason that check exists in the first place. **This
is not a leak**: VRAM returns to baseline (~74.7MB, matching llama.cpp's own
post-exit baseline) within ~2 seconds of process exit, confirmed by
polling — the first check immediately on exit can catch a transient
higher reading before the driver reclaims memory, so wait briefly before
trusting a "VRAM after exit" number close to process termination.

Whether the round-trip verification buffer should be freed immediately
after the check (rather than living for the arena's whole lifetime) is a
memory-budget question relevant to Phase 2 (concurrent requests will each
want their own arena), not something changed here.

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
pixi run mojo build -I ~/projects/modular/max/kernels/src -I src src/qwen_ab_bench.mojo -o /tmp/qwen_ab_bench2
/tmp/qwen_ab_bench2 bench/ab_prompt_short_ids.txt 256   # 3 reps
/tmp/qwen_ab_bench2 bench/ab_prompt_long_ids.txt 8      # 3 reps (8, not 256 -- prefill dominates
                                                         #  the long-prompt case; full 256-decode not
                                                         #  needed to get stable prefill/TTFT numbers)
```

`src/qwen_ab_bench.mojo` mirrors the production `forward()` call sequence
exactly (same binary path as `qwen_generate.mojo`) with `perf_counter_ns`
brackets around (a) the forward pass and (b) the host-side argmax
separately, reported both ways per Finding 2.

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
