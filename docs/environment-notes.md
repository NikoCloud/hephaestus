# Environment Notes --- Hephaestus Dev Environment

## Remote Access
- **Host:** `ssh cachyos` (192.168.1.92, user `nikocloud`)
- **Also reachable via:** Tailscale at `cachyos-x8664.tailfe30a4.ts.net` (100.65.54.20)
- **Working dir on remote:** `~/projects/hephaestus/`
- **Remote shell:** **fish**, not bash. No heredocs. `$` chars get interpreted.
  - **Policy:** All scripts written as files, executed via `pixi run mojo run path/to/script.mojo`. Never pipe code through interactive shell quoting.
- **Mojo:** `pixi run mojo` (Mojo 1.0.0b3.dev2026071006, MAX nightly)
  - `fn` keyword is deprecated --- use `def`
  - `alias` keyword is deprecated --- use `comptime`
  - No `-c` inline eval --- write to a file and `mojo run`
- **Python:** `pixi run python` (3.12, in pixi env). `safetensors` package is NOT installed --- use raw `struct`/`json` to read headers.
- **Torch:** System-wide 2.9.1+rocm6.3, deliberately NOT in pixi.

## GPU
- **Device 0:** AMD Radeon AI PRO R9700, gfx1201, 64 CUs, 32 GB VRAM
- **Device 1:** AMD Radeon RX 9070 XT, gfx1201, 64 CUs, 16 GB VRAM
- **Constraint:** A 27B model may be running on cachyos serving as a local LLM worker agent. It saturates BOTH cards. Must kill 27B before any GPU experiments, then restart it to use the worker again.
- **Dev GPU != orchestrator GPU** --- experimental GPU runs never execute on cards serving the agent dispatching them.

## MAX Kernel Library
- **Location:** `~/projects/modular/max/kernels/src/`
- **Layout module:** `layout/` subdirectory --- `TileTensor`, `TensorLayout`, `row_major`, `PointerStorage`, `DevicePointerStorage`
- **Import paths:** `from layout import TileTensor` / `from layout.tile_layout import row_major`
- **Matmul:** `linalg/matmul/__init__.mojo` -> `matmul()` -> `_matmul_gpu()` -> `gemm_kernel_rdna` on RDNA
- **Key TileTensor constructors:**
  - `TileTensor(ptr=UnsafePointer[Scalar[dtype], origin](), layout=layout)` --- raw pointer (keyword args)
  - `TileTensor(device_buffer, layout)` --- from DeviceBuffer
  - `TileTensor(device_pointer, layout)` --- from DevicePointer
- **Dynamic dimensions:** `row_major(Scalar[DType.int64](n), Idx[4])` --- mixed runtime/static. `Idx[n]` requires compile-time `n`.
- **All weight tensors have static shapes** --- hidden_size, intermediate_size, vocab_size are all compile-time constants. Only activation tensors (seq_len) need dynamic dimensions.
- **Struct fields cannot expose AnyOrigin** --- parameterize struct with `[origin: Origin[mut=True]]` and use `origin=Self.origin` in field types.
- **stack_allocation signature:** `[count: Int, dtype: DType, /, alignment: Int]` --- count is first positional param.

## Model Location
- **4B model:** `/mnt/models/models/qwen3-4b-instruct-2507/`
  - 3 shards: `model-0000{1,2,3}-of-00003.safetensors`
  - Index: `model.safetensors.index.json`
  - Config: `config.json`
- **Tiny random model:** `fixtures/tiny_random/model.safetensors` (single file, no index)
