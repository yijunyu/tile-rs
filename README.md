# tile-rs

**tile-rs** is a Rust compiler framework for writing accelerator kernels *once* and lowering them to many hardware backends. You write kernels in a safe, idiomatic Rust tile DSL; a custom `rustc` codegen backend lowers them through MLIR to whichever target you select — the same source producing numerically-equivalent results across very different accelerators.

The same kernel lowers to **15 backends** through a pluggable registry — NVIDIA/AMD GPUs, Apple GPUs (Metal), Vulkan/SPIR-V, AWS Trainium, Huawei Ascend, and more. Several are validated end-to-end on silicon today (Apple Metal, Vulkan, Ascend NPU); the rest are codegen-proven and scaffolded for on-hardware bring-up.

## Features

- **Write once, target many** — a single Rust tile DSL (`tile_std`) lowers to 15 codegen backends through MLIR
- **Pluggable backends** — a `CodegenTarget` registry picks the target at build time via `TILERS_CODEGEN_PATH`; adding a backend never touches kernel source
- **Memory-safe kernels** — typed, RAII-style APIs structurally prevent whole classes of bugs common in hand-written accelerator C/C++
- **Numerical equivalence** — generated kernels match a CPU reference *on real hardware* (validated on Apple Metal, Vulkan, and Ascend NPU; every backend is checked by a codegen generality test suite)
- **Prebuilt codegen backend** — the LLVM-linked backend ships as a release artifact; a kernel crate points `rustc` at it via `TILERS_CODEGEN_SO` + a bundled `.cargo/config.toml`, with no local LLVM build

## Supported Targets

The same `tile_std` kernel lowers to each backend below, selected with `TILERS_CODEGEN_PATH`. Every backend is proven by a **codegen generality matrix** — each `convert_mlir_to_<backend>` turns one shared MLIR module into syntactically-marked target source (15/15 tested). **On-HW** is checked where generated kernels are additionally validated end-to-end against a CPU reference on that silicon.

| Backend | `TILERS_CODEGEN_PATH` | Target language | Codegen | On-HW |
|---------|----------------------|-----------------|:------:|-------|
| Apple GPU       | `metal`   | Metal Shading Language       | ✅ | ✅ Apple Silicon |
| Vulkan          | `vulkan`  | GLSL → SPIR-V                | ✅ | ✅ MoltenVK |
| Ascend NPU      | `cpp`, `pto` | AscendC C++ / PTO-MLIR    | ✅ | ✅ 910B |
| NVIDIA GPU      | `cuda`    | CUDA C                       | ✅ | — |
| Moore Threads   | `musa`    | MUSA                         | ✅ | — |
| AWS Trainium    | `nki`     | NKI (Python)                 | ✅ | — |
| AMD Ryzen AI    | `aie`     | IRON (Python)                | ✅ | — |
| Cambricon MLU   | `bang`    | BANG-C                       | ✅ | — |
| Intel Gaudi     | `gaudi`   | TPC-C                        | ✅ | — |
| Qualcomm Hexagon| `hexagon` | HVX / QNN                    | ✅ | — |
| Cerebras        | `csl`     | CSL                          | ✅ | — |
| Google TPU      | `tpu`     | JAX / Pallas                 | ✅ | — |
| Tenstorrent     | `ttmetal` | Tensix                       | ✅ | — |
| Portable bridge | `linalg`  | MLIR `linalg` dialect        | ✅ | — |

All 15 emit their target language (codegen-tested); 3 are validated end-to-end on silicon today (Apple Metal, Vulkan, Ascend NPU).

## Architecture

tile-rs is backend-agnostic at its core. A Rust kernel is lowered to MLIR *once*, then a pluggable backend converts that MLIR to the selected target's source — so the same kernel runs on very different hardware:

```
                Rust kernel  (tile_std DSL)
                       │
                       ▼
              rustc_codegen_tile        custom rustc backend: MIR → MLIR
                       │
                       ▼
                     MLIR ──►  CodegenTarget registry
                                    │
        ┌──────────┬──────────┬─────┴────┬──────────┬───────────────┐
        ▼          ▼          ▼          ▼          ▼               ▼
     Metal       CUDA C     SPIR-V     AscendC     NKI       AIE / TPC / …
   Apple GPU    NVIDIA      Vulkan     Ascend    Trainium   Ryzen AI / Gaudi
```

Each backend supplies its own host runtime, registered behind the `CodegenTarget` trait.

### Crates (target-agnostic framework)

| Crate | Purpose |
|-------|---------|
| `tile_std` | Kernel-side DSL: tile intrinsics + buffer API, `#![no_core]` device runtime |
| `tile_std_macros` | Kernel attribute macros |
| `tile_codegen` | The `CodegenTarget` trait + `TargetRegistry` — the pluggable backend skeleton |
| `tile_spec` | Executable Gherkin (Given/When/Then) spec layer + the codegen-generality test suite |
| `tile_hal` | Vendor-neutral Hardware Abstraction Layer (host-side device/stream/buffer + backend selection) |
| `rustc_codegen_tile` | Custom rustc codegen backend (MIR → MLIR → backend source). The 15 backend **emitters** are open source — `src/mlir_to_*.rs`, exercised by `tile_spec`. The LLVM-dependent backend that links them into a runnable `librustc_codegen_tile.so` is distributed as a prebuilt release artifact (it needs LLVM 20; the emitters do not). |

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install.sh | bash
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
