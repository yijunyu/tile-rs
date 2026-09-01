# tile-rs

**tile-rs** is a Rust compiler framework for writing accelerator kernels *once* and lowering them to many hardware backends. You write kernels in a safe, idiomatic Rust tile DSL; a custom `rustc` codegen backend lowers them through MLIR to whichever target you select — the same source producing numerically-equivalent results across very different accelerators.

The same kernel lowers to **15 backends** through a pluggable registry — NVIDIA/AMD GPUs, Apple GPUs (Metal), Vulkan/SPIR-V, AWS Trainium, Huawei Ascend, and more. **Eight are validated end-to-end on real hardware today** — Apple Metal, Vulkan (MoltenVK), Huawei Ascend NPU, NVIDIA GPU, Google TPU, AWS Trainium, Intel Gaudi, and the portable `linalg`→CPU bridge (five with published [pu-rs.org](https://pu-rs.org) benchmark entries); the rest are codegen-proven and scaffolded for on-hardware bring-up.

## Features

- **Write once, target many** — a single Rust tile DSL (`tile_std`) lowers to 15 codegen backends through MLIR
- **Pluggable backends** — a `CodegenTarget` registry picks the target at build time via `TILERS_CODEGEN_PATH`; adding a backend never touches kernel source
- **Memory-safe kernels** — typed, RAII-style APIs structurally prevent whole classes of bugs common in hand-written accelerator C/C++
- **Numerical equivalence** — generated kernels match a CPU reference *on real hardware* (validated on 8 targets: Apple Metal, Vulkan/MoltenVK, Huawei Ascend NPU, NVIDIA GPU, Google TPU, AWS Trainium, Intel Gaudi, and the `linalg`→CPU bridge — five with benchmark entries at [pu-rs.org](https://pu-rs.org); every backend is also checked by a codegen generality test suite)
- **Prebuilt codegen backend** — the LLVM-linked backend ships as a release artifact; a kernel crate points `rustc` at it via `TILERS_CODEGEN_SO` + a bundled `.cargo/config.toml`, with no local LLVM build

## Supported Targets

The same `tile_std` kernel lowers to each backend below, selected with `TILERS_CODEGEN_PATH`. Every backend is proven by a **codegen generality matrix** — each `convert_mlir_to_<backend>` turns one shared MLIR module into syntactically-marked target source (15/15 tested). **On-HW** marks backends whose generated kernels are additionally validated end-to-end against a CPU reference on real hardware — via the [pu-rs.org](https://pu-rs.org) leaderboard (Metal, Ascend, NVIDIA, TPU, Trainium) and/or standalone runs (Vulkan on MoltenVK; Gaudi on a cloud instance; the `linalg` bridge on CPU).

| Backend | `TILERS_CODEGEN_PATH` | Target language | Codegen | On-HW |
|---------|----------------------|-----------------|:------:|-------|
| Apple GPU       | `metal`   | Metal Shading Language       | ✅ | ✅ M2 Max / M4 |
| Vulkan          | `vulkan`  | GLSL → SPIR-V                | ✅ | ✅ Apple Silicon (MoltenVK) |
| Ascend NPU      | `cpp`, `pto` | AscendC C++ / PTO-MLIR    | ✅ | ✅ 910B |
| NVIDIA GPU      | `cuda`    | CUDA C                       | ✅ | ✅ T4 / H20 |
| Google TPU      | `tpu`     | JAX / Pallas                 | ✅ | ✅ v5e |
| AWS Trainium    | `nki`     | NKI (Python)                 | ✅ | ✅ trn1 |
| Intel Gaudi     | `gaudi`   | TPC-C                        | ✅ | ✅ Gaudi (cloud) |
| Portable bridge | `linalg`  | MLIR `linalg` dialect        | ✅ | ✅ CPU |
| Moore Threads   | `musa`    | MUSA                         | ✅ | — |
| AMD Ryzen AI    | `aie`     | IRON (Python)                | ✅ | — |
| Cambricon MLU   | `bang`    | BANG-C                       | ✅ | — |
| Qualcomm Hexagon| `hexagon` | HVX / QNN                    | ✅ | — |
| Cerebras        | `csl`     | CSL                          | ✅ | — |
| Tenstorrent     | `ttmetal` | Tensix                       | ✅ | — |

All 15 emit their target language (codegen-tested); **8 are validated end-to-end on real hardware** today — Apple Metal (M2 Max / M4), Vulkan/SPIR-V (MoltenVK on Apple Silicon), Huawei Ascend NPU (910B), NVIDIA GPU (Tesla T4 / H20, via `mlir_to_cuda`), Google TPU (v5e, via `mlir_to_tpu` Pallas), AWS Trainium (trn1, via `mlir_to_nki`), Intel Gaudi (cloud, via `mlir_to_gaudi`), and the portable `linalg` bridge (CPU). Five carry published [pu-rs.org](https://pu-rs.org) benchmark entries (Metal, Ascend, NVIDIA, TPU, Trainium); Vulkan, Gaudi, and the CPU `linalg` path are validated by standalone runs.

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
