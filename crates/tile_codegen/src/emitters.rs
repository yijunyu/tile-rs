//! Real MLIR->source emitters, included from the `rustc_codegen_tile` tree.
//!
//! GATE: `feature = "emitters"`. These files import `crate::mlir_parse` (the
//! std-only parser, included by `lib.rs` under the same feature) and are part of
//! the LLVM-20 codegen crate, so this module compiles on adablue/910c — NOT in a
//! bare macOS build. The inclusion set mirrors `crates/mlir_to_aie_tests`
//! (the established no-NPU emitter test harness).
//!
//! Each file exposes `convert_mlir_to_<t>(mlir: &str) -> Result<String, String>`
//! (the 14 uniform open targets). AscendC/PTO are NOT here — they are closed and
//! live in `crate::ascend`.

#[path = "../../rustc_codegen_tile/src/mlir_to_gpu.rs"]
mod gpu;
#[path = "../../rustc_codegen_tile/src/mlir_to_musa.rs"]
mod musa;
#[path = "../../rustc_codegen_tile/src/mlir_to_spirv.rs"]
mod spirv;
#[path = "../../rustc_codegen_tile/src/mlir_to_msl.rs"]
mod msl;
#[path = "../../rustc_codegen_tile/src/mlir_to_nki.rs"]
mod nki;
#[path = "../../rustc_codegen_tile/src/mlir_to_aie.rs"]
mod aie;
#[path = "../../rustc_codegen_tile/src/mlir_to_bang.rs"]
mod bang;
#[path = "../../rustc_codegen_tile/src/mlir_to_gaudi.rs"]
mod gaudi;
#[path = "../../rustc_codegen_tile/src/mlir_to_tpu.rs"]
mod tpu;
#[path = "../../rustc_codegen_tile/src/mlir_to_csl.rs"]
mod csl;
#[path = "../../rustc_codegen_tile/src/mlir_to_hexagon.rs"]
mod hexagon;
#[path = "../../rustc_codegen_tile/src/mlir_to_ttmetal.rs"]
mod ttmetal;
#[path = "../../rustc_codegen_tile/src/mlir_to_linalg.rs"]
mod linalg;

pub use aie::convert_mlir_to_aie;
pub use bang::convert_mlir_to_bang;
pub use csl::convert_mlir_to_csl;
pub use gaudi::convert_mlir_to_gaudi;
pub use gpu::convert_mlir_to_gpu;
pub use hexagon::convert_mlir_to_hexagon;
pub use linalg::convert_mlir_to_linalg;
pub use msl::convert_mlir_to_msl;
pub use musa::convert_mlir_to_musa;
pub use nki::convert_mlir_to_nki;
pub use spirv::convert_mlir_to_spirv;
pub use tpu::convert_mlir_to_tpu;
pub use ttmetal::convert_mlir_to_ttmetal;
