//! CLOSED Ascend targets — AscendC (`cpp`) and PTO (`pto`).
//!
//! **Non-open-source.** Behind `feature = "ascend"`. This module is the concrete
//! demonstration of the user's goal — "move ascend-specific codegen to the same
//! level as the other target-specific parts": `AscendTarget` implements the
//! exact same [`CodegenTarget`] trait as every open backend, and joins via the
//! same [`register`] call. The only differences from an open target are (a) it
//! reads the `ub_size` hardware param and (b) it carries cube-kernel metadata —
//! both absorbed by the uniform [`EmitOpts`] / [`EmitOut`] shape, so Ascend is no
//! longer a bespoke signature (`convert_mlir_to_cpp(mlir, ub_size) -> CppOutput`)
//! sitting outside the dispatch.
//!
//! For the open tile-rs release this module (and the two `mlir_to_{cpp,pto}.rs`
//! files it includes) simply move to a separate private crate that depends on
//! `tile_codegen` and calls `register` — no change to the open core.

#[path = "../../rustc_codegen_tile/src/mlir_to_cpp.rs"]
mod cpp_impl;
#[path = "../../rustc_codegen_tile/src/mlir_to_pto.rs"]
mod pto_impl;

use crate::registry::TargetRegistry;
use crate::target::{CodegenTarget, EmitOpts, EmitOut, TargetMeta};

/// Register the closed Ascend targets as peers of the open ones.
pub fn register(r: &mut TargetRegistry) {
    r.register(Box::new(AscendTarget::Cpp));
    r.register(Box::new(AscendTarget::Pto));
}

/// The two Ascend codegen paths, behind one trait impl.
pub enum AscendTarget {
    /// AscendC C++ (`mlir_to_cpp`) → bisheng.
    Cpp,
    /// PTO-MLIR assembly (`mlir_to_pto`) → ptoas → bisheng.
    Pto,
}

impl CodegenTarget for AscendTarget {
    fn name(&self) -> &'static str {
        match self {
            AscendTarget::Cpp => "cpp",
            AscendTarget::Pto => "pto",
        }
    }

    fn emit(&self, mlir_text: &str, opts: &EmitOpts) -> Result<EmitOut, String> {
        match self {
            AscendTarget::Cpp => {
                // The one emitter that needs a hardware param + returns metadata.
                let out = cpp_impl::convert_mlir_to_cpp(mlir_text, opts.hw.ub_size)?;
                Ok(EmitOut {
                    source: out.source,
                    ext: "cpp",
                    meta: TargetMeta {
                        has_cube_kernel: out.has_cube_kernel,
                        kernel_names: out.kernel_names,
                    },
                })
            }
            AscendTarget::Pto => {
                let source = pto_impl::convert_mlir_to_pto(mlir_text)?;
                Ok(EmitOut { source, ext: "pto", meta: TargetMeta::default() })
            }
        }
    }
}
