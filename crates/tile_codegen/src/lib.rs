//! # tile_codegen — the open tile-rs codegen skeleton
//!
//! This crate is the **open core** of tile-rs: the generic, vendor-neutral
//! code-generation infrastructure, with **zero** LLVM / MLIR / melior / rustc
//! dependencies. Everything target-specific plugs in through one trait
//! ([`CodegenTarget`]) and one registry ([`TargetRegistry`]).
//!
//! ## What lives where
//!
//! * **open (this crate)** — the trait, the registry, the shared std-only MLIR
//!   parser (`mlir_parse`, under the `emitters` feature), and the open reference
//!   targets (CUDA, Metal, SPIR-V, NKI, AIE, …).
//! * **closed (the `ascend` feature / ultimately a separate private crate)** —
//!   the AscendC + PTO targets. They implement [`CodegenTarget`] *exactly* like
//!   the open ones and join via [`TargetRegistry::register`]; "moving Ascend to
//!   the same level as the other targets" is therefore a one-line registration,
//!   not a dispatch rewrite.
//!
//! ## Build matrix
//!
//! | features        | builds where      | contains                                   |
//! |-----------------|-------------------|--------------------------------------------|
//! | *(default)*     | anywhere (macOS)  | trait + registry + `DebugTarget` + `pico`  |
//! | `emitters`      | LLVM-20 box       | + the 14 open `convert_mlir_to_*` emitters |
//! | `ascend`        | LLVM-20 box       | + closed AscendC/PTO targets (peers)       |
//!
//! The default build is what keeps the skeleton verifiable standalone — see the
//! tests at the bottom of this file.

pub mod registry;
pub mod target;
pub mod targets;

pub use registry::TargetRegistry;
pub use target::{CodegenTarget, EmitOpts, EmitOut, HardwareParams, TargetMeta};
pub use targets::{DebugTarget, EmitterTarget};

// Under `emitters`: the shared std-only parser the real emitters import as
// `crate::mlir_parse`, plus the emitter source modules. Compiled on an LLVM-20
// box (the emitters live in the `rustc_codegen_tile` tree).
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_parse.rs"]
pub(crate) mod mlir_parse;

// Emitters that other emitters import via crate-root paths must be declared here
// under their `mlir_to_*` names, not aliased inside `mod emitters`:
//   - `mlir_to_pto`: shared PTO/MLIR helper used by 9 emitters (`use crate::mlir_to_pto`)
//   - `mlir_to_gpu`: imported by `mlir_to_musa` (`use crate::mlir_to_gpu`)
// `emitters` re-exports their public `convert_mlir_to_*` entry points.
// PTO is a closed target: the open emitters import `mlir_to_pto`'s shared
// helpers, but its own `convert_mlir_to_pto` entry and PTO-specific translators
// are unused in the open build — allow the resulting dead_code here.
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_to_pto.rs"]
#[allow(dead_code)]
pub(crate) mod mlir_to_pto;

#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_to_gpu.rs"]
pub(crate) mod mlir_to_gpu;

#[cfg(feature = "emitters")]
pub(crate) mod emitters;

// PICO is the one emitter that is NOT behind `emitters`.
//
// The other 14 import `crate::mlir_parse` and are part of the LLVM-20 codegen
// crate, so they can only build on that box. `mlir_to_pico` carries its own
// parser and its own intrinsic model and imports nothing from this crate, so it
// compiles in the default std-only build — which means the PICO target is
// registered, selectable and testable anywhere, including the macOS skeleton
// build that keeps this crate verifiable standalone.
//
// It has no vendor compiler behind it either: the emitted intrinsic program is
// assembled into a loadable `.om` by svp, so there is no `ptoas` step to gate a
// build on.
#[path = "../../rustc_codegen_tile/src/mlir_to_pico.rs"]
pub(crate) mod mlir_to_pico;

// Under `ascend`: the CLOSED AscendC + PTO targets (non-open-source).
#[cfg(feature = "ascend")]
pub(crate) mod ascend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_populated_and_selectable() {
        let r = TargetRegistry::with_builtin();
        assert!(!r.is_empty(), "registry should have at least the debug target");
        assert!(r.select("debug").is_some(), "debug target must be registered");
        assert!(r.select("does-not-exist").is_none());
        assert!(r.names().contains(&"debug"));
    }

    #[test]
    fn debug_target_emits_source() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("debug").expect("debug target");
        let out = t
            .emit("func.func @kernel() { return }", &EmitOpts::default())
            .expect("emit ok");
        assert!(out.source.contains("func.func"));
        assert!(out.source.contains("tile-rs debug target"));
        assert_eq!(out.ext, "mlir.txt");
    }

    #[test]
    fn empty_mlir_is_an_error() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("debug").unwrap();
        assert!(t.emit("   \n  ", &EmitOpts::default()).is_err());
    }

    #[test]
    fn register_adds_a_custom_target() {
        struct Noop;
        impl CodegenTarget for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn emit(&self, _m: &str, _o: &EmitOpts) -> Result<EmitOut, String> {
                Ok(EmitOut::default())
            }
        }
        let mut r = TargetRegistry::new();
        assert!(r.is_empty());
        r.register(Box::new(Noop));
        assert_eq!(r.len(), 1);
        assert!(r.select("noop").is_some());
    }

    #[test]
    fn pico_is_registered_in_the_default_build() {
        // The 16th target, and the only emitter outside the `emitters` feature.
        // If this ever needs an LLVM-20 box, something has started importing
        // `crate::mlir_parse` and the standalone build has quietly lost a target.
        let r = TargetRegistry::with_builtin();
        assert!(r.select("pico").is_some(), "pico must register without any feature");
        assert!(r.names().contains(&"pico"));
    }

    #[test]
    fn pico_lowers_a_kernel_to_an_intrinsic_program() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("pico").expect("pico target");
        let mlir = r#"
module {
  llvm.func @tile_softmax(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1024 : i32) : i32
    %3 = llvm.call @__tile_load_f32(%arg0, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %4 = llvm.call @__tile_softmax_f32(%3, %3, %1, %2) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %4, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let out = t.emit(mlir, &EmitOpts::default()).expect("emit ok");
        assert_eq!(out.ext, "pico.s");
        // There is no `softmax` instruction on PICO, so the listing shows the
        // five the vector engine actually runs rather than one that does not exist.
        for mnemonic in ["vmax", "vsemad", "vexp", "vsum", "vdiv"] {
            assert!(out.source.contains(mnemonic), "missing `{mnemonic}`:\n{}", out.source);
        }
        assert!(out.source.contains("PackPicoOm"), "the build path must name svp's packer");
    }

    #[test]
    fn pico_refuses_an_operation_it_has_no_intrinsic_for() {
        // A target with no vendor compiler behind it cannot fall back on "the
        // compiler will sort it out", so an unknown op is an error, not a stub.
        let r = TargetRegistry::with_builtin();
        let t = r.select("pico").unwrap();
        let mlir = r#"
module {
  llvm.func @k(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    %1 = llvm.call @__tile_conv3d_f32(%arg0) : (!llvm.ptr<1>) -> i32
    llvm.return
  }
}
"#;
        let err = t.emit(mlir, &EmitOpts::default()).unwrap_err();
        assert!(err.contains("no PICO intrinsic lowering"), "{err}");
    }

    #[test]
    fn hardware_params_carry_ub_size() {
        // The one asymmetry (AscendC's ub_size) rides on EmitOpts uniformly.
        let opts = EmitOpts {
            hw: HardwareParams { ub_size: 192 * 1024 },
        };
        assert_eq!(opts.hw.ub_size, 192 * 1024);
    }

    #[test]
    fn emitter_target_adapter_wraps_a_plain_fn() {
        // This is exactly how `rustc_codegen_tile` registers its existing
        // `convert_mlir_to_*` emitters — a plain
        // `fn(&str) -> Result<String, String>` lifted into the trait, keyed by
        // its TILERS_CODEGEN_PATH name.
        fn fake_convert(mlir: &str) -> Result<String, String> {
            if mlir.is_empty() {
                return Err("empty".into());
            }
            Ok(format!("// CUDA\n{mlir}"))
        }
        let mut r = TargetRegistry::new();
        r.register(Box::new(targets::EmitterTarget::new("cuda", "cu", fake_convert)));
        let t = r.select("cuda").expect("cuda target registered");
        let out = t.emit("module {}", &EmitOpts::default()).unwrap();
        assert_eq!(out.ext, "cu");
        assert!(out.source.starts_with("// CUDA"));
        assert!(t.emit("", &EmitOpts::default()).is_err());
    }
}
