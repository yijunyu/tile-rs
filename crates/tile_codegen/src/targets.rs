//! Built-in target registration — the open/closed seam.
//!
//! * The open skeleton always ships a self-contained [`DebugTarget`] so the
//!   trait + registry + emit path is exercisable with zero external deps.
//! * The real MLIR->source emitters (`gpu`, `msl`, `spirv`, `nki`, `aie`, …) are
//!   wired under the `emitters` feature — they live in the full
//!   `rustc_codegen_tile` tree and import `crate::mlir_parse`, so they compile on
//!   an LLVM-20 box. See `register_emitters`.
//! * The CLOSED AscendC + PTO targets come in under `ascend` — registered the
//!   SAME WAY (`register(Box::new(..))`), which is the entire point: "move
//!   ascend-specific codegen to the same level as the other targets" reduces to
//!   one registration call, not a bespoke dispatch arm.

use crate::registry::TargetRegistry;
use crate::target::{CodegenTarget, EmitOpts, EmitOut, TargetMeta};

/// Register every target compiled into this build.
pub fn register_builtin(r: &mut TargetRegistry) {
    r.register(Box::new(DebugTarget));

    #[cfg(feature = "emitters")]
    register_emitters(r);

    #[cfg(feature = "ascend")]
    crate::ascend::register(r);
}

/// Self-contained reference target. Proves the trait/registry/emit path end to
/// end with no LLVM, no parser, no toolchain — this is what makes the open
/// skeleton verifiable on any machine (including this macOS build). Echoes the
/// MLIR back inside a comment banner.
pub struct DebugTarget;

impl CodegenTarget for DebugTarget {
    fn name(&self) -> &'static str {
        "debug"
    }

    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        if mlir_text.trim().is_empty() {
            return Err("empty MLIR module".to_string());
        }
        let lines = mlir_text.lines().count();
        let funcs = mlir_text.matches("func.func").count();
        Ok(EmitOut {
            source: format!(
                "// tile-rs debug target\n// {lines} MLIR lines, {funcs} func.func op(s)\n{mlir_text}\n"
            ),
            ext: "mlir.txt",
            meta: TargetMeta::default(),
        })
    }
}

// ── Real emitters (LLVM-20 build) ─────────────────────────────────────────────
// Each wraps an existing `convert_mlir_to_<t>` from the rustc_codegen_tile tree.
// 14 of 15 share the signature `(mlir: &str) -> Result<String, String>`; only
// AscendC takes `ub_size` and returns richer metadata — handled in `crate::ascend`.
#[cfg(feature = "emitters")]
fn register_emitters(r: &mut TargetRegistry) {
    use crate::emitters::*;
    r.register(Box::new(EmitterTarget::new("gpu", "cu", convert_mlir_to_gpu)));
    r.register(Box::new(EmitterTarget::new("musa", "mu", convert_mlir_to_musa)));
    r.register(Box::new(EmitterTarget::new("spirv", "comp", convert_mlir_to_spirv)));
    r.register(Box::new(EmitterTarget::new("msl", "metal", convert_mlir_to_msl)));
    r.register(Box::new(EmitterTarget::new("nki", "py", convert_mlir_to_nki)));
    r.register(Box::new(EmitterTarget::new("aie", "py", convert_mlir_to_aie)));
    r.register(Box::new(EmitterTarget::new("bang", "mlu", convert_mlir_to_bang)));
    r.register(Box::new(EmitterTarget::new("gaudi", "c", convert_mlir_to_gaudi)));
    r.register(Box::new(EmitterTarget::new("tpu", "py", convert_mlir_to_tpu)));
    r.register(Box::new(EmitterTarget::new("csl", "csl", convert_mlir_to_csl)));
    r.register(Box::new(EmitterTarget::new("hexagon", "c", convert_mlir_to_hexagon)));
    r.register(Box::new(EmitterTarget::new("ttmetal", "cpp", convert_mlir_to_ttmetal)));
    r.register(Box::new(EmitterTarget::new("linalg", "mlir", convert_mlir_to_linalg)));
}

/// Adapter that turns a plain `(mlir: &str) -> Result<String, String>` emitter
/// (the 14 uniform backends) into a [`CodegenTarget`]. Std-only and ALWAYS
/// available — host crates (e.g. `rustc_codegen_tile`) register their own
/// `convert_mlir_to_*` functions through it without enabling the `emitters`
/// feature, so the emitter source can stay in exactly one place.
pub struct EmitterTarget {
    name: &'static str,
    ext: &'static str,
    f: fn(&str) -> Result<String, String>,
}

impl EmitterTarget {
    pub fn new(name: &'static str, ext: &'static str, f: fn(&str) -> Result<String, String>) -> Self {
        Self { name, ext, f }
    }
}

impl CodegenTarget for EmitterTarget {
    fn name(&self) -> &'static str {
        self.name
    }
    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        let source = (self.f)(mlir_text)?;
        Ok(EmitOut { source, ext: self.ext, meta: TargetMeta::default() })
    }
}
