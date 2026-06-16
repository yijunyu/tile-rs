//! The `CodegenTarget` trait — the single seam every backend implements.
//!
//! This is the open tile-rs contract. A target turns a merged MLIR module into
//! target **source** (`emit`). Driving the vendor **toolchain** (source ->
//! object/binary: nvcc / xcrun / bisheng / ptoas) stays host-side in the build
//! crate, because it needs `rustc_session::Session` + the per-vendor settings —
//! deliberately NOT in this trait, so the skeleton is pure, std-only, and
//! testable with no LLVM/CANN/toolchain present (the property that already lets
//! the emitters run 262 tests off-NPU). `emit` is the IP-bearing, unit-testable
//! core; `compile` is mechanical glue layered on top by name.

/// Hardware / codegen knobs a target may read. Host-toolchain targets (CUDA,
/// Metal, …) ignore this; Ascend reads `ub_size`. Std-only so the skeleton stays
/// LLVM-free. This absorbs the one asymmetry in today's backends — `mlir_to_cpp`
/// is the only emitter that takes an extra hardware arg.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HardwareParams {
    /// Unified-buffer size in bytes (Ascend cube/vector unit). `0` = unused.
    pub ub_size: usize,
}

/// Inputs to [`CodegenTarget::emit`] beyond the MLIR text itself. Std-only.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmitOpts {
    pub hw: HardwareParams,
}

/// Optional per-target metadata the host's `compile` step consumes. The generic
/// targets leave it default; Ascend fills it (this is exactly today's
/// `CppOutput { has_cube_kernel, kernel_names }`, lifted into the uniform shape
/// so Ascend stops being a bespoke return type).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TargetMeta {
    /// Ascend: a cube-unit (`__aicore__`) kernel is present in the source.
    pub has_cube_kernel: bool,
    /// Emitted kernel symbol names (Ascend uses these to drive bisheng).
    pub kernel_names: Vec<String>,
}

/// Result of [`CodegenTarget::emit`]: target source + a suggested on-disk
/// extension + optional metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitOut {
    pub source: String,
    /// Suggested file extension for `source` (e.g. `"cu"`, `"metal"`, `"cpp"`).
    pub ext: &'static str,
    pub meta: TargetMeta,
}

/// A code-generation target (CUDA, Metal, SPIR-V, AscendC, …).
///
/// Adding a target is: implement this trait, then `register(Box::new(MyTarget))`
/// on a [`crate::registry::TargetRegistry`]. That is the whole extension surface
/// — no enum to extend, no dispatch `match` arm to add. The closed Ascend
/// backend implements this exactly like the open ones; the only difference is it
/// lives in a feature-gated (and ultimately separate-repo) module.
pub trait CodegenTarget {
    /// Stable id, matched against `TILERS_CODEGEN_PATH` (e.g. `"gpu"`, `"msl"`,
    /// `"cpp"`, `"pto"`). Must be unique within a registry.
    fn name(&self) -> &'static str;

    /// Pure MLIR -> target source. Deterministic: no filesystem, no environment,
    /// no toolchain invocation. This is the unit-testable core.
    fn emit(&self, mlir_text: &str, opts: &EmitOpts) -> Result<EmitOut, String>;
}
