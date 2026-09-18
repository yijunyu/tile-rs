//! # `tile` — the tile-rs kernel swissknife
//!
//! One command for accelerator kernels: identify what a file is, say what it is made
//! of, and move it between the 18 representations tile-rs knows about — lowering,
//! lifting, or optimizing, as the pair of forms dictates.
//!
//! ## What this crate is, and what it deliberately is not
//!
//! It is **pandoc's grammar** applied to kernels: extension-driven defaults, `-f`/`-t`
//! overrides, a route graph, keepable intermediates. It is **not** pandoc's capability.
//! tile-rs reads a few forms and writes many, so the graph is a fan-out, and every part
//! of this crate is built to make that shape discoverable rather than to hide it:
//! the exact counts belong to the table and are printed by `--list-forms`, which computes
//! them — this sentence said "reads 3 and writes 18" until the table reached 20, which is
//! what a number restated in prose beside the thing that knows it always ends up doing.
//! [`routes::list_forms`] prints the matrix, [`routes::plan`] refuses with the nearest
//! routes and the missing capability, and [`forms::Fidelity`] rides on every result.
//!
//! ## Layering
//!
//! ```text
//!   forms      what a representation IS; the table is data
//!   routes     the transformation graph over those forms
//!   profile    what a kernel is made of, over tile_codegen's default features
//!   platform   where we are and what accelerator is here (the ONLY cfg-gated module)
//!   cli        argv -> a request                  bin/tile.rs  request -> rendering
//! ```
//!
//! The daemon (M8) and the UI (M10) are additional renderings of the same request type,
//! not second implementations.
//!
//! ## Two invariants inherited from the compiler side
//!
//! * **Unmeasured refuses.** `HardwareParams` carries `measured`; when it is false every
//!   bound query returns `Err`, because approving a tiling with another chip's
//!   capacities approves precisely the tilings that fail. [`profile`] propagates that
//!   refusal instead of printing a plausible number.
//! * **Emit is pure.** O0/O1/O2 are byte-identical across runs for identical input, the
//!   same contract `tile_spec`'s `backend_emit_purity.feature` holds the emitters to.

// ── The open emitters, included at the CRATE ROOT ─────────────────────────────────
//
// This placement is not a style choice. The emitter sources import each other through
// ABSOLUTE crate paths -- `mlir_to_msl.rs` has `use crate::mlir_parse::...` and
// `use crate::mlir_to_pto::...`, `mlir_to_rvv.rs` has `use crate::mlir_to_linalg::...`
// -- so they must land at `crate::mlir_parse`, not `crate::emit::mlir_parse`. Putting
// the includes inside a submodule does not compile. `tile_spec` gets away with the same
// trick because its `tests/cucumber.rs` IS a crate root.
//
// There is no crate to depend on instead: `crates/rustc_codegen_tile/` has no
// Cargo.toml. It is a bare `src/` of emitter sources, which is why the repo root
// `exclude`s it -- the `crates/*` glob would otherwise fail on a manifest-less
// directory. The path-include is the only door.
//
// Consequence worth stating: `tile` is defined by files it does not own, which change
// on their own cadence. Nothing would notice an emitter changing our output -- so the
// golden files in `testdata/golden/` are simultaneously this crate's correctness tests
// AND its drift alarm against `rustc_codegen_tile`.
// `#[allow(warnings)]` on each: these are 2.5 MB of sources this crate does not own and
// must not "clean up" -- a cosmetic edit here is a silent behaviour change to the
// emitters, and the golden files would be the only thing to notice. Their warnings
// belong to `rustc_codegen_tile`; ours must not be drowned in them, and CI runs clippy
// with `-D warnings`.
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_parse.rs"]
pub(crate) mod mlir_parse;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/access_pattern.rs"]
pub(crate) mod access_pattern;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_aie.rs"]
pub(crate) mod mlir_to_aie;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_bang.rs"]
pub(crate) mod mlir_to_bang;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_csl.rs"]
pub(crate) mod mlir_to_csl;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_gaudi.rs"]
pub(crate) mod mlir_to_gaudi;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_gpu.rs"]
pub(crate) mod mlir_to_gpu;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_hexagon.rs"]
pub(crate) mod mlir_to_hexagon;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_linalg.rs"]
pub(crate) mod mlir_to_linalg;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_msl.rs"]
pub(crate) mod mlir_to_msl;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_musa.rs"]
pub(crate) mod mlir_to_musa;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_nki.rs"]
pub(crate) mod mlir_to_nki;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_pto.rs"]
pub(crate) mod mlir_to_pto;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_rvv.rs"]
pub(crate) mod mlir_to_rvv;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_spirv.rs"]
pub(crate) mod mlir_to_spirv;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_tpu.rs"]
pub(crate) mod mlir_to_tpu;
#[cfg(feature = "emitters")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../rustc_codegen_tile/src/mlir_to_ttmetal.rs"]
pub(crate) mod mlir_to_ttmetal;

pub mod backlog;
pub mod cli;
pub mod corpus;
pub mod daemon;
// The PICO emitter, from the sibling `tile-rs-pico` checkout. Included at the crate root
// like the others, though it needs nothing from `crate::` -- it carries its own MLIR
// parser and imports only std, which is what makes it the one emitter that builds without
// LLVM. The relative path assumes tile-rs and tile-rs-pico are siblings; without the
// checkout, build without `--features pico` and the target refuses cleanly.
#[cfg(feature = "pico")]
#[allow(warnings, clippy::all)]
#[rustfmt::skip]
#[path = "../../../../tile-rs-pico/build/mlir_to_pico.rs"]
pub(crate) mod mlir_to_pico;

pub mod emit;
pub mod engine;
pub mod forms;
pub mod json;
pub mod license;
pub mod lift_cpp;
pub mod lower_rs;
pub mod manifest;
pub mod mlir;
pub mod optimize;
pub mod passes;
pub mod platform;
pub mod profile;
pub mod provision;
pub mod routes;
pub mod run;
pub mod scratch;
pub mod sha256;
pub mod simenv;
pub mod toolchain;
pub mod torchref;
pub mod ui;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The output form to use when the user named none: the detected platform's own, so the
/// artifact runs here. With no accelerator this is the CPU `linalg` bridge — the default
/// therefore never produces an artifact the user cannot execute.
pub fn default_form_for(family: &str) -> &'static forms::Form {
    let id = match family {
        "apple-gpu" => "msl",
        "nvidia" => "gpu",
        "ascend" => "cpp",
        "trainium" => "nki",
        "tpu" => "tpu",
        "amd-npu" => "aie",
        // amd-gpu included: there is no ROCm/HIP target, so it falls back like "none".
        _ => "linalg",
    };
    forms::by_id(id).expect("default form exists")
}

/// The derived output name for a run with no `-o`: `<stem>.opt.<ext>`, beside the input.
pub fn derived_output(input: &str, form: &forms::Form) -> String {
    let stem = match input.rsplit_once('.') {
        Some((s, _)) => s,
        None => input,
    };
    format!("{stem}.opt.{}", form.primary_ext())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_form_id_is_unique_and_stable() {
        // Ids are serialized into MCP responses, attempt records and route reports, so a
        // rename is a breaking change to three things at once.
        let mut ids: Vec<&str> = forms::FORMS.iter().map(|f| f.id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate form id");
    }

    #[test]
    fn the_frozen_form_list_has_not_drifted() {
        // Freezing the list is the point: the drafts drifted between 14/15/16 emitters
        // and 18/19 forms three separate times.
        let ids: Vec<&str> = forms::FORMS.iter().map(|f| f.id).collect();
        assert_eq!(
            ids,
            vec![
                "tile", "mlir", "linalg", "rvv", "pto", "msl", "gpu", "musa", "spirv", "nki",
                "aie", "tpu", "bang", "gaudi", "hexagon", "ttmetal", "cpp", "pico", "csl", "debug",
            ]
        );
    }

    #[test]
    fn no_accelerator_defaults_to_a_form_that_runs_on_a_cpu() {
        assert_eq!(default_form_for("none").id, "linalg");
    }

    #[test]
    fn an_amd_gpu_falls_back_rather_than_getting_the_npu_form() {
        assert_eq!(default_form_for("amd-gpu").id, "linalg");
        assert_eq!(default_form_for("amd-npu").id, "aie");
    }

    #[test]
    fn the_derived_name_marks_the_artifact_as_optimized() {
        let msl = forms::by_id("msl").unwrap();
        assert_eq!(derived_output("softmax.rs", msl), "softmax.opt.metal");
        assert_eq!(
            derived_output("a/b/softmax.rs", msl),
            "a/b/softmax.opt.metal"
        );
    }
}
