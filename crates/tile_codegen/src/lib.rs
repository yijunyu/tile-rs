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
//! | *(default)*     | anywhere (macOS)  | trait + registry + `DebugTarget`           |
//! | `emitters`      | LLVM-20 box       | + the 15 `convert_mlir_to_*` emitters      |
//! | `ascend`        | LLVM-20 box       | + closed AscendC/PTO targets (peers)       |
//!
//! The default build is what keeps the skeleton verifiable standalone — see the
//! tests at the bottom of this file.

pub mod bandit;
pub mod clustering;
pub mod driver;
pub mod hardware;
pub mod frontier;
pub mod hazard;
pub mod layout;
pub mod pointwise;
pub mod registry;
pub mod roofline;
pub mod schedule;
pub mod target;
pub mod targets;

pub use bandit::{ArmStat, BanditState, bandit_enabled};
pub use clustering::{BehaviourVector, Centroids, ClusterTracker, kmeans};
pub use driver::{DriveReport, Pull, RunOutcome, drive, drive_masked, load_or_new, save};
pub use frontier::{Candidate, Flags, Frontier, Strategy, Transform, transforms_for};
pub use hazard::{
    BufId, Buffers, Edge, Hazard, VecOp, barrier_points, hazards, report, unsynced,
};
pub use hardware::{HwSignature, Resource, SATURATION};
pub use layout::{
    BLOCK_BYTES, GuardedTransform, LayoutError, PlanReason, RowLayout, TRANSFORMS, TransferPlan,
    proxy_disagrees,
};
pub use pointwise::{
    AxisClass, Collapsed, InstanceKey, Operand, OperandKind, PointwiseSchema, Promoted,
    PromotionKind, ScalarType, TilePlan, TypeCategory, collapse, elementwise_dtypes, plan_tiles,
    promote_types,
};
pub use registry::TargetRegistry;
pub use roofline::{
    BYTES_PER_GB, BOUND_FRACTION, MIN_TRANSFER_BYTES, ROOFLINE_TRANSFORMS, Regime, Roofline,
    BATCH_FORMULA_CAP_2B, Feasibility, Staging, StagingPlan, TILE_CAP_2B, Traffic,
    TransferShape, achieved_gbps, classify, max_footprint_bytes, plan_staging,
    rows_per_transfer_needed, tile_cap, whole_row_admissible,
};
pub use schedule::{
    KNOBS, K_UNROLL_ARMS, Knob, KnobSite, MAX_K_UNROLL, MAX_TILE, SPLIT_P_ARMS, Schedule,
    TILE_ARMS, TargetKnobs, ValueEffect,
};
pub use target::{
    CodegenTarget, EmitOpts, EmitOut, HardwareParams, NanTest, TargetMeta, TargetSemantics,
};
pub use targets::{DebugTarget, EmitterTarget};

// Under `emitters`: the shared std-only parser the real emitters import as
// `crate::mlir_parse`, plus the emitter source modules. Compiled on an LLVM-20
// box (the emitters live in the `rustc_codegen_tile` tree).
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_parse.rs"]
pub(crate) mod mlir_parse;

// The vectorisation predicate `mlir_to_gpu` consults as `crate::access_pattern`.
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/access_pattern.rs"]
pub(crate) mod access_pattern;

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

// `mlir_to_linalg` is declared at crate root because `mlir_to_rvv` imports it as
// `crate::mlir_to_linalg` (the RVV backend is a thin wrapper over the linalg
// egress). `emitters` re-exports its entry point.
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_to_linalg.rs"]
pub(crate) mod mlir_to_linalg;

#[cfg(feature = "emitters")]
pub(crate) mod emitters;

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
    fn a_nan_test_lowers_to_not_eq_where_ne_is_ordered() {
        // Ascend's `!=` returns FALSE for a NaN operand, exactly like `<`, so
        // `Compare(v, v, NE)` is a NaN test that never fires. The backend has to
        // emit `!(v == v)` instead, and this is where it learns that.
        assert_eq!(TargetSemantics::ascend_arch32().nan_test().unwrap(), NanTest::NotEq);
        // Everywhere else the ordinary spelling is correct.
        assert_eq!(TargetSemantics::default().nan_test().unwrap(), NanTest::Ne);
        // arch35 reworks the vector unit, so arch32's answer does not transfer
        // and the query refuses rather than guessing.
        assert!(TargetSemantics::ascend_arch35_unmeasured().nan_test().is_err());
    }

    #[test]
    fn bitwise_ops_count_in_sixteen_bit_lanes_on_ascend() {
        let a = TargetSemantics::ascend_arch32();
        // An int32 buffer of n elements needs 2n: passing n leaves HALF the
        // destination untouched, which is how this was found.
        assert_eq!(a.bitwise_count(1024, 4).unwrap(), 2048);
        // 16-bit dtypes are already one lane each.
        assert_eq!(a.bitwise_count(1024, 2).unwrap(), 1024);
        // A 1-byte dtype rounds up rather than dropping the odd element.
        assert_eq!(a.bitwise_count(3, 1).unwrap(), 2);
        assert_eq!(a.bitwise_count(4, 1).unwrap(), 2);
        // Elsewhere the count is simply the element count.
        assert_eq!(TargetSemantics::default().bitwise_count(1024, 4).unwrap(), 1024);
        // arch35 refuses: a half-written destination is silent.
        assert!(TargetSemantics::ascend_arch35_unmeasured().bitwise_count(1024, 4).is_err());
    }

    #[test]
    fn semantics_ride_on_emit_opts_beside_the_resource_bounds() {
        // A target reads what its instructions MEAN next to how much fits.
        let opts = EmitOpts {
            hw: HardwareParams::ascend_910b(),
            sem: TargetSemantics::ascend_arch32(),
        };
        assert_eq!(opts.sem.nan_test().unwrap(), NanTest::NotEq);
        assert_eq!(opts.hw.npu_arch, "DAV_2201");
        // and the default target is unaffected by either
        let plain = EmitOpts::default();
        assert_eq!(plain.sem.nan_test().unwrap(), NanTest::Ne);
        assert_eq!(plain.sem.bitwise_count(99, 4).unwrap(), 99);
    }

    #[test]
    fn an_unmeasured_arch_refuses_rather_than_borrowing_another_chips_numbers() {
        let a = HardwareParams::ascend_910b();
        assert_eq!(a.npu_arch, "DAV_2201");
        assert!(a.measured);

        // A 910B number is not a 950 number. Validating a 950 tiling against
        // arch32 capacities would approve exactly the tilings that fail there,
        // so every check refuses instead.
        let b = HardwareParams::ascend_950_unmeasured();
        assert_eq!(b.npu_arch, "DAV_3510");
        assert!(!b.measured);
        for e in [
            b.check_ub(1024).unwrap_err(),
            b.check_repeat(64, 4).unwrap_err(),
            b.check_dma_stride(8, "src").unwrap_err(),
            b.check_cube_tile(16, 16, 16, 2).unwrap_err(),
        ] {
            assert!(e.contains("MEASURED"), "{e}");
            assert!(e.contains("DAV_3510"), "{e}");
        }
        // and the refusal is not "everything is fine": a shape the 910B accepts
        // is still refused for the 950, because nobody has looked.
        assert!(a.check_ub(1024).is_ok());
    }

    #[test]
    fn the_config_capacities_unblock_the_cube_without_relaxing_the_refusal() {
        // The _unmeasured trio must keep refusing: the policy they encode is about
        // whether anybody looked at the SILICON, and a vendor platform config is not
        // that. Flipping `measured` inside them would make the name contradict the
        // value and would silently license planning everywhere.
        for a in [
            HardwareParams::ascend_950_unmeasured(),
            HardwareParams::ascend_950pr_unmeasured(),
            HardwareParams::ascend_950dt_unmeasured(),
        ] {
            assert!(!a.measured);
            assert!(a.check_ub(1024).is_err());
            assert!(a.check_cube_tile(128, 128, 64, 2).is_err());
        }

        // The measured variants carry the config numbers and are opt-in by name.
        let pr = HardwareParams::ascend_950pr();
        let dt = HardwareParams::ascend_950dt();
        for a in [pr, dt] {
            assert!(a.measured);
            assert_eq!(a.npu_arch, "DAV_3510");
            assert_eq!(a.ub_size, 253952);
            assert_eq!(a.l0c_bytes, 262144);
            // L0_A, L0_B and L1 are UNCHANGED from 910B; only UB and L0_C move.
            assert_eq!(a.l0a_bytes, HardwareParams::ascend_910b().l0a_bytes);
            assert_eq!(a.l0b_bytes, HardwareParams::ascend_910b().l0b_bytes);
        }
        // Different silicon ids survive, so a ceiling still cannot be misfiled.
        assert_ne!(pr.chip, dt.chip);
        // The DT eval-image OP gap is not a capacity and must survive measurement.
        assert!(pr.check_vendor_op("Cat").is_ok());
        assert!(dt.check_vendor_op("ConcatD").is_err());

        // THE POINT OF THE EXERCISE: 2x L0_C admits a 2x larger fp32 output block.
        // 256x256 needs 262144B of L0_C -- fits on 950, refused on 910B.
        let b = HardwareParams::ascend_910b();
        assert!(pr.check_cube_tile(256, 256, 64, 2).is_ok());
        assert!(b.check_cube_tile(256, 256, 64, 2).is_err());
        // and the 910B-sized block is still legal on both
        assert!(pr.check_cube_tile(256, 128, 64, 2).is_ok());
        assert!(b.check_cube_tile(256, 128, 64, 2).is_ok());

        // AND THE COUNTER-EXAMPLE, so the extra UB is not read as a tile target:
        // group_norm's float32 working set at the whole-budget tile 11072 is ~238976B.
        // It is INSIDE ub_size and it failed correctness on both 950 boards
        // (job_0fa673447733, job_63f8c3262ae4). check_ub therefore passing it is
        // CORRECT and is not a promise that the tile is sound.
        assert!(pr.check_ub(238_976).is_ok());
    }

    #[test]
    fn pr_and_dt_share_an_isa_target_but_are_different_machines() {
        let pr = HardwareParams::ascend_950pr_unmeasured();
        let dt = HardwareParams::ascend_950dt_unmeasured();
        // Same compile identity — there is no separate DT flag.
        assert_eq!(pr.npu_arch, "DAV_3510");
        assert_eq!(dt.npu_arch, "DAV_3510");
        assert!(!pr.measured && !dt.measured);
        // Different silicon ids, so a ceiling measured on one cannot be
        // filed under the other.
        assert_eq!(pr.chip, "Ascend950PR_9589");
        assert_eq!(dt.chip, "Ascend950DT_9582");
        assert_ne!(pr.chip, dt.chip);
        // Capacity checks still refuse on both (nothing measured on either).
        assert!(pr.check_ub(1024).unwrap_err().contains("Ascend950PR_9589"));
        assert!(dt.check_ub(1024).unwrap_err().contains("Ascend950DT_9582"));
        // Eval-image OP gap is DT-only evidence: PR has the dyn-shape set,
        // DT does not — and that refusal does not depend on `measured`.
        assert!(pr.check_vendor_op("Cat").is_ok());
        let e = dt.check_vendor_op("ConcatD").unwrap_err();
        assert!(e.contains("no OP JSON"), "{e}");
        assert!(e.contains("Ascend950DT_9582"), "{e}");
        // 910B and the generic 950 SKU have no known OP gap.
        assert!(HardwareParams::ascend_910b().check_vendor_op("Cat").is_ok());
        assert!(HardwareParams::ascend_950_unmeasured().check_vendor_op("Cat").is_ok());
    }

    #[test]
    fn c7_the_repeat_field_is_eight_bits() {
        let hw = HardwareParams::ascend_910b();
        // 255 repeats x 64 lanes is the furthest one f32 instruction reaches.
        assert!(hw.check_repeat(16320, 4).is_ok());
        let e = hw.check_repeat(16321, 4).unwrap_err();
        assert!(e.contains("C7") && e.contains("16320"), "{e}");
        // A 2-byte dtype gets 128 lanes, so it reaches twice as far.
        assert!(hw.check_repeat(32640, 2).is_ok());
        assert!(hw.check_repeat(32641, 2).is_err());
        // What a tile search must not propose: 256x256 f32 is 1024 repeats.
        assert!(hw.check_repeat(256 * 256, 4).is_err());
        assert!(hw.check_repeat(64 * 64, 4).is_ok());
    }

    #[test]
    fn c8_the_dma_stride_field_is_sixteen_bits() {
        let hw = HardwareParams::ascend_910b();
        assert!(hw.check_dma_stride(32767, "src").is_ok());
        assert!(hw.check_dma_stride(32768, "src").unwrap_err().contains("C8"));
    }

    #[test]
    fn c9_a_cube_tile_must_fit_l0a_l0b_and_l0c_at_once() {
        let hw = HardwareParams::ascend_910b();
        // The shipped 910B tiling: L0A and L0B both full, L0C full.
        assert!(hw.check_cube_tile(128, 256, 128, 2).is_ok());
        // Fills L0B and starves L0A. It FITS -- C9 is a capacity bound, not a
        // performance one -- and it was the slowest of four measured.
        assert!(hw.check_cube_tile(64, 512, 64, 2).is_ok());
        // 512x512x128 runs out of L0A first: 512*128*2 = 131072B of a 65536B
        // buffer. The checker names the buffer that actually overflows.
        let a = hw.check_cube_tile(512, 512, 128, 2).unwrap_err();
        assert!(a.contains("C9") && a.contains("L0A"), "{a}");
        // A shape that fits both operand buffers and still cannot accumulate:
        // k=16 keeps L0A and L0B at 4096B each, while m*n*4 = 262144B is twice
        // what L0C holds.
        let c = hw.check_cube_tile(256, 256, 16, 2).unwrap_err();
        assert!(c.contains("C9") && c.contains("L0C"), "{c}");
    }

    #[test]
    fn an_unset_limit_never_rejects() {
        let none = HardwareParams::default();
        assert!(none.check_repeat(1 << 20, 4).is_ok());
        assert!(none.check_dma_stride(1 << 20, "src").is_ok());
        assert!(none.check_cube_tile(4096, 4096, 4096, 4).is_ok());
        assert!(none.check_ub(1 << 30).is_ok());
    }

    #[test]
    fn hardware_params_carry_ub_size() {
        // The one asymmetry (AscendC's ub_size) rides on EmitOpts uniformly.
        let opts = EmitOpts {
            hw: HardwareParams { ub_size: 192 * 1024, ..Default::default() },
            ..Default::default()
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
