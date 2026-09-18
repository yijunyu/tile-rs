// SPDX-License-Identifier: Apache-2.0
//
//! `access_pattern` — when is a vector width worth applying?
//!
//! # Why this exists
//!
//! The obvious reading of the Mojo study (`docs/MOJO_TO_TILERS_MIGRATION.md`) is
//! "vectorise everything". That is wrong, and we have the counter-example
//! measured. On an M1 Ultra:
//!
//! | inner loop | example | width-4 vs scalar |
//! |---|---|---|
//! | contiguous, issue-bound | f16 512³ GEMM | **3.35× faster** |
//! | contiguous, bandwidth-saturated | elementwise n=4M | no change (both at roofline) |
//! | dynamic-index gather | DS4 MXFP4 MoE dequant | **1.25× SLOWER** |
//!
//! The third row is the one that matters. `__tile_mul_mv_id_mxfp4_pair_swiglu`
//! spends its inner loop on 16 dynamic-index lookups into a threadgroup table
//! per block per row. A vector width has nothing to hold there: the measured
//! width-4 port ran 155.12 µs against the shipping kernel's 123.67 µs, on
//! identical output. Applying a width unconditionally would pessimise every
//! quantised-dequant kernel in the tree — and quantised matvec is most of what
//! a decode step runs.
//!
//! So the vectorisation policy needs a predicate, and this module is it. It
//! classifies a `__tile_*` intrinsic by the access pattern of its inner loop and
//! answers one question: may a vector width be applied here?
//!
//! # What it is not
//!
//! It does not decide the width — that is [`crate::access_pattern::vector_width`]
//! reading `HardwareParams`. It does not look at shapes; a bandwidth-saturated
//! elementwise kernel is still classified [`AccessClass::Contiguous`] because
//! vectorising it is *harmless* (measured: no change), and gating on shape would
//! need runtime dims the emitter does not have.
//!
//! The distinction that earns its keep is Contiguous-or-Strided vs **Gather**.

use std::fmt;

/// How an intrinsic's inner loop touches memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessClass {
    /// Unit stride. A vector load maps directly onto the loop body.
    Contiguous,
    /// Compile-time-constant non-unit stride — GEMM's `B[k*N + n]`. The width
    /// applies across the stride dimension (give each thread `VEC` columns), which
    /// is exactly the transform measured at 3.35× on f16 512³.
    ConstStrided,
    /// Dynamic index: a table lookup, a routed expert id, a gathered row. A
    /// vector width has nothing to hold. Measured 1.25× *worse* on the DS4 MoE
    /// decode kernel, so this class is a hard no.
    Gather,
    /// Block- or warp-wide reduction. The *load* side is usually contiguous and
    /// may be vectorised; the combine step is a primitive, not a loop.
    Reduction,
    /// Not classified. Never vectorise on speculation — an unrecognised
    /// intrinsic is far more likely to be one of the ~200 fused/quantised
    /// kernels in this tree than a plain elementwise one.
    Opaque,
}

impl fmt::Display for AccessClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AccessClass::Contiguous => "contiguous",
            AccessClass::ConstStrided => "const-strided",
            AccessClass::Gather => "gather",
            AccessClass::Reduction => "reduction",
            AccessClass::Opaque => "opaque",
        };
        f.write_str(s)
    }
}

impl AccessClass {
    /// May a vector width be applied to this loop at all?
    ///
    /// `Reduction` is included because the load side of a reduction is a normal
    /// contiguous sweep — `rms_norm` accumulating `v*v` over a row vectorises
    /// the same way an elementwise kernel does; only the final combine is a
    /// primitive.
    pub fn vectorizable(self) -> bool {
        matches!(
            self,
            AccessClass::Contiguous | AccessClass::ConstStrided | AccessClass::Reduction
        )
    }
}

/// Strip the dtype/width suffix so `__tile_add_f32`, `__tile_add_f16` and
/// `__tile_add_f32_4` all reduce to the same family.
fn family(intrinsic: &str) -> &str {
    // Accept a bare name or any call line — `%5 = llvm.call @__tile_x(...) : ...`
    // included, which is the form the emitters actually iterate over.
    let s = match intrinsic.find("@__tile_") {
        Some(at) => &intrinsic[at + 1..],
        None => intrinsic.trim().trim_start_matches('@'),
    };
    s.split('(').next().unwrap_or(s).trim()
}

/// Does this name carry a quantisation format that decodes through a table or
/// packed nibbles? These are the Gather cases and they are the majority of a
/// decode step.
fn is_quantised(name: &str) -> bool {
    // iq2_xxs / q2_K / q4_K / q6_K / q8_0 / mxfp4 / iq1_s, plus the explicit
    // unpack helpers. q8_0 is included deliberately: its dequant is an affine
    // scale, but the *weight* access is still a packed-block walk, and the
    // measured MXFP4 case shows that is what costs.
    const MARKERS: &[&str] = &[
        "iq2_", "iq1_", "iq3_", "iq4_", "_q2_", "_q3_", "_q4_", "_q5_", "_q6_", "_q8_",
        "mxfp4", "qblock", "_quant", "unpack", "dequantize",
    ];
    MARKERS.iter().any(|m| name.contains(m))
}

/// Classify one `__tile_*` intrinsic.
///
/// Accepts a bare name (`__tile_add_f32`) or a whole call line; anything not
/// recognised is [`AccessClass::Opaque`], which is the safe answer.
pub fn classify(intrinsic: &str) -> AccessClass {
    let name = family(intrinsic);
    if !name.starts_with("__tile_") {
        return AccessClass::Opaque;
    }

    // ── Gather first: a quantised or indexed access wins over any other match,
    // because `__tile_mul_mv_id_mxfp4_pair_swiglu` also contains "mul_mv".
    const GATHER_OPS: &[&str] = &[
        "__tile_gather",
        "__tile_scatter",
        "__tile_get_rows",
        "__tile_set_rows",
        "__tile_embedding",
        "__tile_topk",
        "__tile_argsort",
        "__tile_sample_top_p",
        "__tile_partition_perm",
        "__tile_mul_mm_id_map0",
    ];
    if GATHER_OPS.iter().any(|g| name.starts_with(g)) || is_quantised(name) {
        return AccessClass::Gather;
    }
    // `_id_` marks an expert-routed dispatch: the row is chosen by a routing
    // table, so even an unquantised one is a gather.
    if name.contains("_mv_id_") || name.contains("_mm_id_") {
        return AccessClass::Gather;
    }

    // ── Reductions: contiguous load sweep, primitive combine.
    const REDUCTION_OPS: &[&str] = &[
        "__tile_rms_norm",
        "__tile_layernorm",
        "__tile_softmax",
        "__tile_soft_max",
        "__tile_reduce_",
        "__tile_sum_rows",
        "__tile_absmax",
        "__tile_argmax",
        "__tile_argmin",
        "__tile_matvec",
        "__tile_mul_mv",
        "__tile_gate_up_silu",
        "__tile_swiglu",
        "__tile_v_reduce",
        "__tile_l2dist",
    ];
    if REDUCTION_OPS.iter().any(|r| name.starts_with(r)) {
        return AccessClass::Reduction;
    }

    // ── Strided: matmul's second operand.
    const STRIDED_OPS: &[&str] = &[
        "__tile_matmul",
        "__tile_mmad",
        "__tile_mul_mm",
        "__tile_transpose",
    ];
    if STRIDED_OPS.iter().any(|m| name.starts_with(m)) {
        return AccessClass::ConstStrided;
    }

    // ── Contiguous: the elementwise set, unary and binary.
    const ELEMENTWISE: &[&str] = &[
        "add", "sub", "mul", "div", "max", "min", "exp", "log", "ln", "sqrt", "rsqrt",
        "recip", "reciprocal", "abs", "neg", "sq", "sqr", "square", "sin", "cos", "tan",
        "tanh", "sinh", "cosh", "asin", "acos", "atan", "erf", "sigmoid", "silu", "gelu",
        "relu", "step", "sign", "floor", "ceil", "round", "trunc", "clamp", "scale",
        "fill", "cast", "cvt", "copy", "load", "store", "where", "softplus", "hardswish",
        "hardsigmoid", "fast_gelu", "pow", "fmod", "cbrt", "duplicate", "buf_load",
        "buf_store", "buf_fill",
    ];
    let stem = &name["__tile_".len()..];
    let stem = stem.strip_prefix("v_").unwrap_or(stem);
    if ELEMENTWISE.iter().any(|e| {
        stem == *e || stem.starts_with(&format!("{e}_"))
    }) {
        return AccessClass::Contiguous;
    }

    AccessClass::Opaque
}

/// The width to apply, or `None` for "leave it scalar".
///
/// `preferred` comes from `HardwareParams`; 0 or 1 means the target has not
/// declared one and nothing changes — so wiring a backend up to this module is
/// a no-op until its `HardwareParams` opts in.
pub fn vector_width(class: AccessClass, preferred: usize) -> Option<usize> {
    if preferred <= 1 || !class.vectorizable() {
        return None;
    }
    Some(preferred)
}

/// Some families ship both a width-4 and an explicitly scalar intrinsic
/// (`__tile_exp_f32_4` beside `__tile_exp_f32_scalar`). Where that pair exists
/// the policy is a *selection*, not a rewrite — return the vector sibling's name.
pub fn vector_sibling(intrinsic: &str, width: usize) -> Option<String> {
    let name = family(intrinsic);
    if width != 4 {
        return None; // only the _4 convention exists in the tree today
    }
    if let Some(base) = name.strip_suffix("_scalar") {
        return Some(format!("{base}_4"));
    }
    if name.ends_with("_4") {
        return Some(name.to_string()); // already vectorised
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── the three measured rows from the study ───────────────────────────────

    #[test]
    fn gemm_is_strided_and_vectorizable() {
        // Measured 3.35x from giving each thread VEC columns of the strided B.
        assert_eq!(classify("__tile_matmul_f16"), AccessClass::ConstStrided);
        assert!(classify("__tile_matmul_f16").vectorizable());
        assert_eq!(vector_width(AccessClass::ConstStrided, 4), Some(4));
    }

    #[test]
    fn elementwise_is_contiguous_and_vectorizable() {
        // Measured: no change at n=4M (already at roofline), so vectorising is
        // harmless — the policy allows it rather than trying to predict shape.
        for op in ["__tile_add_f32", "__tile_mul_f32", "__tile_exp_f32", "__tile_silu_f32"] {
            assert_eq!(classify(op), AccessClass::Contiguous, "{op}");
            assert!(classify(op).vectorizable(), "{op}");
        }
    }

    #[test]
    fn the_ds4_kernel_is_a_gather_and_must_not_be_vectorised() {
        // THE regression this module exists for. Measured on an M1 Ultra at real
        // V4-Flash shapes: a width-4 port of this kernel ran 155.12 us against
        // the shipping 123.67 us — 1.25x SLOWER — on identical output.
        let k = "__tile_mul_mv_id_mxfp4_pair_swiglu_f32";
        assert_eq!(classify(k), AccessClass::Gather);
        assert!(!classify(k).vectorizable());
        assert_eq!(vector_width(classify(k), 4), None);
    }

    // ── the rest of the quantised decode surface must land in Gather too ─────

    #[test]
    fn quantised_matvecs_are_gathers() {
        for op in [
            "__tile_matvec_iq2_xxs",
            "__tile_matvec_q2_k",
            "__tile_matvec_q4_k",
            "__tile_matvec_q8_0",
            "__tile_matvec_qblock_coop",
            "__tile_mul_mv_id_iq2_xxs_pair_swiglu",
            "__tile_mul_mv_id_q4_k_f32",
            "__tile_mul_mm_id_iq2_xxs",
            "__tile_mxfp4_unpack_f16_e8m0",
            "__tile_unpack_q4k_scale_min",
            "__tile_laguna_q8_0_matvec",
        ] {
            assert_eq!(classify(op), AccessClass::Gather, "{op} must be a gather");
            assert_eq!(vector_width(classify(op), 4), None, "{op}");
        }
    }

    #[test]
    fn routed_dispatch_is_a_gather_even_unquantised() {
        // `_id_` means the row comes from a routing table.
        assert_eq!(classify("__tile_mul_mv_id_f32"), AccessClass::Gather);
        assert_eq!(classify("__tile_mul_mm_id_map0_ne20_4_full"), AccessClass::Gather);
    }

    #[test]
    fn indexed_ops_are_gathers() {
        for op in [
            "__tile_gather_f32", "__tile_scatter_add", "__tile_get_rows_f32_strided",
            "__tile_embedding_f32", "__tile_topk_f32", "__tile_argsort_f32_i32_desc",
        ] {
            assert_eq!(classify(op), AccessClass::Gather, "{op}");
        }
    }

    // ── reductions ───────────────────────────────────────────────────────────

    #[test]
    fn reductions_vectorise_their_load_side() {
        for op in [
            "__tile_rms_norm_f32", "__tile_layernorm_f32", "__tile_softmax_f32",
            "__tile_matvec_f16", "__tile_reduce_sum_f32", "__tile_gate_up_silu_f16",
        ] {
            assert_eq!(classify(op), AccessClass::Reduction, "{op}");
            assert!(classify(op).vectorizable(), "{op}");
        }
    }

    #[test]
    fn plain_matvec_is_a_reduction_but_quantised_matvec_is_not() {
        // The single most load-bearing pair in this module: same op family,
        // opposite answers, because the weight access is what differs.
        assert_eq!(classify("__tile_matvec_f16"), AccessClass::Reduction);
        assert_eq!(classify("__tile_matvec_q4_k"), AccessClass::Gather);
    }

    // ── the safe default ─────────────────────────────────────────────────────

    #[test]
    fn unknown_intrinsics_are_opaque_and_not_vectorised() {
        for op in [
            "__tile_flash_attn_ext_vec_score", "__tile_indexer_scores_tiled",
            "__tile_dsv4_hc_split_weighted_sum_norm4", "__tile_kv_cache_update",
            "__tile_something_invented",
        ] {
            assert_eq!(classify(op), AccessClass::Opaque, "{op}");
            assert!(!classify(op).vectorizable(), "{op}");
        }
        assert_eq!(classify("not_a_tile_intrinsic"), AccessClass::Opaque);
    }

    #[test]
    fn a_target_that_has_not_opted_in_changes_nothing() {
        // Wiring a backend to this module must be a no-op until its
        // HardwareParams declares a width.
        for w in [0usize, 1] {
            assert_eq!(vector_width(AccessClass::Contiguous, w), None);
            assert_eq!(vector_width(AccessClass::ConstStrided, w), None);
        }
    }

    // ── parsing ──────────────────────────────────────────────────────────────

    #[test]
    fn classifies_from_a_whole_call_line() {
        let line = "%5 = llvm.call @__tile_matmul_f16(%3, %4) : (i32, i32) -> i32";
        assert_eq!(classify(line), AccessClass::ConstStrided);
        let line2 = "llvm.call @__tile_mul_mv_id_mxfp4_pair_swiglu_f32(%a) : (i32) -> i32";
        assert_eq!(classify(line2), AccessClass::Gather);
    }

    #[test]
    fn selects_the_existing_width_4_sibling_where_the_tree_has_one() {
        // The tree already ships `_f32_4` beside `_f32_scalar` for ~15 families,
        // so for those the policy is a selection rather than a codegen rewrite.
        assert_eq!(
            vector_sibling("__tile_exp_f32_scalar", 4).as_deref(),
            Some("__tile_exp_f32_4")
        );
        assert_eq!(
            vector_sibling("__tile_silu_f32_scalar", 4).as_deref(),
            Some("__tile_silu_f32_4")
        );
        // Already vectorised stays put; no sibling convention at other widths.
        assert_eq!(
            vector_sibling("__tile_exp_f32_4", 4).as_deref(),
            Some("__tile_exp_f32_4")
        );
        assert_eq!(vector_sibling("__tile_exp_f32_scalar", 8), None);
        assert_eq!(vector_sibling("__tile_add_f32", 4), None);
    }
}
