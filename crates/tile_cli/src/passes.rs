//! The rewrite passes, over [`crate::mlir`]'s IR.
//!
//! Each pass is a function from a module to a module plus a count of how many times it
//! fired. Zero is a fine and common answer, and it is reported rather than hidden: a user
//! who asked for `-O2` is owed the difference between "this found nothing" and "this does
//! not exist yet".
//!
//! ## What is deliberately not here
//!
//! **Fusion.** The emitters own it: `mlir_to_tpu.rs` fuses SiLU into a following multiply
//! by remembering the previous operation. Fusing here would mean emitting a
//! `__tile_silu_mul_f32` that no emitter's vocabulary contains — turning a kernel that
//! lowers into one that does not. The plan called this pass "fusion" and the codebase
//! says otherwise; the codebase wins.
//!
//! What this layer owes fusion instead is *not to break it*, which is why every pipeline
//! ends with [`crate::mlir::fusion_pairs_intact`].

use crate::mlir::{self, Item, Module};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fired(pub usize);

/// Remove operations whose result nobody reads.
///
/// Only where removal is *observably* nothing: a pure computation, or a load, whose
/// result is dead. An operation that stores, allocates, orders or is simply unknown stays
/// — being wrong about an unknown intrinsic is safe in exactly one direction.
pub fn dead_ops(m: &mut Module) -> Fired {
    let mut fired = 0;
    loop {
        let uses = mlir::use_counts(m);
        let before = m.items.len();
        m.items.retain(|item| {
            let Item::Op(o) = item else { return true };
            let Some(r) = &o.result else { return true };
            if !o.removable_when_unused() {
                return true;
            }
            uses.get(r).copied().unwrap_or(0) > 0
        });
        fired += before - m.items.len();
        if m.items.len() == before {
            break;
        }
    }
    Fired(fired)
}

/// Collapse duplicate `llvm.mlir.constant` definitions.
///
/// The emitters materialise a constant per use site, so a kernel with three `1 : i32`
/// operands carries three definitions. Harmless, and exactly the kind of thing a level
/// that promises "local rewrites" should tidy.
pub fn dedup_constants(m: &mut Module) -> Fired {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut renames = Vec::new();
    for o in m.ops() {
        if o.callee != "llvm.mlir.constant" {
            continue;
        }
        let Some(r) = &o.result else { continue };
        let key = format!("{}|{}", o.args.join(","), o.tail.trim());
        match seen.get(&key) {
            Some(first) => renames.push((r.clone(), first.clone())),
            None => {
                seen.insert(key, r.clone());
            }
        }
    }
    let fired = renames.len();
    for (from, to) in renames {
        mlir::replace_uses(m, &from, &to);
    }
    dead_ops(m);
    Fired(fired)
}

/// Common-subexpression elimination over pure operations.
///
/// Two pure operations with the same callee and the same operands compute the same value,
/// so the second is redundant. Restricted to [`Op::is_pure`], which excludes loads —
/// those need the memory reasoning in [`redundant_loads`].
pub fn cse(m: &mut Module) -> Fired {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut renames = Vec::new();
    for o in m.ops() {
        if !o.is_pure() || o.callee == "llvm.mlir.constant" {
            continue;
        }
        let Some(r) = &o.result else { continue };
        let key = format!("{}({})", o.callee, o.args.join(","));
        match seen.get(&key) {
            Some(first) => renames.push((r.clone(), first.clone())),
            None => {
                seen.insert(key, r.clone());
            }
        }
    }
    let fired = renames.len();
    for (from, to) in renames {
        mlir::replace_uses(m, &from, &to);
    }
    dead_ops(m);
    Fired(fired)
}

/// Reuse a load when the same thing is loaded again with nothing written in between.
///
/// The aliasing rule is the crude one and deliberately so: **any** store, any impure
/// operation, and any line this layer could not parse invalidates every cached load. A
/// precise alias analysis would reuse more; being wrong once would produce a kernel that
/// reads stale data, which is the failure that never shows up in a diff.
pub fn redundant_loads(m: &mut Module) -> Fired {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut renames = Vec::new();
    for item in &m.items {
        match item {
            Item::Raw(r) if !r.trim_start().starts_with("//") && !r.trim().is_empty() => {
                // An unparsed line could be a call, a branch, anything. Forget everything.
                seen.clear();
            }
            Item::Op(o) => {
                if o.is_store() || (!o.is_pure() && !o.is_load()) {
                    seen.clear();
                    continue;
                }
                if !o.is_load() {
                    continue;
                }
                let Some(r) = &o.result else { continue };
                let key = format!("{}({})", o.callee, o.args.join(","));
                match seen.get(&key) {
                    Some(first) => renames.push((r.clone(), first.clone())),
                    None => {
                        seen.insert(key, r.clone());
                    }
                }
            }
            _ => {}
        }
    }
    let fired = renames.len();
    for (from, to) in renames {
        mlir::replace_uses(m, &from, &to);
    }
    dead_ops(m);
    Fired(fired)
}

/// A resource-bound finding against the selected architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundViolation {
    pub op: String,
    pub rule: &'static str,
    pub detail: String,
}

/// Check every tile operation's working set against the target's measured bounds.
///
/// This is the tile-rs-specific pass, and it is a **check**, not a rewrite: re-chunking a
/// tile changes what the kernel computes unless the kernel loops, and this layer has no
/// way to know that it does. Reporting a tiling the hardware cannot hold is the useful
/// half and the honest half.
///
/// Returns `Err` on an architecture nobody has measured. That is commit `f10513c`'s rule
/// arriving in the optimizer: another chip's capacities would approve precisely the
/// tilings that fail here.
pub fn check_bounds(
    m: &Module,
    hp: &tile_codegen::HardwareParams,
    dtype_bytes: usize,
) -> Result<Vec<BoundViolation>, String> {
    if !hp.measured {
        return Err(format!(
            "arch {}: no capacity has been MEASURED on this architecture, so no tiling \
             can be approved or rejected here",
            hp.npu_arch
        ));
    }
    let consts = constant_values(m);
    // Operands of a cube op. A load that feeds a matmul is a CUBE STAGING load
    // -- the emitter lands it in a `loc=mat` (L1) tile, never in UB -- so the
    // vector rules do not bound it either, and its capacity is already checked
    // as L0A/L0B by C9 on the matmul itself. Checking it again against UB is
    // not a second opinion, it is the wrong memory.
    let cube_operands = cube_operand_ssa(m);
    let mut out = Vec::new();
    for o in m.ops() {
        let Some(name) = o.intrinsic() else { continue };
        // The tile intrinsics carry (..., rows, cols) as their last two operands.
        let n = o.args.len();
        if n < 2 {
            continue;
        }
        let (Some(rows), Some(cols)) = (
            consts.get(o.args[n - 2].trim()).copied(),
            consts.get(o.args[n - 1].trim()).copied(),
        ) else {
            continue;
        };
        // WHICH UNIT the op runs on decides which rules bound it. A cube op's
        // operands live in L0A/L0B and its result in L0C; it is not issued as a
        // repeated vector instruction and it does not occupy UB, so the vector
        // rules do not apply and C9 does. Checking everything against the
        // vector unit refused `__tile_matmul_f16` at every shape from 128x128
        // up -- including one since measured correct on hardware (rel 2.2e-07
        // on 910B; 20/20 on 950PR/c310) -- with "16384 elements of 4B is 256
        // repeats of an 8-bit field", a sentence that describes no instruction
        // the cube ever issues. check_cube_tile has existed and been tested as
        // C9 all along; nothing in the pipeline called it.
        // A cube op's TILING IS NOT DECIDED YET at this point. The emitter
        // blocks K and N (and caps Kb by M) when the whole shape will not fit
        // L0, so judging the whole shape here refuses matmuls the blocked path
        // handles perfectly well -- which is what kept 256x256x256 and up out
        // of the emitter, and with them the K/N-blocked path that parallelises
        // over N via get_block_idx. The real tiles are checked AFTER lowering
        // by `check_pto_tiles`, which reads them off the emitted alloc_tile
        // types rather than predicting them.
        if cube_mkn(name, &o.args, &consts).is_some() {
            continue;
        }
        let elems = rows.saturating_mul(cols);
        if elems == 0 {
            continue;
        }
        // A cube staging load keeps the DMA-stride rule -- it really does issue
        // a GM->L1 transfer with a row stride -- and loses UB and repeat.
        let touches_cube = o
            .result
            .as_deref()
            .is_some_and(|r| cube_operands.contains(r))
            || o.args.iter().any(|a| cube_operands.contains(a.trim()));
        if touches_cube {
            if let Err(e) = hp.check_dma_stride(cols * dtype_bytes, "row") {
                out.push(BoundViolation {
                    op: name.to_string(),
                    rule: "DMA stride",
                    detail: e,
                });
            }
            continue;
        }
        for (rule, res) in [
            ("UB", hp.check_ub(elems * dtype_bytes)),
            ("repeat", hp.check_repeat(elems, dtype_bytes)),
            ("DMA stride", hp.check_dma_stride(cols * dtype_bytes, "row")),
        ] {
            if let Err(e) = res {
                out.push(BoundViolation {
                    op: name.to_string(),
                    rule,
                    detail: e,
                });
            }
        }
    }
    Ok(out)
}

/// Bounds over the tiles the emitter ACTUALLY produced.
///
/// `check_bounds` runs before lowering, where a cube op's block shape is not
/// yet chosen; this runs after, where every tile is spelled out as
/// `pto.alloc_tile : !pto.tile_buf<loc=..., dtype=..., rows=R, cols=C, ...>`.
/// The `loc=` says which memory holds it, so nothing has to be inferred.
///
/// `loc=mat` is L1, whose capacity nothing here has measured, so it is counted
/// and reported but not judged -- refusing to check what has not been measured
/// is the rule; charging it to UB, as the emitter used to, is the wrong memory.
pub fn check_pto_tiles(
    pto: &str,
    hp: &tile_codegen::HardwareParams,
) -> Result<Vec<BoundViolation>, String> {
    if !hp.measured {
        return Err(format!(
            "arch {}: no capacity has been MEASURED on this architecture",
            hp.npu_arch
        ));
    }
    let mut out = Vec::new();
    for line in pto.lines() {
        let Some(i) = line.find("pto.alloc_tile") else {
            continue;
        };
        let t = &line[i..];
        let field = |k: &str| -> Option<String> {
            let j = t.find(k)? + k.len();
            Some(
                t[j..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect(),
            )
        };
        let (Some(loc), Some(dt), Some(r), Some(c)) = (
            field("loc="),
            field("dtype="),
            field("rows="),
            field("cols="),
        ) else {
            continue;
        };
        let (Ok(r), Ok(c)) = (r.parse::<usize>(), c.parse::<usize>()) else {
            continue;
        };
        let eb = match dt.as_str() {
            "f64" | "i64" => 8,
            "f32" | "i32" => 4,
            "f16" | "bf16" | "i16" => 2,
            "i8" | "u8" => 1,
            _ => continue,
        };
        let bytes = r * c * eb;
        let (cap, what) = match loc.as_str() {
            "vec" => (hp.ub_size, "UB"),
            "left" => (hp.l0a_bytes, "L0A"),
            "right" => (hp.l0b_bytes, "L0B"),
            "acc" => (hp.l0c_bytes, "L0C"),
            _ => continue, // mat = L1, unmeasured
        };
        if cap != 0 && bytes > cap {
            out.push(BoundViolation {
                op: format!("alloc_tile loc={loc}"),
                rule: "tile capacity",
                detail: format!(
                    "arch {}: a {}x{} {} tile is {}B against {} of {}B",
                    hp.npu_arch, r, c, dt, bytes, what, cap
                ),
            });
        }
    }
    Ok(out)
}

/// SSA names that are read by a cube op, so a load producing one is staging for
/// the cube rather than for the vector unit. `mlir_to_pto` makes the same
/// distinction under the name `detect_blocked_matmul_loads`, which is what
/// decides a `loc=mat` tile; this is that judgement where the bounds live.
fn cube_operand_ssa(m: &Module) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for o in m.ops() {
        let Some(name) = o.intrinsic() else { continue };
        if !name.contains("matmul") {
            continue;
        }
        // (acc_mode, a, b, m, k, n): the operands are everything between the
        // leading mode flag and the trailing three extents.
        let n = o.args.len();
        if n < 4 {
            continue;
        }
        for a in &o.args[1..n - 3] {
            out.insert(a.trim().to_string());
        }
        // The RESULT too: the store that consumes it is the L0C->GM fixpipe,
        // which is the cube's own output path and not a vector store.
        if let Some(r) = &o.result {
            out.insert(r.trim().to_string());
        }
    }
    out
}

/// The (M, K, N) of a cube matmul intrinsic, or `None` if this is not one.
///
/// A matmul carries `(..., m, k, n)` as its last THREE operands, not the
/// `(rows, cols)` pair every vector intrinsic ends with -- so reading the last
/// two off a matmul yields (k, n) and silently checks the wrong tile. A stopgap
/// until an intrinsic registry says this properly.
fn cube_mkn(
    name: &str,
    args: &[String],
    consts: &std::collections::BTreeMap<String, usize>,
) -> Option<(usize, usize, usize)> {
    if !name.contains("matmul") {
        return None;
    }
    let n = args.len();
    if n < 3 {
        return None;
    }
    let g = |i: usize| consts.get(args[i].trim()).copied();
    Some((g(n - 3)?, g(n - 2)?, g(n - 1)?))
}

/// SSA name -> integer value, for the `llvm.mlir.constant` definitions.
pub fn constant_values(m: &Module) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for o in m.ops() {
        if o.callee != "llvm.mlir.constant" {
            continue;
        }
        let (Some(r), Some(a)) = (&o.result, o.args.first()) else {
            continue;
        };
        if let Ok(v) = a.split(':').next().unwrap_or("").trim().parse::<usize>() {
            out.insert(r.clone(), v);
        }
    }
    out
}

/// One entry in the pipeline.
pub struct Pass {
    pub name: &'static str,
    /// The lowest `-O` level that may run it.
    pub level: u8,
    pub run: fn(&mut Module) -> Fired,
}

/// The pipeline, in order. Ordering matters: constant deduplication first, so CSE sees
/// operands that are already equal by name rather than merely equal by value.
pub const PIPELINE: &[Pass] = &[
    Pass {
        name: "dead-op-elimination",
        level: 1,
        run: dead_ops,
    },
    Pass {
        name: "constant-dedup",
        level: 2,
        run: dedup_constants,
    },
    Pass {
        name: "common-subexpression",
        level: 2,
        run: cse,
    },
    Pass {
        name: "redundant-load",
        level: 2,
        run: redundant_loads,
    },
];

/// Passes named by a level that are not written. Reported, never silently skipped.
pub fn unimplemented_at(level: u8) -> Vec<&'static str> {
    let mut v = Vec::new();
    if level >= 2 {
        // Not "not yet": the emitters own fusion by adjacency, so this layer must not.
        v.push("fusion (owned by the emitters, by design)");
        v.push("tile-plan rewriting (bounds are CHECKED, not rewritten)");
        v.push("buffer-reuse");
    }
    if level >= 4 {
        v.push("measured-autotuning");
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::{parse, print};

    fn run_all(src: &str, level: u8) -> (String, Vec<(&'static str, usize)>) {
        let mut m = parse(src);
        let mut fired = Vec::new();
        for p in PIPELINE.iter().filter(|p| level >= p.level) {
            fired.push((p.name, (p.run)(&mut m).0));
        }
        (print(&m), fired)
    }

    const SOFTMAX: &str = include_str!("../testdata/forms/softmax.mlir");

    #[test]
    fn the_canonical_kernel_is_already_optimal_and_is_left_alone() {
        // A pass that "improves" a clean kernel is a pass that is wrong.
        let (out, fired) = run_all(SOFTMAX, 3);
        assert_eq!(out, SOFTMAX);
        assert!(fired.iter().all(|(_, n)| *n == 0), "{fired:?}");
    }

    #[test]
    fn duplicate_constants_collapse_to_one() {
        let src = "  %a = llvm.mlir.constant(4 : i32) : i32\n  \
                   %b = llvm.mlir.constant(4 : i32) : i32\n  \
                   %r = llvm.call @__tile_add_f32(%a, %b) : (i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        let (out, fired) = run_all(src, 2);
        assert_eq!(
            fired
                .iter()
                .find(|(n, _)| *n == "constant-dedup")
                .unwrap()
                .1,
            1
        );
        assert!(!out.contains("%b ="), "the duplicate survived:\n{out}");
        assert!(out.contains("@__tile_add_f32(%a, %a)"), "{out}");
    }

    #[test]
    fn constants_of_different_values_are_not_merged() {
        let src = "  %a = llvm.mlir.constant(4 : i32) : i32\n  \
                   %b = llvm.mlir.constant(8 : i32) : i32\n  \
                   %r = llvm.call @__tile_add_f32(%a, %b) : (i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        let (out, _) = run_all(src, 2);
        assert!(out.contains("%b ="), "{out}");
    }

    #[test]
    fn an_identical_pure_computation_is_computed_once() {
        let src = "  %c = llvm.mlir.constant(1 : i32) : i32\n  \
                   %x = llvm.call @__tile_exp_f32(%c, %c) : (i32, i32) -> i32\n  \
                   %y = llvm.call @__tile_exp_f32(%c, %c) : (i32, i32) -> i32\n  \
                   %z = llvm.call @__tile_add_f32(%x, %y) : (i32, i32) -> i32\n  \
                   llvm.return %z : i32\n";
        let (out, fired) = run_all(src, 2);
        assert_eq!(
            fired
                .iter()
                .find(|(n, _)| *n == "common-subexpression")
                .unwrap()
                .1,
            1
        );
        assert!(out.contains("@__tile_add_f32(%x, %x)"), "{out}");
    }

    #[test]
    fn a_store_is_never_treated_as_a_common_subexpression() {
        // Two identical stores are two stores. Collapsing them is a miscompile.
        let src = "  %c = llvm.mlir.constant(1 : i32) : i32\n  \
                   llvm.call @__tile_store_f32(%c, %c) : (i32, i32) -> ()\n  \
                   llvm.call @__tile_store_f32(%c, %c) : (i32, i32) -> ()\n";
        let (out, _) = run_all(src, 2);
        assert_eq!(out.matches("__tile_store_f32").count(), 2, "{out}");
    }

    #[test]
    fn a_repeated_load_with_nothing_in_between_is_reused() {
        let src = "  %c = llvm.mlir.constant(1 : i32) : i32\n  \
                   %a = llvm.call @__tile_load_f32(%p, %c, %c) : (i32, i32, i32) -> i32\n  \
                   %b = llvm.call @__tile_load_f32(%p, %c, %c) : (i32, i32, i32) -> i32\n  \
                   %r = llvm.call @__tile_add_f32(%a, %b) : (i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        let (out, fired) = run_all(src, 2);
        assert_eq!(
            fired
                .iter()
                .find(|(n, _)| *n == "redundant-load")
                .unwrap()
                .1,
            1
        );
        assert!(out.contains("@__tile_add_f32(%a, %a)"), "{out}");
    }

    #[test]
    fn a_store_between_two_loads_stops_the_reuse() {
        // The crude alias rule: any write invalidates everything. Reusing across it would
        // read stale data, which is the failure that never shows up in a diff.
        let src = "  %c = llvm.mlir.constant(1 : i32) : i32\n  \
                   %a = llvm.call @__tile_load_f32(%p, %c, %c) : (i32, i32, i32) -> i32\n  \
                   llvm.call @__tile_store_f32(%p, %a, %c) : (i32, i32, i32) -> ()\n  \
                   %b = llvm.call @__tile_load_f32(%p, %c, %c) : (i32, i32, i32) -> i32\n  \
                   %r = llvm.call @__tile_add_f32(%a, %b) : (i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        let (out, fired) = run_all(src, 2);
        assert_eq!(
            fired
                .iter()
                .find(|(n, _)| *n == "redundant-load")
                .unwrap()
                .1,
            0
        );
        assert!(
            out.contains("%b ="),
            "the second load was wrongly removed:\n{out}"
        );
    }

    #[test]
    fn an_unknown_call_between_two_loads_also_stops_the_reuse() {
        let src = "  %c = llvm.mlir.constant(1 : i32) : i32\n  \
                   %a = llvm.call @__tile_load_f32(%p, %c, %c) : (i32, i32, i32) -> i32\n  \
                   %m = llvm.call @mystery(%p) : (i32) -> i32\n  \
                   %b = llvm.call @__tile_load_f32(%p, %c, %c) : (i32, i32, i32) -> i32\n  \
                   %r = llvm.call @__tile_add_f32(%a, %b, %m) : (i32, i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        let (_, fired) = run_all(src, 2);
        assert_eq!(
            fired
                .iter()
                .find(|(n, _)| *n == "redundant-load")
                .unwrap()
                .1,
            0
        );
    }

    #[test]
    fn no_pass_separates_a_silu_from_the_mul_that_fuses_it() {
        // The invariant the whole layer is shaped around.
        let src = "  %c = llvm.mlir.constant(1 : i32) : i32\n  \
                   %d = llvm.mlir.constant(1 : i32) : i32\n  \
                   %g = llvm.call @__tile_load_f32(%p, %c, %d) : (i32, i32, i32) -> i32\n  \
                   %s = llvm.call @__tile_silu_f32(%g, %g, %c, %d) : (i32, i32, i32, i32) -> i32\n  \
                   %m = llvm.call @__tile_mul_f32(%s, %s, %g, %c, %d) : (i32, i32, i32, i32, i32) -> i32\n  \
                   llvm.return %m : i32\n";
        let mut mo = parse(src);
        for p in PIPELINE {
            (p.run)(&mut mo);
        }
        mlir::fusion_pairs_intact(&mo).expect("a pass broke the fusion pair");
    }

    #[test]
    fn optimization_is_a_pure_function_of_its_input() {
        let src = "  %a = llvm.mlir.constant(4 : i32) : i32\n  \
                   %b = llvm.mlir.constant(4 : i32) : i32\n  \
                   %r = llvm.call @__tile_add_f32(%a, %b) : (i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        for level in 0..=3u8 {
            assert_eq!(run_all(src, level).0, run_all(src, level).0, "O{level}");
        }
    }

    #[test]
    fn running_the_pipeline_twice_changes_nothing_the_second_time() {
        // A pipeline that keeps finding work is a pipeline that is oscillating.
        let src = "  %a = llvm.mlir.constant(4 : i32) : i32\n  \
                   %b = llvm.mlir.constant(4 : i32) : i32\n  \
                   %x = llvm.call @__tile_exp_f32(%a) : (i32) -> i32\n  \
                   %y = llvm.call @__tile_exp_f32(%b) : (i32) -> i32\n  \
                   %r = llvm.call @__tile_add_f32(%x, %y) : (i32, i32) -> i32\n  \
                   llvm.return %r : i32\n";
        let once = run_all(src, 3).0;
        let twice = run_all(&once, 3);
        assert_eq!(twice.0, once);
        assert!(
            twice.1.iter().all(|(_, n)| *n == 0),
            "second run fired: {:?}",
            twice.1
        );
    }

    #[test]
    fn bounds_are_checked_against_a_measured_architecture() {
        // 910B holds 192 KiB of UB. A 256x256 f32 tile is 256 KiB and cannot be held.
        let src = "  %r = llvm.mlir.constant(256 : i32) : i32\n  \
                   %c = llvm.mlir.constant(256 : i32) : i32\n  \
                   %t = llvm.call @__tile_softmax_f32(%r, %c) : (i32, i32) -> i32\n";
        let hp = tile_codegen::HardwareParams::ascend_910b();
        let v = check_bounds(&parse(src), &hp, 4).unwrap();
        assert!(v.iter().any(|x| x.rule == "UB"), "{v:?}");
    }

    #[test]
    fn a_cube_matmul_is_bounded_by_l0_not_by_the_vector_unit() {
        // The vector rules do not describe the cube. A 128-cubed f16 matmul was
        // refused as "16384 elements of 4B is 256 repeats of an 8-bit field",
        // which is not an instruction the cube issues -- and that kernel has
        // since been measured correct on hardware (rel 2.2e-07 on 910B; 20/20
        // on 950PR/c310). It fits L0A/L0B/L0C, so nothing should be reported.
        let src = "  %z = llvm.mlir.constant(0 : i32) : i32\n  \
                   %d = llvm.mlir.constant(128 : i32) : i32\n  \
                   %a = llvm.call @__tile_load_f16(%d, %d) : (i32, i32) -> i32\n  \
                   %b = llvm.call @__tile_load_f16(%d, %d) : (i32, i32) -> i32\n  \
                   %c = llvm.call @__tile_matmul_f16(%z, %a, %b, %d, %d, %d) : \
                   (i32, i32, i32, i32, i32, i32) -> i32\n";
        let hp = tile_codegen::HardwareParams::ascend_910b();
        // dtype_bytes is deliberately the WRONG 4 here: that is what the
        // module-wide scan infers for an f16 kernel today, and the cube path
        // must not depend on it -- the operand width comes off the name.
        let v = check_bounds(&parse(src), &hp, 4).unwrap();
        assert!(v.is_empty(), "a 128-cubed f16 matmul fits the cube: {v:?}");
    }

    #[test]
    fn a_cube_matmul_too_big_for_l0_is_deferred_not_refused() {
        // Before lowering, the block shape is not chosen: the emitter blocks K
        // and N (capping Kb by M) when the whole shape will not fit L0. So this
        // pass must NOT judge 256-cubed -- judging it is what kept the blocked
        // path, and its get_block_idx parallelism over N, from ever running.
        let src = "  %z = llvm.mlir.constant(0 : i32) : i32\n  \
                   %d = llvm.mlir.constant(256 : i32) : i32\n  \
                   %a = llvm.call @__tile_load_f16(%d, %d) : (i32, i32) -> i32\n  \
                   %b = llvm.call @__tile_load_f16(%d, %d) : (i32, i32) -> i32\n  \
                   %c = llvm.call @__tile_matmul_f16(%z, %a, %b, %d, %d, %d) : \
                   (i32, i32, i32, i32, i32, i32) -> i32\n";
        let hp = tile_codegen::HardwareParams::ascend_910b();
        let v = check_bounds(&parse(src), &hp, 4).unwrap();
        assert!(
            v.is_empty(),
            "the pre-lowering pass must defer a cube op: {v:?}"
        );
    }

    #[test]
    fn the_emitted_tiling_is_what_gets_judged() {
        // ...and after lowering it IS judged, against the memory the tile's own
        // `loc=` names. A 256x256 f16 left tile is 128 KiB against a 64 KiB L0A.
        let hp = tile_codegen::HardwareParams::ascend_910b();
        let over = "%t = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f16, rows=256, cols=256, v_row=256, v_col=256>\n";
        let v = check_pto_tiles(over, &hp).unwrap();
        let x = v.first().expect("a 256x256 f16 L0A tile does not fit");
        assert!(x.detail.contains("L0A"), "{}", x.detail);

        // The block the emitter actually picks for that shape does fit.
        let ok = "%t = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f16, rows=256, cols=128, v_row=256, v_col=128>\n";
        assert!(check_pto_tiles(ok, &hp).unwrap().is_empty());

        // loc=mat is L1, which nothing here has measured: counted, not judged.
        let mat = "%t = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f16, rows=4096, cols=4096>\n";
        assert!(check_pto_tiles(mat, &hp).unwrap().is_empty());
    }

    #[test]
    fn a_vector_op_keeps_its_vector_bounds() {
        // The dispatch must not disarm the vector rules for vector ops: the
        // 256x256 f32 softmax above is still over UB.
        let src = "  %r = llvm.mlir.constant(256 : i32) : i32\n  \
                   %c = llvm.mlir.constant(256 : i32) : i32\n  \
                   %t = llvm.call @__tile_softmax_f32(%r, %c) : (i32, i32) -> i32\n";
        let hp = tile_codegen::HardwareParams::ascend_910b();
        let v = check_bounds(&parse(src), &hp, 4).unwrap();
        assert!(v.iter().any(|x| x.rule == "UB"), "{v:?}");
    }

    #[test]
    fn a_tiling_that_fits_reports_nothing() {
        let src = "  %r = llvm.mlir.constant(1 : i32) : i32\n  \
                   %c = llvm.mlir.constant(64 : i32) : i32\n  \
                   %t = llvm.call @__tile_softmax_f32(%r, %c) : (i32, i32) -> i32\n";
        let hp = tile_codegen::HardwareParams::ascend_910b();
        assert!(check_bounds(&parse(src), &hp, 4).unwrap().is_empty());
    }

    #[test]
    fn an_unmeasured_architecture_refuses_to_judge_a_tiling_at_all() {
        // f10513c's rule, arriving in the optimizer: another chip's capacities would
        // approve precisely the tilings that fail here.
        let hp = tile_codegen::HardwareParams::ascend_950_unmeasured();
        let e = check_bounds(&parse(SOFTMAX), &hp, 4).unwrap_err();
        assert!(e.contains("MEASURED"), "{e}");
    }

    #[test]
    fn fusion_is_named_as_the_emitters_job_not_as_missing_work() {
        // The plan called this "not yet written". The codebase says otherwise, and the
        // report has to say what is true.
        let missing = unimplemented_at(2);
        assert!(
            missing.iter().any(|m| m.contains("owned by the emitters")),
            "{missing:?}"
        );
    }
}
