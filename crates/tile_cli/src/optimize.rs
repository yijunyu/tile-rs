//! `-O` — what each level is permitted to touch.
//!
//! The dividing line between levels is **what the level may use**, not how clever it is,
//! so a user can predict cost and reproducibility from the number alone:
//!
//! | level | may use | deterministic | needs |
//! |---|---|---|---|
//! | O0 | nothing; verbatim | byte-identical | — |
//! | O1 | local/peephole rewrites | byte-identical | — |
//! | O2 | + fusion, tiling, buffer reuse | byte-identical | — |
//! | O3 | + the target toolchain | no | toolchain |
//! | O4 | + measured autotuning | no | device + license |
//!
//! Only O0 and O1 are implemented. **O2 therefore runs O1's passes and says so** — it
//! does not silently pretend to have fused anything, and it does not refuse either,
//! because refusing the DEFAULT level would make the tool unusable to prove a point. The
//! report names every pass that ran and every one that is not written yet, so nobody has
//! to infer what happened from the output.

use std::fmt::Write as _;

/// One pass, and what it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassResult {
    pub name: &'static str,
    /// How many times it fired. Zero is a fine and common answer.
    pub fired: usize,
}

#[derive(Debug, Clone, Default)]
pub struct OptReport {
    pub level: u8,
    pub ran: Vec<PassResult>,
    /// Passes this level is allowed to use that this layer does not do — and, for
    /// fusion, does not do *by design*.
    pub unimplemented: Vec<&'static str>,
    /// Set when a pass separated a producer from the consumer that fuses it. The result
    /// is then the unoptimized input: an unoptimized kernel beats a quietly deoptimized
    /// one, and this must be loud rather than absorbed.
    pub broke_fusion: Option<String>,
}

impl OptReport {
    pub fn changed(&self) -> bool {
        self.ran.iter().any(|p| p.fired > 0)
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = write!(s, "  O{} passes:", self.level);
        if self.ran.is_empty() {
            let _ = write!(s, " none (O0 is verbatim)");
        }
        for p in &self.ran {
            let _ = write!(s, " {}({});", p.name, p.fired);
        }
        if !self.unimplemented.is_empty() {
            let _ = write!(s, "  not done here: {}", self.unimplemented.join(", "));
        }
        if let Some(e) = &self.broke_fusion {
            let _ = write!(
                s,
                "\n  REVERTED: a pass broke a fusion pair and the input was kept — {e}"
            );
        }
        s
    }
}

/// Apply the passes permitted at `level`.
///
/// Pure: same input, same level, byte-identical output, forever. That is the same
/// contract the emitters are held to, and it is what makes `-O0` through `-O2` safe to
/// promise to a build system.
///
/// The passes run over [`crate::mlir`]'s IR rather than over lines of text. A textual
/// pass can count and delete; it cannot know what an operation reads, which is the first
/// thing a rewrite has to know.
pub fn optimize(mlir_text: &str, level: u8) -> (String, OptReport) {
    let mut report = OptReport {
        level,
        ..Default::default()
    };
    if level == 0 {
        return (mlir_text.to_string(), report);
    }

    let mut m = crate::mlir::parse(mlir_text);
    for p in crate::passes::PIPELINE.iter().filter(|p| level >= p.level) {
        let fired = (p.run)(&mut m).0;
        report.ran.push(PassResult {
            name: p.name,
            fired,
        });
    }

    // The invariant the whole layer is shaped around: the emitters fuse SiLU into a
    // following multiply by ADJACENCY, so a pass that separates them costs the fusion
    // downstream — a kernel that still lowers, still computes the right answer, and is
    // slower for a reason invisible in the output. If a pass broke it, the safe answer is
    // the input: an unoptimized kernel is always better than a quietly deoptimized one.
    if let Err(e) = crate::mlir::fusion_pairs_intact(&m) {
        report.broke_fusion = Some(e);
        return (mlir_text.to_string(), report);
    }

    report.unimplemented = crate::passes::unimplemented_at(level);
    (crate::mlir::print(&m), report)
}

/// `-O3`: ask the target's own compiler what it thinks of the emitted source.
///
/// Separate from [`optimize`] because it runs on the OUTPUT, after lowering, and because
/// it is the one part of the pipeline that can be unavailable. Absence is reported, never
/// fatal: refusing a conversion because a check could not run would be worse than not
/// offering the check.
pub fn toolchain_feedback(form: &crate::forms::Form, source: &str) -> crate::toolchain::Feedback {
    crate::toolchain::check(form, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = "\
module {
  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t0, %c1, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
";

    #[test]
    fn o0_is_verbatim() {
        let (out, r) = optimize(LIVE, 0);
        assert_eq!(out, LIVE);
        assert!(r.ran.is_empty());
        assert!(!r.changed());
    }

    #[test]
    fn a_kernel_with_nothing_to_do_comes_back_byte_identical() {
        for level in 1..=3u8 {
            let (out, r) = optimize(LIVE, level);
            assert_eq!(
                out, LIVE,
                "O{level} changed a kernel it had nothing to do to"
            );
            assert!(!r.changed(), "O{level} claimed to fire: {:?}", r.ran);
        }
    }

    #[test]
    fn an_unused_definition_is_removed() {
        let dead = LIVE.replace(
            "    llvm.return\n",
            "    %junk = llvm.mlir.constant(7 : i32) : i32\n    llvm.return\n",
        );
        let (out, r) = optimize(&dead, 1);
        assert!(!out.contains("%junk ="), "the dead op survived:\n{out}");
        assert_eq!(r.ran[0].fired, 1);
        assert_eq!(
            out, LIVE,
            "removing the dead op restores the original exactly"
        );
    }

    #[test]
    fn a_chain_of_dead_ops_goes_in_one_call() {
        let dead = LIVE.replace(
            "    llvm.return\n",
            "    %a = llvm.mlir.constant(7 : i32) : i32\n    \
             %b = llvm.call @__tile_exp_f32(%a) : (i32) -> i32\n    llvm.return\n",
        );
        let (out, r) = optimize(&dead, 1);
        assert!(!out.contains("%a ="), "{out}");
        assert!(!out.contains("%b ="), "{out}");
        assert_eq!(r.ran[0].fired, 2);
    }

    #[test]
    fn a_comment_mentioning_a_name_does_not_keep_it_alive() {
        let dead = LIVE.replace(
            "    llvm.return\n",
            "    %junk = llvm.mlir.constant(7 : i32) : i32  // %junk is never read\n    \
             llvm.return\n",
        );
        let (out, r) = optimize(&dead, 1);
        assert_eq!(r.ran[0].fired, 1, "{out}");
        assert!(!out.contains("%junk ="), "{out}");
    }

    #[test]
    fn an_operation_with_no_result_is_never_removed() {
        let (out, _) = optimize(LIVE, 3);
        assert!(out.contains("__tile_store_f32"), "a store was eliminated");
        assert!(out.contains("llvm.return"), "the terminator was eliminated");
    }

    #[test]
    fn optimization_is_a_pure_function_of_its_input() {
        for level in 0..=3u8 {
            assert_eq!(optimize(LIVE, level).0, optimize(LIVE, level).0, "O{level}");
        }
    }

    #[test]
    fn o2_runs_more_passes_than_o1_and_names_what_it_does_not_do() {
        let o1 = optimize(LIVE, 1).1;
        let o2 = optimize(LIVE, 2).1;
        assert!(o2.ran.len() > o1.ran.len(), "O2 ran no more than O1");
        let rendered = o2.render();
        assert!(rendered.contains("not done here"), "{rendered}");
        // Fusion is not missing work: the emitters own it, by adjacency.
        assert!(rendered.contains("owned by the emitters"), "{rendered}");
    }

    #[test]
    fn a_broken_fusion_pair_reverts_to_the_input_and_says_so() {
        // No pass does this today, so the situation is constructed. The behaviour has to
        // exist before a pass that could cause it does: an unoptimized kernel is always
        // better than a quietly deoptimized one.
        let src = "  %s = llvm.call @__tile_silu_f32(%g, %g, %r, %c) : (i32, i32, i32, i32) -> i32\n  \
                   %x = llvm.call @__tile_exp_f32(%g, %r, %c) : (i32, i32, i32) -> i32\n  \
                   %m = llvm.call @__tile_mul_f32(%s, %s, %x, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n  \
                   llvm.return %m : i32\n";
        let (out, r) = optimize(src, 2);
        assert_eq!(out, src, "the input must be returned unchanged");
        let rendered = r.render();
        let e = r
            .broke_fusion
            .as_deref()
            .expect("the break must be reported");
        assert!(e.contains("adjacent"), "{e}");
        assert!(rendered.contains("REVERTED"), "{rendered}");
    }

    #[test]
    fn higher_levels_never_claim_fewer_gaps() {
        let counts: Vec<usize> = (0..=4u8)
            .map(|l| optimize(LIVE, l).1.unimplemented.len())
            .collect();
        for w in counts.windows(2) {
            assert!(
                w[1] >= w[0],
                "a higher level claimed fewer gaps: {counts:?}"
            );
        }
    }
}
