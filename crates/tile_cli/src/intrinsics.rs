//! `--list-intrinsics <form>`: the whole intrinsic surface for ONE backend.
//!
//! ## Why this exists beside `--list-ops`
//!
//! `--list-ops` is the stronger instrument and answers the smaller question. It probes
//! 32 portable tile-level ops by lowering a real module for each, so every cell is
//! measured -- and its own footer says the denominator is that probe's 32 ops, "NOT the
//! whole intrinsic surface, which is much larger", with single-backend kernel families
//! deliberately outside it. That is the right design for a PORTABILITY table.
//!
//! It leaves no surface that answers *"what can `pto` do for me"*, and the cost of that
//! was measured on a reader in one session: they read the 32-op table, found no `quant`,
//! `dequant`, `scale`, `fixpipe` or `epilogue` row, and concluded a fused int8 dequant
//! matmul was not expressible. It is declared as
//! `__tile_matmul_i8_acc_i32_dequant_f16` -- the dequant is in the name -- and the `pto`
//! emitter names it at nine sites. Four layers of implementation, invisible from the one
//! command a caller runs to ask.
//!
//! A portability table cannot answer a single-backend question, and for Ascend work that
//! is the only question being asked.
//!
//! ## What a row means, and what it does not
//!
//! The table is DERIVED FROM THE TREE by `build.rs` on every build: the intrinsics are
//! every `pub fn __tile_*` `tile_std` declares, and a backend's status is whether its
//! emitter source names that intrinsic -- exactly, or by the family prefix it dispatches
//! a suffix from. So it reports **which arms were written**, which is strictly weaker
//! than `--list-ops`' *this lowered a module*. `SKILL.md` trap 3 is the reason that
//! distinction is printed rather than implied: 42 of 42 `spirv` intrinsics once emitted
//! `p1[gid] = p0[gid]` at exit 0, so neither a name nor an exit code is a lowering.
//!
//! A backend whose emitter source this build could not read is **UNKNOWN**, never zero
//! -- `cpp` (AscendC) and `pico` are closed emitters in sibling checkouts, and a
//! packaged crate has no siblings at all. Scoring those 0 of 509 would be the one wrong
//! answer.

include!(concat!(env!("OUT_DIR"), "/intrinsics_table.rs"));

/// The status of one intrinsic on one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The emitter source names this intrinsic.
    Exact,
    /// The emitter names a family prefix of it and dispatches on the suffix.
    Prefix,
    /// Neither: nobody has written an arm. NOT a claim that it cannot be written.
    Absent,
}

/// Which backends have a derivable surface in this build, in table order.
pub fn backends() -> Vec<&'static str> {
    BACKENDS.iter().map(|(id, _)| *id).collect()
}

/// The emitter source a backend's surface was derived from.
pub fn source_of(form: &str) -> Option<&'static str> {
    BACKENDS.iter().find(|(id, _)| *id == form).map(|(_, p)| *p)
}

/// Every declared intrinsic with its status on `form`, in name order.
pub fn surface(form: &str) -> Option<Vec<(&'static str, &'static str, usize, Arm)>> {
    let bit = 1u32 << BACKENDS.iter().position(|(id, _)| *id == form)?;
    Some(
        INTRINSICS
            .iter()
            .map(|(name, file, line, exact, pref)| {
                let arm = if exact & bit != 0 {
                    Arm::Exact
                } else if pref & bit != 0 {
                    Arm::Prefix
                } else {
                    Arm::Absent
                };
                (*name, *file, *line, arm)
            })
            .collect(),
    )
}

/// How many intrinsics `tile_std` declares. The denominator, taken from the table rather
/// than restated in prose, because a count written beside the data is the thing that
/// drifts.
pub fn declared() -> usize {
    INTRINSICS.len()
}

/// The report, as text. `Err` carries the refusal to print.
///
/// It REFUSES an unknown form rather than printing an empty table, and it refuses a form
/// whose emitter source is not in this build rather than scoring it zero -- the two
/// refusals are different sentences, because "nobody wrote that backend" and "this build
/// cannot see its source" are different facts.
pub fn report(form: &str) -> Result<String, String> {
    if INTRINSICS.is_empty() {
        return Err(format!(
            "this build derived no intrinsic table: crates/tile_std/src/{{lib,tile}}.rs \
             was not readable when it was built, so the surface for {form} is UNKNOWN \
             rather than empty. Build from a checkout that carries crates/tile_std."
        ));
    }
    let known = crate::forms::by_id(form).is_some();
    let Some(src) = source_of(form) else {
        let mut have = backends();
        have.sort_unstable();
        return Err(if known {
            format!(
                "{form} is a form this tool knows, and its emitter source is not in this \
                 build -- it is a closed emitter in a sibling checkout. Its intrinsic \
                 surface is UNKNOWN, not empty.\nDerivable here: {}",
                have.join(" ")
            )
        } else {
            format!(
                "no backend called {form:?}.\nDerivable here: {}\nSee `tile --list-forms` \
                 for every form, readable and writable.",
                have.join(" ")
            )
        });
    };
    let rows = surface(form).expect("source_of agreed");
    let w = rows.iter().map(|r| r.0.len()).max().unwrap_or(40) + 2;

    let mut s = String::new();
    s.push_str(&format!(
        "{form} -- the intrinsic surface tile_std declares, with the arms {src} carries\n\n"
    ));
    s.push_str(&format!("{:<w$}{:<9}declared at\n", "intrinsic", "arm"));
    s.push_str(&"-".repeat(w + 9 + 28));
    s.push('\n');
    let (mut e, mut p) = (0usize, 0usize);
    for (name, file, line, arm) in &rows {
        let tag = match arm {
            Arm::Exact => {
                e += 1;
                "exact"
            }
            Arm::Prefix => {
                p += 1;
                "prefix"
            }
            Arm::Absent => "-",
        };
        s.push_str(&format!("{name:<w$}{tag:<9}{file}:{line}\n"));
    }
    s.push_str(&"-".repeat(w + 9 + 28));
    s.push('\n');
    s.push_str(&format!(
        "{} of {} declared intrinsics are named by {src}: {e} exactly, {p} by a family \
         prefix.\n",
        e + p,
        rows.len()
    ));
    s.push_str(
        "\nDERIVED FROM SOURCE, NOT MEASURED. `exact` means the emitter names that \
         intrinsic;\n`prefix` means it names a family prefix and dispatches on the \
         suffix. Neither is proof\nthat the arm emits anything: 42 of 42 spirv \
         intrinsics once emitted a plain copy at exit\n0. For a MEASURED answer use \
         `tile --list-ops`, which lowers a real module per cell --\nover 32 portable ops \
         rather than these ",
    );
    s.push_str(&format!("{}.\n", rows.len()));
    s.push_str(
        "A dash is a gap nobody has written, never a gap that cannot be written.\n",
    );
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The acceptance test from `docs/cli/spec-data-movement.md` §1: the one question
    /// that cost a session must be answerable in one command.
    #[test]
    fn asking_about_pto_shows_the_fused_int8_dequant_matmul() {
        let r = report("pto").expect("pto has a derivable surface in this build");
        assert!(
            r.contains("__tile_matmul_i8_acc_i32_dequant_f16"),
            "the fused int8 dequant matmul is absent from pto's surface report"
        );
        // Not just present -- present with an arm. A row of dashes would reproduce the
        // wrong conclusion the spec opens with.
        let row = r
            .lines()
            .find(|l| l.starts_with("__tile_matmul_i8_acc_i32_dequant_f16"))
            .expect("the row");
        assert!(
            row.contains("exact") || row.contains("prefix"),
            "pto names this intrinsic in mlir_to_pto.rs, so the row must not read `-`: {row}"
        );
    }

    /// The surface must be much larger than the portable probe, or it is not answering a
    /// different question. 32 is `emit::PROBE_OPS.len()`, read rather than retyped.
    #[test]
    fn the_surface_is_larger_than_the_portable_probe() {
        assert!(
            declared() > 4 * crate::emit::PROBE_OPS.len(),
            "{} declared intrinsics against a {}-op probe -- if these are comparable, \
             this command is redundant",
            declared(),
            crate::emit::PROBE_OPS.len()
        );
    }

    /// An absent source is UNKNOWN. `cpp` is a known form whose AscendC emitter is
    /// closed, so it must refuse in those words rather than print 0 of 509.
    #[test]
    fn a_closed_emitter_is_unknown_and_not_zero() {
        let e = report("cpp").expect_err("cpp's emitter source is not in this tree");
        assert!(e.contains("UNKNOWN"), "{e}");
        assert!(!e.contains("0 of"), "{e}");
    }

    /// And a form nobody has heard of is a different refusal from a closed one.
    #[test]
    fn an_unknown_form_is_refused_with_the_set_that_works() {
        let e = report("vulkan2").expect_err("no such backend");
        assert!(e.contains("no backend called"), "{e}");
        assert!(e.contains("pto"), "the refusal must name what does work: {e}");
    }

    /// The prefix rule must not make every backend cover everything, which is the one
    /// way a derived table flatters itself.
    #[test]
    fn the_prefix_rule_is_not_vacuous() {
        for id in backends() {
            let rows = surface(id).unwrap();
            let pref = rows.iter().filter(|r| r.3 == Arm::Prefix).count();
            assert!(
                pref * 2 < rows.len(),
                "{id}: {pref} of {} rows are PREFIX matches, which means the rule is \
                 matching families far too loosely",
                rows.len()
            );
        }
    }

    /// THE CROSS-CHECK AGAINST THE MEASURED TABLE, in the one direction that matters.
    ///
    /// The two instruments can legitimately disagree one way: an emitter can NAME an
    /// intrinsic and still refuse the probe module, because the probe fixes a shape and
    /// a dtype. The other direction is this table being WRONG -- a backend that
    /// demonstrably lowers an op whose name its source never mentions would mean the
    /// dispatch does not go through the name, and every `-` in that family would be an
    /// under-report rather than a gap. That is the failure a derived table has and a
    /// measured one does not, so it is asserted rather than assumed.
    #[cfg(feature = "emitters")]
    #[test]
    fn nothing_the_probe_lowers_is_missing_from_the_derived_surface() {
        let mut bad = Vec::new();
        for id in backends() {
            let form = crate::forms::by_id(id).unwrap();
            let rows = surface(id).unwrap();
            for (op, shape) in crate::emit::PROBE_OPS {
                if !crate::emit::lowers_op(form, op, *shape) {
                    continue;
                }
                let want = format!("__tile_{op}_f32");
                match rows.iter().find(|r| r.0 == want) {
                    None => bad.push(format!("{id}/{op}: lowered, and tile_std declares no {want}")),
                    Some(r) if r.3 == Arm::Absent => {
                        bad.push(format!("{id}/{op}: the probe lowers it and {want} is not in the emitter source"))
                    }
                    Some(_) => {}
                }
            }
        }
        assert!(bad.is_empty(), "derived surface under-reports:\n{}", bad.join("\n"));
    }

    /// Every backend in the table has a readable source, and every source is a real
    /// emitter path -- the two halves of "derived from the tree".
    #[test]
    fn every_listed_backend_is_a_known_writable_form() {
        for (id, src) in BACKENDS {
            assert!(
                crate::forms::by_id(id).is_some_and(|f| f.writable),
                "{id} is in the intrinsic table and is not a writable form"
            );
            assert!(src.contains("mlir_to_"), "{id}: {src} is not an emitter");
        }
    }
}
