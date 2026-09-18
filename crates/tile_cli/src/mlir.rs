//! The MLIR-to-MLIR rewrite layer: a real IR, not lines of text.
//!
//! Everything before this was textual. Textual works for counting and for deleting whole
//! lines; it does not work for a rewrite, because a rewrite has to know what an operation
//! *reads*, what it *writes*, and whether moving it changes the answer. So this module
//! parses the subset of LLVM-dialect MLIR that tile-rs actually emits into operations,
//! and prints it back.
//!
//! ## The property everything else rests on
//!
//! **Parse then print is byte-identical.** An [`Item::Raw`] holds any line this module
//! does not understand, verbatim; an [`Op`] that no pass touched prints its original text
//! rather than a re-rendering. So the layer can only change what a pass deliberately
//! changed — and the 15 golden files stay stable through a refactor of the parser.
//!
//! ## The constraint that shapes the passes
//!
//! **The emitters own fusion, and they find it by adjacency.** `mlir_to_tpu.rs` fuses
//! SiLU into a following multiply by remembering the previous op (`ctx.last_silu`) and
//! checking whether one of the multiply's operands is its result. Two consequences, both
//! non-obvious and both load-bearing:
//!
//! * A pass here must **never fuse into a new intrinsic**. `__tile_silu_mul_f32` is not
//!   in any emitter's vocabulary, so "fusing" would turn a lowerable kernel into an
//!   unlowerable one. Fusion is the emitter's job and this layer must not take it.
//! * A pass must **never separate a producer from the consumer that fuses it**. Moving an
//!   op between a `silu` and its `mul` silently costs the fusion downstream — the kernel
//!   still lowers, still computes the right answer, and is slower for a reason nobody can
//!   see. [`fusion_pairs_intact`] is checked after every pipeline for exactly this.

use std::collections::BTreeMap;

/// One parsed operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub indent: String,
    /// The SSA name this defines, e.g. `%t0`.
    pub result: Option<String>,
    /// `llvm.call @__tile_load_f32`, `llvm.mlir.constant`, ...
    pub callee: String,
    pub args: Vec<String>,
    /// Everything after the closing paren, including the type signature.
    pub tail: String,
    /// The line exactly as it was read.
    pub raw: String,
    /// Set when a pass rewrote this op, so printing re-renders instead of echoing.
    pub rewritten: bool,
}

impl Op {
    /// The intrinsic name without the `@`, when this is a tile intrinsic call.
    pub fn intrinsic(&self) -> Option<&str> {
        self.callee.strip_prefix("llvm.call @")
    }

    /// Does this operation touch memory or ordering?
    ///
    /// Conservative by construction: an operation is pure only if it is on the list. A
    /// wrong answer here removes a store, and a removed store is a miscompile that stays
    /// invisible until it corrupts a run.
    pub fn is_pure(&self) -> bool {
        if self.callee == "llvm.mlir.constant" {
            return true;
        }
        let Some(name) = self.intrinsic() else {
            return false;
        };
        // Anything that writes, allocates, or orders is impure — and so is anything this
        // list has not been taught about, which is the safe direction to be wrong in.
        // A LOAD is on this list. It has no side effect, but it reads memory, so two
        // identical loads are only interchangeable when nothing wrote in between —
        // reasoning `cse` does not do and `redundant_loads` does. Calling a load pure let
        // CSE collapse a load across a store, which reads stale data.
        const IMPURE: &[&str] = &[
            "load",
            "get_rows",
            "store",
            "buf_alloc",
            "barrier",
            "pipe",
            "scatter",
            "set_rows",
            "fill",
            "kv_cache_update",
            "kv_write",
            "kv_fp8_store",
            "inplace",
            "_mut",
            "compressor_store",
            "init_sort_buf",
            "cache",
            "view",
        ];
        if IMPURE.iter().any(|m| name.contains(m)) {
            return false;
        }
        name.starts_with("__tile_")
    }

    /// A load reads memory: it is not pure, but it IS removable when nothing uses it, and
    /// redundant when repeated with no intervening write.
    pub fn is_load(&self) -> bool {
        self.intrinsic()
            .is_some_and(|n| n.contains("load") || n.contains("get_rows"))
    }

    pub fn is_store(&self) -> bool {
        self.intrinsic()
            .is_some_and(|n| n.contains("store") || n.contains("scatter"))
    }

    /// Operands that are SSA names.
    pub fn ssa_args(&self) -> Vec<&str> {
        self.args
            .iter()
            .map(|a| a.trim())
            .filter(|a| a.starts_with('%'))
            .collect()
    }

    /// Removing an op is only safe when nothing depends on it AND it does nothing.
    pub fn removable_when_unused(&self) -> bool {
        self.is_pure() || self.is_load()
    }

    pub fn render(&self) -> String {
        if !self.rewritten {
            return self.raw.clone();
        }
        let lhs = match &self.result {
            Some(r) => format!("{r} = "),
            None => String::new(),
        };
        format!(
            "{}{lhs}{}({}){}",
            self.indent,
            self.callee,
            self.args.join(", "),
            self.tail
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Op(Op),
    /// Anything not recognized: module headers, function signatures, comments, `}`.
    /// Kept verbatim, which is what makes the round trip exact.
    Raw(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Module {
    pub items: Vec<Item>,
    /// Whether the input ended with a newline, so printing restores it exactly.
    trailing_newline: bool,
}

impl Module {
    pub fn ops(&self) -> impl Iterator<Item = &Op> {
        self.items.iter().filter_map(|i| match i {
            Item::Op(o) => Some(o),
            _ => None,
        })
    }
    pub fn ops_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.items.iter_mut().filter_map(|i| match i {
            Item::Op(o) => Some(o),
            _ => None,
        })
    }
    pub fn op_count(&self) -> usize {
        self.ops().count()
    }
}

/// Split `a, b, c` respecting nesting, so a type argument with its own parens survives.
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | '<' | '[' => {
                depth += 1;
                cur.push(c);
            }
            ')' | '>' | ']' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Find the argument list of a call: the parenthesised group that closes at depth zero.
fn call_parts(body: &str) -> Option<(String, Vec<String>, String)> {
    let open = body.find('(')?;
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in body[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    Some((
        body[..open].trim().to_string(),
        split_args(&body[open + 1..close]),
        body[close + 1..].to_string(),
    ))
}

pub fn parse(text: &str) -> Module {
    let mut m = Module {
        trailing_newline: text.ends_with('\n'),
        ..Default::default()
    };
    for raw in text.lines() {
        let trimmed = raw.trim_start();
        let indent = raw[..raw.len() - trimmed.len()].to_string();

        // A comment is never an operation, however much it looks like one. This is the
        // rule that stopped dead-op elimination reading SSA names out of prose.
        if trimmed.starts_with("//") {
            m.items.push(Item::Raw(raw.to_string()));
            continue;
        }

        let (result, body) = match trimmed.split_once(" = ") {
            Some((lhs, rhs)) if lhs.starts_with('%') => (Some(lhs.trim().to_string()), rhs),
            _ => (None, trimmed),
        };
        let is_call = body.starts_with("llvm.call @") || body.starts_with("llvm.mlir.constant");
        match (is_call, call_parts(body)) {
            (true, Some((callee, args, tail))) => m.items.push(Item::Op(Op {
                indent,
                result,
                callee,
                args,
                tail,
                raw: raw.to_string(),
                rewritten: false,
            })),
            _ => m.items.push(Item::Raw(raw.to_string())),
        }
    }
    m
}

pub fn print(m: &Module) -> String {
    let mut s = String::new();
    for (i, item) in m.items.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        match item {
            Item::Op(o) => s.push_str(&o.render()),
            Item::Raw(r) => s.push_str(r),
        }
    }
    if m.trailing_newline {
        s.push('\n');
    }
    s
}

/// Every SSA name read by any operation, with how many operations read it.
pub fn use_counts(m: &Module) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for o in m.ops() {
        for a in o.ssa_args() {
            *counts.entry(a.to_string()).or_insert(0) += 1;
        }
    }
    // A name mentioned by an unparsed line (a `llvm.return %x`, a branch) is used too.
    for item in &m.items {
        if let Item::Raw(r) = item {
            if r.trim_start().starts_with("//") {
                continue;
            }
            for tok in r.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '%')) {
                if tok.starts_with('%') && tok.len() > 1 {
                    *counts.entry(tok.to_string()).or_insert(0) += 1;
                }
            }
        }
    }
    counts
}

/// Rename every use of `from` to `to`, leaving definitions alone.
pub fn replace_uses(m: &mut Module, from: &str, to: &str) -> usize {
    let mut n = 0;
    for o in m.ops_mut() {
        if o.result.as_deref() == Some(from) {
            continue;
        }
        for a in o.args.iter_mut() {
            if a.trim() == from {
                *a = to.to_string();
                o.rewritten = true;
                n += 1;
            }
        }
    }
    n
}

/// Producer/consumer pairs the emitters fuse by adjacency.
///
/// Today that is SiLU into a following multiply — the SwiGLU arm. The check is written
/// against the *mechanism* (an emitter that remembers the previous op) rather than
/// against that one pair, so a second adjacency-fused pattern joins by adding a row.
const FUSED_ADJACENT: &[(&str, &str)] = &[("silu", "mul")];

/// Is every fusible producer still immediately followed by its consumer?
///
/// A pass that inserts an operation between a `silu` and the `mul` that consumes it costs
/// the fusion downstream. The kernel still lowers and still computes the right answer,
/// and is slower for a reason nobody can see from the output — so this is checked rather
/// than trusted.
pub fn fusion_pairs_intact(m: &Module) -> Result<(), String> {
    let ops: Vec<&Op> = m.ops().collect();
    for (i, op) in ops.iter().enumerate() {
        let Some(name) = op.intrinsic() else { continue };
        for (producer, consumer) in FUSED_ADJACENT {
            // Match the bare op name, so `__tile_silu_f32` matches `silu` while
            // `__tile_silu_mul_batched` does not masquerade as one.
            if !is_intrinsic(name, producer) {
                continue;
            }
            let Some(result) = &op.result else { continue };
            // Find the consumer that reads it.
            let Some(j) = ops
                .iter()
                .position(|o| o.ssa_args().contains(&result.as_str()) && is_consumer(o, consumer))
            else {
                continue;
            };
            if j != i + 1 {
                return Err(format!(
                    "{producer} at op {i} is no longer adjacent to the {consumer} at op {j} \
                     that fuses it; the emitters find this pattern by adjacency, so the \
                     fusion would be silently lost"
                ));
            }
        }
    }
    Ok(())
}

fn is_intrinsic(full: &str, bare: &str) -> bool {
    let Some(rest) = full.strip_prefix("__tile_") else {
        return false;
    };
    // `silu_f32` and `silu_f16` are silu; `silu_mul_batched` is a different intrinsic.
    let stem = rest
        .strip_suffix("_f32")
        .or_else(|| rest.strip_suffix("_f16"))
        .unwrap_or(rest);
    stem == bare
}

fn is_consumer(o: &Op, bare: &str) -> bool {
    o.intrinsic().is_some_and(|n| is_intrinsic(n, bare))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOFTMAX: &str = include_str!("../testdata/forms/softmax.mlir");

    #[test]
    fn parse_then_print_is_byte_identical() {
        // The property every pass rests on: the layer can only change what a pass
        // deliberately changed. Without it, the 15 golden files churn on a refactor.
        assert_eq!(print(&parse(SOFTMAX)), SOFTMAX);
    }

    #[test]
    fn round_trips_awkward_input_unchanged() {
        for src in [
            "",
            "\n",
            "module {}\n",
            "no trailing newline",
            "  // just a comment\n",
            "%x = something.unparseable\n",
            "}\n}\n",
        ] {
            assert_eq!(print(&parse(src)), src, "{src:?} did not survive");
        }
    }

    #[test]
    fn the_canonical_kernel_parses_into_the_operations_it_has() {
        let m = parse(SOFTMAX);
        assert_eq!(m.op_count(), 6, "3 constants, load, softmax, store");
        let names: Vec<&str> = m.ops().filter_map(|o| o.intrinsic()).collect();
        assert_eq!(
            names,
            vec!["__tile_load_f32", "__tile_softmax_f32", "__tile_store_f32"]
        );
    }

    #[test]
    fn a_type_signature_with_parens_does_not_confuse_the_argument_split() {
        let m = parse("    %t = llvm.call @f(%a, %b) : (!llvm.ptr<1>, i32, i32) -> i32\n");
        let o = m.ops().next().unwrap();
        assert_eq!(o.args, vec!["%a", "%b"]);
        assert!(o.tail.contains("-> i32"));
    }

    #[test]
    fn a_comment_is_never_an_operation() {
        let m = parse("  // %t = llvm.call @__tile_load_f32(%a)\n");
        assert_eq!(m.op_count(), 0);
    }

    #[test]
    fn stores_and_allocations_are_never_pure() {
        // The conservative direction. A wrong answer here removes a store, and a removed
        // store is a miscompile that stays invisible until it corrupts a run.
        for name in [
            "__tile_store_f32",
            "__tile_buf_alloc",
            "__tile_scatter_add_f32",
            "__tile_rope_inplace",
            "__tile_pipe_barrier",
            "__tile_kv_cache_update",
        ] {
            let m = parse(&format!("  llvm.call @{name}(%a) : (i32) -> ()\n"));
            assert!(!m.ops().next().unwrap().is_pure(), "{name} was called pure");
        }
    }

    #[test]
    fn an_unknown_call_is_treated_as_impure() {
        // Being wrong about something the list has not been taught is safe in exactly
        // one direction.
        let m = parse("  %x = llvm.call @mystery(%a) : (i32) -> i32\n");
        assert!(!m.ops().next().unwrap().is_pure());
    }

    #[test]
    fn a_load_is_not_pure_however_harmless_it_looks() {
        // It has no side effect, but it READS memory. Treating it as pure let CSE
        // collapse a load across a store, which reads stale data — the failure that
        // never shows up in a diff.
        let m = parse("  %a = llvm.call @__tile_load_f32(%p) : (i32) -> i32\n");
        let o = m.ops().next().unwrap();
        assert!(!o.is_pure(), "a load was called pure");
        assert!(o.is_load());
        assert!(o.removable_when_unused(), "a dead load is still removable");
    }

    #[test]
    fn arithmetic_intrinsics_and_constants_are_pure() {
        for name in ["__tile_add_f32", "__tile_softmax_f32", "__tile_exp_f32"] {
            let m = parse(&format!("  %x = llvm.call @{name}(%a) : (i32) -> i32\n"));
            assert!(
                m.ops().next().unwrap().is_pure(),
                "{name} was called impure"
            );
        }
        let m = parse("  %c = llvm.mlir.constant(1 : i32) : i32\n");
        assert!(m.ops().next().unwrap().is_pure());
    }

    #[test]
    fn a_value_read_only_by_an_unparsed_line_still_counts_as_used() {
        // `llvm.return %t1` is not an Op. Missing it would make the returned value look
        // dead and delete the whole kernel.
        let m = parse("  %t = llvm.mlir.constant(1 : i32) : i32\n  llvm.return %t : i32\n");
        assert_eq!(use_counts(&m).get("%t"), Some(&1));
    }

    #[test]
    fn replacing_a_use_rewrites_the_operand_and_not_the_definition() {
        let mut m = parse(
            "  %a = llvm.mlir.constant(1 : i32) : i32\n  \
             %b = llvm.call @__tile_add_f32(%a, %a) : (i32, i32) -> i32\n",
        );
        assert_eq!(replace_uses(&mut m, "%a", "%z"), 2);
        let printed = print(&m);
        assert!(printed.contains("%a = llvm.mlir.constant"), "{printed}");
        assert!(printed.contains("@__tile_add_f32(%z, %z)"), "{printed}");
    }

    #[test]
    fn an_untouched_op_prints_its_original_bytes_even_beside_a_rewritten_one() {
        let src = "  %a  =  llvm.mlir.constant(1 : i32) : i32\n  \
                   %b = llvm.call @__tile_add_f32(%a, %a) : (i32, i32) -> i32\n";
        let mut m = parse(src);
        replace_uses(&mut m, "%a", "%z");
        // The constant's unusual spacing survives; only the rewritten line is re-rendered.
        assert!(print(&m).contains("%a  =  llvm.mlir.constant"));
    }

    #[test]
    fn an_adjacent_silu_and_mul_are_recognised_as_a_fusion_pair() {
        let src = "  %s = llvm.call @__tile_silu_f32(%g, %g, %r, %c) : (i32, i32, i32, i32) -> i32\n  \
                   %m = llvm.call @__tile_mul_f32(%s, %s, %u, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n";
        assert!(fusion_pairs_intact(&parse(src)).is_ok());
    }

    #[test]
    fn separating_a_silu_from_its_mul_is_reported() {
        // The whole reason this check exists: the kernel would still lower and still be
        // correct, and would be slower for a reason invisible in the output.
        let src = "  %s = llvm.call @__tile_silu_f32(%g, %g, %r, %c) : (i32, i32, i32, i32) -> i32\n  \
                   %x = llvm.call @__tile_exp_f32(%g, %r, %c) : (i32, i32, i32) -> i32\n  \
                   %m = llvm.call @__tile_mul_f32(%s, %s, %u, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n";
        let e = fusion_pairs_intact(&parse(src)).unwrap_err();
        assert!(e.contains("adjacent"), "{e}");
        assert!(e.contains("silently lost"), "{e}");
    }

    #[test]
    fn a_differently_named_intrinsic_does_not_masquerade_as_the_fusible_one() {
        // `__tile_silu_mul_batched` is its own intrinsic, not a silu awaiting a mul.
        let src = "  %s = llvm.call @__tile_silu_mul_batched(%g) : (i32) -> i32\n  \
                   %y = llvm.call @__tile_exp_f32(%s) : (i32) -> i32\n";
        assert!(fusion_pairs_intact(&parse(src)).is_ok());
    }
}
