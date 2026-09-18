//! A shape-specialized Metal kernel must refuse a shape it was not written for.
//!
//! `emit_flash_attn_ext_vec_score_msl` bakes in `DK4_FIXED = 16` — a head dimension of
//! 64 — and uses it as the per-row K stride, while the intrinsic still takes `dk` as an
//! operand. Before the guard, a call with dk=128 emitted that kernel anyway: it read K at
//! half the right stride and returned wrong attention scores with no crash and no
//! refusal. The same applies to the norm4 kernel, which bakes in n_embd=4096.
//!
//! The guard refuses only what it can PROVE: a shape operand that is a known constant and
//! disagrees. A dynamic operand is not evidence of a mismatch, so it still lowers — the
//! last case here pins that, because a guard that refused those would break working
//! callers to look thorough.

use tile_cli::emit::emit_for;
use tile_cli::forms::by_id;

fn score_mlir(dk: &str, dv: &str) -> String {
    format!(
        r#"
module {{
  llvm.func @sk(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {{hacc.entry}} {{
    ^bb0:
    %b = llvm.mlir.constant(0 : i32) : i32
    %dk = llvm.mlir.constant(64 : i32) : i32
    %dv = llvm.mlir.constant(64 : i32) : i32
    %dyn = llvm.load %arg0 : !llvm.ptr<1> -> i32
    %n = llvm.mlir.constant(32 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_vec_score_f32(%b, %b, %b, %b, %b, %b, %b, {dk}, {dv}, %n, %n, %n, %n, %n) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }}
}}
"#
    )
}

fn emit(mlir: &str) -> Result<String, String> {
    emit_for(by_id("msl").expect("msl form"), mlir)
}

#[test]
fn the_shape_it_was_written_for_lowers() {
    let out = emit(&score_mlir("%dk", "%dv")).expect("dk=dv=64 is the specialization");
    assert!(out.contains("DK4_FIXED"), "expected the specialized kernel");
}

#[test]
fn a_head_dimension_it_was_not_written_for_is_refused() {
    let err = emit(&score_mlir("128", "128")).expect_err("dk=128 must be refused");
    assert!(err.contains("specialized to dk=64"), "got: {err}");
    assert!(err.contains("called with dk=128"), "got: {err}");
}

#[test]
fn each_shape_operand_is_checked_separately() {
    // dk is right and dv is wrong: a guard that only looked at the first operand,
    // or that stopped at the first match, would let this through.
    let err = emit(&score_mlir("%dk", "128")).expect_err("dv=128 must be refused");
    assert!(err.contains("specialized to dv=64"), "got: {err}");
}

#[test]
fn a_dynamic_shape_operand_still_lowers() {
    // %dyn is loaded at run time, so nothing is proven about it. `parse_const_arg` would
    // happily turn an SSA name into a number and refuse on it; `known_const` does not.
    emit(&score_mlir("%dyn", "%dv")).expect("an unknown dk is not a proven mismatch");
}
