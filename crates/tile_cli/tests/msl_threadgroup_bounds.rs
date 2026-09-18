//! A threadgroup array indexed by `tid` must be sized for the widest dispatch, not for
//! the tile width the MLIR happened to mention.
//!
//! `sdata` used to be declared `threadgroup float sdata[local_x]`, where local_x came
//! from the MLIR's tile width at EMIT time, while the body writes `sdata[tid]` for every
//! tid < tcount and reduces from `s = tcount/2` — and tcount is `threads_per_threadgroup`,
//! chosen at RUN time from the run shape. Two independent sources for one quantity, equal
//! only by coincidence.
//!
//! When they disagreed the kernel wrote past the end of the array. Measured on an M1
//! Ultra with a kernel emitted for a 64-wide tile and run over a 1024-wide row: softmax
//! came back with a max relative error of 17.8 against torch (our own reference and torch
//! agreed to 6.7e-7, so the disagreement was attributable to the kernel). `-O4` made it
//! worse by preferring threadgroup 1024 — overrunning fastest is still fastest. After the
//! fix the same case gives 6.11e-7 and all three agree.
//!
//! This test is device-free on purpose: it pins the invariant that made the numbers wrong,
//! so it fails on a machine with no GPU too.

use tile_cli::emit::emit_for;
use tile_cli::forms::by_id;

/// Load a `load_cols`-wide tile, reduce it, store `store_cols` wide. The mismatch is what
/// pulled the emitted array size away from the dispatch width.
fn reduce_mlir(load_cols: u32, store_cols: u32) -> String {
    format!(
        r#"
module {{
  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {{hacc.entry}} {{
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %lc = llvm.mlir.constant({load_cols} : i32) : i32
    %sc = llvm.mlir.constant({store_cols} : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %r, %lc) : (!llvm.ptr<1>, i32, i32) -> i32
    %y = llvm.call @__tile_softmax_f32(%a, %a, %r, %lc) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %y, %r, %sc) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }}
}}
"#
    )
}

fn emit(mlir: &str) -> String {
    emit_for(by_id("msl").expect("msl form"), mlir).expect("softmax lowers")
}

#[test]
fn the_shared_array_is_sized_for_the_widest_dispatch() {
    let out = emit(&reduce_mlir(256, 256));
    assert!(
        out.contains("sdata[MAX_TG]") && out.contains("constexpr uint MAX_TG = 1024;"),
        "expected sdata sized by the dispatch ceiling, got:\n{out}"
    );
}

#[test]
fn a_narrow_tile_width_does_not_shrink_the_array() {
    // The case that produced 17.8 max-rel error: emitted for 64, dispatched at 1024.
    let out = emit(&reduce_mlir(1024, 64));
    assert!(
        !out.contains("sdata[64]"),
        "the array was sized by the tile width again; a wider dispatch overruns it:\n{out}"
    );
    assert!(out.contains("sdata[MAX_TG]"), "got:\n{out}");
}

#[test]
fn the_argmin_half_of_the_same_reduction_is_sized_too() {
    // idx_data sat beside sdata carrying the argmin, hardcoded at 256. Fixing only sdata
    // would have left this half overrunning on the same dispatch.
    let mlir = reduce_mlir(256, 256).replace("__tile_softmax_f32", "__tile_argmin_f32");
    let out = emit(&mlir);
    assert!(out.contains("idx_data[MAX_TG]"), "got:\n{out}");
    assert!(!out.contains("idx_data[256]"), "got:\n{out}");
}
