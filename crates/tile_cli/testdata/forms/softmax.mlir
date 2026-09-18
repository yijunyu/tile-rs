// The canonical tile-rs lowering of softmax: LLVM-dialect MLIR carrying `__tile_*`
// intrinsic calls and an `hacc.entry` attribute. This is the shape the emitters consume
// -- the same snippet crates/tile_spec drives the generality matrix with -- so the
// golden files here and the emit-purity contract there are about the same input.
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c1024 = llvm.mlir.constant(1024 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
