// The canonical softmax with a dead definition spliced in: `%junk` is computed and never
// read. It exists so `-O1`'s dead-op elimination has something to actually do, and so
// "identical forms mean optimization, not a copy" can be asserted against a run that
// really does change the bytes.
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c1024 = llvm.mlir.constant(1024 : i32) : i32
    %junk = llvm.mlir.constant(7 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
