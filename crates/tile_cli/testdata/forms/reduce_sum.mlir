module {
  llvm.func @rsum(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %y = llvm.call @__tile_reduce_sum_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
