module {
  func.func @softmax(%arg0: memref<1024xf16>) {
    %0 = pto.load %arg0 : memref<1024xf16>
    %1 = pto.exp %0 : vector<1024xf16>
    pto.store %1, %arg0 : memref<1024xf16>
    return
  }
}
