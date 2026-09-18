module {
  func.func @softmax(%arg0: tensor<1024xf32>, %out: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>]}
         ins(%arg0 : tensor<1024xf32>) outs(%out : tensor<1024xf32>) {
      ^bb0(%in: f32, %o: f32):
        %e = math.exp %in : f32
        linalg.yield %e : f32
    } -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
