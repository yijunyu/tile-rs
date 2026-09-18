module {
  func.func @rms_blk(%x: !pto.ptr<f32>, %y: !pto.ptr<f32>, %rows: index, %cols: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cTW = arith.constant 1024 : index
    %eps = arith.constant 0.000001 : f32
    %one = arith.constant 1.0 : f32
    %vin = pto.make_tensor_view %x, shape = [%rows, %cols], strides = [%cols, %c1] : !pto.tensor_view<?x?xf32>
    %vout = pto.make_tensor_view %y, shape = [%rows, %cols], strides = [%cols, %c1] : !pto.tensor_view<?x?xf32>
    %tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>
    %sq = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>
    %ws = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>
    %acc = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
    %part = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
    %accr = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %partr = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %m0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %m1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %m2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %m3 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %bc = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>
    %outt = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>
    // runtime 1/D
    %colsi = arith.index_cast %cols : index to i32
    %colsf = arith.sitofp %colsi : i32 to f32
    %invd = arith.divf %one, %colsf : f32
    %w0 = arith.minui %cols, %cTW : index
    scf.for %i = %c0 to %rows step %c1 {
      // peeled first block seeds the accumulator, so nothing needs zeroing
      %p0 = pto.partition_view %vin, offsets = [%i, %c0], sizes = [%c1, %w0] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x?xf32>
      pto.tload ins(%p0 : !pto.partition_tensor_view<1x?xf32>) outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
      pto.tmul ins(%tile, %tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>) outs(%sq : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
      pto.trowsum ins(%sq, %ws : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>) outs(%acc : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
      pto.tmuls ins(%acc, %one : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%accr : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      scf.for %cb = %cTW to %cols step %cTW {
        %rem = arith.subi %cols, %cb : index
        %w = arith.minui %rem, %cTW : index
        %p = pto.partition_view %vin, offsets = [%i, %cb], sizes = [%c1, %w] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x?xf32>
        pto.tload ins(%p : !pto.partition_tensor_view<1x?xf32>) outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
        pto.tmul ins(%tile, %tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>) outs(%sq : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
        pto.trowsum ins(%sq, %ws : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>) outs(%part : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
        pto.tmuls ins(%part, %one : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%partr : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.tadd ins(%accr, %partr : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%accr : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      }
      pto.tmuls ins(%accr, %invd : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%m0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tadds ins(%m0, %eps : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%m1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tsqrt ins(%m1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%m2 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.trecip ins(%m2 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%m3 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.trowexpand ins(%m3 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%bc : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
      scf.for %cb2 = %c0 to %cols step %cTW {
        %rem2 = arith.subi %cols, %cb2 : index
        %w2 = arith.minui %rem2, %cTW : index
        %pi = pto.partition_view %vin, offsets = [%i, %cb2], sizes = [%c1, %w2] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x?xf32>
        pto.tload ins(%pi : !pto.partition_tensor_view<1x?xf32>) outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
        pto.tmul ins(%tile, %bc : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>) outs(%outt : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>)
        %po = pto.partition_view %vout, offsets = [%i, %cb2], sizes = [%c1, %w2] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x?xf32>
        pto.tstore ins(%outt : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=1>) outs(%po : !pto.partition_tensor_view<1x?xf32>)
      }
    }
    return
  }
}
