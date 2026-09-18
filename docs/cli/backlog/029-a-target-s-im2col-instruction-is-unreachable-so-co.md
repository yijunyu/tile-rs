id: 29
area: target
title: A target's im2col instruction is unreachable, so conv lowers to a materialised column matrix
opened: 2026-09-10

## wanted
Lower a 2-D convolution to the Ascend cube without materialising the im2col
column matrix. The hardware has `LoadData3DParamsV2<T>` -- padList, l1H, l1W,
channelSize, kExtension, mExtension, kStartPt, mStartPt, strideW/H, filterW/H,
dilationFilterW/H, padValue -- which produces column fractals on the way from
L1 into L0A. The feature map is read once, in its own layout.

## got
No path to it. tile-rs lowers a convolution as im2col + matmul, which is what
the cannbench tree does by hand: a GM buffer of rows x (kh*kw*C), written by
one kernel and read back by another. For a 5x5 conv over 256 channels that is
a 419 MB intermediate against a 16 MB input, and the operator runs at 0.17x of
the reference where the field leader runs at 1.63x.

## workaround
None yet -- recorded before the work rather than after, because the shape of
the gap is already clear and the next session should start from the
instruction. What the fix will need:

  * The intrinsic contract has no place to say "this target has a strided
    windowed load", so the choice between im2col+matmul and a direct-conv
    lowering is not expressible -- it is a property of the TARGET, and the
    declaration the `lowering` cluster keeps asking for is where it belongs.
  * `LoadData3DParamsV2` is arch-templated in the same 3510/5102 way
    FixpipeParams is (#028), so whatever describes it has to carry per-arch
    slot presence, not just names.
  * The layout precondition is C-innermost (NHWC). The reference pays that too
    -- every conv_2d case's published `baseline_kernels` note says
    `TransData x2` or `x3` -- so it is a cost of the operator, not of this
    approach.
  * depthwise_conv_2d and conv_3d_backprop_filter have the same shape.
