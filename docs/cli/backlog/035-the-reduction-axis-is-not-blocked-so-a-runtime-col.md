id: 35
area: emit-structure
title: the reduction axis is not blocked, so a runtime column extent has no lowering
opened: 2026-09-13
closed: 2026-09-13

## wanted
`tile k.mlir -t pto` for a kernel whose column extent is a runtime argument.
The CANN Bench RmsNorm cases run D from 67 to 8192 against one kernel, so a
compile-time width is not a usable contract.

## got
Before this session: `rows=0` tiles and `partition_tensor_view<0x1024xf32>` --
a kernel that compiles, launches and processes nothing. Runtime ROWS is now
supported (a row loop with the extent as its bound). Runtime COLUMNS is now
REFUSED with a diagnostic, because the tile width is a compile-time UB
allocation: a kernel built for one width is wrong for any wider input, and
the emitter cannot see the caller's value to check. Emitting a fixed-width
kernel would be the same silent-wrong the rest of this area is about.

## workaround
None in the emitter. The shape that works is verified end to end against the
real toolchain (ptoas 0.24, cann-9.2.0, 910b) and is checked in as
`crates/tile_cli/testdata/forms/rms_norm_blocked.pto.mlir`:

  * peel the first column block so the accumulator is seeded by a write
    rather than needing a fill -- there is no whole-tile fill op (`pto.tfillpad`
    only re-pads to a BLOCK boundary), and zeroing by multiplying uninitialised
    memory by 0 is not safe against NaN;
  * loop the remaining blocks with `sizes = [%c1, min(TW, cols - cb)]` into a
    `!pto.partition_tensor_view<1x?xf32>`;
  * give the data tiles `pad=1`, which ptoas maps to `PadValue::Zero`, so a
    short tail load does not poison the reduction. The mapping is
    0=Null 1=Zero 2=Max 3=Min, 4 rejected -- and the right pad is the
    reduction's IDENTITY, so a max-reduce wants Min, not Zero;
  * accumulate in ROW-major: `pto.trowsum` must write col-major (verifier),
    `pto.tadd` refuses col-major, and `pto.tmuls` by 1.0 is the layout change
    between them;
  * second pass re-loads and scales, because the value is only known after
    the whole row is reduced;
  * the runtime reciprocal is fine on device: `arith.index_cast` +
    `arith.sitofp` + `arith.divf` lowers to `float v53 = v6 / (float) v4;`.

What the fix needs from mlir_to_pto.rs: a BODY-LEVEL pass. The blocking spans
`__tile_load`, the reduce and `__tile_store`, so it cannot live inside any one
translator -- something has to classify each op as per-block, per-row, or
broadcast-back, which is the dataflow analysis this file does not have. That
is the same missing ownership as #027 and #031: the emitter does not own the
loop nest, so it cannot restructure one.
