id: 36
area: lowering
title: rms_norm's PTO lowering is only correct for a single row
opened: 2026-09-13
closed: 2026-09-14

## wanted
`__tile_rms_norm_f32(%dst, %src, %eps, %rows, %cols)` with rows > 1. The CANN
Bench RmsNorm cases are all multi-row -- 32x128x1024 is 4096 rows -- so a
one-row-only lowering cannot serve the operator it is named for.

## got
Every row receives ROW 0's scale factor. Exactly, not approximately:

    row  applied    correct
      0  0.532340   0.532340
      1  0.532340   0.459091
      7  0.532340   0.516909

Measured on a static 8-row kernel straight out of the emitter -- no blocking,
no batching, no runtime extents. 1 of 8 rows correct.

Nothing announces this. The op is listed as lowered, ptoas compiles it, the
kernel runs, and it is right whenever rows == 1, which is what every test and
golden file in this tree happens to use.

## workaround
None. What the fix needs, all measured:

  * `pto.trowsum` is NOT at fault. Dumping its output for an 8-row batch shows
    all eight per-row sums, correct and distinct (903.36, 1214.62, 1133.64...
    against a host-computed sum of squares).
  * The fault is the conversion after it. `tmuls(col-major 8x1, 1.0f) ->
    row-major reduced tile` collapses every slot to element 0. At one row that
    conversion is trivially a no-op, which is why it has never been noticed.
  * It is a STRIDED SCATTER: the col-major 8x1 holds its values contiguously,
    the row-major reduced tile (`rows=8, cols=8, v_row=8, v_col=1`) needs them
    at stride 8.
  * A row-major 8x1 destination would make the shapes match, and is ILLEGAL:
    "expects result row-major none_box tile row byte size (cols * sizeof(dtype))
    to be 32-byte aligned, but got 4 bytes".
  * `pto.tmov` refuses the shape change outright: "expects A2/A3 non-mat tmov
    to use matching src/dst shapes".

So the open question is narrow and concrete: which PTO op moves a col-major
Rx1 reduction result into a row-major RxN tile at stride N. `pto.ttrans` is
the obvious candidate and is untried.

Found while batching rows to amortise the per-row scalar chain (the emitter is
otherwise batch-ready; BLOCK_ROWS is the knob). Batching is blocked on this,
and so is any multi-row rms_norm at all.
