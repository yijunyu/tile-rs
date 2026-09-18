id: 37
area: lowering
title: trowexpand loses per-row values when the target valid_col is short
opened: 2026-09-14
closed: 2026-09-14

## wanted
`pto.trowsum` over a tile whose valid_col is less than the tile's cols, with
more than one valid row. That is what reduction-axis blocking produces on its
last column block, so batching rows is blocked without it.

## CORRECTED DIAGNOSIS
The reduce is NOT at fault. Dumped at D = 72 with 8 rows, batched:

    trowsum output   8 distinct per-row sums, correct to the last digit
    accumulator      identical, correct
    scalar chain     8 distinct scale factors, all correct

Short valid_col was tested on trowsum directly, both STATIC (v_col=72 on an
8x256 tile) and DYNAMIC (v_row=-1/v_col=-1 with runtime operands 8 and 72).
Both reduce all eight rows correctly -- the only discrepancy in that probe was
a divisor of 1/256 instead of 1/72, an artifact of the hand-edit, and the
applied factors differed per row by exactly sqrt(256/72).

So the collapse is in `pto.trowexpand`, broadcasting back into a tile whose
valid_col is short of its cols: every row then receives row 0's value. Fix the
title accordingly.

`pto.trowexpandmul` as a fused replacement -- `ins(%full, %reduced)` -- makes
it worse (0 of 64 rows at every D, including the ones that pass today), so its
operand order or semantics is not that. Its signature has not been read.

## got (original, about trowsum -- superseded above)
Only the first row of the batch is reduced; every other row receives row 0's
result. The discriminator is not alignment, it is fullness:

    D = 256, 512, 1024   (multiples of BLOCK_COLS=256)   64/64 rows correct
    D = 72               (32-byte aligned, NOT a multiple)  1/8 correct
    D = 67, 68           (unaligned)                        1/8 correct

D = 72 is the case that matters: `72 * 4 = 288` is 32-byte aligned, so this is
not the byte-alignment rule that governs TLOAD and TSTORE. What distinguishes
the working cases is valid_col == cols.

Same collapse signature as the col-major reduction destination fixed in #036,
but a different cause -- that one is fixed and the static multi-row kernel is
now 8/8, while this appears only when the valid extent is short.

## workaround
BLOCK_ROWS stays at 1, so every column block is reduced one row at a time and
valid_col never matters. That costs the amortisation the batching was for: the
per-row scalar chain still runs once per row, and the kernel is 17x slower
than the hand-written one with 48 cores.

What the fix needs:
  * Confirm whether the dynamic valid path is at fault or the static one:
    these tiles carry `v_row=-1, v_col=-1` with runtime operands, so a static
    tile of the same shape with a short valid_col would separate "trowsum
    cannot do short rows" from "trowsum cannot do RUNTIME short rows".
  * TRowReduceOps.hpp is the place to read -- FillTmp and the BinInstrByMode
    paths take srcRptPerRow and validRow/validCol, and the row stride comes
    from the tile's declared cols. A 1L/flat path that assumes contiguity
    across rows would produce exactly this.
  * If PTO genuinely cannot, the emitter's way out is to pad the reduction:
    give the last column block a full-width tile and make the surplus lanes
    the reduction's identity. Pre-zeroing does not work (TLOAD overwrites it,
    see NOTES 241), so that means a separate zeroed tile and a copy of only
    the valid lanes, which costs an op per tail block.
