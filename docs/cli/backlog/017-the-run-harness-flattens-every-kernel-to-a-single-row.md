id: 17
area: run
title: the run harness flattens every kernel to a single row, so row indexing is never tested
opened: 2026-09-03
closed: 2026-09-03

## wanted
A shape sweep: every op at twelve row shapes, to find the bugs that only appear at widths
and row counts the corpus never had. It found two real ones (a power-of-two-only tree fold,
and elementwise kernels that cannot cover a row wider than a threadgroup).

## got
Half the sweep was not measuring what it said. `tile.rs` builds the shape as

    run::Shape::Rows      { rows: 1, cols: kernel.numel() }
    run::Shape::RowReduce { rows: 1, cols: kernel.numel() }
    run::Shape::Rows2     { rows: 1, cols: kernel.numel() }
    run::Shape::Matvec    { rows: 4, cols: kernel.numel() }

`rows` is a literal 1 for three of the four shapes and a literal 4 for the last. The MLIR's
own row count is never read, so a kernel written `%r = 3, %c = 129` is driven as ONE row of
387 — and `softmax_3x129` in the sweep behaved exactly like `softmax_1x387`, which is how
it was noticed.

Consequences, in order of how much they matter:

* **Row indexing is never exercised.** Every emitted kernel computes
  `base = row * num_elements` from its threadgroup position. With one threadgroup that term
  is always 0. A kernel that indexed its row wrongly would pass every test in this repo.
* **The reference agrees with the kernel about the wrong problem.** `reference_output` is
  given the same flattened shape, so both sides compute one 387-wide softmax and agree. The
  three-way comparison cannot see this: it is not a disagreement, it is two right answers
  to a question the MLIR did not ask.
* **`Matvec`'s `rows: 4` is a magic number** with no relation to the source.

## why filed and not fixed
Fixing the shape is small — `matmul_from_mlir` already reads its three dimensions the same
way, and the load intrinsic carries `(rows, cols)` in plain sight. What is NOT small is
what it uncovers: once `rows > 1` is really dispatched, `num_elements` has to mean the row
WIDTH rather than the total, and every emitter that reads it as a total is then wrong.
That is a change to the harness contract and to fifteen emitters at once, and doing it in
the same edit as two unrelated kernel fixes is how a change becomes unreviewable.

It is filed with the evidence in hand rather than half-done, which is what this backlog is
for.

## workaround
Read a passing `-r` result as evidence about a SINGLE row of `rows * cols` elements. It
says nothing about how the kernel indexes rows, because nothing here has ever given it more
than one.

## fixed (2026-09-03), on its own
The issue said the fix was small and what it uncovered was not, and that doing it in the
same edit as two unrelated kernel fixes is how a change becomes unreviewable. It was done
next, alone.

`Shape::rows_cols_from_mlir` reads `(rows, cols)` from the first `__tile_load_*` the way
`matmul_from_mlir` already read its three dimensions. The CLI uses it for `Rows`,
`Rows2`, `RowReduce` and `Matvec`, and falls back to the old flattening when the source
states no constant dimensions -- still correct for one row, and honest about not knowing.

Two things had to move with it:

* **`num_elements` is the row WIDTH**, not the total. Every emitted kernel uses it both as
  the row stride (`base = row * num_elements`) and as the per-row bound, so the total was
  wrong in two places at once. It was only ever right because `rows` was hardcoded to 1.
* **The SPIR-V elementwise path became row-aware.** It had just been given a flat
  grid-stride loop to fix the wide-row bug; that is correct for one row and wrong for
  several. It is one workgroup per row striding within it now, which fixes both.

`matmul_from_mlir` needed nothing -- it had been reading its dimensions from the source all
along, which is why matmul was the one shape this issue did not affect.

## how the fix is known to have worked
Agreement could not show it. Before the change the KERNEL was flat too -- one threadgroup
over one row of `rows * cols` -- so kernel and reference agreed about the wrong problem,
and they agree about the right one now. The verdict reads the same either way.

What distinguishes them is that the two shapes are DIFFERENT computations, and the test
says so: a softmax over four rows of 256 is not a softmax over one row of 1024, and
`reference_output` must now return different numbers for the two. If it did not, `rows`
would not be reaching the reference.

The full 192-kernel shape sweep is 191/192 on both backends with the row counts real. The
one remaining is `log` at 4096 wide, which is one libm against another and not a defect.
