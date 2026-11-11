id: 14
area: lowering
title: bang's row reductions write a whole row and read past a one-element buffer
opened: 2026-09-03
closed: 2026-09-03

## wanted
To read what freezing `reduce_sum` as a golden had actually frozen, across the nine
backends that lower it, before trusting the files as a regression baseline.

## got
Seven are fine and two of those are fine BY CONSTRUCTION, which is worth saying: `linalg`
emits `linalg.reduce ... dimensions = [1]` and `nki` emits
`nisa.tensor_reduce(np.add, t0, axis=(1,))`. They name the reduction instead of hand-rolling
it, so the partial-reduction bug cannot be expressed in them.

`bang` has two defects, both visible in the emitted text.

    __nram__ float nram_1[1];
    int per_task = TILE_SIZE_RSUM / taskDim;
    int offset   = taskId * per_task;
    __memcpy(nram_0, p0 + offset, per_task * sizeof(float), GDRAM2NRAM);
    nram_1[0] = 0.0f;
    for (int _i = 0; _i < per_task; _i++) nram_1[0] += nram_0[_i];
    __memcpy(p1 + offset, nram_1, per_task * sizeof(float), NRAM2GDRAM);

**The write-back reads past the buffer.** `nram_1` holds one float. The copy takes
`per_task * sizeof(float)` FROM it -- 1024 bytes out of a 4-byte array at taskDim=1. Same
in `reduce_max`. `absmax` declares `nram_2[256]` and is not over-reading; `softmax` writes
1024 from `nram_1[1024]` and is correct, because its output really is a row wide.

**The output count is wrong for all three reductions.** A row reduction produces ONE value
per row. These write `per_task` values at `p1 + offset`, so the destination gets a row's
worth where one number was due -- the first of them the sum, the rest whatever followed
`nram_1` in NRAM.

There is a third thing that may or may not be a defect: each task sums only its own
`per_task` slice and nothing combines across tasks, so at taskDim > 1 each writes a partial.
Whether that is wrong depends on how the caller dispatches, which I cannot see from here.

## why filed and not fixed
No Cambricon hardware here, and #001 says the harness reaches Metal only. Writing an MLU
cross-task reduction I cannot run is the plausible-looking work this repo has twice
rejected -- once for the CUDA block reduce, once for the f16 tolerance. The right fix needs
someone who can run it.

The over-read is the exception in principle: `per_task * sizeof(float)` should be
`sizeof(float)` for a one-element result, and that is a one-word change with no numerics to
get wrong. It is still not made here, because the output-count question above has to be
settled in the same edit -- fixing the copy size while leaving the destination stride wrong
would produce a kernel that is differently wrong and looks fixed.

## how it was found, which is the reusable part
By reading a golden file that had just been created. The value of freezing emitted text is
not only that a later diff fails -- it is that the text becomes something a person can
read. Nine kernels for backends nobody here can run were reviewed in about a minute
because they were sitting in the tree as files.

## workaround
Do not use `bang`'s reduce_sum, reduce_max or absmax. The other six lowerings of reduce_sum
are sound on inspection, two of them by construction.

## fixed (2026-09-03)
Not by writing an MLU cross-task reduction — that judgement in "why filed and not fixed"
stands, and nothing here can run one. By removing the need for one.

**A row that fits in NRAM does not need splitting across tasks.** When the kernel contains a
row reduction the prologue emits `per_task = TILE_SIZE_X; offset = 0;` instead of the
elementwise split, so every task reduces the WHOLE row, and the write-back becomes

    if (taskId == 0) __memcpy(p1 + offset, nram_1, sizeof(float), NRAM2GDRAM);

That is correct at any taskDim, needs no cross-task communication, and settles all three
defects in one edit — the over-read, the output count, and the uncombined partials — which
is what this issue said had to happen together.

It duplicates the reduction across tasks. That is a performance cost, not a correctness
one, and a real cross-task reduction is still work for someone who can run it. The emitted
source says so in a comment rather than leaving the next reader to infer it.

`absmax` went with them. It broadcast its result back across the whole tile, so its output
had a different SHAPE from every other lowering of the same intrinsic — and from
`RefOp::Absmax`, which this repo folds to one value per row exactly like `reduce_max`.

Pinned by `bang_row_reductions_write_one_value_and_read_the_whole_row`, which checks the
write-back LINE rather than the file: the load still reads `per_task * sizeof(float)`
legitimately, and the first version of the test failed on that.
