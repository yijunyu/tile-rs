id: 9
area: optimize
title: the descriptor-width and gather-throughput costs cannot be checked against a real kernel
opened: 2026-09-02

## wanted
Have `tile` tell me, before running anything, that cannbench's Transpose tile
shape was going to move memory at a fraction of the achievable rate -- and why.
Two costs decided the whole operator and neither is a resource violation:

  * a tile whose innermost contiguous run is 4 bytes writes one DMA descriptor
    per row and moved memory at 14 GB/s against a vendor kernel's 386;
  * a permute lowered to `Gather` runs at ~1.07 elements/cycle/core, which caps
    the operator below the DMA it is feeding.

## got
`tile submissions/transpose/op_kernel/transpose_kernel.cpp -i` identifies the
form and the dtypes, then:

    kernel: <unnamed>  [0 ops]
    tile plan: not computable (no declared extent)
    hazards: 0 edges, 0 barrier points, 0 unsynchronised
    bounds (n/a): UB ok; repeat ok; DMA stride ok;

Exit 0 -- so this is not a refusal that exit 3 would cluster. Every check
answers "ok" because there is nothing to check against, which is the worst of
the three possible answers: a planner reading this concludes the tiling is fine.
`bounds (n/a)` is honest, but `UB ok` next to it is not.

## workaround
Modelled both costs in `HardwareParams` so they are at least written down once
rather than rediscovered per operator (commit e6c256c, 70 tests):

  * `C11 check_descriptor_run(run_bytes)` -- refuses a run under
    `min_efficient_descriptor_bytes` (128 on DAV_2201, the width measured
    healthy) and names the remedy, which is a different tile SHAPE that folds an
    adjacent axis into the run. Widening the other axes provably cannot fix it.
  * `C12 vector_transpose_block(dtype_bytes)` -- the block
    `scatter_vnchwconv` moves per repeat, so a permute can pick the hardware
    transpose over a gather.

Then applied both by hand in `bench/gen_transpose.py`: two new tile shapes for a
narrow innermost output axis, and the tile permute moved off Gather. Transpose
63.45 @ 0.45x -> 69.58 @ 0.72x, 20/20 throughout.

Things the eventual fix will need, that are not obvious:

  * b16 transposes a 16x16 block per repeat but b32 only 16x8, and a b32
    destination row of 16 elements spans two 32-byte blocks, so it needs TWO of
    the sixteen addresses. Assuming b32 matched b16 failed exactly the b32 tiles.
    CANN's own `cumsum_common_impl.h` states the geometry; guessing did not.
  * the `LocalTensor` overload of `TransDataTo5HD` copies all 32 addresses into
    four stack arrays on EVERY call -- ~7000 cycles, enough to make the hardware
    path 7x SLOWER than the Gather it replaced. The raw `uint64_t[16]` overload
    does not. A cost model that stops at "which instruction" picks right and
    still loses; the overload is part of the cost.
  * the toolkit include dirs are SYMLINKS: `grep -r` and `find` without `-L`
    report that none of these instructions exist at all.

What would have made the tool able to do it: the reader #004 asked for, plus a
tile plan carrying the innermost RUN LENGTH per operand -- not just extents and
dtypes -- since that is the quantity both costs are a function of. Failing a
reader, `-i` should print `unknown`, not `ok`, for checks it has no input for.

## the part that needed no hardware, done (2026-09-03)

The issue asks for two things. The costs themselves need the AscendC reader of #004 and a
tile plan carrying the innermost RUN LENGTH per operand; neither is reachable from here.

The last line asks for something else entirely:

> Failing a reader, `-i` should print `unknown`, not `ok`, for checks it has no input for.

That needed no hardware and no reader, and it is done. `Bounds` has a third variant:

    bounds (n/a): UNKNOWN — nothing to check
      the source declares no extent, so there is no tile size to check a capacity against
      unknown is not ok: a check with no input has not passed, and a tiling
      that was never examined must not read as one that was

It used to print `bounds (n/a): UB ok; repeat ok; DMA stride ok;` directly under
`tile plan: not computable (no declared extent)` — three checks answering ok against a
DEFAULT tile size, one line below the admission that there was no tile.

Two tests, deliberately opposed: one that no declared extent gives `UNKNOWN` and never
`UB ok`, and one that a source WITH an extent is still `Checked`. The second exists because
a change that made everything answer "unknown" would satisfy the first and destroy the
feature.

The issue stays OPEN for the costs, which are what it is really about.
