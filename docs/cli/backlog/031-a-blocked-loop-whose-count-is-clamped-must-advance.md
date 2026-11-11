id: 31
area: emit-structure
title: A blocked loop whose count is clamped must advance by the CLAMPED count, and nothing checks it
opened: 2026-09-11

## wanted
An emitted blocked loop that is correct when the block is short. The shape is
universal in tiling: take `kRow` rows, clamp the count so the block does not
cross a boundary, process `rc` rows, advance.

## got
The emitter (and the hand-written kernel it mirrors) writes the advance from
the NOMINAL block size while the body honours the CLAMPED one:

    for (rb = r0; rb < r1; rb += kRow) {        // nominal
        rc = min(r1 - rb, kRow);
        if ((rb + rc - 1) / M != rb / M)
            rc = (rb/M + 1) * M - rb;           // clamped at a batch boundary
        ... process rc rows ...
    }

Every row between the clamp and the next block is silently never computed. No
fault, no diagnostic -- just missing output.

It needs `M % kRow != 0` AND a block that spans a boundary, so the obvious test
shapes all miss it: batch 1 passes, M divisible by kRow passes, and M small
enough that each core gets one row passes. Shipped in cannbench-tilers commit
64154ee and found only because a downstream operator fell to 2/20.

## workaround
`rb += rc` (commit d9a7feb), plus seven boundary-straddling shapes added to the
regression bench -- B=4 M=17, B=8 M=9, B=3 M=11, B=6 M=13, B=2 M=9, B=7 M=5,
B=4 M=100 against N and K that are not multiples of anything.

What a fix needs: the block-loop is a codegen TEMPLATE, not something each
kernel should re-derive. An emitter that owns "iterate in blocks of N with a
clamp predicate" cannot make this mistake; every hand-written instance can, and
this one did. The generality matrix should also carry a shape where the block
size does not divide the extent, because none of its current shapes expose it.
