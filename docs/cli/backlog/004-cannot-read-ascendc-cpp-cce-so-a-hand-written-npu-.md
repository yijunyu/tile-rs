id: 4
area: lowering
title: cannot READ AscendC (cpp/.cce), so a hand-written NPU kernel cannot be profiled or transformed
opened: 2026-09-01
closed: 2026-09-01

## wanted
Profile and then transform an existing hand-written AscendC cube kernel --
cannbench's tile_mm_core.h -- to add head-aware addressing, so an attention
operator can read Q/K/V still in [B, S, N, D] instead of materialising three
transposes first. NOTES 106 measured those transposes at 4.0 ms of the 5.3 ms
spent preparing Q/K/V, and 73% of MHA's decode cases.

    tile submissions/_mm/op_kernel/tile_mm_core.h -i
    tile .../tile_mm_core.cce -i

## got
The .h is not identified at all:

    tile: ...: cannot tell what kind of kernel .h is, and no content marker
    matched; pass -f <form>

Renamed .cce it IS sniffed, but nothing is read from it:

    form: cpp (sniffed: extension ∩ content)
    kernel: <unnamed>  [0 ops]
    tile plan: not computable (no declared extent)

`--list-forms` confirms why: cpp is `read: no`, and its `lifts-from` column is
empty. Fifteen targets can be WRITTEN and only tile/mlir/linalg can be read, so
any kernel that already exists in a target language is outside the tool. That is
the common case on a machine where the kernels were written before tile-rs
reached that target.

## workaround
Wrote the transformation by hand in the operator's own generator, and used
`tile` only for the part it could do.

Concretely, in cannbench-tilers:

  * `bench/gen_mm.py` gained a SECOND translation unit per dtype
    (`tile_mm_h{f16,f32}.cpp`) whose `__global__` calls the existing
    `TileMM::InitNd`. It has to be a separate TU: a second
    `__global__ __aicore__` kernel beside another faults the first one too.
  * a host entry `tile_mm_run_heads(...)` carrying heads / aInner / aLd /
    bInner / bLd / aHeads / bHeads.
  * `bench/gen_mha.py` passes `heads = N` and skips all three
    `[B,S,N,D] -> [B,N,S,D]` transposes. Q@K reads both operands head-strided;
    P@V only V, because P is the contiguous softmax output.

Measured on 910B2: MHA 55.55 -> 60.26, speedup 0.18x -> 0.38x, 20/20.

Things the eventual fix will need, that are not obvious:

  * Nd2Nz carries the source row stride in a UINT16. Head-strided reads hand it
    N*D rather than K, so the bound must be checked against the ROW stride, not
    the reduction extent. Past it the transfer wraps silently and returns
    garbage rather than an error.
  * ONE `__global__ __aicore__` kernel per translation unit on this toolchain.
  * Adding a defaulted parameter to a shared host function changes its C++
    mangled name and breaks every other operator that declares it -- gqa and
    mla both declare `mha_attend` -- so a signature change is not local.

What would have made the tool able to do it: a READER for cpp/.cce, even a
partial one that recovers the tile plan and the operand addressing. The
transformation itself is an addressing change -- base offset and row stride per
batch index -- which is exactly the kind of thing the tile plan already models
for the forms it can read.

## resolved by the form rule (2026-09-01)

The owner settled what a form is:

> A form should be either a **source** form for the downstream toolchain to consume, or an
> **MLIR** form for further compiler passes to process. If it is in a **binary** form, it
> would be directly loaded and executed by the harness.

That is a rule about the CONSUMER, not about how far down a representation sits, and it is
what `level` could not say. It is now `forms::Role` on every row of the table, with the
obligations derived from it (`owes_a_reader`, `directly_executable`) and enforced by tests.

Same resolution as #003, and the PICO session's evidence explains why this one is the
harder half. `cpp` (AscendC) is a **source** form, so no reader is owed. And a reader
could not be got cheaply even if it were wanted: `pico_lift` works only because PICO's
target language IS a closed 91-symbol intrinsic alphabet, invertible by table. AscendC is
a C++ program, so there is no alphabet to invert -- reading it means a C++ frontend.

The underlying want -- profile and transform a hand-written NPU kernel -- is real and
unmet. It is now correctly filed as "tile-rs has no AscendC frontend", which is a product
decision with a visible cost, rather than as a missing lifter that someone might have
assumed was nearly done.
