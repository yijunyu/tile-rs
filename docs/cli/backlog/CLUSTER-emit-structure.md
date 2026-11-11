# The `emit-structure` cluster, and what it forced

The split that created this area was executed in [`CLUSTER-lowering.md`](CLUSTER-lowering.md)
on 2026-09-12, and the stop rule requires each area at the limit to carry its own account.
This is that account: the three open entries read together, plus the one already closed and
the two measured defects that were never filed, because they are part of the same reading.

## The issues

* **#027** — no host dispatch is emitted, so "the operator must reach the device" cannot be
  a codegen invariant. *tile-rs emits kernels only; the torch dispatch — tile plan, output
  allocation, launch — is written by hand every time, and the launch obligation is checked
  by a regex in another repository (`bench/lint_launch.py`, 31 real instances found across
  three widenings). One omitted launch scores the operator ZERO in full: Transpose lost
  63.31 and StridedSlice 65.64 to one such return each.*
* **#031** — a blocked loop whose count is clamped must advance by the CLAMPED count, and
  nothing checks it. *Shipped in cannbench-tilers 64154ee; every row between the clamp and
  the next block silently never computed; found only when a downstream operator fell to
  2/20. Fixed by hand (`rb += rc`, d9a7feb) plus seven boundary-straddling shapes.*
* **#033** — a hoisted per-window table silently absorbs a loop-variant factor. *Compiles
  clean, runs clean, wrong by exactly the box size whenever the dropped factor ≠ 1 — and 4
  of 20 cases pass precisely because there it is 1, a coincidence indistinguishable from a
  dozen unrelated shape predicates. Cost three misdirected builds.*
* **#035** — the reduction axis is not blocked, so a runtime column extent has no lowering.
  *Closed 2026-09-13 the same day it was filed: runtime ROWS is supported, runtime COLUMNS
  is REFUSED with a diagnostic, because emitting a fixed-width kernel would be the same
  silent-wrong the rest of this area is about. Kept in the reading as the member that was
  caught in time — refusal is what the other two shipped instead of.*

## One design problem

**The emitter owns no structure larger than one instruction.** Every template a kernel
needs — iterate in blocks, reach the device, hoist the invariant — is re-derived by hand,
per kernel, and nothing in the tree checks the re-derivation:

    #027   the emitter does not own the DISPATCH        -> checked by a lint, elsewhere
    #031   the emitter does not own the BLOCK LOOP      -> checked by nothing
    #033   the emitter does not own the HOIST+RESIDUAL  -> checked by nothing

Same sentence three times. That is the area. Each entry looks like a different bug — a
missing host path, an off-by-one advance, a dropped algebraic factor — and the reason they
are one problem is that in each case a human wrote an instance of a structure the tool
should have emitted, the instance disagreed with the structure's contract at one corner,
and there was no owner to disagree with.

This is the complement of `lowering`'s sentence. That area's rule is a contract on the way
IN — *a backend must not accept a call it does not implement the contract of*. This one is
a contract on the way OUT — **a structure that every kernel needs should be emitted once,
not re-derived per kernel.** Both are the same failure in two directions: nothing owns the
thing, so every user of it re-implements it and every disagreement is silent.

## Why each was invisible, and why the tests agreed with the bug

All three shipped with passing tests, and the passing set is not luck — each test suite's
shapes satisfy exactly the bug's precondition:

* #031 needs `M % kRow != 0` AND a block straddling a boundary; batch 1 passes, `M`
  divisible by `kRow` passes, small `M` (one row per core) passes.
* #033 passes exactly where the dropped factor is 1 (`D == od and H == oh`) — 4 of 20,
  and those four fit a wrong gather-indexing hypothesis just as well.
* #027's omission is invisible to function tests by construction: the operator answers
  correctly without launching. Only the grader's anti-cheat — external, empirical — saw it.

So the generality matrix is not a formality here. Each template needs the shape its own
failure requires — a block that does not divide the extent, a factor that is not 1, a
deferred or multi-output dispatch — or the matrix will keep certifying the bug's corner.

## What the area should have been

A small, enumerated set of structural templates the emitter OWNS, which kernels
instantiate instead of rewriting:

1. the **blocked loop** — take N, clamp at a boundary, process, advance by the CLAMPED
   count (#031; shipped wrong, hand-fix exists and is the reference implementation)
2. the **host dispatch** — the operator reaches the device (#027; the workaround's four
   distinctions — pass-through borrow, allocator discharges, deferred linear token,
   multi-output — are the specification a witness type would enforce structurally)
3. the **hoist with residual** — per-factor invariance analysis; the residual is emitted
   at the use site, or the hoist is refused (#033)
4. the **batched pass** — measured, never filed: `prep_split`/`prep_weave` did one row per
   iteration with three barriers around it; batching gave 6.9x and 10.9x
5. the **double-buffered stage** — measured, never filed: `tile_mm`'s inner loop is three
   `PipeBarrier<PIPE_ALL>` per k-step; bounded at ~2x and still unfixed

Items 4 and 5 are recorded here rather than filed as #041/#042 because that is exactly what
the stop rule asks for: measured members of one design problem, read together with it, not
new hats on it. Filing them would not change the design; finishing it would close them.

**The single sentence: a structure that every kernel needs should be emitted once, not
re-derived per kernel.**

## Order

1. The **blocked loop** as an emitter-owned template. Smallest, and the hand-fix plus its
   seven shapes are already in the tree as the correct instance — extraction, not design.
2. The **hoist**: per-factor invariance check, residual emitted at the use site, refusal as
   the fallback. Self-contained, and the failing/passing split gives its test shape for free.
3. The **host dispatch** with a launch witness — the biggest, and the only one that changes
   `tile_spec`; the workaround's four cases are what the witness must distinguish, learned
   by hand across three rounds of false negatives. Started only after 1 and 2 have made the
   template pattern ordinary in the emitter.
4. The **generality matrix** gains one shape per template's escape condition (non-dividing
   block, factor ≠ 1, deferred and multi-output dispatch), so each template is checked at
   the corner where re-derivation used to win.

Closing an entry means the class is template-owned, not that the instance was patched:
#031 is fixed by hand already; it closes when the advance comes from the template.
