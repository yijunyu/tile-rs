id: 13
area: run
title: a fixed tolerance cannot judge an operation whose own precision is not fixed
opened: 2026-09-03
closed: 2026-09-03

## wanted
`matmul` and `matvec` in the list of ops the harness has checked, alongside the seventeen
that agree.

## got
Neither can pass, and neither can this tool's own CPU reference.

`ABS_TOL` is 1e-5 and `REL_TOL` 1e-4, applied to every operation. On a K=64 f32 matmul:

    kernel vs torch   max abs 1.14e-5  max rel 2.72e-4
    kernel vs ours    max abs 1.14e-5  max rel 3.55e-4
    ours   vs torch   max abs 9.54e-6  max rel 3.29e-4

and on a 256-term matvec:

    kernel vs torch   max abs 1.53e-5  max rel 1.60e-7
    ours   vs torch   max abs 3.05e-5  max rel 2.14e-7

The `ours vs torch` row is the point. Our reference computes the same product in a
different summation order and lands outside the gate. So the gate is not measuring the
kernel at all for these ops: no lowering, however correct, reaches it.

Everything elementwise passes easily, because an elementwise op's error does not grow with
anything. A K-term reduction's does — roughly with sqrt(K) in RMS, worse under cancellation
— so one constant cannot serve both.

## what was NOT done, twice
Widening the number. That was tried for f16 (#010), produced a threshold fitted to the one
sample in front of me, and was reverted; the same reasoning applies here and the same
answer. A tolerance chosen so that today's kernel passes measures nothing tomorrow.

## what was done instead
The verdict stopped calling this a fault. It used to read "they are separate faults", which
is false here: nothing is at fault, the three sources merely disagree by the amount f32
summation order disagrees. It now hands the reader the discriminator that actually works:

    all three disagree, including the kernel against the reference — compare the three
    magnitudes above: one much larger than the others is a defect, three of a size is the
    operation's own precision against this tolerance

That distinction is not theoretical. It separated 17.8-against-6.7e-7 (a real overrun),
4.96-against-2.86e-6 (a real partial reduction) and 1.74e2-against-3.05e-5 (a real
scalar-binding bug, caught by this very sentence) from 1.14e-5-against-9.54e-6, which is
arithmetic.

## the fix someone with the numerics call should choose between
  (a) A tolerance that scales with the reduction length the op actually performs — the
      harness knows K from the shape, so `abs_tol * sqrt(K)` is available and defensible.
  (b) Compare against a reference computed in the SAME order as the kernel, making
      summation order not a variable. Costly and exact.
  (c) Report reductions with their own band and stop calling the elementwise number a
      pass/fail for them.

Each is a decision about what "agrees" should mean for a reduction, which is a numerics
call rather than a bug fix.

## workaround
Read the three magnitudes rather than the verdict's tolerance verdict, which is what the
line now tells you to do.

## how it was closed (2026-09-03)
Option (a), with the part that made it a "numerics call" removed: the tolerance is not
CHOSEN, it is computed from `f32::EPSILON` and the terms the operation actually sums.

    error_budget(op, shape, dtype) -> Option<Budget>

For an op that sums, over the same inputs the harness generates:

    magnitude  = max over outputs of the sum of the ABSOLUTE values of its terms
    guaranteed = (u_dtype + gamma_K) * magnitude,  gamma_K = Ku/(1 - Ku),  u = EPS/2
    typical    = (u_dtype + sqrt(K) * u) * magnitude

`guaranteed` is Wilkinson's bound: NO summation order can exceed it. That is the property
a fixed constant never had and the reason this is not a fitted number -- the objection in
"what was NOT done, twice" was to widening a threshold until today's kernel passed, and a
bound that a correct implementation provably cannot cross is a different object.

`typical` is not a gate. A correct kernel may exceed it. It is printed because "admissible
but 8x above typical" is what a needlessly bad summation order looks like, and the report
says so rather than staying silent inside a pass.

Measured after the change, every one a verdict that used to be unreachable:

    matmul  f32  msl   1.14e-5  against a budget of 3.12e-4   all three agree
    matmul  f32  spirv 1.14e-5  against a budget of 3.12e-4   all three agree
    matvec  f32  msl   1.53e-5  against a budget of 3.54e-4   all three agree
    reduce_sum K=256   2.86e-6  against a budget of 4.15e-3   all three agree
    reduce_sum K=1024  1.14e-5  against a budget of 6.65e-2   all three agree

The last one had been failing the old 1e-5 by 14%, which is what judging a 1024-term sum
by a constant chosen for elementwise arithmetic produces.

## what the fix got wrong first, and how that was caught
The first version used f32's unit roundoff for every kernel. The msl f16 matmul errs by
2.18e-2 -- correctly, because `half * half` rounds each product before the f32 accumulator
sees it -- and the newly TIGHTER gate promoted its verdict from a vague "all three
disagree" to a confident "the KERNEL disagrees with torch, look at the lowering".

A tighter tolerance that points confidently at the wrong place is worse than the loose one
it replaced. The budget takes the buffer dtype now (`u = 2^-11` for f16), the f16 matmul
reads "all three agree" against 4.02e-2, and a kernel that accumulated in HALF would still
be 10x outside it -- which is a lowering decision worth surfacing, not looseness to absorb.
This is the second time in this file a number right for one dtype was applied to both;
#010 was the first.

## and it removed two more constants
`ABS_TOL`/`REL_TOL` were not the only ones. `RunReport.tolerance` was the literal `1e-5`,
and the exit code was decided by `worst > 1e-4` -- an order looser, so a kernel could be
reported outside tolerance and still exit 0. All three read the one budget now.
