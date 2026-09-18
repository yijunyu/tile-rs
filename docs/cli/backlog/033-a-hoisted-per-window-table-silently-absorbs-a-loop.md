id: 33
area: emit-structure
title: a hoisted per-window table silently absorbs a loop-variant factor
opened: 2026-09-12

## wanted
Hoisting a reciprocal table out of a loop nest is a legitimate transform: the
W-window sizes are loop-invariant, so `1/(b_k - a_k)` belongs in a table built
once. I wanted the transform to be refused, or the residual factor to be
emitted, when the expression being tabulated is NOT wholly invariant.

The scalar source was

    out[k] = acc * boxRcp / (b - a)

where `1/(b-a)` is invariant across rows and `boxRcp = 1/((d1-d0)*hn)` is a
property of the row. Tabulating the reciprocal captures one factor of a
two-factor product and drops the other. Nothing in the pipeline noticed that
the tabulated expression had a free variable the table cannot index.

## got
A kernel that compiles clean, runs clean, and is wrong by exactly the box size
on every case where the box is not 1. It passes on every case with D == od and
H == oh, because there boxRcp == 1 and the dropped factor is the identity.

That pass/fail split is the dangerous part: 4 of 20 cases passed, and the
passing set is characterised by "the dropped factor happened to be 1", which
is indistinguishable from a dozen unrelated shape predicates. I spent three
builds on a gather-indexing hypothesis (`W % ow == 0`) that fits the same 4
cases and had nothing to do with the bug.

## workaround
Emit the residual explicitly: `Mul(out, seg, rcpv, n)` for the tabulated part,
then a second `Muls(out, out, boxRcp, n)` for the loop-variant part. One extra
vector instruction per row, unmeasurable against the reduction it follows.

What the fix will need:
  * The invariance check is per-FACTOR, not per-expression. Splitting the
    product and asking which factors are invariant in the hoisted loop is the
    whole analysis; it is cheap and it is not being done.
  * When a table is built from a loop-invariant subexpression, the residual
    must be emitted at the use site, not discarded. Refusing the hoist is an
    acceptable fallback; silently tabulating a subset is not.
  * A test shape where the dropped factor is NOT 1 is required. The task set's
    own cases include four where it is, and those four were enough to make the
    kernel look plausible for three iterations.

Related: this is the same shape as #030 and #032 -- a transform checks one
property of a value and treats it as the whole contract.
