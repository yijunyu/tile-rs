id: 10
area: run
title: f16 kernels are compared against inputs they were never given
closed: 2026-09-03
opened: 2026-09-02

## wanted
`__tile_matmul_f16` verified on the GPU like the eleven unary ops, so the count of ops
this tool has actually checked matches the count it has references for.

## got
It cannot pass, and nor can any other f16 kernel.

`input_values_for` carries this comment, which states the rule exactly:

> a reference fed different data from the kernel is a comparison of two unrelated
> numbers, and it fails in a way that looks like a numerical bug

and `metal.swift` breaks it three lines of Swift later:

    if dtype == "half" {
        let p = buf.contents().bindMemory(to: Float16.self, capacity: v.count)
        for i in 0..<v.count { p[i] = Float16(v[i]) }
    }

The kernel is handed the inputs rounded to half. Both references -- this tool's own and
torch -- compute on the f32 originals. The generator was written once and shared "by
construction rather than by two implementations agreeing", and then the harness quantized
one side on the way in.

Measured on a K=64 f16 matmul, after fixing its accumulator (below):

    kernel vs torch   max abs 1.28e-2  max rel 3.48e-2  rmse 4.38e-3
    ours   vs torch   max abs 7.63e-6  max rel 5.34e-6  rmse 1.50e-6

`ours vs torch` shows the reference is sound. The kernel's gap is dominated by inputs it
never saw. The verdict reads "the KERNEL disagrees with torch -- look at the lowering",
which sends the reader to audit a lowering that may be right.

## what was fixed, and what was not
Fixed: the kernel accumulated in `half`. Every hardware path for f16 matmul accumulates in
f32 (simdgroup_matrix here, tensor cores elsewhere), and a 64-term sum in half loses the
low bits of every partial. `float acc` took it from 6.46e-2 to 1.28e-2 max abs and
1.63e-2 to 4.38e-3 rmse. That part was a real defect and is measured.

Not fixed: the input mismatch. Two ways to close it, and choosing between them is a
decision about which inputs are canonical for this harness:

  (a) Quantize at generation. When the kernel's dtype is half, round the generated values
      through f16 before anyone uses them, so all three sides compute on identical
      numbers. Costs: the `i * 1e-4` ramp does not survive f16 (spacing near 2.0 is
      ~0.002), and that ramp is the witness that caught the partial-reduction miscompile.
      An f16 reduction would lose it.

  (b) Round the REFERENCES' inputs instead, in both the Rust reference and the torch
      script, leaving the f32 generator alone.

Both need the dtype threaded to about eight input sites plus the torch script and the test
that pins the script and the harness to the same formula. Neither is a bug fix; both
rewrite what the harness considers the true input.

A tolerance is NOT the answer and was tried and reverted: widening it until this kernel
passes is fitting the threshold to the sample, and no threshold makes a comparison against
different numbers mean anything.

## meanwhile
The report says so. When the buffers are half and the kernel is outside tolerance, it now
prints that the harness rounds the kernel's inputs and the references do not, so part of
the gap is not the lowering.

## workaround
Read `ours vs torch` first. If it is tight, the reference is sound, and an f16 kernel's
remaining gap is partly the input rounding.

## fixed (2026-09-03)
The inputs are rounded to the kernel's buffer type before either reference computes, so
all three sides see identical values.

* Rust: `run::quantize_inputs(&mut v, dtype)`, built on `run::f16_round`.
* torch: `.half().float()` on every input tensor in the generated script.
* the harness: unchanged — it already wrote `Float16(v[i])`, which is the same rounding.

Measured on the msl f16 matmul, before and after:

    before   kernel vs torch  max abs 2.18e-2   max rel 2.86e-1
             ours   vs torch  max abs 9.54e-6   max rel 3.29e-4
    after    kernel vs torch  max abs 1.56e-2   max rel 4.81e-4
             ours   vs torch  max abs 0.00e0    max rel 0.00e0

`ours vs torch` at exactly zero is the check on the fix: the two references now agree bit
for bit, which they can only do if they are computing on the same numbers. And the
kernel's max REL error fell by a factor of 600 — that is how much of the old figure was
the harness rounding for it rather than the lowering.

What remains is the kernel's own f16 arithmetic, which is a different thing and is bounded
separately by `error_budget`'s `u_dtype` term (#013). The report says which of the two it
is looking at now, rather than pointing at the lowering for both.

## the comment that made this worse
`RunReport::render` carried, for a day, a comment stating that "the inputs are quantized at
generation time now, so both sides see identical values and one tolerance is right for both
dtypes". `input_values_for` has no dtype parameter and never had one. The sentence asserted
a fix that had never been made, sitting in the one place a reader would check before
trusting an f16 number — and it contradicted, in the same file, the note the same function
printed to the user.

A wrong comment is worse than no comment where it describes a GUARANTEE someone would
otherwise verify.

## how f16_round is trusted
It is hand-written — there is no `f16` in stable Rust and no dependency for it — which is
the kind of code that looks right and is subtly wrong. It was: the subnormal path had an
exponent one too low and halved every f16 subnormal.

Caught by checking against numpy rather than by re-reading it. 598 values over the
harness's own range plus the edges agreed exactly. That sweep needs numpy and cannot live
in the suite, so its equivalent does: every one of the 63488 finite f16 bit patterns is
asserted to be a fixed point of the rounding, decoded independently of the code under test.
The halved-subnormal bug fails 2048 of them.
