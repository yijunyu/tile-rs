id: 32
area: transform
title: A narrowing conversion is checked for width and not for EXPONENT RANGE
opened: 2026-09-11
closed: 2026-09-11

## wanted
A conversion rule that separates the two things a narrowing cast does.
`bfloat16 -> float16` keeps every mantissa bit (8 into 10) and destroys the
exponent (8 bits into 5): bf16 reaches ~1e-38 where half's smallest normal is
~6e-5, so small values flush to zero. The declaration says "narrow to f16"; the
VALUE RANGE change is written down nowhere.

## got
Nothing checks it, so a lowering routes an operand through half whenever the
target's cube wants half, and the loss is invisible until a grader reports it.
In the cannbench tree mla_prolog cast both matmul operands to half and lost 8
of 20 cases on 950pr -- every failure bfloat16, MERE PASSING at 2.9e-06, MARE
pinned at 0.0078 (two bf16 ulp), reported as 小值域 (small-value-range) errors.

The instructive part: that operator ALREADY carried a hi+lo decomposition
(`mp_mm_split`) written to recover MANTISSA precision on the same path. It
cannot touch exponent range, so a partial fix for one half of the loss read as
a fix for both, and the notes beside it said so.

## workaround
Route the multiply through the exact float32 kernel wherever no cube justifies
the cast (bench/gen_mlaprolog.py in cannbench-tilers). 12/20 -> 19/20 on
hardware, +10.5 banked, and the first board movement of that session.

What a fix needs:
  * a conversion carries a RANGE obligation as well as a width one: f32 -> bf16
    is range-preserving and precision-losing, bf16 -> f16 is the reverse, and
    only the second can turn a finite value into zero;
  * so an emitter can refuse, or demand a scale factor, when a value that may be
    subnormal in the destination is narrowed through it;
  * it is a static property of the two types, not of the data, so the check
    costs nothing at emit time.

This is the type-system half of what NOTES 213 in that tree found on the target
side: there, a capability was read from the wrong architecture's header; here, a
conversion's cost is read as width when it is range. Both are facts the
declaration could carry and does not.
