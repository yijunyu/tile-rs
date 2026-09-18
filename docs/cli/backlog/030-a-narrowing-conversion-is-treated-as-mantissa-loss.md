id: 30
area: transform
title: A narrowing conversion is treated as mantissa loss, but it can silently lose EXPONENT RANGE
opened: 2026-09-11

## wanted
A type or conversion rule that distinguishes the two things a narrowing cast
does. `bfloat16 -> float16` keeps every mantissa bit (8 into 10) and destroys
the exponent (8 bits into 5): bf16 reaches ~1e-38, half's smallest normal is
~6e-5, so small values flush to zero. The declaration says "narrow to f16" and
nothing anywhere says the VALUE RANGE changed.

## got
Nothing checks it, so the lowering routes an operand through half whenever the
cube wants half, and the loss is invisible until a grader reports it. In the
cannbench tree, mla_prolog cast both matmul operands to half for the cube and
lost 8 of 20 cases on 950pr -- every failure bfloat16, MERE PASSING at 2.9e-06,
MARE pinned at 0.0078 (two bf16 ulp), reported as 小值域 (small-value-range)
errors.

Worse, the operator ALREADY carried a hi+lo split (`mp_mm_split`) written to
recover MANTISSA precision on the same path. It cannot touch exponent range, so
the compensation that was there looked like it covered the problem and did not.
That is the specific failure mode: a partial fix for one half of the loss reads
as a fix for both.

## workaround
Route the multiply through the exact float32 kernel wherever there is no cube to
justify the cast (bench/gen_mlaprolog.py in cannbench-tilers). 12/20 -> 19/20 on
hardware, +10.5 banked.

What a fix needs:
  * conversions carry a RANGE obligation as well as a width one -- f32 -> bf16
    is range-preserving and precision-losing, bf16 -> f16 is the reverse, and
    only the second can turn a finite value into zero;
  * so the emitter can refuse, or demand a scale, when a value that may be
    subnormal in the destination is narrowed through it;
  * the check is cheap and static: it is a property of the two types, not of
    the data.
