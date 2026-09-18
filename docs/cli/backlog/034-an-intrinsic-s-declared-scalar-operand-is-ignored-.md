id: 34
area: lowering
title: an intrinsic's declared scalar operand is ignored and a constant baked in its place
opened: 2026-09-12

## wanted
`__tile_rms_norm_f32(%dst, %src, %eps, %rows, %cols)` declares an epsilon
operand. I wanted the value the caller declared to be the value the kernel
computes with.

## got
`translate_rms_norm_pto` never read args[2]. It set `let eps = 1.0e-6_f32` and
emitted that. Two calls differing ONLY in epsilon produced byte-identical PTO.

Nothing announced this. The op is listed as lowered, the kernel compiles, runs,
and is numerically right on every case that happens to use 1e-6 -- eleven of
the twenty CANN Bench RmsNorm cases. The other nine (1e-3 through 1e-12) are
silently normalised with the wrong denominator.

A second, worse layer sat underneath: `format_f32_decimal` renders via
`{:.9}`, so 1e-12 trimmed to "0" and emitted "0.0". A zero epsilon is not a
slightly wrong epsilon -- it turns `mean + eps` into `mean` and the rsqrt of a
near-zero mean into an infinity.

## workaround
Fixed in the emitter rather than worked around: read args[2] through the
existing `ctx.resolve_float` (which `translate_clamp_pto` already used), and
fall back to scientific notation in `format_f32_decimal` when the decimal
rendering would flatten a nonzero value to zero.

What the general fix will need:
  * This is #024's shape one level down. There, a declared DTYPE was accepted
    and ignored; here a declared SCALAR OPERAND is. In both the signature is
    the promise and the translator is free to not keep it, with no diagnostic.
  * The cheap structural guard is arity/use coverage: a translator that never
    reads args[i] for a declared operand i is almost certainly dropping it.
    That is checkable mechanically over the ~70 `__tile_*` arms in
    mlir_to_pto.rs, and would have caught this without knowing what rms_norm
    means.
  * The cheap test-level guard is differential: emit the same intrinsic twice
    varying one operand, assert the outputs differ. Cheap enough to generate
    for every intrinsic that takes a scalar.

Verified on hardware: with the fix, 1e-12 reaches the generated AscendC as
9.99999996E-13f and 1e-6 as 9.99999997E-7f (ptoas 0.24, cann-9.2.0, 910b).
