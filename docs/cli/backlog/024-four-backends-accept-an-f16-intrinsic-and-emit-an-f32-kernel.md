id: 24
area: lowering
title: four backends accept an f16 intrinsic and emit an f32 kernel, byte for byte
opened: 2026-09-04

## wanted
An emitter handed `__tile_load_f16` / `__tile_exp_f16` / `__tile_store_f16` to produce a
kernel whose buffers are f16 — or to refuse.

## got
A kernel byte-identical to the f32 lowering. The emitter looked at the intrinsic's NAME,
found an arm, and ignored its WIDTH. On hardware that kernel reads four bytes where two
were meant: twice the buffer's length, and every value a pair of f16s reinterpreted as one
f32.

Measured by emitting the same kernel at both widths for every backend and comparing:

| backend | f16 pairs identical to f32 | what it emits |
|---|---|---|
| gpu | 6 of 6 | `const float* __restrict__ p0` |
| musa | 6 of 6 | `const float* __restrict__ p0` |
| hexagon | 2 of 2 | `hvx_vexp_f32` over `const float*` |
| csl | 2 of 2 | `var input0_tile: [N]f32;` |

`musa` is the starkest: its emitter contains **zero** mentions of `half`, `__half` or
`"f16"`. It has no f16 support at all and emits a plausible f32 kernel instead of refusing,
which is **#005 in the dtype dimension** — and `emit::ignored_intrinsics`, the detector
built for #005, cannot catch it, because the name IS handled and only the width is dropped.

Two backends are identical for a GOOD reason and are not in this entry: `tpu` emits
`jax.ShapeDtypeStruct(x0.shape, x0.dtype)` and takes the dtype from the array at run time,
and `ttmetal` names no dtype anywhere because its circular buffers carry their own format.
A bare diff reports those exactly as it reports the four above, which is why the check
requires a named mechanism rather than a clean diff.

`bang` was on this list until this session. Its fix was one flag: `ctx.dtype = "f16"` was
set only inside the matmul branch, while the whole `half` path — `"f16" => "half"`,
`"f16" => "sizeof(half)"` — had been present all along. **The four above do not have that
fix available**, which is the difference between a bug and a capability gap.

`nki` had one arm of twelve in this state (`nl.log` computed its dtype and never emitted
it) and it is now fixed.

## workaround
`tests/dtype_is_not_ignored.rs` emits every kernel at both widths and compares. The 16
pairs above are recorded in `known_debt` with the reason each is not a one-line fix, so the
gate is green on the tree as it stands, fails on a NEW backend that starts ignoring the
dtype, and **also fails when one on the list is fixed** — the entry then becomes a false
record and must be removed. The debt can only shrink deliberately.

Until then: do not pass an f16 intrinsic to gpu, musa, hexagon or csl. Refusing would be
better than emitting, and is the smaller change of the two available.
