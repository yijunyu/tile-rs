id: 23
area: lowering
title: the f16 matvec reads its operands in the opposite order from the f32 one, and reads out of bounds
opened: 2026-09-04

## wanted
One operand order for `__tile_matvec_*`, so a caller that works in f32 works in f16.

## got
The f16 lowering reads its two buffers the other way round from the f32 one, so driven
through the harness's ABI it reads past the end of buffer 1 and returns values that are
not the matvec. Found by `emulate_msl.rs` when the f16 sweep was added: the kernel ran on
the GPU and reported a max relative error of exactly 1.0, which is what "returned zero"
looks like.

## workaround
`emulate_msl.rs` drives f16 for every other operation Metal lowers and SKIPS matvec,
naming this entry. Driving it under a swapped ABI would hide the disagreement rather
than record it. The f32 matvec is unaffected and is driven at three shapes and three
threadgroup sizes.

## What the two kernels do

Both are emitted for the same intrinsic family, `__tile_matvec_f32` / `__tile_matvec_f16`,
from MLIR of the same shape. Their inner loops are exact mirrors:

```c
// f32:  p0 is the MATRIX, p1 is the vector
uint base = row * num_elements;
acc += (float)p0[base + i] * (float)p1[i];

// f16:  p0 is the VECTOR, p1 is the matrix
uint base = row * K;
sum += p0[i] * float(p1[base + i]);
```

The f32 form matches the harness: `Shape::Matvec::in_sizes()` is `[rows*cols, cols]` —
buffer 0 holds the matrix, buffer 1 the `cols`-long vector — and the f32 kernel agrees
with `reference_output` on every shape and every threadgroup size.

Driven through that same ABI, the f16 kernel reads buffer 1 at offsets up to
`(rows-1)*cols` while that buffer holds only `cols` elements. **It reads out of bounds**,
and the values it returns are not the matvec.

## Why this is a decision and not a typo

`mlir_to_msl.rs` declares the intent:

```rust
KernelType::MatvecF16 => 3, // activation(f32), weight(f16), output(f32)
```

So the f16 path is a DS4-derived kernel with its own convention — activation first, weight
second — rather than a careless swap. Two defensible answers exist and they are not
interchangeable:

1. **The f16 kernel is right and the harness is wrong** to drive `__tile_matvec_f16` with
   the f32 operand order. Then `Shape::Matvec` needs a dtype-dependent `in_sizes`, and the
   declared signature of `__tile_matvec_f16` should say which operand is which.
2. **The intrinsic's operand order is the contract** and the f16 lowering should match the
   f32 one. Then this kernel is wrong wherever it is called through the intrinsic rather
   than through DS4's own call site.

Choosing (2) may break the DS4 kernel this was written for, which is why it is not being
chosen here. This is the same shape as #019: one intrinsic, two lowerings, two orders, and
nothing in the declaration to say which is meant.

## A second defect in the same kernel, independent of the above

```c
uint num_simd_groups = tpg / 32;      // f16 matvec
uint nsg = (tcount + 31u) / 32u;      // f32 matvec, next door
```

Truncating division undercounts whenever the threadgroup size is not a multiple of 32, so
the partial group's contribution is dropped from the cross-SIMD fold. At `tcount = 33` that
is one whole group of the two. The f32 kernel already uses the ceiling form, so the correct
shape is established in the same file.

This one has no ABI question attached and could be fixed on its own — but it cannot be
verified until the operand order above is settled, because the kernel cannot be driven
end to end in the meantime. Fixing code that no test can reach is how the first defect
survived.

## What the harness does in the meantime

`emulate_msl.rs` drives f16 for every other operation it lowers and skips `matvec`,
naming this issue. The f32 matvec is driven at three shapes and three threadgroup sizes
and is unaffected.
