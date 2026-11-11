id: 15
area: lowering
title: gaudi's row reduction reuses the elementwise loop and never accumulates across it
opened: 2026-09-03
closed: 2026-09-03

## wanted
To finish reading the `reduce_sum` goldens after `bang`'s turned up #014.

## got
Eight of nine are sound, and HOW is worth recording: six cannot express a partial-reduction
bug at all — `linalg.reduce`, `nisa.tensor_reduce`, `jnp.sum`, `pto.trowsum`, a single pico
`vsum` opcode — because they NAME the reduction and the width belongs to the operand. `aie`
hand-rolls it correctly. `msl` hand-rolls it correctly, having shipped this bug once.

`gaudi` emits, for a reduction:

    while (index[0] < end[0]) {
        float256 v_0 = v_f32_ld_tnsr(index, input0);
        float   v_1 = v_f32_reduce_add(v_0);
        v_f32_st_tnsr(index, output0, v_1);
        index[0] += get_index_space_stride()[0];
    }

That is the ELEMENTWISE template with a reduce in the middle. It stores inside the loop
and carries no accumulator across iterations, so each pass writes one vector's sum at its
own index. A row reduction produces one value per row.

## what I got wrong first, and the method that misled me
The first version of this issue said the kernels emitted at cols=256 and cols=1024 were
"byte-identical", and offered a cheap check: emit a reduction at two widths and diff.

Both were wrong.

The files are NOT identical — they differ in a comment, `// rsum: rows=1, cols=256`. I had
compared them with `rtk proxy diff`, and `diff` is one of rtk's OWN subcommands, not the
diff program: it printed nothing and exited 0 on every pair, including `linalg`, whose two
outputs differ in plain sight (`tensor<1x256xf32>` against `tensor<1x1024xf32>`).
`/usr/bin/diff` reports both correctly, exit 1.

The tell was there and I read past it: the check said every backend was identical. A result
that uniform is a broken instrument, not a discovery about nine independent emitters.

And the check is not valid even done properly. `msl` and `tpu` ARE identical at both
widths, and both are correct: msl reads `num_elements` at dispatch, tpu takes
`jax.ShapeDtypeStruct(x0.shape, ...)` at run time. Width-independent TEXT is the right
answer for a kernel that obtains its width at run time. The diff cannot separate that from
a baked-in constant, so it produces false positives on two of the best-behaved backends.

## what is actually true
The executable content of gaudi's reduction does not change with the row width, and
`float256` is the native TPC vector, which the emitter's own header documents. So the loop
processes one vector per iteration — correct for elementwise, and for a reduction correct
ONLY if the index space is configured one iteration per row.

That configuration is not in the emitted file and is not recorded anywhere in this
repository. So this is a question, not a proven defect, and it is filed as one.

## what would settle it
Someone with a Gaudi toolchain running the kernel over a row wider than one vector. If the
index space is per-row the kernel is right and this issue closes; if it walks vectors, the
stores overwrite and the answer is the last vector's sum.

## workaround
Until it is settled, treat gaudi's reduce_sum, reduce_max and absmax as unverified for rows
wider than one vector.

## fixed (2026-09-03)
Structurally, and inventing no intrinsic: every call in the new shape is one this emitter
already emitted.

The accumulator is hoisted above the loop, each iteration folds its vector's result into
it, and the single store moves below the loop:

    int5 out_index = get_index_space_offset();
    float _acc = 0.0f;
    while (index[0] < end[0]) {
        float256 v_0 = v_f32_ld_tnsr(index, input0);
        float   v_1 = v_f32_reduce_add(v_0);
        _acc += v_1;
        index[0] += get_index_space_stride()[0];
    }
    v_f32_st_tnsr(out_index, output0, _acc);

`reduce_max` gets `-INFINITY` and a max fold. The elementwise template is untouched and
still stores inside the loop, which is correct for an op producing one value per element.

The test checks POSITIONS, not substrings — the accumulator must be declared before the
loop and the store must come after the loop's advance. A test that only looked for `_acc`
would pass on a kernel that still stored every iteration, which is the bug.

What is still not claimed: that this runs. There is no Habana device here and
`[exact, unvalidated on hardware]` is still what the route line prints for gaudi.
