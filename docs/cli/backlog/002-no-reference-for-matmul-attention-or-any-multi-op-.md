id: 2
area: run
title: no reference for matmul, attention or any multi-op kernel
opened: 2026-09-01
closed: 2026-09-01

## wanted
`-r` on a matmul or an attention kernel — the kernels that actually matter for a model.

## got
"no operation this harness has a reference for. It knows softmax, exp, sigmoid, relu,
sqrt, log, neg and abs over one row."

## workaround
Checked those kernels by eye against the emitted source only, which proves nothing
about the numbers. The refusal is deliberate — numbers with nothing to compare them
against are worse than no numbers — but the referenceable set is far too small to be
useful on real work. matmul is the obvious next one and torch already has it; the
blocker is that the harness passes exactly two buffers and a length, so a two-input
kernel does not fit the ABI it assumes.

## fixed
The blocker named above -- "the harness passes exactly two buffers and a length, so a
two-input kernel does not fit the ABI it assumes" -- was the whole of it. Widened:

* `run::Shape` replaces the `(rows, cols)` pair, which had no way to say that two
  matrices of different sizes produce a third of a third size. `Shape::Matmul { m, k, n }`
  does, and `matmul_from_mlir` reads the three dimensions out of the intrinsic call's own
  operands rather than asking the user for numbers that must then agree with the kernel.
* The Swift harness takes buffer sizes, an output size and scalars on the command line
  instead of assuming one input. It stayed dumb -- `tile` still owns the reference and
  the comparison -- which is what keeps a second harness a transcription.
* Scalars are bound BY NAME (`M`, `N`, `K`, `num_elements`), and an unbindable one is
  refused rather than defaulted. A zero `K` makes the inner loop run no iterations and
  the kernel returns all-zeros, successfully: a wrong answer wearing a green exit code.
* The torch matmul script had `m = rows; k = cols; n = cols` -- square by assumption. It
  now takes the real shape, so a non-square matmul is not compared against a different
  product.

Two bugs fell out of doing it for real, both found by running rather than by reading:

* `KernelAbi::parse_msl` ended the parameter list at the first `)`, which is the one
  inside `[[ buffer(0) ]]`. Every kernel therefore looked like it had one parameter and
  no output buffer, and the first real matmul run was refused with "expected exactly one
  output buffer, found 0". Now matched by depth.
* `RunReport.checked` was recorded and never rendered. `compare` maxes the error over the
  values that came back, so an empty result set gives zero error and the verdict line
  reads "all three agree" -- a green report for a kernel that produced nothing. The
  PICO session independently hit this shape on a board rig that returned the previous
  model's output for a model that had failed to load. `render` now says so.

Verified end to end on the M1 Ultra with a NON-square f16 matmul (64x128x64, so the two
input buffers differ in size and neither matches the output):

    run: Apple M1 Ultra - matmul over 4096 elements
      kernel vs torch 2.13.0 (via uv)  max abs 0.00e0
      verdict: all three agree
      target   : median 78.04 us over 50 runs
      reference: median 4660.21 us over 50 runs

The 59.7x ratio those numbers give is INFLATED and should not be quoted: it came from a
debug build, whose reference is unoptimized. The same matmul on the same machine from a
release build of the same commit reports 5.2x -- identical device time, a baseline eleven
times faster. Later confirmed across three Apple GPUs from the installed release binary:
5.2x (M1 Ultra), 6.0x (M2 Max), 7.4x (M4). The report now carries a warning under
`debug_assertions` saying so, because the direction of that error always flatters the tool.

The exact agreement is real rather than a vacuous comparison: 4096 values are read back,
all products are multiples of 0.0625 with magnitudes under 64, which f16 represents
exactly. Checked by dumping the harness output directly.
