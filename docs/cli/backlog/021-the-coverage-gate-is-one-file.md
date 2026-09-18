id: 21
area: ci
title: the coverage gate fails by 7 points and 69% of the gap is one file
opened: 2026-09-04

## wanted
`scripts/coverage.sh --gate 88`, which CI runs, to pass -- or a gate that says
something true.

## got
A clean local run totals **80.74%** against a threshold of **88**:

    tile_codegen              96.72%       1238/1280
    tile_hal                  98.45%         635/645
    tile_spec                 80.23%     44832/55881
    tile_std_macros           66.02%         136/206
    TOTAL                     80.74%     46841/58012

The workflow's own comment says the ratchet was "set BELOW the achieved total
(90.28% line on a clean run)", so this has drifted, not broken.

`tile_spec` dominates the total because its cucumber harness `#[path]`-includes
the emitter sources, so the number is mostly "how much of the emitters does the
spec suite reach".

## the obvious idea, measured and rejected
`tile_cli` is measured NOWHERE by coverage.sh -- zero mentions -- despite having
1169 tests and `#[path]`-including the SAME emitter sources. It looks like the
missing half of the measurement.

It is not. Measured, on one consistent line model, over the 16 emitter files both
runs see:

    tile_cli alone   82.44%   (32408/39309)
    tile_spec alone  79.10%   (35003/44249)
    UNION            79.59%   (35218/44249)

The union adds **215 lines**. Adding the CLI suite to the gate would move the
total by well under a point. The two suites reach almost exactly the same code.

(The percentages above are computed from llvm-cov's line SEGMENTS, whose
denominator differs from the line totals llvm-cov reports -- 39309 vs 62090 for
tile_cli. The DELTA between the three figures is the trustworthy part, not the
absolutes. And the first attempt at this was wrong: tile_cli reaches the emitters
via `.../tile_cli/src/../../rustc_codegen_tile/...` and tile_spec via
`.../tile_spec/tests/../../...`, so comparing raw filenames made every set
disjoint and turned the union into a sum. Paths are realpath'd now.)

## where the gap actually is
    uncovered  union%   lines  file
         6223   70.7%   21270  mlir_to_msl.rs
          519   87.3%    4092  mlir_to_pto.rs
          444   81.4%    2392  mlir_to_bang.rs
          375   81.3%    2001  mlir_to_gpu.rs
          262   91.6%    3111  mlir_to_gaudi.rs
          260   88.9%    2351  mlir_to_spirv.rs
          246   89.7%    2396  mlir_to_aie.rs
          219   86.7%    1642  mlir_to_nki.rs
          179   87.9%    1480  mlir_to_tpu.rs
          141   88.3%    1208  mlir_parse.rs
    total uncovered across the emitters: 9031

**`mlir_to_msl.rs` is 69% of the entire gap** -- 6223 uncovered lines, 70.7%
against 81-92% for every other backend. It is also by far the largest, at 21270
lines, and it is where the DS4 kernel families live: 274 KernelType variants, most
of them specialized or batched kernels that neither suite constructs.

The declared-intrinsic gate added this session compiles 236 of them, but that
exercises the EMISSION path for one call shape each; the specialized bodies behind
`Dsv4*`, the batched arms and the flash-attention variants are not reached.

## what the fix needs
A decision between two honest options, not a patch:

* **Move the gate** to what the surface actually scores, and ratchet from there.
  Truthful immediately; concedes the 20%.
* **Cover mlir_to_msl.rs.** 6223 lines is where the work is, and the corpus built
  this session is the obvious lever: it already generates a call per declared
  intrinsic, so extending it to the specialized shapes would reach the bodies.
  Note there is already a branch attempting this -- `ci/add-emitter-tests`,
  "cover the emitters added since the partition" -- whose runs are also failing.

Whoever picks this up should look at that branch first.
