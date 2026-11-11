id: 7
area: corpus
title: a failing spec suite scored coverage and passed CI
opened: 2026-09-01
closed: 2026-09-02

## wanted
Add the unhandled-intrinsic test to tile_spec (the test backlog #006 said was
missing). For that to be worth anything, tile_spec has to run.

## got
`cargo test --test cucumber` in crates/tile_spec was ALREADY failing before any
change of mine:

    UNDEFINED STEP in scenario 'an unmeasured architecture refuses rather than
    borrowing numbers': Given "the tile_codegen HardwareParams model"

Verified against a clean tree with my files removed: same failure. Two features --
npu_resource_bounds and npu_target_semantics -- had been committed with no step
definitions at all, and two of their steps were written across two lines, which
the parser cannot take either.

And CI reported green throughout. scripts/coverage.sh runs cargo-llvm-cov in a
subshell, does not check its exit status (the script sets `-uo pipefail`, not
`-e`, deliberately), and `check_json` only asks whether a report file exists.
cargo-llvm-cov writes one whether the tests passed or not. So a red suite scored
a coverage percentage and cleared the `--gate 88`.

## workaround
Fixed rather than worked around, because the request depended on it:

* defined the missing steps against the real tile_codegen model -- HardwareParams
  C7/C8/C9 and TargetSemantics nan_test/bitwise_count -- rather than rewording the
  features into something easier to satisfy;
* split the two multi-line steps;
* added `check_status` to coverage.sh so a non-zero exit sets FAILED and prints
  "coverage below is of a red suite". Verified by breaking a row and watching the
  gate catch it.

tile_spec is green for the first time in this tree: 771 scenarios.

Evidence worth keeping: this is the failure mode tile_cli's own tagging discipline
exists to prevent, and it happened here in a worse form. A red suite teaches people
to ignore red; a red suite that CI calls green teaches them the suite is fine. Any
gate that measures a property OF a test run must first check that the run passed --
coverage, benchmark timings and mutation scores all have this shape.

## closed (2026-09-02)

Marked closed rather than left open: the body already records the fix and the
verification -- undefined steps defined, multi-line steps split, `check_status` added to
coverage.sh, and the gate confirmed by breaking a row and watching it catch it. tile_spec
has been green ever since.

It stayed open only because nothing set the field. An issue whose text says "fixed" and
whose status says "open" is the same class of defect as the rest of this backlog: the
record disagreeing with the world. `tile backlog`'s verdict is computed from the field,
not from the prose, so leaving it open would have kept counting a solved problem against
the cluster rule.
