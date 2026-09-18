id: 3
area: lowering
title: no lifter exists for any target-source form
opened: 2026-09-01
closed: 2026-09-01

## wanted
`tile k.metal -o k.rs` — lift a hand-written Metal kernel back to tile-rs so it can be
retargeted. This is half of what a "swissknife" implies.

## got
"tile-rs can emit \"msl\" but cannot read it — no msl frontend has been written."
tile-rs reads 3 forms and writes 19.

## workaround
Rewrote the kernel by hand in the tile DSL, using the emitted Metal as a reference for
what the intrinsic sequence had to be. That is tractable for a small elementwise kernel
and hopeless for anything with a schedule. Evidence for the eventual fix: the round trip
tile -> mlir -> msl is stable and byte-exact, so a lifter only has to reach `mlir`, not
`tile` — mlir_to_msl.rs is effectively the specification for what an msl_to_mlir would
have to invert.

## resolved by the form rule (2026-09-01)

The owner settled what a form is:

> A form should be either a **source** form for the downstream toolchain to consume, or an
> **MLIR** form for further compiler passes to process. If it is in a **binary** form, it
> would be directly loaded and executed by the harness.

That is a rule about the CONSUMER, not about how far down a representation sits, and it is
what `level` could not say. It is now `forms::Role` on every row of the table, with the
obligations derived from it (`owes_a_reader`, `directly_executable`) and enforced by tests.

For this issue it is decisive. `msl`, `cu`, `.cce`, `.pico.s` and the rest are **source**
forms: they exist to be handed to a downstream toolchain, and tile-rs owes correct text
plus the name of the toolchain that takes it. It does not owe a reader. So "no lifter
exists for any target-source form" describes the design rather than a defect, and the
16 `read: no` rows in `--list-forms` are complete rather than partial.

What would reopen it: someone wanting a *frontend* for a specific target source. That is a
separate product with its own justification, not a milestone this one is missing. The
honest framing survives in the tool's own output, which now prints the role column and
says which absences are gaps.

Retained from the PICO session's evidence: even where a lifter exists (`pico_lift`), it is
a signature lookup against a fixed catalog and returns CANDIDATES, because the intrinsic
map is not injective. So a general lifter would not be the same kind of object as the
emitters, and building one is not "finishing" the table.
