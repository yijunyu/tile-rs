id: 8
area: lowering
title: .pico.s stops at the op program; nothing in tile turns it into instruction words
opened: 2026-09-01
closed: 2026-09-01

## wanted


## got


## workaround
(none recorded — an entry with no workaround is a report, not evidence)

## resolved by the form rule (2026-09-01)

The owner settled what a form is:

> A form should be either a **source** form for the downstream toolchain to consume, or an
> **MLIR** form for further compiler passes to process. If it is in a **binary** form, it
> would be directly loaded and executed by the harness.

That is a rule about the CONSUMER, not about how far down a representation sits, and it is
what `level` could not say. It is now `forms::Role` on every row of the table, with the
obligations derived from it (`owes_a_reader`, `directly_executable`) and enforced by tests.

`.pico.s` is a **source** form: svp (via `scripts/pico_wrap.py`, which drives 13 native
schedulers) is the downstream toolchain that consumes it. So stopping at the op program is
where a source form is supposed to stop, and tile-rs does not owe instruction words.

What the rule does say is that instruction words plus a container would be a **binary**
form -- a NEW row with `Role::Binary`, not more of `.pico.s`. It is worth adding only when
the harness is to load and execute it directly, which is the one thing binary forms are
for. Today no form is binary at all, and `forms::role_tests` asserts that, so `-r`
compiles source at runtime; adding the first binary form should flip that test and give
the harness a path that skips the compiler.

Two constraints recorded for whoever writes it, both from the PICO session:
`instr_size` must be an EVEN instruction-word count (the loader refuses an odd one before
running, and nothing in the container hints at it), and a schedule must be labelled
UNMEASURED rather than qualified-vs-derived until measurement separates them.
