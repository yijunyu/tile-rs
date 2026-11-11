id: 18
area: optimize
title: pico reports "bounds (n/a)" then three "ok"s, and a tile plan from another chip
opened: 2026-09-04

## wanted
`tile k.mlir -t pico --cross pico -i` to either check the PICO resource bounds or
say clearly that it cannot. The sibling tile-rs-pico checkout carries the real
profile (Hi3403 `ub_bytes: 2097152`, and an explicit UbGeometry that errors
rather than guessing on an unnamed machine), so the numbers exist.

## got
Exit 0, and this line for every PICO kernel:

    tile plan: 8192 x 8 cores -> 0 full + 4608 tail, 1 blocks, 1 rounds/block
    bounds (n/a): UB ok; repeat ok; DMA stride ok;

Two problems. The tile plan is another target's geometry -- PICO has a 2 MiB UB,
not 8192 x 8. And the bounds line prints `(n/a)` followed by three `ok`s, which
reads as "checked and fine" at a glance when nothing was checked. `doctor`
already states the risk exactly ("a bound derived from another chip's numbers
would approve exactly the tilings that fail"), so the warning exists but the
per-kernel output contradicts its tone.

## workaround
Do not read `tile -i`'s bounds or tile-plan lines for PICO at all. The authority
is the sibling repo: `pico_codegen`'s Hi3403 profile discloses `ub_bytes:
2097152`, and `UbGeometry` errors on a machine with no named profile rather than
guessing -- which is the behaviour this line should have.

What the fix needs:

* PICO's UB is **2 MiB (2,097,152 B)** on the Hi3403 / SS928 target, and there is
  a separate four-RAM tier at 128 KiB. A profile, not a constant.
* L0A/L0B/L0C on PICO are **programmatic over UB bank groups**
  (`L0Model::ProgrammaticOverUb`), not physical as on Ascend, so a bound derived
  from an Ascend-shaped L0 model would be wrong in kind, not just in size.
* The resident weight block must stay **under 256 KiB** -- that is the constraint
  that actually binds MiniCPM5's tiles, and none of `UB ok / repeat ok / DMA
  stride ok` expresses it.
* Print `not checked` rather than `ok` when the profile is absent. Three `ok`s
  after an `(n/a)` is the part that misleads.
