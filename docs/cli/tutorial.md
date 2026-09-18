# `tile` in ten minutes

A single command that identifies, profiles, lowers, lifts, optimizes and runs
accelerator kernels across twenty forms. This is the short version; `tile --help`
is accurate and shorter still, and worth reading before guessing at a flag.

**Try the options without installing anything:**
[the tile playground](https://adablue.duckdns.org/artifacts/tile-playground/) —
pick an input form, an output form and some flags, and it shows you the exact
command and what the tool would do with it. Its tables are exported from the real
binary, so they cannot drift from it.

## 1. Ask what is possible before asking for anything

```sh
tile --list-forms      # what can be read, written, optimized, lifted
tile doctor            # this machine: its accelerators and SDKs
```

The form matrix is the thing to internalise. tile-rs **reads 3 forms and writes
20**. Most targets are write-only: a conversion into one is a one-way trip,
because the receiving toolchain owns the language from there. That asymmetry
explains almost every refusal you will see.

## 2. The grammar is the argument list

There is no `convert` subcommand. What you pass decides what happens:

| you type | it does |
|---|---|
| one input, no `-o` | profile it, then optimize it for this machine |
| input and output **differ** | lower (down the stack) or lift (up it) |
| input and output **match** | optimize — needs a reader, so `tile`, `mlir`, `linalg` only |

```sh
tile k.rs -i                     # what it IS: dtypes, extents, tile plan, hazards, bounds
tile k.rs -t msl --route         # the candidate routes, their cost and their fidelity
tile k.rs -t msl -o k.metal      # do it
tile k.rs -t msl -o k.metal -r   # ...and run it, check it against a reference, time it
```

`-i` is the one to reach for first. It needs no toolchain and no target: dtypes
and promotion, collapsed extents, the tile plan, RAW/WAR/WAW edges with their
barrier points, and the target's resource bounds all come out of the source.

## 3. Lowering works anywhere; two things do not

No accelerator, no vendor SDK, no LLVM — `tile k.mlir -t bang -o k.mlu` runs on a
laptop. Only two rows are machine-specific:

- `-O3` hands the result to the target's own compiler, so that compiler must be
  installed. `-O4` additionally measures on real hardware.
- `-r` runs and checks the kernel. **Metal only today** (backlog #001).

## 4. A lift is not a lowering run backwards

Lifting is program synthesis, so the graph has exactly two lift edges and each was
justified on its own:

- `pto -> tile` — declared, **not implemented**. `tile` will tell you so.
- `cpp -> tile` — real. It shells out to `ascendc-to-rs`, so it needs that binary:

```sh
tile softmax.cce --via tile -t tile -o softmax.rs
# route: cpp -> tile  -O2  [synthesised]
```

Two rules come with it. **`--via` is required**: a lift never composes into a
lowering automatically, because a synthesised intermediate feeding a lowering is
the kind of wrongness that stays invisible until it corrupts a run. And the result
is marked **`[synthesised]`** — a lifted kernel is a reconstruction, not a
recovery, and it is only as good as the lift.

If the lifter is missing you get exit 4 and the command that fixes it:

```
tile: the route cpp -> tile  [synthesised] needs toolchain:ascendc-to-rs, ...
  the AscendC lifter is built from source, not downloaded:
    cargo install --path crates/ascendc_to_rs   (in the ascend-rs checkout)
```

## 5. Read the refusals literally

The exit codes are grouped by **who can fix it, and how** — the difference between
3 and 4 is the difference between a missing capability and a missing package:

| | means | what to do |
|---|---|---|
| 2 | a nearby command works | type that one — often just adding `--via` |
| 3 | tile-rs cannot do this **yet** | nobody has written it. Never a claim of impossibility |
| 4 | exists, but not here | install the toolchain, or use a build that has the target |
| 6 | needs hardware that is not here | run it where the device is |

When you hit a 3, record it before working around it — `tile backlog add` — and
come back afterwards to write down what you did instead. An entry with no
workaround is a report; an entry with one is evidence.

## 6. Two more worth knowing

```sh
tile k.rs -s                     # what was tried before on this kernel, and what it cost
tile k.rs -t msl -o k.metal -k   # keep the intermediates
TILE_SIMULATE=bare tile ...      # run as if nothing were installed, to reproduce a bug report
```

`TILE_SIMULATE` can only ever *remove* a capability, never grant one, and a
simulated run always says so on stderr.

## Where things are

| | |
|---|---|
| the tool | `crates/tile_cli` |
| the playground | `crates/tile_playground` — `./deploy.sh --build` |
| the backlog | `docs/cli/backlog/`, one file per issue, reviewed like code |
| the design | `docs/cli/design.md`; `plan.md` for what is scheduled |
| what actually runs | `docs/cli/progress.md` |
| specimens | `crates/tile_cli/testdata/forms/`, one per form |
