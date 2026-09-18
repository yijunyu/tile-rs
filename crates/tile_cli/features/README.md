# The `tile` CLI executable specification

These eleven `.feature` files are the **normative requirement** for the `tile`
command. They are staged here rather than under `crates/tile_spec/features/`
because `crates/tile_spec/tests/cucumber.rs` globs that directory and asserts
every scenario green — dropping undefined steps into it turns the suite red the
moment the file lands.

**Landing plan.** They move to `crates/tile_cli/features/` in the same change that
creates `crates/tile_cli` and its harness. The harness takes `tile_spec` as a
path dev-dependency and reuses its zero-dependency runner
(`tile_spec::gherkin::{Runner, StepKind, World}`), so the CLI specs are executable
from the first milestone rather than prose that rots.

## Tags, and what CI enforces

A scenario with no tag MUST be green in CI. Everything else carries exactly one
tag saying why it is not:

| tag | meaning |
|---|---|
| `@planned` | the behaviour is specified but not yet built; CI skips and counts it |
| `@requires-rustc-backend` | needs the prebuilt LLVM-linked codegen `.so`; runs only where doctor finds it |
| `@requires-lifter` | needs a lifter that does not exist for that form yet |
| `@requires-device` | needs real accelerator hardware |
| `@requires-license` | needs a knowledge-base license key |
| `@requires-stats-build` | **conditional, not permanent** — the scenario is real and runs in a build with `--features stats`. It is skipped and counted in a build without the crypto, rather than passing vacuously |

CI gates on untagged scenarios only, and separately reports the tagged counts so
"how much of the spec is real" is a number, not an impression. A tag is removed in
the same change that makes its scenario pass.

**On landing, every scenario is tagged `@planned`** — 120 of them today, across 11
files. Each milestone's change removes the tag from exactly the scenarios it makes
pass, so the untagged count is a running measure of how much of the tool exists. That
ordering matters: tags added later, to quiet a red suite, are how a team learns to
ignore red.

`docs/cli/check-features.py` is the structural check — it mirrors what
`tile_spec`'s zero-dependency runner accepts (no multi-line steps, every
`Scenario Outline` has `Examples`, every `<placeholder>` has a column, every table row
matches its header, only known tags) and it runs in CI before anything else:

```
$ python3 docs/cli/check-features.py docs/cli/features
OK: 120 scenarios, 1 tagged, 119 must be green in CI
```

## The files

| file | covers |
|---|---|
| `01_form_detection` | extension + magic + `-f`; the four overloaded extensions |
| `02_transform_dispatch` | lower / lift / optimize; routes; fidelity classes; `--list-forms` |
| `03_intermediates` | `-k`, `--keep-dir`, scratch lifecycle |
| `04_profile_and_optimize` | `-i`, `-O0..4`, the default single-input run, unmeasured refusal |
| `05_platform_and_targets` | `doctor`, native default, `--cross`, the amd-gpu/amd-npu split |
| `06_toolchains` | on-demand acquisition, checksums, what is refused and why |
| `07_stats_and_license` | `-s`, the two databases, the license gate |
| `08_daemon_and_mcp` | `-d`, MCP parity, `-m`, daemon hardening |
| `09_ui` | `-u`, headless degradation |
| `10_portability_and_grammar` | triples, cfg confinement, option grammar, exit codes |
| `11_io_edges` | stdin/stdout, many inputs, symlinks, non-UTF8, concurrency, crash sweep |
