# Grilling the `tile` CLI design and plan

Three adversarial passes: one on the executable spec, one on the milestone plan, one on
the pandoc analogy itself. Everything marked **[verified]** was checked on disk on this
branch, not inferred.

---

## The eight things that would have sunk it

**G1 — M2's mechanism, as first written, does not compile.** [verified] The emitters use
absolute crate paths (`mlir_to_msl.rs:48` `use crate::mlir_parse::…`; `mlir_to_rvv.rs:42`
`use crate::mlir_to_linalg::…`). `tile_spec` gets away with `#[path]`-including them
because `tests/cucumber.rs` **is** a crate root; putting the same includes in
`src/emit.rs` lands them at `crate::emit::mlir_parse` and every internal import fails.
→ Includes go at `crates/tile_cli/src/lib.rs`. The trick itself transplants fine — the
workspace exclusion was never load-bearing, because `#[path]` knows nothing about
workspaces.

**G2 — The root toolchain pin reaches every subdirectory.** [verified] `rust-toolchain.toml`
pins `nightly-2025-08-04` and is the only such file in the tree;
`cd crates/tile_codegen && rustc --version` reports that nightly. A CLI that must build on
stable, on six triples, cannot inherit a deliberately frozen nightly whose own comment says
"do not bump casually". → `tile_cli` is `exclude`d **and** carries its own
`rust-toolchain.toml` pinning stable. As a root member it would also make `cargo build` at
the root try to build `tile_std`, which ICEs without the out-of-tree backend (the repo has
the ICE file checked in).

**G3 — `rusqlite` would break requirement (3) on day one.** `libsqlite3-sys` compiles C,
which the plan's own "no non-Rust compiled dep" CI gate would red-flag. The design made it
worse by adding encryption-at-rest (sqlcipher is more C). → The storage decision moves to
the front of M5: export the corpus to a serialized pure-Rust format at release time,
encrypted with RustCrypto's `chacha20poly1305`. That also makes the license gate mechanical
instead of decorative.

**G4 — The license gate as designed is theatre, and half-admitted it.** A licensed user can
dump the whole corpus through `-s` itself; offline expiry is unenforceable; and the key sits
beside the ciphertext. → Gate **delivery**, not the file: the license fetches signed
snapshots, or the crown-jewel queries live server-side. Ship it as honest product
segmentation and say so, or spend the effort server-side — but do not spend milestone time
on obfuscation.

**G5 — The plan contradicted itself on the default write.** M2's exit criterion had
`tile kernel.mlir` writing `kernel.metal` unprompted; its own Trap 3 argued never to write
to cwd without `-o`. → Settled: write `<stem>.opt.<ext>`, **never overwrite without
`--force`**, print the path as the last line. Both an explicit `-o` and the derived name
obey the same no-clobber rule. Still flagged for the user, because the requirement asks for
the automatic behaviour explicitly.

**G6 — M4 was 3–5x under-scoped and partly unimplementable.** O1/O2 presuppose an MLIR→MLIR
rewrite layer that does not exist [verified: `mlir_parse.rs` is a parse-for-emission helper;
the emitters are one-shot text→source transforms; the fusion and merge-over-scatter
doctrines live inside lowering, not as reusable passes]. → Split three ways: route surface
and O0/O1, then the rewrite layer with O2/O3, then O4 after a device and the database exist.

**G7 — The ten normative feature files were wired to nothing.** The plan wrote four new
features into `crates/tile_spec/features/` — which [verified] `tests/cucumber.rs` globs and
asserts all-green, so they turn the suite red the moment they land — and never executed
`01`–`11`. → Features live at `crates/tile_cli/features/`, executed by a harness reusing
`tile_spec::gherkin` [verified: `pub mod gherkin; pub use gherkin::{Runner, StepKind,
World}` — std-only, zero-dep]. Every milestone names which files go green.

**G8 — The cross-triple matrix cannot be an "empty job".** [verified] CI today is
ubuntu-latest and macos-14. `platform/linux.rs`, requirement (2) and requirement (3) all
break silently on the dev Mac. → Three-triple `cargo check` and the `cargo deny` gate are
real jobs from M1; M10 keeps only packaging and armv7.

---

## The analogy itself

**The formats are not peers.** Pandoc's formats round-trip through one shared document
model. `tile → mlir → msl` is a monotone descent: each hop discards the information the
level above had. Lift is not "the reverse edge with a different cost" — it is program
synthesis. → Forms carry an explicit `level`; lift edges are a separate, sparse relation,
and **they do not auto-compose into lowering routes**. A synthesised intermediate silently
feeding a lowering is exactly the invisible-wrongness this tool exists to prevent.

**"Same type means optimize" is 11% populated.** It is real for `tile` and the mlir family.
For the other 16 forms, `msl → msl` means writing a Metal frontend — a second product, not
a milestone. tile-rs has **1.5 readers and 16 writers**. → The tool is a *fan-out*, not yet
a swissknife. Adopt pandoc's grammar; do not inherit its promise. Say so in `--help`, and
make `tile --list-forms` print the reader/writer matrix, because a sparse graph is fine and
a sparse *undiscoverable* graph is what makes a tool feel broken.

**There was no vocabulary for trust.** A mangled heading is visible; a numerically wrong
kernel is invisible until it corrupts a run. → Every edge carries a fidelity class —
`exact`, `validated`, `unvalidated`, `synthesised` — printed on every conversion. The
one-line footer `route: tile → mlir → msl [exact, validated on M2 Max]` is worth more than
anything else the tool prints.

**`rvv` is not a peer node.** Per the README it "has no translator of its own; LLVM already
owns RVV codegen, so it wraps the linalg egress and stamps the RISC-V triple". Modelling it
as a distinct node makes the planner offer `linalg → rvv` and `mlir → rvv` as if they were
different lowerings when one is a relabelling. → Model it as `linalg` plus an attribute.
Ask the same question of `pto` before freezing the taxonomy: [verified] `mlir_to_pto` is
imported by the `msl` and `gpu` emitters, which makes it a shared substrate, not a leaf.

---

## Also found

* **`crates/rustc_codegen_tile` has no `Cargo.toml`** [verified] — a bare `src/` of 15
  emitters plus `mlir_parse.rs`. That is *why* it is excluded (the `crates/*` glob would
  fail on a manifest-less directory) and it means the `#[path]`-include is the only access
  path. It also means **any edit to those files silently changes CLI behaviour with no
  version boundary** — which is the second reason the golden-file suite must exist from M2.
* **`mlir_to_pto` is open and mandatory** [verified: `msl` and `gpu` both import it]. The
  "pto is closed" story was wrong; only the richer target *registration* is closed.
* **`--features ascend` does not exist** [verified: `tile_codegen/Cargo.toml` declares
  `default` and `emitters` only; `ascend` appears in a comment and in `check-cfg`].
  "Available only when built with it" must be a `cfg(feature)` probe, not a flag someone
  types.
* **There are no `examples/`** [verified: `git ls-files | grep -c '^examples/'` = 0; 86
  tracked files total]. Every exit criterion citing an example kernel was unrunnable. M1
  must create `testdata/forms/` — without a corpus the sniffer spec cannot execute at all.
* **The repo's entire dependency graph is 18 crates** [verified from `Cargo.lock`], and
  `tile_spec` hand-rolled a Gherkin parser rather than take one dependency. The CLI as
  specified wants argv, SQLite, async, MCP, HTTP, egui, an unpacker, TLS, Ed25519 and an
  AEAD — naively 18 → 400+, in a repo whose reviewers refused *one*. → Dependency budget
  per milestone, everything behind a feature, default binary under 15 crates.
* **`amd` was a category error.** A ROCm GPU is not a Ryzen AI NPU, and there is no HIP
  target in the registry — so the draft would have defaulted a Radeon to IRON Python. →
  `amd-gpu` falls back to `linalg` and says why; `amd-npu` gets `aie`.
* **The profiler is not a stub** [verified]. `tile_codegen`'s *default* features already
  give `plan_tiles`, `collapse`, `promote_types`, `hazards`/`barrier_points`/`unsynced`,
  and `HardwareParams::check_{ub,repeat,dma_stride,cube_tile}`. So `-i` reports the tile
  plan, the RAW/WAR/WAW edges, the required barriers and the resource verdicts — on any
  machine, with no toolchain. This is what makes M1 worth shipping alone, and it gives the
  two most recent commits' rules ("NPU resource bounds as codegen invariants", "the hazard
  model it lacked") their first user-facing surface.
* **Unmeasured must refuse.** Commit `f10513c` made every `HardwareParams` query return
  `Err` when `measured` is false, because the previous commit had copied 910B capacities
  onto an unrun chip — "approving precisely the tilings that fail there". The CLI is the
  first place a user meets that rule, and it must propagate the refusal, not smooth it into
  a plausible number. Now specified in `04` and `05`. The refusal is also **chip-keyed**:
  950PR and 950DT are different machines behind one ISA target, so each SKU carries its
  silicon id, DT's known-missing eval-image OP JSON, and no borrowed core count
  (`cannbench-tilers/NOTES-dt-pr-diff.md` is the evidence table).
* **A third of the Then-steps were unfalsifiable** — "no network request is made", "no
  dynamic library beyond libc was loaded", "valid tile-rs source", "a strict superset of
  the first's categories". Each needs a mechanisable rewrite (an injected recorder, a
  syn-parse plus a re-lowering round trip) or cutting.
* **One feature file did not parse** — a multi-line `Then` step in `04`. Fixed, and CI
  parses all eleven files before anything else runs.
* **Tagging discipline was missing.** Untagged scenarios must be green; everything else
  carries exactly one tag saying why (`@planned`, `@requires-rustc-backend`,
  `@requires-lifter`, `@requires-device`, `@requires-license`). CI gates untagged scenarios
  and reports the tagged counts separately, so "how much of the spec is real" is a number.
  Otherwise this becomes a permanently red suite that teaches the team to ignore red.
* **The `.so` has no stable ABI.** Provisioning must pin `(nightly, .so)` as one unit, and
  the per-arch release inventory must be audited *before* M4 is committed — if there is no
  linux-aarch64 artifact today, two milestones' exit criteria are unmeetable.
* **`-v`/`-V` inverts the universal convention.** Implemented as specified, with long forms
  and a warning alias. Expect the bug reports anyway.
* **`cargo publish` of `tile_cli` is permanently impossible** with out-of-package `#[path]`
  sources. Accepted; it ships as a release binary.
