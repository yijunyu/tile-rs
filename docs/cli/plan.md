# `tile` — milestone plan (reconciled, post-grill)

This supersedes the first-draft plan. It folds in two adversarial reviews and ten
facts verified on this branch. Where the first draft and the design doc disagreed,
**this document is the contract**; §0 lists the reconciliations.

Facts this plan is built on, all verified on `tile-cli-harness`:

* `crates/rustc_codegen_tile` has **no `Cargo.toml`** — it is a bare `src/` of 15
  `mlir_to_*.rs` files plus `mlir_parse.rs`. There is no crate to depend on; the
  `#[path]`-include is the only route to the emitters, and it transplants out of
  `tile_spec` because `#[path]` knows nothing about workspaces.
* Those emitters use **absolute crate paths** (`mlir_to_msl.rs:48` `use crate::mlir_parse`,
  `mlir_to_rvv.rs:42` `use crate::mlir_to_linalg`), so the includes must sit at a **crate
  root**. `tile_spec` gets away with it because `tests/cucumber.rs` *is* one.
* `mlir_to_pto.rs` is **open and mandatory** — `msl` and `gpu` both `use crate::mlir_to_pto`.
  PTO's *emitter* is open; only its target *registration* is closed.
* `tile_codegen`'s manifest declares `default` and `emitters` only. **There is no
  `ascend` feature** on the open tree; it exists in a comment and in `check-cfg`.
* The root `rust-toolchain.toml` pins `nightly-2025-08-04` and is the **only** toolchain
  file in the tree, so it reaches every subdirectory, excluded crates included.
* The repo's whole dependency graph is **18 crates**, and `tile_spec` hand-rolled a
  Gherkin parser rather than take one dependency.
* There is no `examples/` directory on this branch; the repo is 86 tracked files.
* CI is two runners (ubuntu-latest, macos-14), not five triples.
* `tile_codegen`'s default features already provide the whole profiler: `plan_tiles`,
  `collapse`, `promote_types`, `hazards`/`barrier_points`/`unsynced`, and
  `HardwareParams::check_{ub,repeat,dma_stride,cube_tile}`.

---

## 0 — Reconciliations (the contract, settled)

| # | Question | Settled as |
|---|---|---|
| R1 | Where does `tile_cli` live? | **`exclude`d from the root workspace**, with its own `crates/tile_cli/rust-toolchain.toml` pinning **stable**. A root member would inherit the frozen nightly and would make `cargo build` at the root try to build `tile_std`, which ICEs without the out-of-tree backend. |
| R2 | Where do the `#[path]` includes go? | **`crates/tile_cli/src/lib.rs`**, at the crate root, as `pub(crate) mod`s — not inside `src/emit.rs`. |
| R3 | Is `pto` open? | The **emitter is open and required**; the richer PTO/AscendC *target registration* is closed. `-t pto` works in the default build; `-t cpp` needs the closed registration and refuses distinguishably. |
| R4 | Input form override | **`-f/--from` exists.** `-t` is the output form only; using it as an input tiebreaker was incoherent. |
| R5 | `-O` semantics | Design's table wins, with one change: **O3 = toolchain-assisted**, **O4 = measured on hardware**. Route search is not a level; it is what the planner does at every level. |
| R6 | Default single-input write | **Settled by the user:** writes `<stem>.opt.<ext>` beside the input, prints the path as the last line, and **refuses to overwrite without `--force`**. The requirement's automatic optimization is kept; the clobber the grills feared is made impossible instead of being avoided by doing less. |
| R7 | Knowledge-base storage | **Settled by the user:** the corpus is **exported at release time** from `impact.db` to a serialized pure-Rust format, sealed with RustCrypto's `chacha20poly1305`. No SQLite, no C, and the gate becomes mechanical at the delivery boundary. The corpus is a versioned snapshot, not a live database — that is the accepted trade. |
| R13 | What "pure Rust" forbids | **Settled by the user:** *no dependency in `tile`'s build graph runs a C or C++ compiler.* This rules out `rusqlite`/`libsqlite3-sys` and `ring`; it selects `rustls` + `webpki-roots`, RustCrypto (`chacha20poly1305`, `ed25519-dalek`), and `miniz_oxide`/`flate2` with the Rust backend. Vendor runtimes are still `dlopen`ed at runtime — that is detection, not a build dependency. Machine-checked from M1. |
| R8 | Feature-file home | `crates/tile_cli/features/`, executed by a harness reusing `tile_spec::gherkin`. Staged today at `docs/cli/features/`. Nothing is written into `crates/tile_spec/features/`. |
| R9 | Lift composition | **Lifts do not auto-compose into lowering routes.** A route through a synthesised intermediate must be named with `--via`. |
| R10 | `amd` | Split into **`amd-gpu`** (no tile-rs target; falls back to `linalg` and says so) and **`amd-npu`** (`aie`). Defaulting a Radeon to IRON Python was a category error. |
| R11 | Daemon transport | **stdio by default, no port.** `--listen` requires a token, because it would serve licensed data. |
| R12 | Dependency budget | Every dependency behind a feature. Default binary target: **< 15 crates**. |
| R14 | Exit codes | Grouped by **remedy**, not by subsystem. `3` = tile-rs cannot do this *yet* (unwritten); `4` = it exists but not in this build or on this machine (acquirable); `2` = a nearby command works, which is where the `--via` refusal now lands. The old single `3` covered all three and told a caller to retry forever or give up a download early. |

---

## Load-bearing decisions (expensive to change later)

**D1 — Form identity is a struct with a stable string id and a level.**
`Form { id: &'static str, level: u8, ext: &[&str], magic: &[&str], readable: bool,
writable: bool, fidelity: Fidelity }`. The id is serialized into MCP responses, attempt
records and route reports, so it is frozen at M1 and covered by a compat test. `level`
is what makes lower/lift/optimize a *derived* fact rather than three code paths.

**D2 — The transformation graph is data.** Edges carry kind, capability requirements
(`toolchain:<id>`, `device:<family>`, `license`, `feature:<name>`) and a fidelity class.
The planner filters by satisfiable capabilities, then shortest-path. `--route`, `--via`,
`-O`, the daemon and the UI all read this one graph.

**D3 — Fidelity is a first-class output.** `exact` (mechanically checked emit),
`validated` (checked against a CPU reference on that hardware — the 8 On-HW backends),
`unvalidated` (emitted, never run there), `synthesised` (lifted; needs review). Printed
on every conversion. A kernel that is numerically wrong is invisible until it corrupts a
run, so the trust level is not an optional detail.

**D4 — Unmeasured refuses.** `HardwareParams`/`TargetSemantics` carry `measured`; when
false, every bound query returns `Err`. The CLI propagates that refusal instead of
printing a plausible number. This is commit `f10513c`'s rule and it is the single most
important behaviour the CLI inherits. The params block is also **chip-keyed**: PR and
950DT share `dav-3510` but are different silicon and different eval images, so
`ascend_950pr_unmeasured` / `ascend_950dt_unmeasured` name their SKU, DT carries its
known-missing vendor OP JSON (`Cat`/`Arange`/`ConcatD`), and neither may borrow the
other's core count (`cores_for` returns 0 / unknown rather than 910B's 48). One graded
DT job that prints `ubSize`/`nAiv`/`nAic` is what flips either SKU to `measured: true`.

**D5 — Core is synchronous.** Async exists only in the daemon binary and the ds4
supervisor. Retrofitting async through the core is the classic infection.

**D6 — CLI, MCP and UI are three renderings of one request type.** No parallel logic.

**D7 — Golden files from M2, in root CI, on every push.** They are simultaneously the
CLI's correctness tests and the **emitter-drift alarm**: `tile_cli` is defined by files
it does not own, which change on their own cadence, and nothing else would notice.

---

## Milestones

### M0 — Freeze the contract  *(half a day)*
Reconcile design + plan into the single grammar, `-O` table, exit codes, form list and
storage decision above. Answer the three open decisions in §Open decisions. Freeze the
form-id list and the number of forms (the drafts drifted between 14/15/16 emitters and
18/19 forms).
**Exit:** `docs/cli/design.md` and this file agree on every flag, and `--help` output is
written out longhand as a golden file before any code exists.
**Retires:** the largest risk in the project, which is two documents disagreeing about
what is being built.

### M1 — Skeleton, taxonomy, sniff corpus, doctor, and a real CI matrix ✅ **DONE**

> **Delivered.** `crates/tile_cli` builds on stable, 65 tests green, `cargo fmt --check`
> and `cargo clippy -D warnings` clean, dependency graph **3 crates** against a budget of
> 15. `cargo check --target aarch64-unknown-linux-musl` passes from a Mac, so the Linux
> probe module is checked rather than assumed. 28 of 125 specification scenarios are live
> and executing (75 cases, counting Examples rows); the rest carry `@planned` and are
> counted. See [`progress.md`](progress.md).
>
> Two bugs the work surfaced that the design had not:
> * **`rvv` was unreachable.** An rvv module carries `linalg.` as well as `riscv64`, so
>   the general form won by table order. Fixed by giving `Form` a `refines` field: a form
>   that refines another beats the one it refines. The corpus caught this on its first run,
>   which is the entire argument for having a corpus.
> * **Two scenarios in different files shared one step text.** Steps resolve in
>   registration order, so the generic `When I run "tile ..."` silently swallowed specific
>   invocations and those scenarios asserted nothing while reporting green. The catch-all
>   is now registered last, deliberately and with a comment saying why.
>
> Two scenarios were rewritten rather than faked: the lift-composition pair now uses the
> one lift edge that exists (`pto -> tile`) instead of a hypothetical `msl -> tile`, and
> the hazard-report scenario reads MLIR, where the SSA form makes dependences visible.
> Their `.rs` and `msl` variants remain, tagged `@planned`.

**Goal:** `tile k -i` identifies any form and prints a profile worth reading; `tile doctor`
reports the machine.
**Touches:** `crates/tile_cli/{Cargo.toml, rust-toolchain.toml, src/lib.rs, src/bin/tile.rs}`;
`src/forms.rs` (D1 + sniffers); `src/profile.rs`; `src/platform/{mod,macos,linux,windows}.rs`;
`crates/tile_cli/testdata/forms/` (**one specimen per form**, ~19 small files — the repo has
no `examples/`, so the corpus must be created); `crates/tile_cli/features/` + harness over
`tile_spec::gherkin`; `.github/workflows/tile-cli.yml`.
**The profile is not a stub.** From `tile_codegen` default features: operand dtypes and the
promotion that will apply, collapsed axes and element count, the tile plan (tile x cores),
the RAW/WAR/WAW hazard edges with required barrier points and any *unsynced* edge, and the
UB / repeat / DMA-stride / cube-tile verdicts — or D4's refusal.
**Flags:** `-h`, `-v`, `-V`, `-f`, `-i`, `doctor`, `--list-forms`, `--json`.
**Reqs:** (2), (3), (4).
**Exit:** `cargo test -p tile_cli sniff_corpus` — every specimen resolves to its known form,
plus the negative case; `tile doctor --json` diffs against a golden file per fixture (canned
`system_profiler` / `nvidia-smi` / `npu-smi` output parsed into a struct, so Linux probes are
testable on a Mac); `cargo check --target` green on 3 triples; `cargo deny` gate proving no
dependency compiles C; features `01`, `05`, `10` green.
**Not:** any conversion, install or database access.

### M2 — Tier-0 emit: MLIR → the open target sources ✅ **DONE**

> **Delivered.** The 15 emitters are `#[path]`-included at `src/lib.rs` — the crate root,
> because they import each other through absolute `crate::` paths — behind a default-on
> `emitters` feature. `tile k.mlir -t msl -o k.metal` emits real MSL on a clean macOS
> checkout with no LLVM. 15 golden files freeze every emitter's output byte-for-byte and
> double as the drift alarm against `rustc_codegen_tile`, which changes on its own cadence
> with no version boundary between it and this crate. **880 tests** (the included emitter
> sources bring their own with them); 34 of 127 spec scenarios live.
>
> Three things the work found:
> * **`rvv` was routed wrongly.** The graph had `mlir → linalg → rvv`, reading "rvv wraps
>   the linalg egress" as a routing fact. It is not: the wrapping happens *inside*
>   `convert_mlir_to_rvv`, which consumes the same tile MLIR every emitter does. The bad
>   hop planned cleanly and failed at emit time with "No entry-point kernel functions
>   found", because linalg output carries no `hacc.entry`. `rvv refines linalg` is about
>   identifying a file, not producing one — conflating the two cost a working route.
> * **The failure path leaked its scratch.** Every early return between creating the
>   scratch and finishing it now cleans up, and a run killed outright is swept by the
>   next one — "no orphans" is a property of the run that follows, not the one that died.
> * **The corpus MLIR was the wrong dialect.** The emitters consume LLVM-dialect modules
>   with `__tile_*` intrinsics and an `hacc.entry` attribute, not `func.func`/`tensor<>`.
>   The profiler's extent scan was reading the `1` out of `!llvm.ptr<1>` and calling a
>   1024-element kernel `numel 1`.

### M2 (original scope) — Tier-0 emit: MLIR → the open target sources
**Goal:** the pandoc moment, with zero external dependencies.
**Touches:** `src/lib.rs` (the 16 crate-root `#[path]` mods, R2); `src/routes.rs` (D2, edges
`mlir → src(*)`); `src/output.rs`; golden outputs in `testdata/golden/`.
**Flags:** `-t`, `-o`, `-`/stdin, `-o -`, `--force`.
**Reqs:** (5) — with no `-t`, the destination is the detected platform's form, named to match
`tile_hal::BackendKind::from_codegen_path` so the HAL can run the result.
**Exit:** `tile testdata/forms/softmax.mlir -o out.metal` on a clean macOS checkout with no
LLVM produces a **byte-identical match to a golden file**; `-o out.py` fails naming
`nki|aie|tpu`; a build without the closed registration refuses `-t cpp` with "not compiled
into this build", distinct from "no such form"; features `02`, `11` green.
**Cost to accept:** ~2.5 MB of emitter source compiles into every build (`mlir_to_msl.rs`
alone is 1.0 MB). Measure it here; if it hurts, extract a `tile_emitters` crate so the cost
is cached once.
**Not:** `.rs` input, optimization, lifting, compiling to binaries.

### M3 — Route surface and intermediates ✅ **DONE**

> **Delivered.** `-O1` and above put a real **optimize hop** on the front of the route for
> a readable input. That is the design decision that makes the rest work: the optimization
> becomes a nameable, keepable step rather than a claim in a report, so `--route` shows it,
> `-k` writes it out, and a user can diff what the passes did instead of taking the
> report's word for it. O1 implements dead-op elimination, iterated to a fixed point and
> conservative about anything with no result (that is where the side effects live).
> **O2 runs O1's passes and names what is not written yet** — it does not pretend to have
> fused, and it does not refuse the default level to make a point about not having fused.
>
> Three bugs, all found by the spec rather than by inspection:
> * **`-o out.mlir` was refused as ambiguous** — `.mlir` is claimed by four forms — even
>   when the input was already `mlir`. The input's own form now settles it, which is what
>   made the whole same-form optimization case reachable.
> * **Dead-op elimination read SSA names out of comments.** The fixture whose comment said
>   "`%junk` is never read" kept `%junk` alive. Dataflow now stops at `//`.
> * **`--route` listed a different route from the one that would run**, because the
>   prologue was applied by `plan_with_opt` and not by the listing. A `--route` that
>   disagrees with `-o` is the one thing it must never be.

### M3 (original scope) — `--route`, `--via`, `-k`, `-O0`/`-O1`
*(was M4's first third; it needs only M2 and de-risks the planner before provisioning)*
**Touches:** `src/routes.rs` (planner, cycles, cost), `src/scratch.rs` (run-scoped dir, sweep
on crash), `src/optimize.rs` (O0 verbatim; O1 local rewrites, mlir-family only).
**Flags:** `--route`, `--via`, `-k`, `--keep-dir`, `-O0`, `-O1`.
**Exit:** `tile k.mlir -t rvv --route` prints a diffable route list; a property test asserts
the planner terminates on the full graph; `-k` leaves exactly the named hops and a failed hop
leaves hop 1 and no scratch; a SIGKILL mid-run is swept on the next run; feature `03` green.
**Not:** O2 and above.

### M4 — Provisioning engine ✅ **DONE**

> **Delivered.** A pinned manifest (`assets/provision.toml`, a strict TOML subset parsed
> without a dependency), verified acquisition into `~/.tile-rs/toolchains/<id>/<version>`,
> a `layout-version` file, `tile install [<id>]`, and `--offline` / `--no-install` /
> `--yes`.
>
> **SHA-256 is written out, not taken as a dependency.** Requirement (3) forbids a crate
> that compiles C, and the obvious hashing crates pull one in through their assembly
> backends. Writing a *cipher* by hand would be indefensible; a hash is different — fully
> specified, holds no secret, used only for integrity — and it is checked against FIPS
> 180-4's own vectors plus every padding boundary, so it is validated against the standard
> rather than against itself.
>
> **The fetch shells out, and that is the honest answer to a real conflict.** Every mature
> TLS stack compiles C (`ring` and `aws-lc-rs` both), so an HTTPS client inside our graph
> is not available *at all* under requirement (3). Rather than quietly relax the
> requirement, the fetch uses the system's `curl` or `wget` — exactly as accelerator
> detection uses the system's `nvidia-smi`: a runtime tool, not a build dependency. The
> guarantee does not depend on it, because **the transport is not trusted; the hash is.**
> `file://` needs no fetcher at all, which is what makes the whole test group hermetic.
>
> Two policy distinctions that the spec forced into the open:
> * **`--offline` is about the network, `--no-install` is about doing nothing.** Offline
>   still installs a `file://` artifact; conflating them makes `--offline` mean "do less
>   work", which is not what anyone reaches for it to mean.
> * **Every shipped manifest entry is either barriered or unpinned, deliberately.** CUDA,
>   CANN and Xcode carry a barrier and a remedy; the `rustc_codegen_tile` entries carry a
>   placeholder hash and a note saying so, because the per-arch release inventory has not
>   been audited. An entry that *looks* pinned but is not is worse than one that admits it,
>   so `pinned()` returns false and the tool refuses to download rather than fetching
>   something it cannot verify.

### M4 (original scope) — Provisioning engine
**Touches:** `src/provision.rs`, `assets/provision.toml` (id, version, per-(os,arch) URL,
sha256, unpack rule, verify command), `~/.tile-rs/` with a **`layout-version` file**.
**Flags:** `tile install <id>`, `--offline`, `--no-install`, `--yes`.
**Reqs:** (1), (6).
**Audit first, before committing the milestone:** a rustc codegen backend has **no stable
ABI**, so the `.so` and the nightly must be provisioned as **one pinned unit**. Inventory the
`codegen-release.yml` artifacts now — if there is no linux-aarch64 or armv7 `.so`, say so
here rather than discovering it in M9.
**Exit:** `tile install llvm-20` twice — installs once, second run is a no-op; a corrupted
archive is rejected before unpacking and the partial file removed; two concurrent installs
of the same id produce one verified install; `--offline` makes no network call; a
EULA/root/login-gated toolchain refuses with the exact command; feature `06` green.
**Not:** vendor SDK installation, ever.

### M5 — The knowledge base ✅ **DONE**

> **Delivered.** Two stores, one storage layer: `~/.tile-rs/attempts` (yours, never gated)
> and a sealed corpus snapshot (licensed). Both use the same tab-separated line format,
> which diffs, greps, and cannot grow a query engine by accident. `-s` reports prior
> attempts for the kernels **the profile found in the file**, not for its filename.
>
> **Not SQLite, and that was the whole storage decision.** `rusqlite` compiles C and would
> fail requirement (3)'s gate on day one; encrypting it (sqlcipher) makes it worse. The
> `stats` feature adds 26 crates, verified C-free — the default binary stays at 2.
>
> **The doctrine is enforced, not documented.** `headroom` NULL prints `unassessed` and
> `0.0` prints `closed`, and a test asserts the two cells are not the same text. A record
> carrying a headroom with **no stated basis is refused on the way in** — a number whose
> provenance nobody recorded is indistinguishable from a guess. Every report states the
> snapshot's export timestamp and revision, because answering from month-old measurements
> without saying so breaks the same rule as inventing one.
>
> **The gate is honest about itself.** A license check on a file on the user's disk is
> decoration, and so is encrypting it and shipping the key beside it — a licensed user can
> dump the corpus through `-s` itself. So the module's own doc says the gate is on
> *delivery*, and what it genuinely buys is stated: the corpus is sealed with
> XChaCha20-Poly1305 so it is not one `strings` away and a tampered snapshot fails whole;
> a token is unforgeable (Ed25519, verified offline against a compiled-in key). **The
> shipped public key is all zeros** — this repository does not hold the private half, so
> the build fails closed rather than looking like it verified something. And expiry is
> checked *after* the signature: a forged expired token is a forgery, not an expiry, and
> telling its holder to renew would be misleading.

### M5 (original scope) — The knowledge base: `-s`, and the storage decision
**The blocker to settle first (R7):** `rusqlite` compiles C and would fail our own
requirement-(3) gate on day one. Options, in preference order:
1. **Export at release time** to a serialized pure-Rust format (the corpus is read-only to
   this tool), encrypted with `chacha20poly1305` from RustCrypto. Pure Rust, small, and it
   makes the license gate *mechanical* rather than decorative.
2. A pure-Rust SQLite reader (`limbo`), if the schema must stay live.
3. Keep SQLite behind a non-default `sqlite` feature for dev use only, with the plaintext
   `$TILE_KERNEL_IMPACT_DB (developer backdoor)` path as an env-var-gated developer backdoor.
**Two databases:** `~/.tile-rs/attempts.db` (the user's own, never gated, written by O4) and
the licensed corpus (read-only, gated).
**Doctrine:** `headroom IS NULL` prints **unassessed**, `0.0` prints **closed**; no number is
ever written that was not measured; no dispatch share is written by the conversion path.
**Flags:** `-s`, `tile license status`.
**Exit:** `tile testdata/forms/softmax.rs -s` prints prior attempts matched by content sha256
or alias; unlicensed, the local attempts still print and the conversion still succeeds; an
expired/tampered/unknown-key token fails with exit 5 naming the defect; feature `07` green.

### M6 — `.rs` inputs ✅ **DONE** — the release inventory was the blocker, and it resolved

> **`tile softmax.rs -o k.metal` produces Metal that `xcrun metal` compiles.** The front
> door works. `gh release list` showed one release with one asset — aarch64-apple-darwin —
> so the manifest now carries its real digest
> (`231f85f1…`, 97 MB) and **no entry at all for the platforms with no published asset**: a
> URL that 404s is worse than an honest "no manifest entry here".
>
> `tile install rustc_codegen_tile` downloads it, verifies it with our own SHA-256 against
> the published digest, and unpacks it — **verified before it is opened**, because an
> archive is an instruction set for writing files.
>
> **Three ways this fails silently, now refused before the build:**
> * **`RUSTFLAGS` set.** Cargo replaces `build.rustflags` wholesale rather than merging, so
>   the backend never loads and the kernel builds with the stock LLVM backend — quietly,
>   with no error and no emitted source. The release notes warn about it; this refuses
>   rather than letting you discover it after a two-minute build.
> * **`--target` omitted.** Without it the backend flag reaches build scripts and
>   proc-macro dependencies, which fail with `found invalid metadata files for crate core`
>   — a message about `core` for a mistake about scope. Undocumented; cost a debugging
>   session; now always passed.
> * **`TILERS_CODEGEN_PATH` given a form id.** `msl` is not `metal`, `gpu` is not `cuda`,
>   `spirv` is not `vulkan` — and an unrecognised value is not rejected, it falls through
>   to the Ascend default and fails with `Could not determine ASCEND_HOME_PATH`.
>   `forms::codegen_path` is the translation, written down so nobody discovers it twice.
>
> `testdata/forms/softmax.rs` is now a **real kernel**, not a sketch — and its own trap
> comment ("safe for it to mention `__global__` and `kernel void`") caught the profiler
> reading a kernel called `in` out of "kernel void in a comment". Fourth appearance of
> reading code out of prose.

### M6 (original scope) — `.rs` inputs over the provisioner
**Touches:** `src/lower_rs.rs` (subprocess rustc with `TILERS_CODEGEN_SO` +
`TILERS_CODEGEN_PATH`, capturing the MLIR intermediate), routes edges `tile → mlir → *`
tagged `needs: rustc-so`.
**Exit:** on a machine with no `.so`, `tile testdata/forms/softmax.rs -o out.cu` provisions
the pinned (nightly, `.so`) unit, reports the sha256, and emits CUDA matching a golden file;
the second run provisions nothing; with `--no-install` it refuses with the one command that
fixes it.

### M7 — `-O2`/`-O3` ✅ **DONE**

> **Delivered.** `src/mlir.rs` is a real IR — operations with results, callees, operands
> and types — and its load-bearing property is that **parse-then-print is byte-identical**:
> an unrecognised line is kept verbatim and an untouched operation prints its original
> text, so the layer can only change what a pass deliberately changed and the 15 golden
> files survive a parser refactor. `src/passes.rs` holds the pipeline: dead-op elimination
> (O1), constant dedup, CSE and redundant-load reuse (O2), plus the tile-rs-specific
> bounds check. `src/toolchain.rs` is O3, and on this Mac it is a real answer today —
> `xcrun metal` compiles the MSL this tool just emitted and accepts it.
>
> **The plan's "fusion" pass does not exist, and must not.** The emitters own fusion and
> find it *by adjacency*: `mlir_to_tpu.rs` remembers the previous operation and fuses SiLU
> into a following multiply. Emitting a `__tile_silu_mul_f32` here would produce an
> intrinsic no emitter's vocabulary contains — turning a kernel that lowers into one that
> does not. So what this layer owes fusion is **not to break it**: every pipeline ends with
> `fusion_pairs_intact`, and if a pass ever separates a producer from its consumer the
> result is **the unoptimized input**, loudly. An unoptimized kernel beats a quietly
> deoptimized one.
>
> **Bounds are checked, not rewritten.** Re-chunking a tile changes what the kernel
> computes unless it loops, and this layer cannot know that it does. Reporting a tiling the
> hardware cannot hold is the useful and the honest half — and on an unmeasured
> architecture the check *refuses*, which is `f10513c`'s rule reaching the optimizer.
>
> Two bugs the tests caught: **loads were classified pure**, so CSE collapsed a load across
> a store and read stale data; and the cfg-gate scanner **read `cfg(target_os)` out of a
> comment about cfg gates** — the same "read past the prose" mistake dead-op elimination
> made, now fixed in three places including the tagging script.

### M7 (original scope) — `-O2`/`-O3`
**The honest scope:** O1/O2 need an **MLIR→MLIR rewrite layer that does not exist** in the
open tree — `mlir_parse.rs` is a parse-for-emission helper, and the fusion and
merge-over-scatter doctrines live *inside* lowering, not as reusable passes. Building that
layer is the milestone; the levels are its surface. O3 adds toolchain feedback.
**Exit:** each O2 rewrite has a before/after golden pair; `-O2` output is byte-identical
across runs; an unmeasured architecture blocks O2 with D4's refusal rather than borrowing
another arch's bounds; feature `04` green except `@requires-device`.

### M8 — Daemon + MCP ✅ **DONE**

> **Delivered.** `tile -d` serves MCP over stdio: `initialize`, `tools/list`, `tools/call`,
> `ping`, with seven tools — `doctor`, `list_forms`, `identify`, `profile`, `routes`,
> `convert`, `install`. **No async runtime and no dependencies**: JSON-RPC over stdin is a
> request/response loop, and a blocking read is exactly the right shape for it, so D4's
> synchronous core holds. JSON is ~250 hand-written lines rather than a dozen crates
> against a fifteen-crate budget — strict, and refusing malformed input rather than
> guessing, because the thing on the other end is an agent.
>
> Three rules that only exist in daemon mode:
> * **stdout belongs to the protocol.** A test drives `-d -VVV` and asserts every stdout
>   line parses as JSON, because a well-meaning log line there looks like a parser bug on
>   the client's side.
> * **Nothing is acquired on an agent's behalf.** A `convert` needing a toolchain refuses
>   and names the `install` tool the client may call explicitly. An MCP client asking a
>   question must not be able to start a multi-gigabyte download.
> * **stdio only.** `--listen` is refused by the parser, so there is no code path that
>   binds — "binds localhost" is not a security story for a server that will one day hand
>   out license-gated data.
>
> Parity is checked, not asserted: an end-to-end test runs `convert` over MCP and the
> equivalent CLI invocation and compares the bytes.

### M8 (original scope) — Daemon + MCP
**Touches:** `src/daemon/{mod,mcp}.rs`. stdio only unless `--listen` + token (R11).
**Reqs:** (7).
**Exit:** `tools/list` shows the verbs; an MCP `convert` and the equivalent CLI run produce
byte-identical output; `-VVV` in stdio mode writes nothing to stdout; **no toolchain is
auto-installed on an agent's request**; SIGKILL leaves no orphan scratch; feature `08` green.

### M9 — `-m` engine mapping ✅ **the unblocked half**

> **Delivered.** `engine_for` maps each accelerator family to its `ds4-rs-*` engine,
> `locate` searches `TILE_ENGINE_PATH`, `$HOME` and a sibling directory, and a checkout is
> one only if it has a manifest. `-m` requires `-d`, because an engine provisioned beside
> nothing is an engine nobody can reach. A failure to provision **never costs the kernel
> tools**: the daemon reports it and serves on.
>
> **Blocked:** cloning, building and supervising an engine needs the sibling repositories
> and a machine with the accelerator. A supervisor whose failure modes nobody has watched
> is worse than none, so the refusal names exactly what is missing instead. Note that
> `amd-gpu` maps to `ds4-rs-amd` even though tile-rs has no ROCm *codegen* target — the
> engine and the backend are different questions.
>
> **Also blocked:** `-O4`, which needs real hardware to measure on.

### M9 (original scope) — `-O4` measured  ∥  `-m` ds4-rs supervision
**O4:** candidates are run on the native device, each attempt recorded in `attempts.db` with
its measurement; absent a device it fails with exit 6 and offers O3; under `--cross` it is a
usage error. **`-m`:** map family → engine (`amd-gpu` → `ds4-rs-amd`), locate or fetch,
build, launch, health-check, supervise; `-m` without `-d` is a usage error.
**Reqs:** (8).
**Exit:** `tile_model_status` returns a JSON diff against a golden shape; `tile -d` without
`-m` starts no model process.

### M10 — UI ✅ **the server half**

> **Delivered.** `-u` builds the view from the core, binds an ephemeral **loopback** port,
> prints the address on stdout so it can be piped, opens a browser where there is one, and
> says so where there is not. The page is self-contained — no external host, so a viewer
> with no connectivity sees everything — and kernel source is escaped, because a kernel is
> untrusted text.
>
> **Why a server before a window.** The server half has to exist either way (the design
> already said the bundle is served on loopback), it is the half a headless box can
> **test** — CI can fetch a page and assert what is on it; it cannot look at a window — and
> it has no build dependency. An egui-wasm bundle needs a wasm toolchain, which is a build
> prerequisite, and M4's rule is that nothing is acquired that cannot be verified. The
> bundle slots in behind this same server as a static asset.
>
> Two scenarios stay `@planned`: `--ui native` (eframe) and serving the wasm bundle.

### M10b — Release packaging ✅ **DONE**

> **Delivered.** `.github/workflows/tile-release.yml` builds `tile` for four triples with
> `--features stats`, runs the tests **on what is being shipped**, checks the binary
> actually starts and converts a kernel on the machine it was built for, then packages it
> with a sha256 — the same checksum discipline `tile install` applies to what it
> downloads.
>
> **Tagged `tile-v*`, not `v<x>+nightly-<date>`.** `codegen-release.yml` encodes the
> nightly in its tag because a `.so` built against one rustc does not load into another.
> `tile` has no such pin, so inheriting that scheme would tell users a version number
> means something it does not.
>
> **A release carries every feature.** The corpus reader and the license verifier are what
> a licence holder is buying; a release that silently lacked them would report "this build
> has no corpus support" to someone who paid.
>
> **Not done, and not doable here:** teaching `scripts/install.sh` to install `tile`. Its
> own header says it is deployed from the private repo at `ci/tile-rs-public/install.sh`
> and must be edited there. This workflow publishes the assets that installer would fetch.

### M10 (original scope) — UI, packaging, release
`-u` serves the egui-wasm bundle on loopback and opens a browser; `--ui native` uses eframe;
headless prints the URL (exit 0) rather than failing; a build without the `ui` feature says
so (exit 7). Then release binaries per triple, `scripts/install.sh` integration keeping
`install-smoke.yml` green, and the armv7 cross build. **`tile` versions independently of the
codegen backend** — it must not inherit `codegen-release.yml`'s `v<x>+nightly-<date>` scheme.
Features `09` green.

---

## Sequencing

```
M0 ─► M1 ─► M2 ─► M3 ─┬─► M4 ─► M6 ─► M7 ─► M8 ─┬─► M9 ─► M10
                      └─► M5 ──────────────────┘
```
Hard blocks: M2 needs M1's taxonomy · M3 needs M2's edges · M6 needs M4 · M7 needs M6 for
`.rs` inputs (its MLIR core needs only M3) · M8 needs M5 for license parity over MCP ·
M9's O4 needs M5 and a device · M10's stats panel needs M5.
Parallel: **M4 ∥ M5** genuinely (disjoint files, disjoint risks). M9's two halves are
independent of each other.

## Flag → milestone (no orphans)

| M1 | `-h` `-v` `-V` `-f` `-i` `--json` `doctor` `--list-forms` |
| M2 | `-t` `-o` `-` `-o -` `--force` |
| M3 | `--route` `--via` `-k` `--keep-dir` `-O0` `-O1` |
| M4 | `install` `--offline` `--no-install` `--yes` |
| M5 | `-s` `license status` |
| M7 | `-O2` `-O3` |
| M8 | `-d` `--listen` |
| M9 | `-O4` `-m` `--cross` |
| M10 | `-u` `--ui native` `--ui-port` |

## Requirement → milestone

(1) M4 · (2) M1, enforced by the M1 CI matrix · (3) M1 gate + **M5's storage decision, which
is where it is actually at risk** · (4) M1 · (5) M2 · (6) M4 · (7) M8 · (8) M9.

## What is still missing, and is now owned

* **Golden/snapshot suite** — M2, in root CI, doubling as the emitter-drift alarm.
* **Error messages as a deliverable** — for most form pairs the refusal *is* the product.
  Every refusal in features `02`, `06`, `11` is golden-tested. Owned by M3.
* **`~/.tile-rs` layout version + migration** — M4.
* **Telemetry stance** — one line in `--help` from M1: no telemetry; downloads only from the
  pinned manifest, and every fetch is printed before it starts.
* **Adding the 17th backend** — `docs/cli/adding-a-backend.md` plus a test that fails when
  the registry names, the form table, the include list and the provision manifest diverge.
  Owned by M2.
* **Form-id compatibility test** — M1.
* **`cargo publish` of `tile_cli` is permanently impossible** with out-of-package `#[path]`
  sources. Recorded, accepted; the binary ships as a release artifact.

## Decisions taken (no longer open)

All three questions the grills left hanging are now answered, and every milestone above
reflects the answers:

1. **The default write** (R6) — `tile softmax.rs` profiles, optimizes, and writes
   `softmax.opt.metal` beside the input, printing the path last. `--force` is required to
   replace an existing file, for the derived name and for an explicit `-o` alike.
2. **The corpus** (R7) — exported at release time to a sealed pure-Rust blob. M5's first
   task is the exporter and the reader, not a database binding. `~/.tile-rs/attempts.db`
   (the user's own records, never gated) uses the same serialized format, so there is one
   storage layer, not two.
3. **Pure Rust** (R13) — no dependency compiles C. The `cargo deny` build-graph gate lands
   in M1, before there is anything to grandfather in, and it is what makes the claim
   checkable rather than remembered.

The one consequence worth stating plainly: because the corpus is a **snapshot**, `-s` can
be stale relative to `$TILE_KERNEL_IMPACT_DB (developer backdoor)`. The snapshot therefore carries
its export timestamp and source revision, and `-s` prints both. A tool that quietly answers
from month-old measurements would violate the same doctrine as inventing a headroom.
