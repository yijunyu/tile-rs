# `tile` — the tile-rs kernel swissknife

> Status: **design + executable-spec draft**. The `.feature` files under
> `docs/cli/features/` are the normative requirement. They move to
> `crates/tile_cli/features/` (with a step harness mirroring `crates/tile_spec`)
> in Milestone 1, at which point they become executable and CI-enforced.

`tile` is to accelerator kernels what `pandoc` is to documents: one command that
reads a kernel in *some* representation, works out what it is, works out what
you asked for, and produces it — lowering, lifting or optimizing as required.

```
tile softmax.rs -o softmax.metal          # lower: tile-rs -> Metal
tile softmax.rs                           # profile + optimize for THIS machine
tile softmax.rs -i                        # profile only
tile softmax.mlir -t rvv --route          # what routes exist, and what they cost
tile softmax.rs -O3 -o softmax.opt.rs     # optimize, with toolchain feedback
tile softmax.metal -o softmax.rs          # lift: refuses today, see §2a
tile -d -m qwen3                          # daemon + MCP + provisioned LLM
```

Read §2a before the lift example raises expectations: tile-rs today has **1.5 readers
and 16 writers**. The tool is a fan-out with pandoc's grammar, not yet pandoc's
capability, and the design says so out loud rather than letting the first user discover
it as a wall of refusals.

---

## 1. The form taxonomy

A **form** is a kernel representation. It is identified by a stable id, not by a
file extension — because extensions collide badly in this domain.

| form | ext | family | discriminating magic |
|---|---|---|---|
| `tile` | `.rs` | source | **reserved** — `.rs` is *always* tile-rs |
| `mlir` | `.mlir` | IR | generic; dialect decides the refinement below |
| `linalg` | `.mlir` | IR | contains `linalg.` |
| `pto` | `.mlir` | IR (closed) | PTO module header |
| `rvv` | `.mlir` | IR | `riscv64` triple stamp |
| `msl` | `.metal` | target src | `kernel void` |
| `gpu` | `.cu` | target src | `__global__` |
| `musa` | `.mu` | target src | `musa_runtime.h` |
| `spirv` | `.comp` | target src | `layout(set = 0` |
| `nki` | `.py` | target src | `@nki.jit` |
| `aie` | `.py` | target src | `from aie.iron` |
| `tpu` | `.py` | target src | `pallas` |
| `bang` | `.mlu` | target src | `__mlu_entry__` |
| `gaudi` | `.c` | target src | `tpc-clang` |
| `hexagon` | `.c` | target src | `hvx_` |
| `ttmetal` | `.cpp` | target src | `void MAIN` |
| `cpp` | `.cce`,`.cpp` | target src (closed) | `AscendC::` / `__aicore__` |
| `csl` | `.csl` | target src | `comptime` |
| `debug` | `.mlir.txt` | diagnostic | tile-rs debug banner |

Four extensions are overloaded — `.py` by 3 forms, `.mlir` by 4, `.c` by 2,
`.cpp` by 2. **Extension alone is therefore not a decision procedure**, which is
exactly why the pandoc analogy needs both halves of pandoc's interface:

* `-f, --from <form>` — force the input form (**added to the requirement**; the
  requirement named only `-t`, but sniffing a hand-written `.py` is a heuristic
  and heuristics need an override).
* `-t, --to <form>` — force the output form.

Resolution order for an input: **explicit `-f`** → **magic sniff** →
**extension** → error listing the candidate forms. For an output: **explicit
`-t`** → **extension of `-o`** → **native target of the detected platform**.

The magic table is data, not code: one `FormSpec` row per form, so adding a
17th backend is a table entry, mirroring how `TargetRegistry::register` makes
adding a codegen target a one-line change.

## 2. What the tool does is a function of (in-form, out-form)

| relation | transformation | example |
|---|---|---|
| in ≠ out, in higher | **lowering** | `tile` → `msl` |
| in ≠ out, out higher | **lifting** | `msl` → `tile` |
| in = out | **optimization** | `tile` → `tile`, `pto` → `pto` |
| out omitted, single input | **profile + optimize** for the detected platform | |

These are edges of one **transformation graph**. Nodes are forms; edges carry a
kind (`lower`/`lift`/`optimize`), a set of *capability requirements*
(`toolchain:<id>`, `device:<family>`, `license`), and a cost. Route selection is
a shortest-path over the subgraph whose capability requirements are satisfiable
under the current flags. Consequences that fall out for free:

* **Multiple routes** between the same pair are natural (`tile → mlir → linalg →
  rvv` vs `tile → mlir → rvv`). `--route` prints them; `--via a,b` pins one.
* **Intermediates are real files** on the route, written to a scratch dir and
  deleted on exit unless `-k/--keep`.
* An impossible request fails with *"no route from X to Y"* plus the nearest
  routes and what capability each one is missing — never a silent no-op.

## 2a. Fidelity, and what the graph can actually do

Pandoc's formats round-trip through one shared document model. `tile -> mlir -> msl` is a
**monotone descent**: each hop discards what the level above knew. So lifting is not the
reverse edge with a different cost — it is program synthesis, and it exists today for
almost nothing. Two rules follow, and both are load-bearing:

* **Lifts do not auto-compose into lowering routes.** `msl -> cpp` is refused even when an
  `msl -> tile` lifter exists; the composed route must be named with `--via tile`. A
  synthesised intermediate silently feeding a lowering is the invisible-wrongness this
  tool exists to prevent.
* **Same-form optimization exists only for the forms tile-rs can read** — `tile` and the
  mlir family. `tile k.metal -o k2.metal` would mean writing a Metal frontend; it refuses
  with exit 3 and names the forms where optimization does exist.

Because a mangled heading is visible and a numerically wrong kernel is not, every edge
carries a **fidelity class**, printed on every conversion and in `--route`:

| class | meaning |
|---|---|
| `exact` | the mechanically-checked emit path (the generality matrix + emit-purity contract) |
| `validated` | additionally checked against a CPU reference on that hardware, **and it says where that check is recorded** — `(docs/cli/INTEGRATION.md)` for a run this repo holds, `— inherited, no run recorded here` for a claim it does not |
| `unvalidated` | emitted correctly, never run on that target |
| `synthesised` | produced by a lift; needs review before use |

`route: tile -> mlir -> msl [exact, validated on M2 Max]` as the last line of every run is
worth more than anything else the tool prints. And `tile --list-forms` prints the full
reader/writer matrix, because a sparse graph is fine but a sparse *undiscoverable* graph
is what makes a tool feel broken.

## 3. `-O` levels

The dividing line between levels is **what the level is allowed to touch**, so a
user can predict cost and reproducibility from the number alone.

| level | may use | deterministic | needs |
|---|---|---|---|
| `-O0` | nothing; verbatim translation | yes | — |
| `-O1` | local/peephole rewrites | **byte-identical** | — |
| `-O2` | + fusion, tiling plans, buffer reuse, resource-bound checks (`plan_tiles`, `pointwise`, `hazard`) | **byte-identical** | — |
| `-O3` | + the target toolchain (compile, resource/asm feedback, may auto-install) | no | toolchain |
| `-O4` | + measured autotuning on real hardware; records attempts | no | device + license |

**Default is `-O2`** — the highest level that needs no network, no toolchain, no
device, and no clock. `-O` bare means `-O2`. `-O1`/`-O2` inherit the repo's
existing purity contract (`backend_emit_purity.feature`): same input, byte-identical
output, forever.

## 4. Platform, accelerators, and the default target

`tile doctor` (and the header of any verbose run) reports: OS/arch/libc; each
accelerator family found (Apple GPU, NVIDIA, AMD, Ascend, Trainium, …), *how* it
was detected; and whether that family's SDK/compiler is present and its version.

Detection is runtime `dlopen` of vendor libraries plus vendor CLI probes — never
a build-time link. This is what keeps requirement (3) "pure Rust" honest: **no
C/C++ compiles in our build graph**; vendor `.so`/`.dylib` are loaded at runtime
if and only if present.

Accelerator families are named precisely, because `amd` is two different machines: an
**`amd-gpu`** (ROCm/HIP) has no tile-rs target at all and falls back to `linalg` with an
explanation, while an **`amd-npu`** (Ryzen AI) gets `aie`. Defaulting a Radeon to IRON
Python would emit NPU source for a GPU.

The **default output target is the detected native one**, so what comes out can
be run here. With no accelerator present the default is `linalg` → CPU, which is
always runnable — the default therefore never produces a dead artifact.
`--cross <platform>` (or naming a non-native `-t`) opts into cross generation
and suppresses any run/measure step.

## 5. Toolchains: on demand, never a scavenger hunt

Toolchain acquisition is manifest-driven: id, version, per-(os,arch) URL, sha256,
unpack rule, install prefix, and a *verification command*. Rules:

* Installs are **user-scoped** (`~/.tile-rs/toolchains/<id>/<version>`), never
  root, never onto system paths, always sha256-verified before unpacking.
* Acquisition happens **only when a route needs it** (or ahead of time via
  `tile install <id>`), never all at once.
* Anything demanding root, a EULA click, or a vendor login (CUDA, CANN, Xcode)
  is **not** silently installed. It is refused with the exact command to run and
  the exact URL — which is what requirement (6) actually asks for. Pretending
  otherwise would be a lie the tool cannot keep.
* `--no-install` / `--offline` / `--yes` control the policy. **In daemon mode
  auto-install is off by default**: an MCP client should not be able to trigger
  a multi-gigabyte download by asking a question.

## 6. The knowledge base and the license gate

Two databases, deliberately:

| store | path | contents | gate |
|---|---|---|---|
| local | `~/.tile-rs/attempts` | *your* conversions and measurements | none |
| knowledge | `corpus.sealed` | the curated cross-target corpus, exported and sealed at release time | **license** |

Both use the same serialized pure-Rust format, so there is one storage layer. The corpus
is a **snapshot**, not a live database: it carries its export timestamp and source
revision, and `-s` prints both, because answering from month-old measurements without
saying so would violate the same doctrine as inventing a headroom.

`-s/--stats` joins both: prior attempts on the kernels named in the arguments,
their targets, states, headroom and outcomes. Unlicensed, `-s` reports only the
local db and says in one line what the licensed corpus would add and how to get
a key — and **the conversion still runs**. A missing license degrades a feature;
it never blocks the tool.

The license is an offline Ed25519-signed token (subject, expiry, feature set),
verified against an embedded public key. **A license check on a plaintext file on the
user's disk is decoration** — but so is encrypting a file and shipping the key beside it
in the token: a licensed user can dump the whole corpus through `-s` itself, and offline
expiry is unenforceable against a copy already decrypted. So the gate is on **delivery,
not the file**: the license fetches signed corpus snapshots, and anything that must
genuinely stay private is answered server-side rather than shipped. What remains
client-side is honest product segmentation, and the design says so rather than pretending
otherwise.

The corpus is **not SQLite in the shipped product**. `rusqlite` pulls `libsqlite3-sys`,
which compiles C and breaks requirement (3) and our own CI gate; adding encryption on top
(sqlcipher) makes it worse. The corpus is exported at release time to a serialized
pure-Rust format, sealed with RustCrypto's `chacha20poly1305`.

Doctrine inherited from `kernel-impact`: `headroom IS NULL` means UNASSESSED and
`0.0` means CLOSED; the tool never writes a headroom or dispatch share it did not
measure.

## 7. Shape of the program

```
crates/tile_cli/            EXCLUDED from the root workspace, with its own
                            rust-toolchain.toml pinning STABLE.
  src/lib.rs                the crate root -- and the only legal home for the
                            #[path] includes of rustc_codegen_tile's emitters,
                            which use absolute `crate::` paths.
  src/forms.rs              form table, magic, levels, fidelity
  src/routes.rs             the transformation graph + planner
  src/profile.rs            the -i report, over tile_codegen default features
  src/optimize.rs           the -O pipeline
  src/platform/             THE ONLY cfg-gated module
  src/provision.rs          toolchain manifest, download, verify, install
  src/impactdb.rs           corpus reader          src/license.rs
  src/daemon/               MCP; the only place tokio appears
  src/ui/                   egui; feature `ui`, off by default
  features/                 the executable spec, run by a harness reusing
                            tile_spec::gherkin
  testdata/forms/           one specimen per form -- the repo has no examples/
  testdata/golden/          byte-exact outputs; also the emitter-drift alarm
```

**Why excluded, and why its own toolchain file.** The root `rust-toolchain.toml` pins
`nightly-2025-08-04` — deliberately, because `tile_std` is `#![no_core]` and only that
nightly's matching codegen backend satisfies its intrinsics — and rustup resolves that
file by walking *up* from the working directory, so it reaches every subdirectory,
excluded crates included. A CLI that must build on stable across six triples cannot
inherit a frozen nightly, and as a root member `cargo build` at the root would try to
build `tile_std` and ICE. Excluded plus its own stable pin is the only arrangement in
which both claims survive.

One core, three front-ends — requirement (7) is a consequence of the layering
rather than a second implementation. Platform specifics live behind
`cfg(target_os/target_arch)` inside `src/platform/` only; every other module is
portable (requirement 2).

**Dependency budget.** This repo's entire dependency graph is 18 crates, and `tile_spec`
hand-rolled a Gherkin parser rather than take one dependency. The CLI naively wants argv,
SQLite, async, MCP, HTTP, egui, an unpacker, TLS, Ed25519 and an AEAD — 400+ crates, in a
repo whose reviewers refused *one*. So every dependency sits behind a feature, and the
default binary — forms, routes, emitters, profile, doctor — targets **under 15 crates**.
`stats` adds the corpus reader, `net` the downloader, `daemon` the async runtime, `ui`
egui. Nobody who wants `tile k.mlir -o k.metal` pays for any of them.

**Pure Rust, stated precisely.** No crate in `tile`'s build graph runs a C or C++
compiler, and nothing vendor-specific is linked at build time. Vendor runtimes are
`dlopen`ed at *runtime*, if and only if they are present. That distinction is what makes
requirement (3) achievable rather than aspirational — and it is what rules out
`rusqlite`.

`-u/--ui` builds the wasm bundle path: serve the prebuilt egui-wasm bundle on a
loopback port and open the browser; `--ui native` uses eframe directly. On a
headless box it prints the URL instead of failing.

`-d/--daemon` speaks MCP over stdio (default) or `--listen <addr>`. `-m/--model`
provisions a matching `ds4-rs-{metal,cuda,amd,ascend}` engine beside the daemon
and **requires** `-d`.

## 8. Option grammar

```
tile [OPTIONS] <INPUT>...

  -o <FILE>         explicit output path (wins over any extension inference)
                    "-" reads stdin (needs -f); "-o -" writes the result to stdout
      --force       overwrite an existing output (never implicit)
      --list-forms  print the reader/writer/optimize/lift matrix
  -f, --from <FORM> force input form         -t, --to <FORM>  force output form
  -O[0-4]           optimization level (bare -O = -O2; default -O2)
  -i, --info-only   profile and report; perform no transformation
  -k, --keep        keep intermediate representations
      --keep-dir D  where to keep them (implies -k)
      --route       print the candidate routes and exit
      --via a,b     pin the intermediate forms
      --cross P     generate for a non-native platform
  -s, --stats       prior attempts/statistics for the kernels in the arguments
  -u, --ui          open the egui UI       -d, --daemon   MCP daemon mode
  -m, --model <M>   provision an LLM engine alongside the daemon (requires -d)
      --no-install / --offline / --yes     toolchain acquisition policy
  -V, --verbose     repeatable (-VV, -VVV)
  -v, --version     -h, --help
```

Note the deliberate inversion of the common Unix convention: here `-v` is
*version* and `-V` is *verbose*, as specified. Both long forms are unambiguous.

### `-r`: running the kernel, and what a speedup is measured against

Emitting a kernel proves it compiles. Running it proves it computes the right answer, and
is the only way to say anything about speed that is not a guess. `-r` compiles the emitted
source at runtime, dispatches it, checks the numbers, and times it.

```
run: Apple M1 Ultra — softmax over 1024 elements
  accuracy:
    kernel vs torch 2.4.1     max abs 6.5e-9  max rel 1.9e-6  rmse 2.4e-9
    kernel vs ours            max abs 6.5e-9  max rel 1.9e-6  rmse 2.4e-9
    ours   vs torch           max abs 1.2e-9  max rel 3.0e-7  rmse 8.0e-10
    verdict: all three agree
  target   : median 17.75 us over 50 runs (min 17.62, max 18.00, 5 warmup)
  reference: median 29.42 us over 50 runs (min 26.17, max 35.83, 5 warmup)
  speedup  : 1.7x against the reference
             (the reference is a naive single-threaded scalar loop written
              by this tool — this is NOT a GPU-versus-CPU figure)
```

**Three sources, not two.** A kernel and a reference from the same repository can agree
perfectly and both be wrong. PyTorch on the CPU is an outside opinion, and with three
sources a disagreement becomes *attributable*:

| kernel vs torch | ours vs torch | what it means |
|---|---|---|
| close | close | everything agrees |
| far | close | **the kernel is wrong** — the lowering |
| far | far | **our reference is wrong**, and the kernel inherited it |
| close | far | our reference is wrong and the kernel does not follow it |

Two sources cannot tell the second row from the third, and blaming a lowering for a bad
reference is a long afternoon.

**Getting torch takes no second command.** Three ways, in order of what they cost:
a `python3` that already has it; `uv run --with torch`, which resolves and caches torch
itself so there is no virtualenv to create and nothing for anyone to type; or **uv itself
provisioned first**, from the same pinned manifest as every other toolchain. Not
`curl -LsSf https://astral.sh/uv/install.sh | sh` — that pipe is unverified, runs whatever
the server sends, and installs wherever it likes. uv arrives as a checksummed tarball into
`~/.tile-rs/toolchains/uv/<version>`, announced before it is fetched and obeying
`--offline` and `--no-install` like everything else. Its absence still degrades the report
to the two-way comparison and **says so**, rather than silently dropping the stronger claim
and leaving the weaker one looking authoritative.

**What the numbers are, stated every time.** Four rules keep a measurement from becoming a
claim it cannot support:

* **Never report a ratio without both measurements.** When `-O0` and the chosen level
  produce identical source, the line is "identical source, nothing to compare" — not
  "1.00x", which implies a measurement that came out even.
* **Every timing carries its sample count and spread.** A median with no spread is a
  number with no error bar. Device time comes from the command buffer's own
  `GPUStartTime`/`GPUEndTime`; wall clock would fold in encoding and scheduling and report
  them as kernel cost.
* **Say what the baseline is, every time it is quoted.** The reference is a naive
  single-threaded scalar loop written by this tool. A tuned, vectorised, multi-threaded CPU
  implementation would be far faster, so this is *not* a GPU-versus-CPU figure — and the
  report says so on the line below the number, not in a footnote.
* **Do not handicap the baseline.** The reference allocated a scratch row inside the timed
  loop at first, which measured the allocator and called it the reference's cost. Fixing it
  moved a reported speedup from 1.9x to 1.7x. A test now asserts nothing allocates between
  the clock starting and stopping.

A kernel whose operation has no reference is **refused rather than run**: numbers with
nothing to compare them against are worse than no numbers. So is one using two
referenceable operations, because the reference would have to guess the order they compose
in and would then fail for the wrong reason.

### Testing the absences

Almost every interesting failure here is an *absence* — no emitter compiled in, no
toolchain, no vendor SDK, no accelerator, no network, no license. None of those paths run
on a developer's machine, because a developer's machine has everything. So they rot, and
the first person to meet them is a user on a bare box.

`TILE_SIMULATE` makes absence a value:

```
TILE_SIMULATE=no-emitters   tile k.mlir -t msl -o out.metal   # exit 4
TILE_SIMULATE=bare          tile doctor                        # the linalg fallback
```

The vocabulary is `no-emitters`, `no-toolchains`, `no-devices`, `no-vendor-clis`,
`no-network`, `no-license`, and `bare` for all of them. Two rules keep it from becoming a
bug generator of its own:

* **A spec can only ever remove.** Letting it grant a capability would let a test claim
  something the machine lacks, and the run would fail somewhere less honest.
* **A simulated run announces itself**, on stderr, every time. Someone will leave the
  variable set; without the banner they meet a refusal they cannot explain and file it
  against the wrong component.

Where the OS can take something away for real, the tests use that instead of the flag —
an empty `PATH` genuinely hides every vendor CLI, and a redirected `HOME` genuinely has
nothing provisioned. And the emitters get a third treatment: CI builds
`--no-default-features` for real, because a simulation of a build feature is only as good
as its fidelity to the build. That build behaves identically to the simulated one, which
is what makes the simulation trustworthy.

### Exit codes, grouped by remedy

Codes are grouped by **who can fix it and how**, because that is the only grouping a
script or an agent can act on.

| code | meaning | what the caller should do |
|---|---|---|
| `0` | ok | — |
| `1` | the transformation ran and failed | read the error |
| `2` | the command as typed cannot be satisfied, but a nearby one can | type the other command |
| `3` | tile-rs cannot do this **yet** | nothing today; the capability is unwritten |
| `4` | it exists, but not here | install the toolchain, or fetch a build that has the target |
| `5` | a license is required | obtain a key |
| `6` | a device is required and none was detected | run it where the hardware is |
| `7` | the UI or daemon could not start | check the subsystem |

The pair that carries the most weight is **3 against 4**. "tile-rs has no Metal
frontend" and "this binary was built without the Ascend targets" are not the same
answer: the second is one download from working, the first is a piece of work nobody has
done. Collapsing them tells a caller to retry forever, or to give up one download early.

And `3` is **never a claim of impossibility**. Sixteen of the nineteen forms are
write-only today, so most refusals this tool gives are of that kind — which makes the
tense of the message part of the contract, not a nicety. A refusal reads:

```
tile: cannot go from "msl" to "msl" yet.
  tile-rs can emit "msl" but cannot read it — no msl frontend has
  been written. That is missing work, not a missing possibility.
  With one, the route would be:  msl -> mlir -> msl  [synthesised]
  and results through it would be marked [synthesised], because a lifted
  kernel is a reconstruction, not a recovery.
  tile-rs reads tile, mlir, linalg today. Adding a reader is a form-table row plus a
  frontend; `tile --list-forms` shows which forms have one.
```

The route in that message is computed, not written: the planner re-runs the search with
the missing frontend hypothetically present and reports what *would* work. The graph
itself stays honest — an edge is in it only if it is real — while the refusal names the
work that would unlock the pair. `Need::Frontend` is the one need no build can satisfy,
and `Need::availability()` is the function the exit codes are derived from.

Note also which case is **not** a `3`: a route that exists and is takeable but passes
through a lift is exit `2`, because a different command works right now. It is a
statement about the command, not about tile-rs.


## Why `validated` carries its evidence (#012)

The class was printed on the last line of every conversion, and by this document's own
account it is "worth more than anything else the tool prints". Nine backends carried it;
this repo holds a run for three — CPU, Apple GPU, and Vulkan/MoltenVK since the harness
was written.

The other six came from projects that DO have the hardware, so those claims may well be
true. Downgrading them here would have replaced a possibly-true claim with a
definitely-wrong one. So the claim is not changed — it is attributed, and the reader can
tell the two apart at a glance instead of having to know which is which.

`tile doctor --targets` splits the same way: **measured HERE**, **claimed UPSTREAM**, and
**unmeasured**. It used to be two groups, with all nine validated backends in the first.

An evidence pointer must name a file that exists; a test opens every one. A pointer that
cannot be followed would be worse than the unattributed claim it replaced, because it looks
checkable. That test failed on its first run, on a path this change itself had got wrong.
