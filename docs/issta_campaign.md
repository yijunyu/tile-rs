# ISSTA testing campaign — tile-rs emitter group

Methodology follows the sibling `docs/issta_campaign.md` files (kunlun, precc-jev-wt):
apply software-testing techniques (fuzzing, metamorphic testing, differential oracles,
test-adequacy checks) to the emitters, fix every defect they surface, give each fix a
permanent regression gate, and wire the gates into CI.

Harness: `crates/tile_cli/tests/issta_fuzz.rs` (never-panic fuzz corpus),
`issta_metamorphic.rs` (surface-equivalence MRs), `emulate_msl.rs` (the numerical
oracle: MSL vs a CPU reference), `msl_threadgroup_bounds.rs` (resource bounds),
`emulator_coverage.rs` (declared-vs-actual coverage). Tests gate on
`cfg!(feature = "emitters")` exactly like `golden.rs`.

## Findings and fixes

### A. Silent wrong answers — the `emulate_msl` 431-bad class

The oracle flagged 431 kernel/shape disagreements on first run (emission succeeded,
numbers were wrong — the worst of the three outcomes, since nothing crashes). Root
causes, each generalized to the whole class rather than patched per report:

1. **Flat-gid indexing.** Loops iterated `for (i = gid; ...)` with global-element
   addressing while the dispatch contract is one threadgroup per row with
   `num_elements` = row width. Fixed to the canonical strided loop
   `for (uint i = tid; i < num_elements; i += tcount)` over `base + i`, across
   `emit_elementwise_chain_msl`, `emit_binop/unary/scale/cast/max/reduce_max/absmax_msl`,
   all three ggml unaries, and the canned sigmoid/softplus/clamp/silu/silu_mul.
2. **Serial halving folds.** Threadgroup reductions folded with `s <<= 1` but without a
   per-step barrier (racy), or collapsed into a `tid == 0` serial scan. Replaced with
   the doubling fold
   `for (s = 1; s < tcount; s <<= 1) { if (tid % (2*s) == 0 && tid + s < tcount) <fold>;
   barrier; }` in softmax, layernorm, l2dist, argmin/argmax (plus a new
   `emit_threadgroup_fold_max` helper), and the argmax fused-final paths.
3. **Arity-2 dup-form operand bug.** The duplicate-operand chain form read its second
   operand from index 1 even when the call carried five operands (`b_idx` now
   `args.len() >= 5 ? 2 : 1`).
4. **Shared array sized by dtype.** `sdata`/`local_x` were sized per `msl_type`; folds
   over f16 values in a too-narrow slot overflowed the declared extent. `sdata` is now
   always `threadgroup float sdata[MAX_TG]` (`MAX_TG = 1024`) — folds compare/sum
   exactly, f16 values are exactly representable in f32.
5. **Half accumulation.** `emit_matmul_f16_msl`, `emit_softmax_msl` and
   `emit_rms_norm_msl` accumulated in `half` (products round once, sums round again).
   All three now accumulate in `float` and round only on store. Softmax half path must
   not stage `exp(x-max)` in a half `p1` between passes (double rounding against the
   one-ulp tolerance): pass 3 recomputes `exp`; the float path stages, which is exact.
6. **rms_norm epsilon never reached the kernel.** `MslContext.rms_eps` captures the
   literal from the call (5-arg form only, `args[2]`), `emit_rms_norm_msl` bakes it,
   and `run::eps_reaches_the_kernel` still verifies the emitted source carries the
   requested value. Contract wrappers pass `"1e-6"`; dispatch passes `ctx.rms_eps`.

30 pinned unit-test needles in `mlir_to_msl.rs` asserted the *buggy* idioms; they were
updated to pin the fixed ones (13 in `emit_tail_tests`, 17 in `tests::`), each with a
comment recording the reversal.

### B. Never-panic fuzzing — u32-overflow panics

Three `start > end`-style overflow panics on adversarial extents (`mlir_to_aie.rs:270`,
`mlir_to_pto.rs:1312`, `mlir_to_bang.rs:1089`), fixed as a class: `extent_product`
gates (×34 sites), saturating arithmetic in pto, Result-propagation in aie.

### C. Refusals are answers

* **matvec is refused** (`__tile_matvec_f32` has no classify arm). Decision: leave
  refused. The oracle counts 3 refusals for 3 matvec shapes and still requires
  `checked >= 100` and `bad.is_empty()` — both hold. `emulator_coverage.rs` now
  declares `msl/Matvec` `NotLowered` instead of `Driven` (it was claiming coverage the
  backend does not provide).
* The three refusals are printed on every run so a silent fourth cannot hide.

### D. Defects found while making the whole suite green

These were red at HEAD — the campaign ran the *entire* suite, not just its own tests:

1. **Missing specialization guard** (`msl_specialization_guard`, 2/4 red). The test
   pins a guard that was never in the source: `dk=128` emitted the score kernel baked
   for `dk=64` and read K at half the right stride. Implemented: `MslContext::known_const`
   (proves a constant only via the const map, a decimal, or an MLIR `%c<N>` name — an
   unreadable operand is never evidence) and `check_flash_vec_score_dims` (operands 7/8
   of the 14-operand score form; proven mismatch → `... is specialized to dk=64, but
   was called with dk=128`). Dynamic operands still lower.
2. **The refusing-default-arm scenario** (`spec.rs`) asserted the refusal message
   contains `COPY`; the message said `Copy`. The message now names the COPY arm it
   avoids, which is what it always meant to say.
3. **Golden re-bless**: `softmax.msl.metal` reflects the fixed emitter (doubling folds,
   f32 accumulation, staged float path) — `TILE_BLESS=1 cargo test --test golden`.
4. **No-emitters build red at HEAD** (33 tests across 15 targets). Every one now carries
   the `golden.rs` gate: `if !cfg!(feature = "emitters") { eprintln!("skipped: ...");
   return; }`. `cargo test --no-default-features` went 15 failed targets → 0.
5. **Coverage gate red at HEAD** (`coverage.yml`, `--gate 78`: 76.42% measured in a
   clean worktree). Fixed by testing emitters that had never been called: 23 new
   `t_emit_*` smoke tests (the `check(|o| emit_x(o), needle, "emit_x")` pattern —
   non-empty, deterministic, needle must be rendered output) covering the self-contained
   `fn emit_x(out: &mut String)` family plus `emit_mul_mv_q4_K_f32_ggml_ds4` (which
   pulls in the shared `emit_q4k_compute_core`) and `ported_contract_table()` (the
   obligation-table generator). Gate now reads **79.71% (55112/69137)** — ~1.7 points
   of headroom, matching the workflow's intended margin.
6. **Flaky `lift_cpp` unit test** (surfaced only after the suite grew): two tests
   raced on the process-global `TILE_ASCENDC_TO_RS` — `override_must_exist` could
   observe the other test's probe binary and fail `assert!(locate().is_none())`.
   Locked with the repo's standard per-module `ENV_LOCK` (same shape as `backlog.rs`
   / `lower_rs.rs`). Three consecutive full-lib runs clean.

## Results

| Gate | Before | After |
|---|---|---|
| `emulate_msl` | 431 bad, 0/3 pass | **0 bad, 3/3 pass** (534 compared, 3 refused) |
| `msl_threadgroup_bounds` | 0/3 | **3/3** |
| `msl_specialization_guard` | 2/4 | **4/4** |
| `cargo test --locked` (default) | red (4 targets) | **green — 23 targets, 1383 tests** |
| `cargo test --locked --no-default-features` | red — 15 targets | **green — 23 targets** |
| `cargo test --locked --features stats` | — | green |
| `cargo clippy --all-targets -- -D warnings` | 2 errors (issta_fuzz) | clean |
| `cargo fmt --check` | 1 diff | clean |
| `coverage.sh --gate 78` (coverage.yml) | red — 76.42% | **green — 79.71%** (55112/69137) |
| `tile_spec` standalone | — | green — 14 + 966 tests |
| CI ISSTA step (`issta_fuzz` + `issta_metamorphic`) | absent | added to `tile-cli.yml` |
| spec (181 scenarios), doctor, identify, exit-4 contract, no-C graph, dep budget | — | verified locally |

## CI red after the push (follow-up)

The push (`478a80b`) was green locally and red on CI — three jobs, every failure
pre-existing at tag `tile-v0.1.0` (coverage's first-ever run excepted), so the campaign
had measured its own gates against a machine that could not see them: stable had moved
1.96 → 1.98 under the lint job, and this Mac had no Metal Toolchain until
`xcodebuild -downloadComponent MetalToolchain` installed one.

| Job | Failure | Root cause | Fix |
|---|---|---|---|
| lint | 2 clippy errors in `sha256.rs` | the `chunks_exact().as_chunks()` lint is new in clippy 1.98 | `msg.as_chunks::<64>()` / `as_chunks::<4>()` |
| coverage | 216 `-D warnings` errors in `tile_spec`'s `cucumber` target | `setup-rust-toolchain` defaults `RUSTFLAGS=-D warnings`; this job never passed the `rustflags: ""` opt-out `tile-cli.yml` uses | pass `rustflags: ""` (same rationale comment) |
| test (ubuntu) | every `gpu`/`musa` file: `static declaration of '__expf' follows non-static` | glibc's `<math.h>` declares `__expf` and `__logf` (every `X` also as `__X`); the stub's `static inline` definitions collide with them | `#define __expf expf` / `#define __logf logf` — libc spelling, no definition |
| test (macOS) | 9 of 91 MSL fixtures fail `xcrun metal -c` | four emitter defects + one optimizer defect, below | emitter and optimizer fixes |

The nine fixture failures, by class:

1. **`-O2`'s dead-op elimination deleted five ds4 `mul_mv_id_*` kernels.** The MLIR
   discards the call's result, `Op::is_pure` said any `__tile_*` call was pure, and a
   result nobody reads looked like nothing to delete — so the call went, the constants
   feeding it cascaded after it, and the emitter classified the hollow kernel as a copy
   (`p1[gid] = p0[gid]` writing a `const` buffer). The matvec/matmul families are now
   in the IMPURE list: they write their destination through pointer arguments, and the
   unused result says nothing about the buffers. Regression tests in `mlir.rs` and
   `optimize.rs`.
2. **`tile_topk` declared its VALUES buffer `const`.** The default writability rule
   ("everything but the last is writable") left `p1` read-only while the insertion-sort
   arm initialises and updates it. TopK now has an explicit entry: `p0` const, `p1`/`p2`
   writable — the buffer count comment always said `input, out_values, out_indices`.
3. **The shared prologue against arms that do not want it.** `DraftVerify` declares its
   own `uint base = row * cols` (a redefinition on top of the prologue's), and
   `kv_cache_update`'s signature carries `num_heads/max_seq/head_dim/position` instead
   of `num_elements` (the prologue read an identifier that was never bound — the same
   shape as the `causal_mask` bug its list comment already documents). Both, and the
   prefill sibling, join `has_own_indexing`.
4. **`cast_bf16_f32` read a `float*` with `as_type<ushort>`** — a cast between types of
   different size, which Metal rejects outright. The hand-written template's source is
   `ushort*`; through a `float*` the same intent is `(as_type<uint>(p0[gid]) & 0xFFFFu)
   << 16`. Both copies of the emitter (the shadowing one in `mlir_to_msl.rs` and the
   canned split) and the `t_emit` expectation moved together.

Two instrument problems surfaced while reproducing: the `tile` on `~/.cargo/bin` was
four days stale (repro must use the workspace build), and the two metal-compile tests
lacked the `cfg!(feature = "emitters")` guard every other test in the file has —
invisible while the toolchain was absent, a hard failure the moment it was installed.

With those in, every gate in the table above is green again locally, including
`cargo test --no-default-features` (23 targets) and `--features stats` (23 targets).

`emit_fill_msl`, concat/scatter/gather, `splitk_reduce`, `silu_mul_fused`,
quantize/dequantize, rope/kv-cache, `emit_repeat`, `emit_copy_msl`, and the f16
layernorm half-accumulation path have no oracle coverage yet; they are named here so
the absence is a choice. The tolerance model the oracle enforces is documented in
`run::error_budget`: products round to the buffer type, the accumulator is assumed f32,
`Fixed{abs: 1e-5, rel: max(1e-4, unit_roundoff(dtype))}`, per-element, zero non-finite.

## Environment notes

* `crates/tile_std` has `[lib] test = false`; the root workspace pins
  nightly-2025-08-04 while `tile_cli` runs on stable — run cargo from `crates/tile_cli`.
* The cross-check job needs `rustup target add` for each non-host target; only the host
  target is verifiable on this machine.
* Do not confuse the parallel session's uncommitted `crates/tile_codegen` work with
  campaign output.
