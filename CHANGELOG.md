# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **`tile_std` now source-compiles across the whole supported nightly range**
  (pinned `nightly-2025-08-04` through the current `1.99` cycle), not just the
  pinned toolchain. The `#![no_core]` surface tracks compiler-internal renames
  via commit-date `cfg` gates emitted from a new `crates/tile_std/build.rs`, so
  a nightly bump no longer requires editing the sources by hand. Gated changes:
  - `#[derive(Clone)]` now references `core::clone::TrivialClone`; the marker is
    supplied for `no_core`.
  - `#[rustc_do_not_implement_via_object]` → `#[rustc_dyn_incompatible_trait]`.
  - `#[rustc_macro_transparency = "semitransparent"]` → `"semiopaque"`.
  - Scalar float intrinsics (`sqrtf32`, `floorf32`, …) changed from `unsafe fn`
    to `safe [const] fn`; `fabsf32` was removed (unused here) by 2026-04.
  - `fmt` runtime types (`Count`, `Placeholder`, `UnsafeArg`) are no longer lang
    items and resolve by path.
  - `Deref`/`Receiver` impls use `?Sized` (relaxes to `PointeeSized`) to match
    the `Sized` → `MetaSized` → `PointeeSized` hierarchy split.
  - **1.99 cycle:** `#[rustc_layout_scalar_valid_range_start]` removed
    (`NonNull`/`NonZeroU8` keep correctness, minus the `Option` niche);
    `drop_in_place` lang item renamed to `drop_glue`; `#![rustc_coherence_is_core]`
    restricted to the crate root.
  - The `format_args!` builtin's repacked `Arguments` ABI is not reimplemented;
    the two panic-formatting paths are stubbed (on-device panics are `loop {}`
    and kernels never format).

### Added

- **Toolchain-drift CI guard** (`.github/workflows/toolchain-drift.yml`): a hard
  gate that `tile_std` builds on the pinned nightly and checks on the current
  nightly, so an accidental pin bump or new compiler drift fails loudly. The
  deliberate pin and how to move it are documented in `rust-toolchain.toml`.
- `tile_codegen`'s `emitters` feature now builds and tests standalone
  (`cargo test -p tile_codegen --features emitters`): the shared `mlir_to_pto` /
  `mlir_to_gpu` modules are declared at crate root under the names their peers
  import, and the golden-template tests skip gracefully when the `deepseek_metal`
  fixture is absent.

### Fixed

- Coverage and toolchain-drift CI no longer force `RUSTFLAGS="-D warnings"`, which
  had promoted benign lints (non-`snake_case` GGML quant names, `unused`
  warnings) and the inherent `generic_const_exprs` incomplete-feature warning to
  hard errors, failing the coverage gate and the drift build.

[Unreleased]: https://github.com/yijunyu/tile-rs/compare/v0.1.2...HEAD
