//! `tile_spec` — executable Gherkin (Given/When/Then) spec layer for tile-rs.
//!
//! This crate is the **behaviour contract** for the open tile-rs surface: the
//! `CodegenTarget` trait + `TargetRegistry` ([`tile_codegen`]) and the 14 open
//! pure-`emit` backends. Each use case is a `.feature` file under `features/`,
//! written as Given/When/Then scenarios, and **every scenario maps 1:1 to an
//! executable step** that drives real `emit`/registry/trait code — so the specs
//! are tests, not prose.
//!
//! ## Why a hand-rolled runner instead of the `cucumber` crate
//!
//! The repo builds against a **vendored, offline** crate registry (`vendor/`),
//! and `cucumber`/`gherkin` are not vendored. Rather than vendor a large async
//! dependency tree, this crate ships a small, **std-only, zero-dependency**
//! Gherkin parser + step runner ([`gherkin`]). That keeps the open OSS skeleton
//! dependency-free (a feature, not a bug, for the publishable surface) while
//! still parsing standard `.feature` syntax: `Feature`, `Background`,
//! `Scenario`, `Scenario Outline` + `Examples`, and `Given`/`When`/`Then`/`And`/
//! `But` steps with `<placeholder>` substitution. The step-registration API
//! mirrors cucumber's: register `Given(pattern, fn)` etc., then run a feature.
//!
//! The harness that wires the open backends to these steps lives in
//! `tests/cucumber.rs` (so it can `#[path]`-include the real `mlir_to_*`
//! emitters with no LLVM, exactly like the generality-matrix tests do).

pub mod gherkin;

pub use gherkin::{Runner, StepKind, World};
