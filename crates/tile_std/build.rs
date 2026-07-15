//! Detect the rustc nightly's commit date and emit `cfg` flags so the
//! `#![no_core]` surface can track compiler-internal renames while still
//! building on the pinned `nightly-2025-08-04`.
//!
//! Add a new gate here (mirroring the pattern below) whenever a lang item,
//! intrinsic, or `rustc_*` attribute changes name/shape across the supported
//! nightly range, then `#[cfg]` the two spellings in `src/core.rs`.

use std::process::Command;

fn main() {
    let date = rustc_commit_date();

    println!("cargo::rustc-check-cfg=cfg(rustc_dyn_incompatible_trait_attr)");
    println!("cargo::rustc-check-cfg=cfg(rustc_float_intrinsics_safe)");

    // The scalar float intrinsics (sqrtf32, expf32, floorf32, ...) changed from
    // `unsafe fn` to `safe [const] fn` right after the pinned toolchain: they are
    // still `unsafe` on 2025-08-04 but `safe` by 2025-11-24. Gate at 2025-09-01.
    let float_intrinsics_safe = matches!(date, Some(d) if d >= (2025, 9, 1));
    if float_intrinsics_safe {
        println!("cargo::rustc-cfg=rustc_float_intrinsics_safe");
    }

    // The `#[rustc_do_not_implement_via_object]` marker attribute was renamed to
    // `#[rustc_dyn_incompatible_trait]`. It is absent on 2026-01-20 and present
    // by 2026-02-28, so gate at 2026-02-01. When the commit date is unknown
    // (stable rustc, or `--version --verbose` unavailable) assume the pinned
    // older toolchain and keep the legacy spelling.
    let has_new_attr = matches!(date, Some(d) if d >= (2026, 2, 1));
    if has_new_attr {
        println!("cargo::rustc-cfg=rustc_dyn_incompatible_trait_attr");
    }
}

/// Parse `commit-date: YYYY-MM-DD` from `rustc --version --verbose`.
fn rustc_commit_date() -> Option<(u32, u32, u32)> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(&rustc)
        .args(["--version", "--verbose"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let date = text
        .lines()
        .find_map(|l| l.strip_prefix("commit-date:"))?
        .trim();
    let mut it = date.split('-').filter_map(|s| s.trim().parse::<u32>().ok());
    Some((it.next()?, it.next()?, it.next()?))
}
