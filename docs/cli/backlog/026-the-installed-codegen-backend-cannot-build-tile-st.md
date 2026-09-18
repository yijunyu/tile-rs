id: 26
area: provision
closed: 2026-09-08
title: the installed codegen backend cannot build tile_std on this machine (E0786 on core)
opened: 2026-09-07

## wanted
tile crates/tile_cli/testdata/forms/softmax.rs -o /tmp/sm.metal
to lower the corpus specimen through the real backend, so that the effect of
rustc's MIR optimization level on the emitted MLIR could be measured (the
kernel build is `cargo build` with no --release, so mir-opt-level sits at its
debug default and most MIR passes are off).

## got
error[E0786]: found invalid metadata files for crate `core`
      --> crates/tile_std/src/buf.rs:46:5
    error: cannot find attribute `derive` in this scope
      --> crates/tile_std/src/lib.rs:57:3
    error: could not compile `tile_std` (lib) due to 212 previous errors

`tile install` reports `rustc_codegen_tile 0.0.1+nightly-2025-08-04 installed`
and `nightly-2025-08-04` is installed and active, so the pin looks satisfied
and the failure is not the documented "--target omitted" case: lower_rs.rs
passes --target always. Same symptom, different cause.

## resolution: DUPLICATE of #025

Filed without checking the backlog first, which is the one step the skill asks
for before recording a gap. #025 -- "the backend's rlib reader parses an ar
archive as tar" -- is the same E0786 on `core`, diagnosed by disassembling
`AscendMetadataLoader::get_rlib_metadata` and watching it walk an ar archive
with `tar::archive::Archive::_entries`. It was closed 2026-09-06.

What #025 already says, and what I would have read: "the fix is merged
upstream, not provisioned ... so a `tile` user hits this until a new backend is
published." `tile install` reports rustc_codegen_tile 0.0.1 here, which
predates it, and 471a3696 on this branch points provisioning at 0.0.2.

So there was nothing new in this entry except a second person hitting a known
defect on an old backend. The conclusion below still stands and is the only
part worth keeping.

## workaround
Not found in-session. The question the lowering was meant to answer was
instead settled by reading lower_rs.rs directly: it writes
`-Zcodegen-backend=<lib>` into .cargo/config.toml and runs `cargo build
--target <triple>` with no --release and no -Zmir-opt-level, which establishes
that MIR passes ARE in the path and ARE at their lowest setting -- without
needing to run the lowering.
