id: 25
area: toolchain
title: the backend's rlib reader parses an ar archive as tar, so a kernel crate may not depend on anything
opened: 2026-09-05
closed: 2026-09-06

## wanted
`tile k.rs -t msl -o k.metal` to lower a kernel whose `tile_std` is an ordinary
`#![no_std]` crate -- one that depends on the real `core` rather than vendoring
4409 lines of it.

## got
210 errors and no output. Four of them are the real one:

    error[E0786]: found invalid metadata files for crate `core`
      = note: Failed to read rlib at ".../libcore-18c70292b0158369.rlib":
              numeric field did not have utf-8 text: <bytes> when getting cksum
              for !<arch>  #1/12  0 0 0 0 54340 `  __.SYMDEF

The other 206 are cascade: no core means no prelude, so `#[derive]` and every
trait path fail too.

`crates/rustc_codegen_tile/src` holds the 16 emitters and no driver, so this was
read out of the shipped dylib instead. Disassembling
`<rustc_codegen_tile::link::AscendMetadataLoader as
rustc_metadata::creader::MetadataLoader>::get_rlib_metadata` gives the whole
function in seven calls:

    std::fs::OpenOptions::_open
    tar::archive::Archive<dyn Read>::_entries          <-- walks it AS A TAR
    <tar::archive::EntriesFields as Iterator>::next
    tar::entry::EntryFields::path
    <Cow<Path> as PartialEq<&Path>>::eq                 against "lib.rmeta"
    std::io::default_read_to_end
    drop_in_place<tar::entry::Entry<std::fs::File>>

`lib.rmeta` is in the dylib's string table, so the loader knows which member it
wants; it opens the container with the wrong format. An rlib is a Unix `ar`
archive, so the tar reader takes the first 512 bytes as a tar header and hits
`!<arch>` where the octal checksum should be. NOT macOS-specific: an rlib is
`ar` on every platform. `get_dylib_metadata` next door parses nothing at all --
it opens the file and its only other calls are two `format!`s.

This is NOT an instance of the lowering cluster, and it was refiled once that
became clear. Nothing was mis-lowered and nothing plausible-but-wrong was
emitted: the toolchain refused, correctly and loudly, BEFORE any emitter ran.
Filing it under `lowering` -- the command a user runs to hit it -- took that
area to six and fired the stop rule on a pattern that is not there. `toolchain`
is the sixth area, and the reasoning is recorded in CLUSTER-lowering.md rather
than left as a convenient relabel.

## workaround
Keep `tile_std` on `#![no_core]`, which is what it does today. That is not a
patch, it is the reason the bug has never been hit: with no_core the crate has
no external dependency, so `get_rlib_metadata` is never called at all. The stub
is what keeps the path unreachable.

Branch `tile-std-no-std-spike` (`62bd0ad6`) carries the alternative, measured:
`tile_std` on stable rustc 1.96.0, `core.rs` deleted, 23 feature gates to zero,
-4445 lines. It compiles on stable and on the pinned nightly, `emitted_parses`
is 16 of 16 against it, and a real kernel builds through `#[tile_kernel]` -- it
only cannot be LOWERED.

The fix is small and the member name is already right: read the rlib with
`object::read::archive::ArchiveFile`, or delegate to
`rustc_metadata::locator::DefaultMetadataLoader`, instead of `tar::Archive`.

To re-check once that lands, from this repo, no LLVM needed -- the provisioned
backend bundles its own:

    T=crates/tile_cli/target/release/tile
    cp crates/tile_cli/testdata/forms/softmax.rs /tmp/k.rs && cd /tmp
    env -u RUSTFLAGS TILE_STD_PATH=<main>/crates/tile_std  $T k.rs -t msl -o old.metal
    env -u RUSTFLAGS TILE_STD_PATH=<spike>/crates/tile_std $T k.rs -t msl -o new.metal
    diff old.metal new.metal      # identical == the spike is mergeable

`old.metal` already exists and is 1701 bytes of correct MSL; only the second run
fails today.

## how it was actually fixed (2026-09-06)

Not by the one-line change predicted above. The rlib reader WAS the first wall
and reading `ar` was necessary, but it was not sufficient, and each fix only
revealed the next:

  1. `get_rlib_metadata` reads `ar` now, via `object::read::archive::ArchiveFile`.
     Two steps, not one: pulling `lib.rmeta` out gives a Mach-O, not metadata --
     rustc hides it in a section so the linker strips it. Measured on this
     toolchain's libcore: member is Mach-O `cf fa ed fe`, its only section is
     `__DWARF/.rmeta`, and THAT payload starts with rustc's `rust` magic.
     Returning the member would have compiled and still failed.

  2. `add_upstream_native_libraries` called `fatal` on any upstream native
     library, and the real `core` declares one (`compiler-rt`, Static). So the
     mere presence of a `core` dependency was fatal.

  3. `compile_ascend.rs` unpacks every dependency rlib and parses each member as
     MLIR -- "rlibs are archives that we made previously", an assumption never
     checked, true only while `#![no_core]` meant there were no dependencies.

2 and 3 were the same mistake in different words: treating the PRESENCE of a
dependency as proof something is NEEDED from it. Both now name what they cannot
link and continue. That is not the "plausible output" pattern CLUSTER-lowering.md
warns about, and the distinction is worth keeping: anything actually CALLED from
a skipped crate survives into the merged MLIR as a call with no definition and
the target compiler refuses it. The failure stays loud, it just moves to where
the code is missed rather than where a dependency was declared.

Fixed in the private codegen tree, on origin/main: 4c9e43f68 and a1d1014ba.

The diff this entry asked for is EMPTY, and the commands above are exactly how it
was taken:

    diff final_old.metal final_new.metal            IDENTICAL, 1701 bytes each
    diff old.metal final_new.metal                  IDENTICAL to the pre-fix baseline
    xcrun metal -c final_new.metal                  compiles

So `tile_std` on STABLE rustc 1.96.0 -- `#![no_core]` dropped, core.rs deleted,
23 feature gates to zero, -4445 lines -- emits byte-identical Metal. That branch
is `tile-std-no-std-spike` in tile-rs-priv.

ONE CAVEAT ON THIS BEING CLOSED: the fix is merged upstream, not provisioned. The
backend at ~/.tile-rs/toolchains is the 21-June build and still has all three
defects, so a `tile` user hits this until a new backend is published. Closed
because the tool has grown into it, which is what the README asks; reopen if the
provisioned artifact lags long enough to matter.
