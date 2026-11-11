id: 42
area: lowering
title: -Zcrate-attr=register_tool(tile) collides with tile_std's own declaration, so no .rs kernel lowers
opened: 2026-10-28

## wanted
tile <lifted>.rs -t msl -o out.metal
on any kernel lifted from AscendC by ascendc-to-rs, to test the GLM-5.3 and
Kimi K3 kernel set on Metal before taking the same sources to a 910b.

## got
exit 1, no output file, for every kernel tried:

    error: tool `tile` was already registered
      --> <crate attribute>:1:18
      --> /Users/yijun/ascend-rs-priv/crates/tile_std/src/lib.rs:24:5
    error[E0636]: the feature `register_tool` has already been enabled
    error: could not compile `tile_std` (lib) due to 2 previous errors

`crates/tile_cli/src/lower_rs.rs:213-214` injects
`-Zcrate-attr=feature(register_tool)` and `-Zcrate-attr=register_tool(tile)`.
A crate-attr injection applies to EVERY crate in the build, and tile_std
already declares both in its own lib.rs (both copies do -- ~/tile-rs's and
~/ascend-rs-priv's, which is the one resolved here). So the dependency fails to
compile before the kernel is reached.

NOT staleness, which was the first hypothesis and is the recorded trap on this
tool: the binary on PATH was from Oct 23 while tile_std gained `register_tool`
on Oct 24, so it looked like exactly that. Rebuilding to Oct 28 12:01 changes
nothing -- the current CLI injects it too.

Also wrong in the tile-rs skill: the documented rebuild
`cargo install --path . --features pico,stats` cannot work, because
~/tile-rs/Cargo.toml is a VIRTUAL MANIFEST. It has to be
`--path crates/tile_cli`. That cost one failed install (exit 101) before the
real one.

## workaround
FIXED in `crates/tile_cli/src/lower_rs.rs`: the two attributes are prepended to
the kernel source written into the scratch crate's `src/lib.rs` instead of
injected through `.cargo/config.toml`'s `[build] rustflags`, which applies to
every crate in the graph. Guarded with `source.contains("register_tool")` so a
hand-written kernel that already declares them does not hit the same error.
`tile <lifted>.rs -t msl` now exits 0 and writes output.

## what this unblocked, and the WORSE defect it exposed
With lowering running, every kernel lifted from the AscendC corpus emits an
ELEMENTWISE COPY, at exit 0, with no warning:

    kernel void mp_rmsnorm_kernel(...) {
        uint gid = base + tid;
        if (gid < num_elements) p1[gid] = p0[gid];
    }

That is the body for a kernel whose source carries 37 tile_std ops. The same
2183 bytes come out of BOTH lifts of the same file -- `ascendc-to-rs` from
~/kunlun (90 ops) and tile's own `--via tile` cpp lift (108 ops) -- so it is not
a lift defect.

Positive control, which is what makes this attributable: tile's own
`testdata/forms/softmax.rs` lowers to a REAL three-pass softmax with
threadgroup barriers and a shared-memory reduction. The discriminator is not the
`[0 ops]` counter, which reads 0 for the specimen too. It is the line above it:

    specimen:        tile plan: 8192 x 32 cores -> 0 full + 1024 tail
    every lifted:    tile plan: not computable (no declared extent)

**With no declared extent the msl backend emits a copy instead of refusing.**
There is no CLI flag to supply one (`--help` has no extent/shape/dim option), so
a kernel whose extents are runtime values -- which is every kernel lifted from
AscendC, where they come from GM tiling params -- cannot be lowered truthfully
and is not told so.

This is the tool violating its own stated principle. Its `--via` refusal reads:
"a lift never composes into a lowering automatically -- a synthesised
intermediate feeding a lowering is exactly the kind of wrongness that stays
invisible until it corrupts a run." A lowering that silently substitutes a copy
is that same wrongness one stage later, and here it is the default.

Cost on the day: this was found while inventorying the GLM-5.3 and Kimi K3
kernel set for the Ascend track, with the intent of testing on Metal first and
then PTO on a 910b. Had the copy not been read, both arms would have timed and
"validated" memcpys at exit 0 on every instrument.

## what would have made the tool able to do it
Either refuse -- exit 3 with "no declared extent, so no tile plan; supply one or
lower a form that declares it" -- or accept an extent (`--extent R,C` or a
per-kernel attribute) and lower against it. Refusing is the smaller change and
is the one consistent with the `--via` policy. Emitting a copy is the only
option that cannot be checked by the caller.
