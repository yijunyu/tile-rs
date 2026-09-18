# Prebuilt `pto_to_ascendc` binaries

One executable per release triple, packed into `tile-<triple>.tar.gz` by
`.github/workflows/tile-release.yml`. The source is deliberately **not** in this
public tree — only the binaries ship.

| File | Triple |
|------|--------|
| `pto_to_ascendc-aarch64-apple-darwin` | macOS Apple Silicon |
| `pto_to_ascendc-x86_64-apple-darwin` | macOS Intel |
| `pto_to_ascendc-aarch64-unknown-linux-gnu` | Linux aarch64 |
| `pto_to_ascendc-x86_64-unknown-linux-gnu` | Linux x86_64 |

Usage (after unpacking the release tarball):

```sh
pto_to_ascendc <post-pass.pto> [-o out.cpp]
```

Input is what `ptoas --enable-insert-sync --emit-pto-ir` writes. A module that
has not been through `ptoas` has no `addr =` on its `alloc_tile`s and is
rejected rather than lowered.

Do not edit these files by hand. Replace them only by rebuilding from the
private source checkout and copying the four artifacts above.
