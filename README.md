# tile-rs

`tile-rs` provides prebuilt tools for working with accelerator kernels. Install the
`tile` command and its companion runtime tools from the published release assets;
no local Rust or LLVM build is required.

## Install `tile`

```sh
curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-tile.sh | sh
```

The installer downloads the release archive for your platform, verifies its
checksum when available, and installs `tile` into `$HOME/.local/bin`. Set
`TILE_BIN_DIR` to choose another directory or `TILE_TAG` to pin a release.

To install the companion command-line tools as well:

```sh
curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-all.sh | sh
```

## Install the Rust codegen backend

```sh
curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install.sh | sh
```

The backend is distributed as a prebuilt artifact and requires the Rust nightly
named by its release tag. See the release notes for platform and toolchain details.

## Releases

Release pages contain downloadable binaries and checksums. The project builds and
validates release artifacts in CI before publishing them.

## License

The distribution metadata and installer scripts are available under either the
MIT License or Apache License, Version 2.0. See [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
