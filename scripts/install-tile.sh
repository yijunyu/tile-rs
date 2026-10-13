#!/bin/sh
# One-line install for the `tile` CLI -- the tile-rs kernel swissknife (identify,
# profile, lower, lift and optimize accelerator kernels across 16 backends).
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-tile.sh | sh
#
# `tile` builds on stable from crates/tile_cli, a standalone workspace member with its
# own rust-toolchain.toml -- unlike ascendc-rs/pto-rs it carries no nightly pin, and
# unlike those two the source for THIS part of the tool (tile_cli itself) is not secret;
# what IS closed-source and bundled alongside it is pto_to_ascendc (see
# ascendc-rs-release.yml and vendor/pto_to_ascendc/README.md), published only as a
# prebuilt binary inside this same tarball.
#
# Three platforms ship in tile-v0.0.1, not the four the matrix in tile-release.yml
# builds: x86_64-apple-darwin (Intel macOS) is missing because that leg's own test
# suite (the_specification_runs, unrelated to this release) currently fails there --
# the torch wheels tile's own tests fetch don't cover macosx_14_0_x86_64 at all. See
# that release's own NOTES for the detail; published manually from the three matrix
# legs that built and tested clean rather than blocking on an unrelated gap.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="yijunyu/tile-rs"
DEST="${TILE_BIN_DIR:-$HOME/.local/bin}"
TAG="${TILE_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-tile: $*"; exit 1; }

# --- which build do we need? -------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os-$arch" in
    Darwin-arm64)   TRIPLE=aarch64-apple-darwin ;;
    Darwin-x86_64)  die "no build is currently published for Intel macOS -- that leg's
  own test suite fails there for a reason unrelated to this release (see
  https://github.com/$REPO/releases/tag/tile-v0.0.1). Released triples are
  aarch64-apple-darwin, x86_64-unknown-linux-gnu and aarch64-unknown-linux-gnu." ;;
    Linux-x86_64)   TRIPLE=x86_64-unknown-linux-gnu ;;
    Linux-aarch64)  TRIPLE=aarch64-unknown-linux-gnu ;;
    *) die "unsupported platform $os-$arch. Released triples are aarch64-apple-darwin,
  x86_64-unknown-linux-gnu and aarch64-unknown-linux-gnu." ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar  >/dev/null 2>&1 || die "tar is required"

# --- resolve the release -----------------------------------------------------
if [ -z "$TAG" ]; then
    TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=50" 2>/dev/null |
          tr ',' '\n' |
          sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(tile-v[^"]*\)".*/\1/p' |
          head -1)
fi
[ -n "$TAG" ] || die "no tile-v* release found on $REPO.
  Set TILE_TAG=tile-v0.0.1 to pick one explicitly, or check:
    https://github.com/$REPO/releases"

case "$TAG" in
    tile-v*) : ;;
    *) die "TILE_TAG='$TAG' is not a tile build. This repository also publishes
  ascendc-rs, ascendc-to-rs, pto-rs, pto-diff, pto-to-tilers and the codegen backend;
  the tag for this one looks like tile-v0.0.1." ;;
esac

ASSET="tile-$TRIPLE.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"
SUMS_URL="https://github.com/$REPO/releases/download/$TAG/SHA256SUMS"

# --- fetch, verify, unpack ----------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "install-tile: fetching $TAG for $TRIPLE"
curl -fsSL --retry 2 -o "$TMP/$ASSET" "$URL" ||
    die "download failed: $URL
  Does that release have an asset for $TRIPLE? Check:
    https://github.com/$REPO/releases/tag/$TAG"

# This release publishes SHA256SUMS (unlike the single-binary CLIs, which verify by
# running instead); check against it when sha256sum/shasum is available rather than
# skipping verification silently.
if command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1; then
    curl -fsSL --retry 2 -o "$TMP/SHA256SUMS" "$SUMS_URL" 2>/dev/null || true
    if [ -s "$TMP/SHA256SUMS" ]; then
        EXPECTED=$(grep "$ASSET" "$TMP/SHA256SUMS" | awk '{print $1}' | head -1)
        if [ -n "$EXPECTED" ]; then
            if command -v sha256sum >/dev/null 2>&1; then
                ACTUAL=$(sha256sum "$TMP/$ASSET" | awk '{print $1}')
            else
                ACTUAL=$(shasum -a 256 "$TMP/$ASSET" | awk '{print $1}')
            fi
            [ "$EXPECTED" = "$ACTUAL" ] ||
                die "checksum mismatch for $ASSET: expected $EXPECTED, got $ACTUAL"
            say "install-tile: checksum verified"
        fi
    fi
fi

tar -xzf "$TMP/$ASSET" -C "$TMP" || die "could not unpack $ASSET"
[ -f "$TMP/tile" ] || die "the archive did not contain a tile binary"

mkdir -p "$DEST"
chmod +x "$TMP/tile"
mv "$TMP/tile" "$DEST/tile"
# pto_to_ascendc ships bundled alongside tile (closed-source, see the header above) --
# install it too when present, since `tile` shells out to it for the pto->ascendc lower.
if [ -f "$TMP/pto_to_ascendc" ]; then
    chmod +x "$TMP/pto_to_ascendc"
    mv "$TMP/pto_to_ascendc" "$DEST/pto_to_ascendc"
fi

# --- verify it runs ----------------------------------------------------------
"$DEST/tile" --version >/dev/null 2>&1 ||
    die "installed to $DEST/tile but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "tile installed: $DEST/tile ($("$DEST/tile" --version))"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Profile and optimize a kernel for this machine, with no further setup:"
say "    tile kernel.rs"
say ""
say "-O4 adds measured autotuning (a knob search over schedule choices) when a"
say "device and license are present; without them it falls back and says so."
say ""
say "Run 'tile doctor' to see what this machine can target, or 'tile --help' for"
say "the full flag list."
