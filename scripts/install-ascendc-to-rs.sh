#!/bin/sh
# One-line install for the `ascendc-to-rs` CLI.
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-ascendc-to-rs.sh | sh
#
# ascendc_to_rs itself is private, so an installer hosted there could not be curled --
# publishing the binary and this script from the public repo is the whole reason that
# works, same as ascendc-rs and pto-rs.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="yijunyu/tile-rs"
DEST="${ASCENDC_TO_RS_BIN_DIR:-$HOME/.local/bin}"
TAG="${ASCENDC_TO_RS_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-ascendc-to-rs: $*"; exit 1; }

# --- which build do we need? -------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os-$arch" in
    Darwin-arm64)  TRIPLE=aarch64-apple-darwin ;;
    Linux-x86_64)  TRIPLE=x86_64-unknown-linux-gnu ;;
    Darwin-x86_64) die "no build is published for Intel macOS. Released triples are
  aarch64-apple-darwin and x86_64-unknown-linux-gnu." ;;
    *) die "unsupported platform $os-$arch. Released triples are aarch64-apple-darwin
  and x86_64-unknown-linux-gnu." ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar  >/dev/null 2>&1 || die "tar is required"

# --- resolve the release -----------------------------------------------------
if [ -z "$TAG" ]; then
    TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=50" 2>/dev/null |
          tr ',' '\n' |
          sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(ascendc-to-rs-v[^"]*\)".*/\1/p' |
          head -1)
fi
[ -n "$TAG" ] || die "no ascendc-to-rs-v* release found on $REPO.
  Set ASCENDC_TO_RS_TAG=ascendc-to-rs-v0.0.1 to pick one explicitly, or check:
    https://github.com/$REPO/releases"

case "$TAG" in
    ascendc-to-rs-v*) : ;;
    *) die "ASCENDC_TO_RS_TAG='$TAG' is not an ascendc-to-rs build. This repository also
  publishes ascendc-rs, pto-rs and the codegen backend; the tags look like
  ascendc-to-rs-v0.0.1." ;;
esac

ASSET="ascendc-to-rs-$TRIPLE.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

# --- fetch, unpack -----------------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "install-ascendc-to-rs: fetching $TAG for $TRIPLE"
curl -fsSL --retry 2 -o "$TMP/$ASSET" "$URL" ||
    die "download failed: $URL
  Does that release have an asset for $TRIPLE? Check:
    https://github.com/$REPO/releases/tag/$TAG"

tar -xzf "$TMP/$ASSET" -C "$TMP" || die "could not unpack $ASSET"
[ -f "$TMP/ascendc-to-rs" ] || die "the archive did not contain an ascendc-to-rs binary"

mkdir -p "$DEST"
chmod +x "$TMP/ascendc-to-rs"
mv "$TMP/ascendc-to-rs" "$DEST/ascendc-to-rs"

# --- verify it runs ----------------------------------------------------------
# No --version flag, unlike ascendc-rs/pto-rs -- --help is the start check.
"$DEST/ascendc-to-rs" --help >/dev/null 2>&1 ||
    die "installed to $DEST/ascendc-to-rs but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "ascendc-to-rs installed: $DEST/ascendc-to-rs"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Lifting an AscendC kernel to tile-rs works now, with no further setup:"
say "    ascendc-to-rs kernel.cpp -o kernel.rs"
say ""
say "Run 'ascendc-to-rs --help' for the full flag list."
