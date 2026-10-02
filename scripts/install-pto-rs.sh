#!/bin/sh
# One-line install for the `pto-rs` CLI.
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-pto-rs.sh | sh
#
# Source of truth is ci/tile-rs-public/ in the private monorepo; deploy.toml maps this to
# scripts/install-pto-rs.sh in the PUBLIC yijunyu/tile-rs repo, which is what makes the
# URL above fetchable without a token. pto-rs itself is private, so an installer hosted
# there could not be curled — publishing the binary and this script from the public repo
# is the whole reason that works.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="yijunyu/tile-rs"
DEST="${PTO_RS_BIN_DIR:-$HOME/.local/bin}"
TAG="${PTO_RS_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-pto-rs: $*"; exit 1; }

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
# No jq: read the tag out of the redirect that /releases/latest issues. Falls back to a
# caller-supplied PTO_RS_TAG, which is also the escape hatch if the newest release is a
# prerelease (GitHub's /latest skips those).
if [ -z "$TAG" ]; then
    TAG=$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
          "https://github.com/$REPO/releases/latest" 2>/dev/null |
          sed -n 's|.*/tag/\(.*\)$|\1|p')
fi
[ -n "$TAG" ] || die "could not determine the latest release tag.
  Set PTO_RS_TAG=pto-rs-v0.0.1 (prereleases are not returned by /releases/latest)."

case "$TAG" in
    pto-rs-v*) : ;;
    *) die "the latest release is '$TAG', which is not a pto-rs build.
  This repository also publishes the codegen backend. Set PTO_RS_TAG to a pto-rs-v* tag." ;;
esac

ASSET="pto-rs-$TRIPLE.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

# --- fetch, unpack -----------------------------------------------------------
# NOTE: no pinned digest here, deliberately and unlike the codegen backend installer:
# that one checks a sha256 recorded for one frozen release, whereas this resolves
# whatever is newest, so there is no digest to know in advance. The binary is verified by
# running it below instead.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "install-pto-rs: fetching $TAG for $TRIPLE"
curl -fsSL --retry 2 -o "$TMP/$ASSET" "$URL" ||
    die "download failed: $URL
  Does that release have an asset for $TRIPLE? Check:
    https://github.com/$REPO/releases/tag/$TAG"

tar -xzf "$TMP/$ASSET" -C "$TMP" || die "could not unpack $ASSET"
[ -f "$TMP/pto-rs" ] || die "the archive did not contain a pto-rs binary"

mkdir -p "$DEST"
chmod +x "$TMP/pto-rs"
mv "$TMP/pto-rs" "$DEST/pto-rs"

# --- verify it runs ----------------------------------------------------------
"$DEST/pto-rs" --version >/dev/null 2>&1 ||
    die "installed to $DEST/pto-rs but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "pto-rs installed: $DEST/pto-rs ($("$DEST/pto-rs" --version))"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Emitting PTO from MLIR works now, with no further setup:"
say "    pto-rs kernel.mlir -o kernel.pto"
say ""
say "For Rust input, fetch the codegen backend once (98 MB):"
say "    pto-rs install"
say "    pto-rs kernel.rs -o kernel.pto"
say ""
say "Run 'pto-rs --check' to see what is present and what is missing."
