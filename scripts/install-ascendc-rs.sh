#!/bin/sh
# One-line install for the `ascendc-rs` CLI.
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-ascendc-rs.sh | sh
#
# Source of truth is ci/tile-rs-public/ in the private monorepo; deploy.toml maps this to
# scripts/install-ascendc-rs.sh in the PUBLIC yijunyu/tile-rs repo, which is what makes the
# URL above fetchable without a token. ascendc-rs itself is private, so an installer hosted
# there could not be curled — publishing the binary and this script from the public repo
# is the whole reason that works.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="yijunyu/tile-rs"
DEST="${ASCENDC_RS_BIN_DIR:-$HOME/.local/bin}"
TAG="${ASCENDC_RS_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-ascendc-rs: $*"; exit 1; }

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
# Resolve the newest ascendc-rs-v* release from the releases API, NOT from the redirect that
# /releases/latest issues. Two reasons, both load-bearing:
#   * /releases/latest SKIPS prereleases, and every codegen-backend release in this repo
#     is one — so that redirect currently lands on /releases and yields no tag at all.
#   * this repository publishes two unrelated things, the CLI and the codegen backend, so
#     "latest" is the wrong question anyway. Filtering on the tag prefix is the right one.
# No jq: the API response is split on commas and the tag read with sed. The API returns
# releases newest-first, so the first match is the newest CLI release.
if [ -z "$TAG" ]; then
    TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=50" 2>/dev/null |
          tr ',' '\n' |
          sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(ascendc-rs-v[^"]*\)".*/\1/p' |
          head -1)
fi
[ -n "$TAG" ] || die "no ascendc-rs-v* release found on $REPO.
  Set ASCENDC_RS_TAG=ascendc-rs-v0.0.1 to pick one explicitly, or check:
    https://github.com/$REPO/releases"

case "$TAG" in
    ascendc-rs-v*) : ;;
    *) die "ASCENDC_RS_TAG='$TAG' is not a ascendc-rs build. This repository also publishes the
  codegen backend; the CLI tags look like ascendc-rs-v0.0.1." ;;
esac

ASSET="ascendc-rs-$TRIPLE.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

# --- fetch, unpack -----------------------------------------------------------
# NOTE: no pinned digest here, deliberately and unlike the codegen backend installer:
# that one checks a sha256 recorded for one frozen release, whereas this resolves
# whatever is newest, so there is no digest to know in advance. The binary is verified by
# running it below instead.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "install-ascendc-rs: fetching $TAG for $TRIPLE"
curl -fsSL --retry 2 -o "$TMP/$ASSET" "$URL" ||
    die "download failed: $URL
  Does that release have an asset for $TRIPLE? Check:
    https://github.com/$REPO/releases/tag/$TAG"

tar -xzf "$TMP/$ASSET" -C "$TMP" || die "could not unpack $ASSET"
[ -f "$TMP/ascendc-rs" ] || die "the archive did not contain a ascendc-rs binary"

mkdir -p "$DEST"
chmod +x "$TMP/ascendc-rs"
mv "$TMP/ascendc-rs" "$DEST/ascendc-rs"

# --- verify it runs ----------------------------------------------------------
"$DEST/ascendc-rs" --version >/dev/null 2>&1 ||
    die "installed to $DEST/ascendc-rs but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "ascendc-rs installed: $DEST/ascendc-rs ($("$DEST/ascendc-rs" --version))"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Emitting AscendC from MLIR works now, with no further setup:"
say "    ascendc-rs kernel.mlir -o kernel.cpp"
say ""
say "For Rust input, fetch the codegen backend once (98 MB):"
say "    ascendc-rs install"
say "    ascendc-rs kernel.rs -o kernel.cpp"
say ""
say "Run 'ascendc-rs --check' to see what is present and what is missing."
