#!/bin/sh
# One-line install for the `pto-diff` CLI.
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-pto-diff.sh | sh
#
# pto_to_rust itself is private, so an installer hosted there could not be curled --
# publishing the binary and this script from the public repo is the whole reason that
# works, same as ascendc-rs and pto-rs.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="yijunyu/tile-rs"
DEST="${PTO_DIFF_BIN_DIR:-$HOME/.local/bin}"
TAG="${PTO_DIFF_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-pto-diff: $*"; exit 1; }

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
          sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(pto-diff-v[^"]*\)".*/\1/p' |
          head -1)
fi
[ -n "$TAG" ] || die "no pto-diff-v* release found on $REPO.
  Set PTO_DIFF_TAG=pto-diff-v0.0.1 to pick one explicitly, or check:
    https://github.com/$REPO/releases"

case "$TAG" in
    pto-diff-v*) : ;;
    *) die "PTO_DIFF_TAG='$TAG' is not a pto-diff build. This repository also publishes
  ascendc-rs, ascendc-to-rs, pto-rs and the codegen backend; the tags look like
  pto-diff-v0.0.1." ;;
esac

ASSET="pto-diff-$TRIPLE.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

# --- fetch, unpack -----------------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "install-pto-diff: fetching $TAG for $TRIPLE"
curl -fsSL --retry 2 -o "$TMP/$ASSET" "$URL" ||
    die "download failed: $URL
  Does that release have an asset for $TRIPLE? Check:
    https://github.com/$REPO/releases/tag/$TAG"

tar -xzf "$TMP/$ASSET" -C "$TMP" || die "could not unpack $ASSET"
[ -f "$TMP/pto-diff" ] || die "the archive did not contain a pto-diff binary"

mkdir -p "$DEST"
chmod +x "$TMP/pto-diff"
mv "$TMP/pto-diff" "$DEST/pto-diff"

# --- verify it runs ----------------------------------------------------------
# No --version flag -- --help is the start check.
"$DEST/pto-diff" --help >/dev/null 2>&1 ||
    die "installed to $DEST/pto-diff but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "pto-diff installed: $DEST/pto-diff"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Checking a stage-2 MLIR kernel works now, with no further setup:"
say "    pto-diff kernel_stage2.mlir"
say ""
say "Run 'pto-diff --help' for the full flag list."
