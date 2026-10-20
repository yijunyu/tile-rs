#!/bin/sh
# One-line install for the `pto-to-tilers` CLI.
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-pto-to-tilers.sh | sh
#
# pto_to_rust itself is private, so an installer hosted there could not be curled --
# publishing the binary and this script from the public repo is the whole reason that
# works, same as ascendc-rs, pto-rs, ascendc-to-rs and pto-diff.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="yijunyu/tile-rs"
DEST="${PTO_TO_TILERS_BIN_DIR:-$HOME/.local/bin}"
TAG="${PTO_TO_TILERS_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-pto-to-tilers: $*"; exit 1; }

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
          sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(pto-to-tilers-v[^"]*\)".*/\1/p' |
          head -1)
fi
[ -n "$TAG" ] || die "no pto-to-tilers-v* release found on $REPO.
  Set PTO_TO_TILERS_TAG=pto-to-tilers-v0.0.1 to pick one explicitly, or check:
    https://github.com/$REPO/releases"

case "$TAG" in
    pto-to-tilers-v*) : ;;
    *) die "PTO_TO_TILERS_TAG='$TAG' is not a pto-to-tilers build. This repository also
  publishes ascendc-rs, ascendc-to-rs, pto-rs, pto-diff and the codegen backend; the
  tags look like pto-to-tilers-v0.0.1." ;;
esac

ASSET="pto-to-tilers-$TRIPLE.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

# --- fetch, unpack -----------------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "install-pto-to-tilers: fetching $TAG for $TRIPLE"
curl -fsSL --retry 2 -o "$TMP/$ASSET" "$URL" ||
    die "download failed: $URL
  Does that release have an asset for $TRIPLE? Check:
    https://github.com/$REPO/releases/tag/$TAG"

tar -xzf "$TMP/$ASSET" -C "$TMP" || die "could not unpack $ASSET"
[ -f "$TMP/pto-to-tilers" ] || die "the archive did not contain a pto-to-tilers binary"

mkdir -p "$DEST"
chmod +x "$TMP/pto-to-tilers"
mv "$TMP/pto-to-tilers" "$DEST/pto-to-tilers"

# --- verify it runs ----------------------------------------------------------
# No --version flag -- --help is the start check.
"$DEST/pto-to-tilers" --help >/dev/null 2>&1 ||
    die "installed to $DEST/pto-to-tilers but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "pto-to-tilers installed: $DEST/pto-to-tilers"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Lifting a PTO-MLIR kernel to tile-rs works now, with no further setup:"
say "    pto-to-tilers kernel.pto -o kernel.rs"
say ""
say "Every run prints a coverage line (modelled/unmodelled op count) -- check it, not"
say "just the exit code: the parser accepts malformed input without erroring."
say ""
say "Run 'pto-to-tilers --help' for the full flag list."
