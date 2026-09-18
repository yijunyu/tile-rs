#!/usr/bin/env bash
# tile-rs installer — one-liner:
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install.sh | bash
#
# Installs the prebuilt rustc_codegen_tile backend (the per-platform shared
# library that lowers tile_std kernels to a vendor source) WITHOUT building it
# from LLVM source. This is the artifact published by the tile-rs codegen release.
#
# The backend is ABI-pinned to one exact nightly; the release tag encodes it:
#   v<ver>+nightly-2025-08-04   (e.g. v0.0.1+nightly-2025-08-04)
# Per-platform asset: tile-rs-codegen-<triple>.tar.gz
#   aarch64-apple-darwin / x86_64-unknown-linux-gnu / aarch64-unknown-linux-gnu
# macOS ships .dylib files; Linux ships .so.
#
# Source of truth: maintained in the PRIVATE repo at ci/tile-rs-public/install.sh
# and deployed here via deploy.toml. Edit it there, not in the public tree.
set -euo pipefail

REPO="yijunyu/tile-rs"
CODEGEN_NIGHTLY="nightly-2025-08-04"
CODEGEN_TAG="${TILERS_CODEGEN_TAG:-v0.0.1+${CODEGEN_NIGHTLY}}"

info()  { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
error() { printf '\033[1;31m==> %s\033[0m\n' "$*"; exit 1; }

host_triple() {
    local os arch
    arch="$(uname -m)"
    case "$(uname -s)" in
        Darwin) os="apple-darwin" ;;
        Linux)  os="unknown-linux-gnu" ;;
        *) error "unsupported OS: $(uname -s)" ;;
    esac
    case "$arch" in
        arm64|aarch64) arch="aarch64" ;;
        x86_64|amd64)  arch="x86_64" ;;
        *) error "unsupported arch: $arch" ;;
    esac
    if [ "$os" = "apple-darwin" ]; then echo "${arch}-apple-darwin"; else echo "${arch}-${os}"; fi
}

main() {
    # NOTE: `tmp` is intentionally NOT local — the `trap ... EXIT` below fires after
    # main() returns, so a function-local `tmp` would be out of scope (unbound under
    # `set -u`) and the temp dir would leak. Keeping it global lets the trap clean up.
    local triple tag tarball url dest
    triple="$(host_triple)"
    tag="$CODEGEN_TAG"
    tarball="tile-rs-codegen-${triple}.tar.gz"
    dest="${TILERS_CODEGEN_HOME:-$(pwd)/tile-rs-codegen}"

    info "Installing tile-rs codegen backend ($triple, $tag)..."

    # The backend dlopens rustc internals and is ABI-pinned to this exact nightly.
    if command -v rustup &>/dev/null; then
        info "Installing pinned toolchain $CODEGEN_NIGHTLY (rustc-dev + llvm-tools + rust-src)..."
        rustup toolchain install "$CODEGEN_NIGHTLY" \
            --component rustc-dev --component llvm-tools --component rust-src 2>/dev/null || true
    else
        warn "rustup not found — install $CODEGEN_NIGHTLY manually (the backend is ABI-pinned to it)."
    fi

    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    # Try gh first (handles the '+' in the tag + private repos cleanly), but ALWAYS
    # fall back to a plain curl of the public release URL ('+' → %2B) if gh is
    # absent OR fails (e.g. unauthenticated gh in CI / on a fresh machine). Only
    # error if BOTH fail. The release is public, so curl needs no auth.
    url="https://github.com/${REPO}/releases/download/${tag//+/%2B}/${tarball}"
    if command -v gh &>/dev/null && gh release download "$tag" --repo "$REPO" --pattern "$tarball" --dir "$tmp" 2>/dev/null; then
        info "Downloaded $tarball via gh."
    else
        info "Downloading $url"
        curl -fSL "$url" -o "$tmp/$tarball" || error "download failed (gh and curl both failed): $url"
    fi

    rm -rf "$dest"; mkdir -p "$dest"
    tar xzf "$tmp/$tarball" --strip-components=1 -C "$dest"
    info "Installed codegen backend to $dest"
    info "Next: see $dest/USAGE.md — set TILERS_CODEGEN_SO + TILERS_CODEGEN_PATH and build with cargo +$CODEGEN_NIGHTLY"
}

main "$@"
