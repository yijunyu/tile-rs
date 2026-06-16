#!/usr/bin/env bash
# tile-rs installer — one-liner: curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install.sh | bash
set -euo pipefail

VERSION="${ASCEND_RS_VERSION:-0.1.1}"
REPO="yijunyu/tile-rs"
ARCH="$(uname -m)"
TARBALL="tile-rs-${VERSION}-${ARCH}.tar.gz"
INSTALL_DIR="${ASCEND_RS_HOME:-$(pwd)/ascend-rs-${VERSION}}"
CANN_PATH="${ACLRS_CANN_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
DOCKER_IMAGE="ghcr.io/trusted-programming/ascend-ci:latest"

# ── Codegen-backend install mode ─────────────────────────────────────────────
# Installs the prebuilt rustc_codegen_tile backend (the per-platform shared
# library that lowers tile_std kernels to a vendor source) WITHOUT building it
# from source. This is the artifact published by the tile-rs codegen release.
#
#   curl -fsSL .../install.sh | TILERS_CODEGEN=1 bash
#
# Releases are ABI-pinned to one exact nightly; the tag encodes it:
#   v<ver>-pre+nightly-2025-08-04   (e.g. v0.0.1-pre+nightly-2025-08-04)
# The asset is per-platform: tile-rs-codegen-<triple>.tar.gz, where <triple> is
# aarch64-apple-darwin / x86_64-unknown-linux-gnu / aarch64-unknown-linux-gnu.
# macOS ships .dylib files; Linux ships .so.

CODEGEN_NIGHTLY="nightly-2025-08-04"
CODEGEN_TAG="${TILERS_CODEGEN_TAG:-v0.0.1-pre+${CODEGEN_NIGHTLY}}"

host_triple() {
    local os arch
    arch="$(uname -m)"
    case "$(uname -s)" in
        Darwin) os="apple-darwin" ;;
        Linux)  os="unknown-linux-gnu" ;;
        *) echo "unsupported OS: $(uname -s)" >&2; return 1 ;;
    esac
    case "$arch" in
        arm64|aarch64) arch="aarch64" ;;
        x86_64|amd64)  arch="x86_64" ;;
    esac
    # Apple uses aarch64-apple-darwin (no "unknown")
    if [ "$os" = "apple-darwin" ]; then echo "${arch}-apple-darwin"; else echo "${arch}-${os}"; fi
}

install_codegen() {
    local triple tag tarball url tmp dest
    triple="$(host_triple)"
    tag="$CODEGEN_TAG"
    tarball="tile-rs-codegen-${triple}.tar.gz"
    dest="${TILERS_CODEGEN_HOME:-$(pwd)/tile-rs-codegen}"

    # gh download is the most robust (handles `+` in the tag and private repos);
    # fall back to a direct URL with the `+` percent-encoded as %2B.
    info "Installing tile-rs codegen backend ($triple, $tag)..."
    if command -v rustup &>/dev/null; then
        rustup toolchain install "$CODEGEN_NIGHTLY" \
            --component rustc-dev --component llvm-tools --component rust-src 2>/dev/null || true
    else
        warn "rustup not found — install $CODEGEN_NIGHTLY manually (the backend is ABI-pinned to it)"
    fi

    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
    if command -v gh &>/dev/null; then
        gh release download "$tag" --repo "$REPO" --pattern "$tarball" --dir "$tmp" \
            || error "gh release download failed for $tarball @ $tag"
    else
        url="https://github.com/${REPO}/releases/download/${tag//+/%2B}/${tarball}"
        info "Downloading $url"
        curl -fSL "$url" -o "$tmp/$tarball" || error "download failed: $url"
    fi

    rm -rf "$dest"; mkdir -p "$dest"
    tar xzf "$tmp/$tarball" --strip-components=1 -C "$dest"
    info "Installed codegen backend to $dest"
    info "Next: see $dest/USAGE.md (set TILERS_CODEGEN_SO + TILERS_CODEGEN_PATH, use cargo +$CODEGEN_NIGHTLY)"
    exit 0
}

info()  { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
error() { printf '\033[1;31m==> %s\033[0m\n' "$*"; exit 1; }

# Codegen-backend mode: explicit opt-in (TILERS_CODEGEN=1), or auto on macOS
# (which has no Ascend NPU / CANN / Ascend Docker flow — the codegen release is
# the only thing install.sh can deliver there).
if [ "${TILERS_CODEGEN:-0}" = "1" ] || { [ "$(uname -s)" = "Darwin" ] && [ "${TILERS_CODEGEN:-auto}" != "0" ]; }; then
    install_codegen
fi

# ── Detect environment ───────────────────────────────────────────────────────

has_npu() { command -v npu-smi &>/dev/null && npu-smi info &>/dev/null; }
has_cann() { [ -d "$CANN_PATH" ] && [ -f "$CANN_PATH/bin/setenv.bash" ]; }
has_docker() { command -v docker &>/dev/null; }
has_rust() { command -v rustup &>/dev/null; }

MODE="unknown"
if has_npu; then
    MODE="npu"
    info "Ascend NPU detected (npu-smi found)"
elif has_cann; then
    MODE="sim"
    info "CANN toolkit found at $CANN_PATH (no NPU — will use simulator)"
elif has_docker; then
    MODE="docker"
    info "No CANN or NPU — will use Docker simulator"
else
    error "No NPU, no CANN toolkit, and no Docker found. Install Docker or CANN first."
fi

# ── Install Rust if needed ───────────────────────────────────────────────────

if [ "$MODE" != "docker" ] && ! has_rust; then
    info "Installing Rust nightly toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-08-04
    source "$HOME/.cargo/env"
fi

# ── Download release ─────────────────────────────────────────────────────────

info "Downloading tile-rs v${VERSION} for ${ARCH}..."
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/v${VERSION}/${TARBALL}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

curl -fSL "$DOWNLOAD_URL" -o "$TMPDIR/$TARBALL"
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
tar xzf "$TMPDIR/$TARBALL" --strip-components=1 -C "$INSTALL_DIR"

info "Installed to $INSTALL_DIR"

# ── Run smoke test ───────────────────────────────────────────────────────────

run_native() {
    source "$CANN_PATH/bin/setenv.bash"
    export LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
    if [ "$MODE" = "sim" ]; then
        local sim_lib="$CANN_PATH/${ARCH}-linux/simulator/${ACLRS_SOC_VERSION:-Ascend310P1}/lib"
        [ -d "$sim_lib" ] && export LD_LIBRARY_PATH="$sim_lib:$LD_LIBRARY_PATH"
        export ACLRS_RUN_MODE=sim
        export ACLRS_SOC_VERSION="${ACLRS_SOC_VERSION:-Ascend310P1}"
        export CAMODEL_LOG_PATH="$INSTALL_DIR/sim_log"
    else
        export ACLRS_RUN_MODE=npu
    fi
    cd "$INSTALL_DIR/examples/acl_hello_world"
    cargo run --release 2>&1
}

run_docker() {
    info "Pulling Docker image $DOCKER_IMAGE ..."
    docker pull "$DOCKER_IMAGE"

    CACHE_DIR="${ASCEND_RS_CACHE:-$HOME/.cache/ascend-rs}"
    mkdir -p "$CACHE_DIR/target"
    docker run --rm \
        -v "$INSTALL_DIR:/workspace" \
        -v "$CACHE_DIR/target:/workspace/target" \
        -v "$HOME/.cargo:/root/.cargo" \
        -v "$HOME/.rustup:/root/.rustup" \
        -w /workspace/examples/acl_hello_world \
        -e ACLRS_CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest \
        -e ACLRS_RUN_MODE=sim \
        -e ACLRS_SOC_VERSION=Ascend310P1 \
        -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
        -e CAMODEL_LOG_PATH=/workspace/sim_log \
        -e CARGO_TARGET_DIR=/workspace/target \
        "$DOCKER_IMAGE" bash -c '
            export PATH="/root/.cargo/bin:$PATH"
            SYSROOT=$(rustc --print sysroot)
            SIM_LIB=/usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/simulator/Ascend310P1/lib
            export LD_LIBRARY_PATH="/usr/lib/llvm-20/lib:$SIM_LIB:$SYSROOT/lib:${LD_LIBRARY_PATH:-}"
            export LIBRARY_PATH="$SYSROOT/lib:${LIBRARY_PATH:-}"
            source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
            cargo run --release 2>&1
        '

    # Fix ownership of all root-created files so user can rm -rf later
    docker run --rm -v "$INSTALL_DIR:/workspace" -v "$CACHE_DIR:/cache" alpine \
        sh -c "chown -R $(id -u):$(id -g) /workspace /cache"
}

info "Running hello_world smoke test (mode: $MODE)..."
if [ "$MODE" = "docker" ]; then
    run_docker
else
    run_native
fi

info "Smoke test passed! tile-rs v${VERSION} is ready."
info "Install location: $INSTALL_DIR"
