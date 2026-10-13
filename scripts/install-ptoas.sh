#!/bin/sh
# One-line install for Huawei's official `ptoas` (PTO Assembler & Optimizer) -- step 4
# of the pipeline, previously just a pair of links to hiascend.com / the PTOAS repo.
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-ptoas.sh | sh
#
# Not ours: this installs HUAWEI'S own tool from their own public GitHub release
# (hw-native-sys/PTOAS), the same posture as the links it replaces -- it does not
# touch our private source or PRIVATE_SOURCE_TOKEN, and there is no build step: the
# project ships prebuilt wheels.
#
# NO CANN, NO NPU, NO ACCOUNT, NO EULA: confirmed by running the installed binary on a
# Mac with no Ascend hardware and no CANN toolkit present. ptoas is a pure host-side
# compiler (parses .pto, lowers to EmitC/Linalg) -- it does not need the device stack
# the rest of CANN does.
#
# `pip install ptoas` ON PLAIN PyPI INSTALLS THE WRONG PACKAGE, confirmed by actually
# running it: PyPI's `ptoas` is an unrelated, name-squatted "minimal example package
# (with pybind11)", version 0.0.1 -- it installs cleanly, rc=0, no error, the same
# silent-success shape this project keeps running into elsewhere. This script never
# touches plain PyPI; it installs directly from the GitHub release's own wheel URL.
#
# Needs one of python3.10/3.11/3.12 on PATH -- the only interpreters the project
# publishes wheels for. Installs into its own venv so it cannot collide with anything
# already on the machine.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -eu

REPO="hw-native-sys/PTOAS"
VENV_DIR="${PTOAS_VENV_DIR:-$HOME/.local/share/ptoas-venv}"
DEST="${PTOAS_BIN_DIR:-$HOME/.local/bin}"
TAG="${PTOAS_TAG:-}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install-ptoas: $*"; exit 1; }

# --- which wheel platform tag do we need? ------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os-$arch" in
    Darwin-arm64)  PLAT=macosx_26_0_arm64 ;;
    Darwin-x86_64) PLAT=macosx_15_0_x86_64 ;;
    Linux-x86_64)  PLAT=manylinux_2_34_x86_64 ;;
    Linux-aarch64) PLAT=manylinux_2_34_aarch64 ;;
    *) die "unsupported platform $os-$arch. Released platform tags are macosx_26_0_arm64,
  macosx_15_0_x86_64, manylinux_2_34_x86_64 and manylinux_2_34_aarch64." ;;
esac

# --- which python do we have? (only 3.10/3.11/3.12 are published) -----------
PYBIN="" PYTAG=""
for v in 3.12 3.11 3.10; do
    c="python$v"
    if command -v "$c" >/dev/null 2>&1; then
        PYBIN="$c"
        PYTAG="cp${v%.*}${v#3.}"   # "3.11" -> "cp311"
        break
    fi
done
[ -n "$PYBIN" ] || die "need python3.10, python3.11 or python3.12 on PATH -- the
  project's released wheels cover only those three interpreters. Check what's
  installed with 'command -v python3.10 python3.11 python3.12'."

command -v curl >/dev/null 2>&1 || die "curl is required"

# --- resolve the release ------------------------------------------------------
if [ -z "$TAG" ]; then
    TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null |
          sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
fi
[ -n "$TAG" ] || die "could not resolve the latest release from $REPO.
  Set PTOAS_TAG=v0.67 to pick one explicitly, or check:
    https://github.com/$REPO/releases"

VER="${TAG#v}"
WHEEL="ptoas-${VER}-${PYTAG}-${PYTAG}-${PLAT}.whl"
URL="https://github.com/$REPO/releases/download/$TAG/$WHEEL"

say "install-ptoas: $TAG, $PYBIN ($PYTAG), $PLAT"

# --- install into an isolated venv -------------------------------------------
"$PYBIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1 || true
"$VENV_DIR/bin/pip" install --quiet "$URL" ||
    die "wheel install failed: $URL
  Does that release have this asset? Check:
    https://github.com/$REPO/releases/tag/$TAG"

mkdir -p "$DEST"
ln -sf "$VENV_DIR/bin/ptoas" "$DEST/ptoas"

# --- verify it runs ----------------------------------------------------------
"$DEST/ptoas" --help >/dev/null 2>&1 ||
    die "installed to $DEST/ptoas but it does not run.
  Wrong platform build, or a missing system library."

say ""
say "ptoas installed: $DEST/ptoas ($TAG, via $VENV_DIR)"

case ":$PATH:" in
    *":$DEST:"*) : ;;
    *) say ""
       say "$DEST is not on your PATH. Add it:"
       say "    export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Assembling PTO-dialect MLIR to AscendC works now, with no CANN install and no"
say "NPU required for the assembler itself:"
say "    ptoas --enable-insert-sync kernel.pto -o kernel.cpp"
say ""
say "Run 'ptoas --help' for the full flag list."
