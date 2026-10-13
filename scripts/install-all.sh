#!/bin/sh
# One-line install for the whole pipeline, steps 0-7:
#
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install-all.sh | sh
#
# Step 0 (install-agent.sh) installs an agent CLI (claude by default) but never
# INVOKES one on its own -- that costs real API usage, so this script's step-0 call
# carries no --author/-o and just installs. Set AGENT=codex or AGENT=icode before
# running this script to install one of those instead of claude for step 0; see
# install-agent.sh's own header for why icode is not the default.
#
# This is deliberately an ORCHESTRATOR, not a reimplementation: each tool already has its
# own tested install-<name>.sh (platform detection, release resolution, checksum or
# run-it verification), and duplicating that logic here would be a second copy to drift
# out of sync with the first the next time one of them changes. This script fetches and
# runs each one in turn and reports a summary -- nothing more.
#
# Written as seven near-identical explicit blocks rather than a data-driven loop over a
# list: a `cmd | while read` loop runs its body in a SUBSHELL in POSIX sh, so `$?` read
# after the loop (or any variable set inside it) does not reach the code after `done` --
# a real bug in an earlier draft of this script, caught by actually running it rather
# than by reading it. Explicit blocks have nothing to get that wrong.
#
# One tool failing does not stop the rest: ptoas needs python3.10/3.11/3.12 specifically
# (see install-ptoas.sh's own header) and tile ships only 3 of 4 platforms (see
# install-tile.sh's own header) -- a machine missing one of those prerequisites should
# still get the other six.
#
# Per-tool install directories are still controlled by each tool's own env var
# (ASCENDC_TO_RS_BIN_DIR, PTO_RS_BIN_DIR, TILE_BIN_DIR, PTOAS_BIN_DIR,
# PTO_TO_TILERS_BIN_DIR, PTO_DIFF_BIN_DIR, ASCENDC_RS_BIN_DIR) if set individually; this
# script also honours TILE_RS_INSTALL_DIR as a single override for all seven at once,
# and TILE_RS_INSTALL_SKIP as a space-separated list of tool names to leave out (e.g.
# "ptoas tile" on a machine with no python or no minutes for a 4-platform CI matrix's
# worth of download).
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the machine
# has. No bashisms, no jq, no gh.
set -u

BASE="https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts"
RESULTS=""   # lines of "name rc", appended as each tool finishes

say() { printf '%s\n' "$*" >&2; }

command -v curl >/dev/null 2>&1 || { say "install-all: curl is required"; exit 1; }

should_skip() {
    # Is $1 present as a whole word in TILE_RS_INSTALL_SKIP?
    case " ${TILE_RS_INSTALL_SKIP:-} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# $1 = tool name, $2 = script filename, $3 = that script's own BIN_DIR env var name
run_one() {
    name="$1" script="$2" var="$3"
    if should_skip "$name"; then
        say ""
        say "install-all: skipping $name (TILE_RS_INSTALL_SKIP)"
        RESULTS="$RESULTS
$name skipped"
        return
    fi
    say ""
    say "install-all: === $name ==="
    if [ -n "${TILE_RS_INSTALL_DIR:-}" ]; then
        env "$var=$TILE_RS_INSTALL_DIR" sh -c "curl -fsSL '$BASE/$script' | sh"
    else
        sh -c "curl -fsSL '$BASE/$script' | sh"
    fi
    rc=$?
    RESULTS="$RESULTS
$name $rc"
}

# Step 0 does not fit run_one's signature (it takes AGENT, not a BIN_DIR var, and its
# default behaviour -- install only, no --author/-o -- is already the safe one).
say ""
say "install-all: === agent (step 0) ==="
AGENT="${AGENT:-claude}" sh -c "curl -fsSL '$BASE/install-agent.sh' | sh"
rc=$?
RESULTS="$RESULTS
agent $rc"

run_one ascendc-to-rs install-ascendc-to-rs.sh ASCENDC_TO_RS_BIN_DIR
run_one pto-rs        install-pto-rs.sh        PTO_RS_BIN_DIR
run_one tile          install-tile.sh          TILE_BIN_DIR
run_one ptoas         install-ptoas.sh         PTOAS_BIN_DIR
run_one pto-to-tilers install-pto-to-tilers.sh PTO_TO_TILERS_BIN_DIR
run_one pto-diff      install-pto-diff.sh      PTO_DIFF_BIN_DIR
run_one ascendc-rs    install-ascendc-rs.sh    ASCENDC_RS_BIN_DIR

say ""
say "install-all: ================ summary ================"
FAILED=""
OLD_IFS=$IFS
IFS='
'
for line in $RESULTS; do
    [ -n "$line" ] || continue
    rname=${line%% *}
    rrc=${line#* }
    case "$rrc" in
        skipped) say "  skip   $rname" ;;
        0)       say "  ok     $rname" ;;
        *)       say "  FAILED $rname (exit $rrc)"; FAILED="$FAILED $rname" ;;
    esac
done
IFS=$OLD_IFS

if [ -n "$FAILED" ]; then
    say ""
    say "install-all: one or more tools failed:$FAILED"
    say "Each has its own install-<name>.sh with a more specific error above."
    exit 1
fi

say ""
say "install-all: done. Run 'tile doctor' to see what this machine can target."
