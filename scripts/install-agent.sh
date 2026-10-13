#!/bin/sh
# install-agent.sh — install an AI coding-agent CLI, and optionally use it to do
# step 0 of the pipeline: author an AscendC kernel from a PyTorch reference.
#
#   curl -fsSL .../install-agent.sh | sh
#       install only (default agent: claude). Never invokes an agent on its own --
#       that costs real API usage (or, for icode, needs infra most machines don't
#       have -- see below), so this script only does it when asked.
#
#   curl -fsSL .../install-agent.sh | sh -s -- --agent codex
#   curl -fsSL .../install-agent.sh | sh -s -- --agent icode
#       install codex or icode instead of claude.
#
#   curl -fsSL .../install-agent.sh | sh -s -- --author kernel_ref.py -o kernel.cpp
#       install, then have the agent author kernel.cpp from kernel_ref.py's PyTorch
#       reference and write only that. The prompt asks for code only, run in each
#       CLI's print-and-exit mode, and THIS script redirects the final answer to -o
#       -- works the same for all three agents, since all three stream progress to
#       stderr and put only the final answer on stdout (icode's `-q` flag matches
#       claude/codex's own default behaviour here).
#
# WHY ICODE IS NOT THE DEFAULT, stated because it was asked for and investigated,
# not assumed: icode (https://github.com/openJiuwen-ai/iCode, Huawei-authored, real
# headless mode: `icode run "<prompt>" -a Code`) ships with two model profiles and
# NEITHER works on a generic machine. `lingxi-default` points at
# lingxi-infer.rnd.huawei.com, a Huawei-internal R&D domain -- DNS failure outside
# their network, confirmed by actually running it, not inferred from docs.
# `local-mind` expects an activated, locally-running `ds4-server` (the same server
# deploy/ds4/install.sh installs) and fails with an invalid-key error otherwise,
# also confirmed by running it. So `--agent icode` installs cleanly but the
# authoring step will only succeed inside Huawei's network OR after you have run
# deploy/ds4/install.sh, activated a licence, and pointed icode's `local-mind`
# profile at it (`icode models` / `icode run ... -m local-mind`) -- this script
# does not do either of those for you, since the second one needs a paid key only
# you can supply.
#
# Modeled on the ds4-localmind installer's shape (explicit flag loop, a `need()`
# command check, a "Next:" trailer) rather than on the other seven scripts in this
# directory: those install OUR CLOSED-source tools from OUR OWN releases, with a
# platform+release to resolve ourselves. This one does neither -- it defers entirely
# to each agent vendor's own official installer, the same "orchestrate, don't
# reimplement" principle install-all.sh uses one layer up.
#
# POSIX sh on purpose: this runs before anything is installed, on whatever the
# machine has. No bashisms, no jq.
set -eu

AGENT="${AGENT:-claude}"
AUTHOR_FILE=""
OUT_FILE=""
PROMPT_OVERRIDE=""
MODEL=""

say() { printf '%s\n' "$*" >&2; }
die() { say "install-agent: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "needs $1 -- install it and re-run"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --agent)   AGENT="$2"; shift 2 ;;
        --author)  AUTHOR_FILE="$2"; shift 2 ;;
        -o)        OUT_FILE="$2"; shift 2 ;;
        --prompt)  PROMPT_OVERRIDE="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;   # icode only: -m profile, e.g. local-mind
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "unknown option $1 (try --help)" ;;
    esac
done

case "$AGENT" in
    claude|codex|icode) ;;
    *) die "--agent must be claude, codex or icode, got '$AGENT'" ;;
esac

need curl

# --- install the chosen agent's CLI, via ITS OWN official installer ---------------
case "$AGENT" in
    claude)
        if command -v claude >/dev/null 2>&1; then
            say "install-agent: claude already on PATH ($(claude --version 2>/dev/null || echo present))"
        else
            say "install-agent: installing Claude Code (native installer, no Node needed)"
            curl -fsSL https://claude.ai/install.sh | bash
        fi
        command -v claude >/dev/null 2>&1 || die "claude installed but is not on PATH yet --
  open a new shell, or source whatever the installer printed, and re-run"
        ;;
    codex)
        if command -v codex >/dev/null 2>&1; then
            say "install-agent: codex already on PATH ($(codex --version 2>/dev/null || echo present))"
        else
            need npm
            say "install-agent: installing Codex CLI (npm install -g @openai/codex)"
            npm install -g @openai/codex
        fi
        command -v codex >/dev/null 2>&1 || die "codex installed but is not on PATH yet --
  check 'npm config get prefix'/bin is on PATH, and re-run"
        ;;
    icode)
        if command -v icode >/dev/null 2>&1; then
            say "install-agent: icode already on PATH ($(icode --version 2>/dev/null || echo present))"
        else
            need uv
            say "install-agent: installing iCode (uv tool install iCode-TUI)"
            uv tool install iCode-TUI
        fi
        command -v icode >/dev/null 2>&1 || die "icode installed but is not on PATH yet --
  check uv's tool bin dir ('uv tool dir --bin') is on PATH, and re-run"
        ;;
esac

say ""
say "install-agent: $AGENT is installed."

# --- no authoring task given: install only, and say what's next -------------------
if [ -z "$AUTHOR_FILE" ] && [ -z "$PROMPT_OVERRIDE" ]; then
    case "$AGENT" in
        claude) EXAMPLE='claude -p "author an AscendC kernel for ..."' ;;
        codex)  EXAMPLE='codex exec "author an AscendC kernel for ..."' ;;
        icode)  EXAMPLE='icode run "author an AscendC kernel for ..." -a Code -q' ;;
    esac
    cat >&2 <<NEXT

Next, step 0 (pytorch -> ascendc) by hand:
    $EXAMPLE

Or let this script build the prompt from a PyTorch reference:
    install-agent.sh --agent $AGENT --author kernel_ref.py -o kernel.cpp
NEXT
    if [ "$AGENT" = icode ]; then
        say ""
        say "icode's built-in model profiles need either Huawei-internal network access"
        say "(lingxi-default) or a running, licensed ds4-server (local-mind) -- see this"
        say "script's own header comment. Run 'icode models' to see what's configured."
    fi
    exit 0
fi

# --- build the prompt ---------------------------------------------------------------
if [ -n "$PROMPT_OVERRIDE" ]; then
    PROMPT="$PROMPT_OVERRIDE"
else
    [ -f "$AUTHOR_FILE" ] || die "--author $AUTHOR_FILE: no such file"
    [ -n "$OUT_FILE" ] || die "--author needs -o <output.cpp> too"
    # The reference's TEXT goes into the prompt rather than just its path: this keeps
    # the agent call a pure text completion (no file-read tool call for it to make, no
    # permission prompt for a non-interactive run to get stuck on), and it is step 0's
    # actual input either way -- golden.py (or whatever the file is) IS the operator's
    # definition in this benchmark, per the pipeline's own step-0 note.
    REF_SRC="$(cat "$AUTHOR_FILE")"
    PROMPT="Author a correct AscendC C++ kernel (not Python, not pseudocode) that
implements the following PyTorch reference exactly. Output ONLY the AscendC C++ source
-- no explanation, no markdown fences, no chat. Start directly with the first
#include or declaration.

PyTorch reference ($AUTHOR_FILE):
$REF_SRC"
fi

say ""
say "install-agent: running $AGENT non-interactively..."

# < /dev/null on every invocation: without it, a CLI that probes stdin for piped
# input (claude does, with a 3s wait before giving up) stalls every run by that
# much when there is nothing to pipe -- found by running it, not by reading --help.
case "$AGENT" in
    claude) RESULT="$(claude -p "$PROMPT" < /dev/null)" ;;
    codex)  RESULT="$(codex exec "$PROMPT" < /dev/null)" ;;
    icode)
        if [ -n "$MODEL" ]; then
            RESULT="$(icode run "$PROMPT" -a Code -m "$MODEL" -q < /dev/null)"
        else
            RESULT="$(icode run "$PROMPT" -a Code -q < /dev/null)"
        fi
        ;;
esac

# A model does not reliably honour "no markdown fences" just because the prompt asks
# -- confirmed: claude wrapped real output in a ```cpp ... ``` fence despite being
# told not to. Strip one leading/trailing fence line mechanically rather than trust
# the instruction, since the whole point of asking for code-only output is to skip
# a human re-editing it afterwards.
FIRST_LINE=$(printf '%s\n' "$RESULT" | head -1)
case "$FIRST_LINE" in
    '```'*)
        RESULT=$(printf '%s\n' "$RESULT" | sed '1d')
        LAST_LINE=$(printf '%s\n' "$RESULT" | tail -1)
        case "$LAST_LINE" in
            '```') RESULT=$(printf '%s\n' "$RESULT" | sed '$d') ;;
        esac
        ;;
esac

if [ -n "$OUT_FILE" ]; then
    printf '%s\n' "$RESULT" > "$OUT_FILE"
    say "install-agent: wrote $OUT_FILE"
    say ""
    say "This is a first draft from an LLM, not a verified kernel -- the pipeline's own"
    say "step 0 note says so: it is the only step that is not reproducible, and the"
    say "grader measures fidelity to the PyTorch reference, not to anything in this"
    say "chain. Lift it with ascendc-to-rs and read the coverage line before trusting it."
else
    printf '%s\n' "$RESULT"
fi
