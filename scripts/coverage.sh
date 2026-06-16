#!/usr/bin/env bash
# coverage.sh — line+region coverage for the OPEN, no-toolchain tile-rs crates.
#
# Measures only the publishable surface that builds & tests with NO LLVM / CANN /
# rustc-dev present (the same property that lets the emitter unit tests run
# off-NPU). One command -> per-crate % + a weighted TOTAL, and a CI gate.
#
# Requires: cargo-llvm-cov (`cargo install cargo-llvm-cov`) + the `llvm-tools`
# rustup component. Both ship on the dev boxes and the GitHub `coverage` job.
#
# Usage:
#   bash scripts/coverage.sh                 # print per-crate + TOTAL table
#   bash scripts/coverage.sh --gate 50       # exit non-zero if TOTAL line% < 50
#   bash scripts/coverage.sh --html          # also write target/llvm-cov/html
#   COVERAGE_GATE=50 bash scripts/coverage.sh # gate via env (used by CI)
#
# NOT measured here (documented in docs/TILE_RS_COVERAGE.md):
#   * tile_std / tile_ir — `no_std`+`no_core` kernel-side crates with
#     `[lib] test = false`; they need the bare-metal nightly target and carry no
#     host-runnable tests, so host line-coverage is N/A by construction.
#   * The full `rustc_codegen_tile` rustc backend — needs LLVM 20 + rustc-dev.
#     Its *pure emit* functions ARE covered here via the `codegen_tests` local
#     copies of `mlir_to_*.rs` / `mlir_parse.rs` (byte-identical to canonical).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GATE="${COVERAGE_GATE:-}"
WANT_HTML=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gate) GATE="$2"; shift 2 ;;
    --gate=*) GATE="${1#*=}"; shift ;;
    --html) WANT_HTML=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUTDIR="$(mktemp -d)"
trap 'rm -rf "$OUTDIR"' EXIT

# Sum line/region tallies in Python across every per-crate JSON we produce.
TALLY="$OUTDIR/tally.py"
cat > "$TALLY" <<'PY'
import json, sys, glob, os
rows = []
tot_lc = tot_lt = tot_rc = tot_rt = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    name = os.path.splitext(os.path.basename(path))[0]
    d = json.load(open(path))
    t = d["data"][0]["totals"]
    lc = t["lines"]["covered"]; lt = t["lines"]["count"]
    rc = t["regions"]["covered"]; rt = t["regions"]["count"]
    tot_lc += lc; tot_lt += lt; tot_rc += rc; tot_rt += rt
    lpct = 100.0*lc/lt if lt else 0.0
    rpct = 100.0*rc/rt if rt else 0.0
    rows.append((name, lpct, lc, lt, rpct, rc, rt))
print("%-22s %9s %15s %9s" % ("crate", "line%", "(cov/total)", "region%"))
print("-"*62)
for name, lpct, lc, lt, rpct, rc, rt in rows:
    print("%-22s %8.2f%% %15s %8.2f%%" % (name, lpct, "%d/%d"%(lc,lt), rpct))
print("-"*62)
TL = 100.0*tot_lc/tot_lt if tot_lt else 0.0
TR = 100.0*tot_rc/tot_rt if tot_rt else 0.0
print("%-22s %8.2f%% %15s %8.2f%%" % ("TOTAL", TL, "%d/%d"%(tot_lc,tot_lt), TR))
with open(os.path.join(sys.argv[1], "TOTAL_LINE"), "w") as f:
    f.write("%.4f" % TL)
PY

check_json() {
  local label="$1"
  if [ ! -s "$OUTDIR/$label.json" ]; then
    echo "!! $label produced no coverage; see below" >&2
    tail -20 "$OUTDIR/$label.err" >&2
    rm -f "$OUTDIR/$label.json"
    FAILED=1
  fi
}

# run_cov <label> <crate-dir> <extra cargo-llvm-cov args...>
# For an EXCLUDED standalone crate (own empty [workspace]): run from its dir so
# the standalone manifest + the package-root filename filter resolve correctly.
run_cov() {
  local label="$1"; local dir="$2"; shift 2
  echo ">> coverage: $label (standalone)" >&2
  ( cd "$dir" && cargo llvm-cov --json --output-path "$OUTDIR/$label.json" "$@" >/dev/null 2>"$OUTDIR/$label.err" )
  check_json "$label"
}

# run_cov_p <label> <pkg> <extra cargo-llvm-cov args...>
# For a workspace-MEMBER crate: run from the repo root via `-p` (running from
# the member dir would see the nested standalone [workspace] roots and error).
run_cov_p() {
  local label="$1"; local pkg="$2"; shift 2
  echo ">> coverage: $label (-p $pkg)" >&2
  ( cd "$REPO_ROOT" && cargo llvm-cov -p "$pkg" --json --output-path "$OUTDIR/$label.json" "$@" >/dev/null 2>"$OUTDIR/$label.err" )
  check_json "$label"
}

FAILED=0

# --- The open core: trait + registry (std-only). ---
run_cov tile_codegen "$REPO_ROOT/crates/tile_codegen"

# --- The HAL: backend-agnostic dispatch traits (a workspace member). ---
run_cov_p tile_hal tile_hal

# --- The kernel-boundary proc-macro (`#[aiv_kernel]`). A proc-macro crate, but
#     its REAL logic (view-type matching, ident extraction, prelude synthesis)
#     lives in `syn`/`proc_macro2`-typed helpers that ARE host-testable with no
#     rustc/trybuild fixture. The `proc_macro::TokenStream` entry itself can only
#     run inside a real expansion, so it stays uncovered here (documented gap).
run_cov_p tile_std_macros tile_std_macros

# --- The executable GWT spec layer + the 14 open pure emitters + shared MLIR
#     parser. `tile_spec`'s cucumber harness `#[path]`-includes the canonical
#     `rustc_codegen_tile/src/mlir_to_*.rs` (no LLVM), so running its tests both
#     drives the GWT scenarios AND runs every emitter's in-source unit tests --
#     this single run reports line/region coverage for ALL 14 open backends,
#     `mlir_parse`, and the std-only `gherkin` runner. (The `codegen_tests`
#     crate carries byte-identical LOCAL copies of the same emitters for the
#     generality-matrix tests; covering them here via the canonical paths avoids
#     double-counting the same source twice under two filenames.) ---
run_cov tile_spec "$REPO_ROOT/crates/tile_spec"

echo
python3 "$TALLY" "$OUTDIR"
echo

if [ "$WANT_HTML" = 1 ]; then
  ( cd "$REPO_ROOT/crates/tile_codegen" && cargo llvm-cov --html >/dev/null 2>&1 ) || true
  echo "HTML report: $REPO_ROOT/crates/tile_codegen/target/llvm-cov/html/index.html" >&2
fi

if [ "$FAILED" = 1 ]; then
  echo "ERROR: one or more crates produced no coverage (build/test failure)." >&2
  exit 1
fi

if [ -n "$GATE" ]; then
  TOTAL_LINE="$(cat "$OUTDIR/TOTAL_LINE")"
  awk -v t="$TOTAL_LINE" -v g="$GATE" 'BEGIN{ if (t+0 < g+0) { printf "GATE FAIL: total line coverage %.2f%% < threshold %s%%\n", t, g; exit 1 } else { printf "GATE OK: total line coverage %.2f%% >= threshold %s%%\n", t, g } }'
  exit $?
fi
