#!/usr/bin/env python3
"""Structural check for the tile CLI .feature files.
Mirrors what tile_spec's zero-dependency gherkin runner accepts, so a file that
passes here will not surprise the harness. Intended to run in CI before anything else."""
import re, glob, sys, os
KEY = ('Given ', 'When ', 'Then ', 'And ', 'But ')
TAGS = {'@planned','@requires-rustc-backend','@requires-lifter','@requires-device',
        '@requires-license','@requires-stats-build','@requires-toolchain',
        '@requires-backend','@requires-hosts','@requires-wasm-runtime',
        '@requires-wasm-target','@requires-cross'}
errs, total, tagged = [], 0, 0
for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv)>1 else '.', '*.feature'))):
    name = os.path.basename(f)
    lines = open(f).read().split('\n')
    seen_feature = in_desc = in_outline = saw_examples = False
    outline_line = 0; cols = None; ph = set(); pending_tags = []
    for i, l in enumerate(lines, 1):
        t = l.strip()
        if not t or t.startswith('#'):
            continue
        if t.startswith('Feature:'):
            seen_feature = True; in_desc = True; continue
        if t.startswith('@'):
            in_desc = False
            bad = [x for x in t.split() if x not in TAGS]
            if bad: errs.append(f"{name}:{i} unknown tag(s) {bad}")
            pending_tags = t.split(); continue
        if t.startswith(('Scenario Outline:', 'Scenario:', 'Background:')):
            in_desc = False
            if in_outline and not saw_examples:
                errs.append(f"{name}:{outline_line} Scenario Outline with no Examples")
            in_outline = t.startswith('Scenario Outline:')
            outline_line = i; saw_examples = False; ph = set(); cols = None
            if not t.startswith('Background:'):
                total += 1
                if pending_tags: tagged += 1
            pending_tags = []; continue
        if in_desc:
            continue
        if t.startswith('Examples:'):
            if not in_outline: errs.append(f"{name}:{i} Examples outside a Scenario Outline")
            saw_examples = True; cols = None; continue
        if t.startswith('|'):
            cells = [c.strip() for c in t.strip('|').split('|')]
            if cols is None:
                cols = len(cells)
                missing = ph - set(cells)
                if in_outline and missing:
                    errs.append(f"{name}:{i} placeholder(s) with no column: {sorted(missing)}")
            elif len(cells) != cols:
                errs.append(f"{name}:{i} row has {len(cells)} cells, header has {cols}")
            continue
        if t.startswith(KEY):
            if in_outline: ph |= set(re.findall(r'<([^>]+)>', t))
            continue
        errs.append(f"{name}:{i} not a step or keyword (multi-line step?): {t[:60]!r}")
    if in_outline and not saw_examples:
        errs.append(f"{name}:{outline_line} Scenario Outline with no Examples")
    if not seen_feature:
        errs.append(f"{name}: no Feature: line")
if errs:
    print("PARSE ERRORS:"); [print("  " + e) for e in errs]; sys.exit(1)
print(f"OK: {total} scenarios, {tagged} tagged, {total-tagged} must be green in CI")
