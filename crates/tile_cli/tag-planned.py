#!/usr/bin/env python3
"""Tag every scenario `@planned` that is not in the KEEP list.

The discipline (features/README.md): an untagged scenario MUST be green in CI, and
everything else carries exactly one tag saying why it is not. Tags are added ONCE, on
landing; each milestone's change REMOVES the tag from the scenarios it makes pass. Tags
added later, to quiet a red suite, are how a team learns to ignore red.
"""
import sys, os, re

KEEP_FILE = os.path.join(os.path.dirname(__file__), "features", ".live")

def live_set():
    if not os.path.exists(KEEP_FILE):
        return set()
    out = set()
    for line in open(KEEP_FILE):
        line = line.split("#")[0].strip()
        if line:
            out.add(line)
    return out

def main(d):
    live = live_set()
    changed = 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".feature"):
            continue
        path = os.path.join(d, fn)
        lines = open(path).read().split("\n")
        out, i = [], 0
        while i < len(lines):
            l = lines[i]
            t = l.strip()
            if t.startswith("Scenario:") or t.startswith("Scenario Outline:"):
                name = t.split(":", 1)[1].strip()
                key = f"{fn}::{name}"
                # Look BACK past comments and blanks for the tag. Reading only the
                # immediately preceding line means a scenario with an explanatory comment
                # above it looks untagged, gets a second tag, and can never be untagged
                # again -- the same "read past the prose" mistake dead-op elimination and
                # the cfg scanner both made.
                at = len(out) - 1
                while at >= 0 and (not out[at].strip() or out[at].strip().startswith("#")):
                    at -= 1
                already = at >= 0 and out[at].strip().startswith("@")
                # A conditional tag is not a "not built yet" tag: the scenario IS live in
                # a build that has the feature. Leave it alone.
                if already and "requires-stats-build" in out[at]:
                    out.append(l)
                    i += 1
                    continue
                if key in live:
                    if already:
                        out.pop(at)        # untag: this one is live now
                        changed += 1
                elif not already:
                    indent = l[: len(l) - len(l.lstrip())]
                    out.append(indent + "@planned")
                    changed += 1
            out.append(l)
            i += 1
        open(path, "w").write("\n".join(out))
    print(f"{changed} scenario tag(s) adjusted; {len(live)} live")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "features"))
