# What `tile` could not do

One file per issue, in git, reviewed like code. A backlog in a database nobody reads is a
backlog nobody acts on.

**Only gaps hit while doing real work belong here.** [`../plan.md`](../plan.md) already
says what is scheduled; this says what actually bit somebody. An entry written
speculatively is noise in the one place that should be nothing but evidence.

```bash
tile backlog                 # grouped by area, with a verdict. exit 0 healthy, 3 stop
tile backlog add             # reads the entry from stdin
tile backlog close <id>
```

The `tile-rs` skill (`~/.claude/skills/tile-rs/SKILL.md`) is what fires this: try the
tool, and only when it refuses with **exit 3** — *tile-rs cannot do this yet* — record the
gap, solve it by hand, then write the workaround back into the entry.

## The workaround is the point

An entry with no workaround is a report. An entry with one is evidence, and the eventual
fix starts from evidence rather than a fresh guess. Record what you did instead, anything
the fix will need (a signature, an ABI, an env var, a name that is not what it looks
like), and what would have let the tool do it.

## The stop rule

Raw count is a weak signal — a dozen scattered gaps is a healthy tool with an honest
record. **Repetition in one area is the signal that matters.**

* **Three open in one area** is not three bugs. It is one design problem wearing three
  hats, and fixing them individually is three ways of not addressing it.
* **Twelve open overall** means the backlog grows faster than it shrinks and the tool is
  costing more than it saves.

Either fires and `tile backlog` exits 3. The right response is to say so and propose the
redesign, not to add a fourth entry. Close issues as the tool grows into them, or the rule
becomes a ratchet that only ever says stop.
