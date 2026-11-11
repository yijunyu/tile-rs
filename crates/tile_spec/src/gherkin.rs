//! A small, std-only Gherkin parser + step runner. Zero external deps.
//!
//! Supported `.feature` syntax (the practical subset):
//!   * `Feature:` / free-text description lines
//!   * `Background:` (steps run before every scenario in the feature)
//!   * `Scenario:` and `Scenario Outline:`
//!   * `Examples:` tables (header row + data rows) bound to a `Scenario Outline`
//!   * Steps: `Given` / `When` / `Then` / `And` / `But`
//!   * `# comments` and blank lines
//!   * `<placeholder>` substitution from the Examples row
//!
//! Step matching: a step is registered with a *pattern* that may contain
//! `{word}` capture holes. At run time the holes match a single whitespace-free
//! token (or, for the final hole, the rest of the line) and the captured strings
//! are passed to the step fn in order. This is the std-only analogue of
//! cucumber's `#[given("...{}...")]`.

use std::collections::BTreeMap;

/// Which Gherkin keyword introduced a step. `And`/`But` inherit the previous
/// step's effective kind, but for *matching* we treat every step uniformly —
/// the kind is retained only for diagnostics and registration ergonomics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    Given,
    When,
    Then,
}

/// Per-scenario shared state handed to every step. Backends stash emitted source
/// and bookkeeping here so later `Then` steps can assert on it.
#[derive(Default)]
pub struct World {
    /// Free-form string slots (e.g. "emitted_source", "target", "second_run").
    pub strings: BTreeMap<String, String>,
    /// Free-form boolean slots (e.g. "emit_ok").
    pub flags: BTreeMap<String, bool>,
}

impl World {
    pub fn set(&mut self, k: &str, v: impl Into<String>) {
        self.strings.insert(k.to_string(), v.into());
    }
    pub fn get(&self, k: &str) -> &str {
        self.strings.get(k).map(|s| s.as_str()).unwrap_or("")
    }
    pub fn has(&self, k: &str) -> bool {
        self.strings.contains_key(k)
    }
    pub fn set_flag(&mut self, k: &str, v: bool) {
        self.flags.insert(k.to_string(), v);
    }
    pub fn flag(&self, k: &str) -> bool {
        *self.flags.get(k).unwrap_or(&false)
    }
}

type StepFn = Box<dyn Fn(&mut World, &[String])>;

struct Step {
    pattern: String,
    f: StepFn,
}

/// A parsed step inside a scenario.
#[derive(Clone, Debug)]
struct ParsedStep {
    kind: StepKind,
    text: String,
}

/// A parsed scenario (after Outline expansion, one per Examples row).
#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: String,
    steps: Vec<ParsedStep>,
}

/// A parsed `.feature` file.
#[derive(Clone, Debug, Default)]
pub struct Feature {
    pub name: String,
    background: Vec<ParsedStep>,
    pub scenarios: Vec<Scenario>,
}

/// The runner: holds registered steps and executes features.
#[derive(Default)]
pub struct Runner {
    steps: Vec<Step>,
}

impl Runner {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Register a step. The `kind` is advisory (matching is kind-agnostic so an
    /// `And` continuation finds the right fn regardless of which keyword it
    /// chained from). `pattern` may contain `{}` capture holes.
    pub fn step(
        &mut self,
        _kind: StepKind,
        pattern: &str,
        f: impl Fn(&mut World, &[String]) + 'static,
    ) -> &mut Self {
        self.steps.push(Step {
            pattern: pattern.to_string(),
            f: Box::new(f),
        });
        self
    }

    pub fn given(&mut self, p: &str, f: impl Fn(&mut World, &[String]) + 'static) -> &mut Self {
        self.step(StepKind::Given, p, f)
    }
    pub fn when(&mut self, p: &str, f: impl Fn(&mut World, &[String]) + 'static) -> &mut Self {
        self.step(StepKind::When, p, f)
    }
    pub fn then(&mut self, p: &str, f: impl Fn(&mut World, &[String]) + 'static) -> &mut Self {
        self.step(StepKind::Then, p, f)
    }

    /// Find the registered step whose pattern matches `text`, returning the
    /// captured args. Patterns are tried in registration order; the first that
    /// matches wins.
    fn resolve(&self, text: &str) -> Option<(&Step, Vec<String>)> {
        for s in &self.steps {
            if let Some(args) = match_pattern(&s.pattern, text) {
                return Some((s, args));
            }
        }
        None
    }

    /// Run every scenario in `feature`. Panics (test failure) with a precise
    /// message on the first undefined or failing step. Returns the number of
    /// scenarios executed.
    pub fn run_feature(&self, feature: &Feature) -> usize {
        for scenario in &feature.scenarios {
            let mut world = World::default();
            let all_steps = feature.background.iter().chain(scenario.steps.iter());
            for step in all_steps {
                let (s, args) = self.resolve(&step.text).unwrap_or_else(|| {
                    panic!(
                        "UNDEFINED STEP in scenario '{}': {:?} \"{}\"\n  (no registered pattern matches)",
                        scenario.name, step.kind, step.text
                    )
                });
                // A failing assertion inside the step fn panics with its own
                // message; we add scenario context via a catch + re-panic is
                // overkill — the default panic already names the file/line.
                (s.f)(&mut world, &args);
            }
        }
        feature.scenarios.len()
    }

    /// Convenience: parse + run a feature from raw text. Returns scenario count.
    pub fn run_str(&self, feature_text: &str) -> usize {
        let feature = parse_feature(feature_text);
        self.run_feature(&feature)
    }
}

/// Match `text` against `pattern`. `{}` in the pattern captures one token
/// (or, if it is the last segment, the remainder). Returns the captured
/// substrings, or `None` if the literal parts don't line up.
fn match_pattern(pattern: &str, text: &str) -> Option<Vec<String>> {
    let mut caps = Vec::new();
    let mut p = pattern;
    let mut t = text;
    loop {
        match p.find("{}") {
            None => {
                // No more holes: literal tails must be equal.
                return if p == t { Some(caps) } else { None };
            }
            Some(idx) => {
                let lit = &p[..idx];
                if !t.starts_with(lit) {
                    return None;
                }
                t = &t[lit.len()..];
                p = &p[idx + 2..];
                if p.is_empty() {
                    // Trailing hole: capture the rest.
                    caps.push(t.to_string());
                    return Some(caps);
                }
                // Capture up to the next literal char of the pattern.
                let next_lit = p.chars().next().unwrap();
                let stop = t.find(next_lit).unwrap_or(t.len());
                caps.push(t[..stop].to_string());
                t = &t[stop..];
            }
        }
    }
}

/// Parse a `.feature` file into a [`Feature`], expanding `Scenario Outline` +
/// `Examples` into one concrete [`Scenario`] per data row.
pub fn parse_feature(text: &str) -> Feature {
    let mut feature = Feature::default();
    let mut background: Vec<ParsedStep> = Vec::new();

    // State machine over lines.
    enum Mode {
        Top,
        Background,
        Scenario,
        Outline,
        Examples,
    }
    let mut mode = Mode::Top;
    let mut cur_name = String::new();
    let mut cur_steps: Vec<ParsedStep> = Vec::new();
    let mut outline_steps: Vec<ParsedStep> = Vec::new();
    let mut outline_name = String::new();
    let mut ex_header: Vec<String> = Vec::new();
    let mut last_kind = StepKind::Given;

    let flush_scenario = |feature: &mut Feature, name: &str, steps: &[ParsedStep]| {
        if !steps.is_empty() {
            feature.scenarios.push(Scenario {
                name: name.to_string(),
                steps: steps.to_vec(),
            });
        }
    };

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = strip_kw(line, "Feature:") {
            feature.name = rest.trim().to_string();
            mode = Mode::Top;
            continue;
        }
        if let Some(_rest) = strip_kw(line, "Background:") {
            mode = Mode::Background;
            continue;
        }
        if let Some(rest) = strip_kw(line, "Scenario Outline:") {
            // flush any pending plain scenario
            flush_scenario(&mut feature, &cur_name, &cur_steps);
            cur_steps.clear();
            outline_steps.clear();
            outline_name = rest.trim().to_string();
            ex_header.clear();
            mode = Mode::Outline;
            continue;
        }
        if let Some(rest) = strip_kw(line, "Scenario:") {
            flush_scenario(&mut feature, &cur_name, &cur_steps);
            cur_steps.clear();
            cur_name = rest.trim().to_string();
            mode = Mode::Scenario;
            continue;
        }
        if let Some(_rest) = strip_kw(line, "Examples:") {
            ex_header.clear();
            mode = Mode::Examples;
            continue;
        }

        // Step lines (Given/When/Then/And/But).
        if let Some((kind, body)) = parse_step_line(line, &mut last_kind) {
            let step = ParsedStep { kind, text: body };
            match mode {
                Mode::Background => background.push(step),
                Mode::Scenario => cur_steps.push(step),
                Mode::Outline => outline_steps.push(step),
                Mode::Top => { /* steps before any scenario: ignore */ }
                Mode::Examples => { /* steps after Examples are invalid; ignore */ }
            }
            continue;
        }

        // Table rows (Examples). Format: | a | b | c |
        if line.starts_with('|') {
            let cells: Vec<String> = line
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();
            if matches!(mode, Mode::Examples) {
                if ex_header.is_empty() {
                    ex_header = cells;
                } else {
                    // Expand one concrete scenario from this row.
                    let row = &cells;
                    let mut steps = Vec::with_capacity(outline_steps.len());
                    for s in &outline_steps {
                        let mut txt = s.text.clone();
                        for (h, v) in ex_header.iter().zip(row.iter()) {
                            txt = txt.replace(&format!("<{h}>"), v);
                        }
                        steps.push(ParsedStep {
                            kind: s.kind,
                            text: txt,
                        });
                    }
                    let name = if row.is_empty() {
                        outline_name.clone()
                    } else {
                        format!("{} [{}]", outline_name, row.join(", "))
                    };
                    feature.scenarios.push(Scenario { name, steps });
                }
            }
            continue;
        }
        // Anything else (description prose) is ignored.
    }

    // Flush trailing plain scenario.
    flush_scenario(&mut feature, &cur_name, &cur_steps);
    feature.background = background;
    feature
}

fn strip_kw<'a>(line: &'a str, kw: &str) -> Option<&'a str> {
    line.strip_prefix(kw)
}

/// Parse a step line, resolving `And`/`But` to the previous step's kind.
fn parse_step_line(line: &str, last_kind: &mut StepKind) -> Option<(StepKind, String)> {
    for (kw, kind) in [
        ("Given ", StepKind::Given),
        ("When ", StepKind::When),
        ("Then ", StepKind::Then),
    ] {
        if let Some(rest) = line.strip_prefix(kw) {
            *last_kind = kind;
            return Some((kind, rest.trim().to_string()));
        }
    }
    for kw in ["And ", "But ", "* "] {
        if let Some(rest) = line.strip_prefix(kw) {
            return Some((*last_kind, rest.trim().to_string()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_pattern_literal() {
        assert_eq!(match_pattern("hello world", "hello world"), Some(vec![]));
        assert_eq!(match_pattern("hello world", "hello mars"), None);
    }

    #[test]
    fn match_pattern_one_hole() {
        let c = match_pattern("target {}", "target gpu").unwrap();
        assert_eq!(c, vec!["gpu".to_string()]);
    }

    #[test]
    fn match_pattern_mid_hole() {
        let c = match_pattern("emit {} contains {}", "emit gpu contains __global__").unwrap();
        assert_eq!(c, vec!["gpu".to_string(), "__global__".to_string()]);
    }

    #[test]
    fn match_pattern_no_match_returns_none() {
        assert!(match_pattern("target {}", "select gpu").is_none());
    }

    #[test]
    fn world_slots_roundtrip() {
        let mut w = World::default();
        w.set("k", "v");
        w.set_flag("ok", true);
        assert_eq!(w.get("k"), "v");
        assert!(w.has("k"));
        assert!(!w.has("missing"));
        assert_eq!(w.get("missing"), "");
        assert!(w.flag("ok"));
        assert!(!w.flag("nope"));
    }

    #[test]
    fn parse_plain_scenario() {
        let f =
            parse_feature("Feature: demo\n  Scenario: s1\n    Given a\n    When b\n    Then c\n");
        assert_eq!(f.name, "demo");
        assert_eq!(f.scenarios.len(), 1);
        assert_eq!(f.scenarios[0].name, "s1");
        assert_eq!(f.scenarios[0].steps.len(), 3);
        assert_eq!(f.scenarios[0].steps[0].kind, StepKind::Given);
        assert_eq!(f.scenarios[0].steps[1].kind, StepKind::When);
        assert_eq!(f.scenarios[0].steps[2].kind, StepKind::Then);
    }

    #[test]
    fn and_inherits_previous_kind() {
        let f = parse_feature("Feature: x\n Scenario: s\n  Given a\n  And b\n  Then c\n  But d\n");
        let s = &f.scenarios[0];
        assert_eq!(s.steps[1].kind, StepKind::Given); // And after Given
        assert_eq!(s.steps[3].kind, StepKind::Then); // But after Then
    }

    #[test]
    fn comments_and_blanks_ignored() {
        let f = parse_feature("# c\nFeature: x\n\n  # inner\n  Scenario: s\n    Given a\n");
        assert_eq!(f.scenarios.len(), 1);
        assert_eq!(f.scenarios[0].steps.len(), 1);
    }

    #[test]
    fn background_runs_before_each_scenario() {
        let f = parse_feature(
            "Feature: x\n Background:\n  Given base\n Scenario: s1\n  Then a\n Scenario: s2\n  Then b\n",
        );
        assert_eq!(f.background.len(), 1);
        assert_eq!(f.scenarios.len(), 2);
    }

    #[test]
    fn scenario_outline_expands_examples() {
        let f = parse_feature(
            "Feature: x\n Scenario Outline: emit <t>\n  When emit to <t>\n  Then out has <sig>\n Examples:\n  | t | sig |\n  | gpu | __global__ |\n  | msl | kernel |\n",
        );
        assert_eq!(f.scenarios.len(), 2);
        assert_eq!(f.scenarios[0].steps[0].text, "emit to gpu");
        assert_eq!(f.scenarios[0].steps[1].text, "out has __global__");
        assert_eq!(f.scenarios[1].steps[0].text, "emit to msl");
        assert!(f.scenarios[0].name.contains("gpu"));
    }

    #[test]
    fn runner_executes_steps_and_counts() {
        let mut r = Runner::new();
        r.given("a value {}", |w, a| w.set("v", &a[0]))
            .when("doubled", |w, _| {
                let n: i32 = w.get("v").parse().unwrap();
                w.set("v", (n * 2).to_string());
            })
            .then("it is {}", |w, a| assert_eq!(w.get("v"), a[0]));
        let n = r.run_str(
            "Feature: math\n Scenario: dbl\n  Given a value 21\n  When doubled\n  Then it is 42\n",
        );
        assert_eq!(n, 1);
    }

    #[test]
    #[should_panic(expected = "UNDEFINED STEP")]
    fn undefined_step_panics() {
        let r = Runner::new();
        r.run_str("Feature: x\n Scenario: s\n  Given nothing is registered\n");
    }

    #[test]
    #[should_panic]
    fn failing_assertion_in_step_panics() {
        let mut r = Runner::new();
        r.then("it equals {}", |_w, _a| assert_eq!(1, 2));
        r.run_str("Feature: x\n Scenario: s\n  Then it equals 1\n");
    }

    #[test]
    fn star_step_inherits_kind() {
        let f = parse_feature("Feature: x\n Scenario: s\n  Given a\n  * b\n");
        assert_eq!(f.scenarios[0].steps.len(), 2);
        assert_eq!(f.scenarios[0].steps[1].kind, StepKind::Given);
    }
}
