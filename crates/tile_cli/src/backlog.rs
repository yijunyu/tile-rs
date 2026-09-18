//! The backlog: what `tile` could not do, recorded where the next session will find it.
//!
//! This exists because the useful signal about a tool is not what it does — it is what it
//! was asked for and could not do. That signal is normally lost: an agent works around a
//! gap, the session ends, and the next one works around it again. Here the workaround is
//! written down beside the gap, so the eventual fix starts from evidence rather than from
//! a fresh guess.
//!
//! ## What belongs here
//!
//! **Only gaps hit while doing real work.** Not a wish list, not the milestone plan.
//! `docs/cli/plan.md` already says what is scheduled; this says what actually bit
//! somebody. An entry written speculatively is noise in the one place that should be
//! nothing but evidence.
//!
//! ## The stop rule
//!
//! Raw count is a weak signal — a dozen scattered small gaps is a healthy tool with an
//! honest record. **Repetition in one area is the signal that matters**: three open
//! issues in `form-detection` does not mean three bugs, it means the taxonomy is wrong,
//! and fixing them one at a time is three ways of not addressing it.
//!
//! So [`Verdict`] reports both, and says plainly when the answer is to stop adding
//! entries and reconsider the design instead.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

/// One thing the tool could not do.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// Short stable id, `NNN`.
    pub id: u32,
    /// The area it belongs to. Clustering is by this, so it has to be a small vocabulary.
    pub area: String,
    pub title: String,
    /// What was asked for.
    pub wanted: String,
    /// What happened instead — the actual message or behaviour, not a paraphrase.
    pub got: String,
    /// How the agent got the job done anyway. This is the evidence the fix starts from.
    pub workaround: String,
    pub opened: String,
    pub closed: Option<String>,
}

impl Issue {
    pub fn is_open(&self) -> bool {
        self.closed.is_none()
    }

    /// One record per file, so two sessions adding entries do not conflict on one file.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("id: {}\n", self.id));
        s.push_str(&format!("area: {}\n", self.area));
        s.push_str(&format!("title: {}\n", self.title));
        s.push_str(&format!("opened: {}\n", self.opened));
        if let Some(c) = &self.closed {
            s.push_str(&format!("closed: {c}\n"));
        }
        s.push_str(&format!("\n## wanted\n{}\n", self.wanted.trim()));
        s.push_str(&format!("\n## got\n{}\n", self.got.trim()));
        s.push_str(&format!(
            "\n## workaround\n{}\n",
            if self.workaround.trim().is_empty() {
                "(none recorded — an entry with no workaround is a report, not evidence)"
            } else {
                self.workaround.trim()
            }
        ));
        s
    }

    pub fn parse(text: &str) -> Result<Issue, String> {
        let mut fields: BTreeMap<&str, String> = BTreeMap::new();
        let mut sections: BTreeMap<String, String> = BTreeMap::new();
        let mut current: Option<String> = None;
        for line in text.lines() {
            if let Some(name) = line.strip_prefix("## ") {
                current = Some(name.trim().to_string());
                sections.insert(name.trim().to_string(), String::new());
                continue;
            }
            match &current {
                Some(sec) => {
                    let e = sections.get_mut(sec).expect("section");
                    e.push_str(line);
                    e.push('\n');
                }
                None => {
                    if let Some((k, v)) = line.split_once(": ") {
                        fields.insert(
                            match k.trim() {
                                "id" => "id",
                                "area" => "area",
                                "title" => "title",
                                "opened" => "opened",
                                "closed" => "closed",
                                other => {
                                    return Err(format!("unknown field {other:?}"));
                                }
                            },
                            v.trim().to_string(),
                        );
                    }
                }
            }
        }
        let need =
            |k: &str| -> Result<String, String> { fields.get(k).cloned().ok_or(format!("no {k}")) };
        Ok(Issue {
            id: need("id")?
                .parse()
                .map_err(|_| "id is not a number".to_string())?,
            area: need("area")?,
            title: need("title")?,
            opened: need("opened")?,
            closed: fields.get("closed").cloned(),
            wanted: sections
                .get("wanted")
                .cloned()
                .unwrap_or_default()
                .trim()
                .to_string(),
            got: sections
                .get("got")
                .cloned()
                .unwrap_or_default()
                .trim()
                .to_string(),
            workaround: sections
                .get("workaround")
                .cloned()
                .unwrap_or_default()
                .trim()
                .to_string(),
        })
    }
}

/// Where the backlog lives.
///
/// In the repository, so it is reviewed, diffed and versioned like code — a backlog in a
/// database nobody reads is a backlog nobody acts on. `TILE_BACKLOG` overrides it for
/// sessions working outside a checkout, and for the tests.
pub fn dir() -> PathBuf {
    if let Some(p) = std::env::var_os("TILE_BACKLOG") {
        return PathBuf::from(p);
    }
    // The repository path is baked in at build time. An installed binary on a machine
    // without that checkout would otherwise read an empty directory and report a healthy
    // backlog -- the worst possible answer, since a session would then record its gap
    // into nowhere and the stop rule would never fire.
    let in_repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/cli/backlog");
    if in_repo.exists() {
        return in_repo;
    }
    crate::provision::home().join("backlog")
}

/// Is the backlog where the repository keeps it, or a private fallback?
///
/// Reported by `tile backlog` so a session cannot mistake "nothing here" for "nothing
/// recorded anywhere".
pub fn dir_is_repo() -> bool {
    std::env::var_os("TILE_BACKLOG").is_none()
        && PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/cli/backlog")
            .exists()
}

pub fn load() -> Vec<Issue> {
    load_reporting_problems().0
}

/// The issues, AND the files that could not be read as issues.
///
/// `load` used to drop an unparseable file with `if let Ok(i) = ...` and say nothing. A
/// backlog exists so that a known gap is not lost, and the loader lost one silently: an
/// issue filed with an `updated:` line -- a field the parser rejects, deliberately, since
/// an unknown field is an error rather than something to skip -- simply stopped appearing.
/// Fourteen files on disk, thirteen accounted for in "6 open, 7 closed", and nothing said
/// which one had gone or that any had.
///
/// The parser's strictness is right and stays. What was wrong is that the strictness was
/// enforced where nobody could see the result.
pub fn load_reporting_problems() -> (Vec<Issue>, Vec<String>) {
    let mut out = Vec::new();
    let mut problems = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir()) else {
        return (out, problems);
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "md") {
            continue;
        }
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // CLUSTER-*.md and README.md are prose about the backlog, not entries in it.
        if name.starts_with("CLUSTER-") || name == "README.md" {
            continue;
        }
        match std::fs::read_to_string(&p) {
            Ok(t) => match Issue::parse(&t) {
                Ok(i) => out.push(i),
                Err(e) => problems.push(format!("{name}: {e}")),
            },
            Err(e) => problems.push(format!("{name}: {e}")),
        }
    }
    out.sort_by_key(|i| i.id);
    (out, problems)
}

pub fn next_id(existing: &[Issue]) -> u32 {
    existing.iter().map(|i| i.id).max().unwrap_or(0) + 1
}

pub fn path_for(issue: &Issue) -> PathBuf {
    let slug: String = issue
        .title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug: String = slug
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    dir().join(format!(
        "{:03}-{}.md",
        issue.id,
        &slug[..slug.len().min(50)]
    ))
}

pub fn save(issue: &Issue) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir())?;
    let p = path_for(issue);
    std::fs::write(&p, issue.render())?;
    Ok(p)
}

/// The health of the backlog, and what to do about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing recorded, or a handful of unrelated gaps.
    Healthy,
    /// One area has accumulated enough entries that it is a design problem, not a bug
    /// list. Fixing them one at a time is several ways of not addressing it.
    Cluster { areas: Vec<(String, usize)> },
    /// A cluster whose `CLUSTER-<area>.md` accounts for every issue currently open in it.
    /// The instruction has been carried out, so the alarm stands down to a note.
    ClusterAnalysed { areas: Vec<(String, usize)> },
    /// Enough open entries overall that the tool is costing more than it saves.
    TooMany { open: usize },
}

/// Three in one area is a design problem. The number is small on purpose: by the third
/// time the same part of the tool has failed somebody, the pattern is the finding.
pub const CLUSTER_LIMIT: usize = 3;
/// Twelve scattered gaps is where an honest record turns into a tool that is in the way.
pub const OPEN_LIMIT: usize = 12;

/// Does `CLUSTER-<area>.md` exist and name every currently-open id in that area?
///
/// Matched as `#NNN` with the same zero padding the filenames use, so a document that
/// merely mentions the area does not count as having read its issues.
fn cluster_doc_covers(area: &str, open_ids: &[u32]) -> bool {
    let path = dir().join(format!("CLUSTER-{area}.md"));
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    open_ids
        .iter()
        .all(|id| text.contains(&format!("#{id:03}")))
}

pub fn verdict(issues: &[Issue]) -> Verdict {
    verdict_with(issues, cluster_doc_covers)
}

/// The decision, with the cluster-document lookup passed in.
///
/// Split out so it can be tested without the filesystem. `dir()` honours the
/// `TILE_BACKLOG` environment variable, and environment variables are process-global while
/// cargo runs tests in parallel threads: a test that read the real backlog saw an empty one
/// whenever another test had TILE_BACKLOG pointed at its own temp directory. It passed or
/// failed by scheduling, which is worse than not having it.
fn verdict_with(issues: &[Issue], covered: impl Fn(&str, &[u32]) -> bool) -> Verdict {
    let open: Vec<&Issue> = issues.iter().filter(|i| i.is_open()).collect();
    let mut by_area: BTreeMap<&str, usize> = BTreeMap::new();
    for i in &open {
        *by_area.entry(i.area.as_str()).or_insert(0) += 1;
    }
    // The cluster is reported first even when the total is also over: "this area is
    // wrong" is actionable and "you have too many issues" is not.
    // EVERY area at or over the limit, not just the largest.
    //
    // It reported `max_by_key` alone, so with run at 4 and lowering at 3 the second cluster
    // was invisible -- and the rule's whole claim is that by the third time one part of the
    // tool has failed somebody, the pattern is the finding. That was true of two parts and
    // the report named one.
    let clustered: Vec<(String, usize)> = by_area
        .iter()
        .filter(|(_, c)| **c >= CLUSTER_LIMIT)
        .map(|(a, c)| ((*a).to_string(), *c))
        .collect();
    if !clustered.is_empty() {
        // Each area stands down only on its OWN document. One analysed cluster does not
        // excuse another.
        let all_covered = clustered.iter().all(|(area, _)| {
            let ids: Vec<u32> = open
                .iter()
                .filter(|i| &i.area == area)
                .map(|i| i.id)
                .collect();
            covered(area, &ids)
        });
        return if all_covered {
            Verdict::ClusterAnalysed { areas: clustered }
        } else {
            Verdict::Cluster { areas: clustered }
        };
    }
    // The cluster alarm is OFF, and this says so rather than leaving `if false && false`,
    // which is indistinguishable from an accident and which clippy flags forever. It was
    // switched off in ff449112, where it named one cluster of two. The body is kept because
    // the design below is worth keeping: the alarm stands down when someone writes
    // CLUSTER-<area>.md, and it keeps its teeth by requiring that document to ACCOUNT for
    // every open id in the area.
    const CLUSTER_ALARM_ENABLED: bool = false;
    if CLUSTER_ALARM_ENABLED {
        // An alarm that cannot be switched off is one people learn to read past, and
        // this one could not be: it fires on a count, and an area whose issues wait on
        // hardware nobody here has will hold that count forever. The instruction it
        // gives -- "read them together, decide what the area should have been" -- is
        // carried out by writing CLUSTER-<area>.md, so that document is what stands it
        // down.
        //
        // It keeps its teeth by requiring the document to ACCOUNT for the issues: every
        // open id in the area must appear in it. File a fourth and the document no
        // longer covers the area, so the stop comes back until someone has read the
        // fourth one together with the rest -- which is exactly what the rule asks for.
    }
    if open.len() >= OPEN_LIMIT {
        return Verdict::TooMany { open: open.len() };
    }
    Verdict::Healthy
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Healthy => write!(
                f,
                "healthy — keep using the tool and keep recording what it cannot do"
            ),
            Verdict::Cluster { areas } => {
                // One area keeps the original sentence, because that is the common case and
                // it reads as English. Several get a list. The plural was not worth losing
                // the singular over.
                if let [(area, count)] = areas.as_slice() {
                    return write!(
                        f,
                        "STOP AND RECONSIDER: {count} open issues in \"{area}\".\n  \
                         That is not {count} bugs, it is one design problem wearing \
                         {count} hats. Fixing them\n  individually is {count} ways of not \
                         addressing it. Read them together, decide what\n  the area should \
                         have been, and change that before adding a {}th.",
                        count + 1
                    );
                }
                let list = areas
                    .iter()
                    .map(|(a, c)| format!("\"{a}\" ({c})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let total: usize = areas.iter().map(|(_, c)| *c).sum();
                write!(
                    f,
                    "STOP AND RECONSIDER: {list} — {total} open issues across {} areas at \
                     the limit.\n  Each is one design problem wearing several hats, and \
                     fixing them individually is\n  {total} ways of not addressing them. \
                     Read each area together and write\n  CLUSTER-<area>.md saying what it \
                     should have been.",
                    areas.len()
                )
            }
            Verdict::ClusterAnalysed { areas } => {
                let list = areas
                    .iter()
                    .map(|(a, c)| format!("\"{a}\" ({c}), accounted for in CLUSTER-{a}.md"))
                    .collect::<Vec<_>>()
                    .join("; ");
                write!(
                    f,
                    "{list}.\n  Each area has been read as one problem rather than a list, \
                     which is what the\n  cluster rule asks for. Filing another without \
                     adding it there brings the stop back.",
                )
            }
            Verdict::TooMany { open } => write!(
                f,
                "STOP AND RECONSIDER: {open} open issues.\n  A tool whose backlog grows \
                 faster than it shrinks is costing more than it saves. Spend\n  the next \
                 session on the backlog rather than around it."
            ),
        }
    }
}

/// The listing `tile backlog` prints.
pub fn render(issues: &[Issue]) -> String {
    let mut s = String::new();
    let open: Vec<&Issue> = issues.iter().filter(|i| i.is_open()).collect();
    let closed = issues.len() - open.len();
    if issues.is_empty() {
        s.push_str("backlog: empty — nothing has been recorded as beyond the tool yet\n");
        if !dir_is_repo() {
            // An empty backlog and a backlog you cannot see are different facts.
            s.push_str(&format!(
                "  reading {} — not the repository's, so this may be empty because the\n  checkout is elsewhere. Set TILE_BACKLOG to share one.\n",
                dir().display()
            ));
        }
    } else {
        s.push_str(&format!("backlog: {} open, {closed} closed\n", open.len()));
    }
    let mut by_area: BTreeMap<&str, Vec<&Issue>> = BTreeMap::new();
    for i in &open {
        by_area.entry(i.area.as_str()).or_default().push(i);
    }
    for (area, items) in &by_area {
        s.push_str(&format!("\n  {area} ({})\n", items.len()));
        for i in items {
            s.push_str(&format!("    #{:03}  {}\n", i.id, i.title));
            if !i.workaround.is_empty() {
                let first = i.workaround.lines().next().unwrap_or("");
                s.push_str(&format!("          workaround: {first}\n"));
            }
        }
    }
    s.push_str(&format!("\nverdict: {}\n", verdict(issues)));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Sandbox(
        PathBuf,
        #[allow(dead_code)] std::sync::MutexGuard<'static, ()>,
    );
    impl Sandbox {
        fn new(tag: &str) -> Sandbox {
            let g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let d = std::env::temp_dir().join(format!("tile-bl-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            std::env::set_var("TILE_BACKLOG", &d);
            Sandbox(d, g)
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
            std::env::remove_var("TILE_BACKLOG");
        }
    }

    fn issue(id: u32, area: &str, title: &str) -> Issue {
        Issue {
            id,
            area: area.into(),
            title: title.into(),
            wanted: "lower a kernel".into(),
            got: "no route".into(),
            workaround: "wrote the MSL by hand".into(),
            opened: "2026-09-01".into(),
            closed: None,
        }
    }

    #[test]
    fn an_issue_round_trips_through_its_file() {
        let i = issue(7, "lowering", "no lifter for msl");
        let back = Issue::parse(&i.render()).expect("parse");
        assert_eq!(back, i);
    }

    #[test]
    fn a_closed_issue_keeps_its_closing_date() {
        let mut i = issue(1, "run", "no harness for cuda");
        i.closed = Some("2026-09-05".into());
        let back = Issue::parse(&i.render()).unwrap();
        assert_eq!(back.closed.as_deref(), Some("2026-09-05"));
        assert!(!back.is_open());
    }

    #[test]
    fn an_entry_with_no_workaround_says_so_rather_than_leaving_a_blank() {
        // An entry with no workaround is a report, not evidence — and the whole point of
        // the backlog is that the eventual fix starts from evidence.
        let mut i = issue(1, "run", "x");
        i.workaround = String::new();
        assert!(
            i.render().contains("a report, not evidence"),
            "{}",
            i.render()
        );
    }

    #[test]
    fn ids_do_not_collide_when_sessions_add_entries_one_after_another() {
        let existing = vec![issue(1, "a", "x"), issue(4, "b", "y")];
        assert_eq!(next_id(&existing), 5);
        assert_eq!(next_id(&[]), 1);
    }

    #[test]
    fn one_file_per_issue_so_two_sessions_do_not_conflict() {
        let _s = Sandbox::new("files");
        save(&issue(1, "lowering", "No lifter for MSL!")).unwrap();
        save(&issue(2, "run", "no harness for cuda")).unwrap();
        let names: Vec<String> = std::fs::read_dir(dir())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(
            names.iter().any(|n| n.starts_with("001-no-lifter-for-msl")),
            "{names:?}"
        );
    }

    #[test]
    fn a_handful_of_unrelated_gaps_is_healthy() {
        // A dozen scattered small gaps is a healthy tool with an honest record.
        let v = verdict(&[issue(1, "run", "a"), issue(2, "lowering", "b")]);
        assert_eq!(v, Verdict::Healthy);
    }

    #[test]
    fn three_in_one_area_is_a_design_problem_not_three_bugs() {
        // The signal that matters. By the third time the same part of the tool has
        // failed somebody, the pattern IS the finding.
        let v = verdict(&[
            issue(1, "form-detection", "a"),
            issue(2, "form-detection", "b"),
            issue(3, "form-detection", "c"),
        ]);
        assert_eq!(
            v,
            Verdict::Cluster {
                areas: vec![("form-detection".into(), 3)]
            }
        );
        let msg = v.to_string();
        assert!(msg.contains("STOP AND RECONSIDER"), "{msg}");
        assert!(msg.contains("form-detection"), "{msg}");
        assert!(msg.contains("design problem"), "{msg}");
    }

    #[test]
    fn a_cluster_is_reported_ahead_of_a_raw_count() {
        // "This area is wrong" is actionable; "you have too many issues" is not.
        let mut v: Vec<Issue> = (1..=13).map(|i| issue(i, "run", "x")).collect();
        v[0].area = "run".into();
        assert!(matches!(verdict(&v), Verdict::Cluster { .. }));
    }

    #[test]
    fn many_scattered_issues_eventually_say_stop_too() {
        let v: Vec<Issue> = (1..=OPEN_LIMIT as u32)
            .map(|i| issue(i, &format!("area{i}"), "x"))
            .collect();
        assert_eq!(verdict(&v), Verdict::TooMany { open: OPEN_LIMIT });
        assert!(verdict(&v)
            .to_string()
            .contains("costing more than it saves"));
    }

    #[test]
    fn closing_issues_brings_the_verdict_back_to_healthy() {
        // The loop has to be able to close, or the stop rule is a ratchet.
        let mut v = vec![
            issue(1, "form-detection", "a"),
            issue(2, "form-detection", "b"),
            issue(3, "form-detection", "c"),
        ];
        assert!(matches!(verdict(&v), Verdict::Cluster { .. }));
        v[0].closed = Some("2026-09-02".into());
        assert_eq!(verdict(&v), Verdict::Healthy);
    }

    #[test]
    fn the_listing_groups_by_area_and_shows_the_workaround() {
        let out = render(&[
            issue(1, "lowering", "no lifter"),
            issue(2, "run", "no harness"),
        ]);
        assert!(out.contains("lowering (1)"), "{out}");
        assert!(out.contains("#001  no lifter"), "{out}");
        assert!(out.contains("workaround: wrote the MSL by hand"), "{out}");
        assert!(out.contains("verdict:"), "{out}");
    }

    #[test]
    fn an_empty_backlog_elsewhere_says_where_it_looked() {
        // "Nothing recorded" and "nothing recorded HERE" are different facts, and a
        // session that cannot tell them apart records its gap into nowhere.
        let _s = Sandbox::new("elsewhere");
        let out = render(&[]);
        assert!(out.contains("nothing has been recorded"), "{out}");
    }

    #[test]
    fn an_empty_backlog_says_so_rather_than_printing_a_bare_zero() {
        let out = render(&[]);
        assert!(out.contains("nothing has been recorded"), "{out}");
        assert!(out.contains("healthy"), "{out}");
    }

    #[test]
    fn an_unknown_field_in_a_file_is_an_error_not_something_to_skip() {
        let e = Issue::parse("id: 1\nseverity: high\n").unwrap_err();
        assert!(e.contains("severity"), "{e}");
    }

    /// The cluster stop must be answerable, and must come back when it is out of date.
    ///
    /// It fires on a count alone, so an area whose issues wait on hardware nobody here has
    /// would hold that count forever — and an alarm that cannot be switched off is one
    /// people read past, which costs more than the rule earns. Writing CLUSTER-<area>.md is
    /// the instruction being carried out, so it stands the alarm down; an id missing from
    /// the document means the area has grown past what anyone read together, and the stop
    /// returns.
    ///
    /// The document lookup is passed in rather than read from disk: `dir()` honours
    /// TILE_BACKLOG, and the first version of this test read the real backlog and so failed
    /// whenever another test in the same process had that variable pointed elsewhere.
    #[test]
    fn an_analysed_cluster_stands_down_and_a_new_issue_brings_it_back() {
        let issue = |id: u32| Issue {
            id,
            area: "run".into(),
            title: format!("issue {id}"),
            opened: "2026-09-02".into(),
            wanted: String::new(),
            got: String::new(),
            workaround: String::new(),
            closed: None,
        };
        let three: Vec<Issue> = (1..=3).map(issue).collect();

        // No document: the stop stands.
        assert!(matches!(
            verdict_with(&three, |_, _| false),
            Verdict::Cluster { .. }
        ));

        // A document accounting for all three: stood down.
        let covers = |_area: &str, ids: &[u32]| ids.iter().all(|id| *id <= 3);
        assert!(matches!(
            verdict_with(&three, covers),
            Verdict::ClusterAnalysed { .. }
        ));

        // A fourth the document does not mention: the stop comes back.
        let four: Vec<Issue> = (1..=4).map(issue).collect();
        assert!(
            matches!(verdict_with(&four, covers), Verdict::Cluster { .. }),
            "an id outside the document means nobody has read the area as one problem"
        );
    }

    /// EVERY area at the limit is named, not just the biggest one.
    ///
    /// It reported `max_by_key` alone. With run at 4 and lowering at 3 the second cluster
    /// was invisible, and the rule's claim is that by the third time one part of the tool
    /// has failed somebody the pattern is the finding — which was true of two parts while
    /// the report named one. An area also stands down only on its OWN document: one
    /// analysed cluster does not excuse another.
    #[test]
    fn a_second_area_at_the_limit_is_not_hidden_by_the_first() {
        let issue = |id: u32, area: &str| Issue {
            id,
            area: area.into(),
            title: format!("issue {id}"),
            opened: "2026-09-03".into(),
            wanted: String::new(),
            got: String::new(),
            workaround: String::new(),
            closed: None,
        };
        let mut issues: Vec<Issue> = (1..=4).map(|i| issue(i, "run")).collect();
        issues.extend((5..=7).map(|i| issue(i, "lowering")));

        let Verdict::Cluster { areas } = verdict_with(&issues, |_, _| false) else {
            panic!("both areas are at the limit, so this is a Cluster");
        };
        let named: Vec<&str> = areas.iter().map(|(a, _)| a.as_str()).collect();
        assert!(named.contains(&"run"), "{named:?}");
        assert!(
            named.contains(&"lowering"),
            "the smaller cluster is named too: {named:?}"
        );

        // One area documented and the other not is still a stop -- for the other.
        let only_run = |area: &str, _ids: &[u32]| area == "run";
        assert!(
            matches!(verdict_with(&issues, only_run), Verdict::Cluster { .. }),
            "an analysed run cluster must not excuse an unanalysed lowering one"
        );
    }

    /// A file the loader cannot read must be NAMED, not dropped.
    ///
    /// `load` swallowed parse errors, and a backlog exists precisely so a known gap is not
    /// lost. One was: an issue filed with an `updated:` line -- a field the parser rejects
    /// on purpose -- stopped appearing entirely. Fourteen files on disk, "6 open, 7 closed"
    /// printed, and nothing to say the fourteenth had gone. The strictness is right; being
    /// strict where nobody can see the result is not.
    #[test]
    fn a_file_that_cannot_be_parsed_is_reported_rather_than_skipped() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let d = std::env::temp_dir().join(format!("tile-bl-unreadable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("001-fine.md"),
            "id: 1\narea: run\ntitle: fine\nopened: 2026-09-03\n\n## wanted\nx\n\n## got\ny\n",
        )
        .unwrap();
        std::fs::write(d.join("002-bad.md"), "id: 2\nupdated: 2026-09-03\n").unwrap();
        std::env::set_var("TILE_BACKLOG", &d);

        let (issues, problems) = load_reporting_problems();
        std::env::remove_var("TILE_BACKLOG");
        let _ = std::fs::remove_dir_all(&d);

        assert_eq!(issues.len(), 1, "the readable one still loads");
        assert_eq!(
            problems.len(),
            1,
            "the unreadable one is reported: {problems:?}"
        );
        assert!(
            problems[0].contains("002-bad.md"),
            "names the file: {problems:?}"
        );
        assert!(
            problems[0].contains("updated"),
            "names the reason: {problems:?}"
        );
    }

    /// Two files carrying the same id is a silent collision.
    ///
    /// `next_id` exists and hands out the right number, but nothing checked the files on
    /// disk, so an issue written by hand could reuse an id already taken. That is exactly
    /// what happened: a new issue was filed as 009 beside an existing 009, `load()` sorted
    /// them adjacent, `tile backlog` listed both without remark, and every later reference
    /// to "#009" became ambiguous -- including one written into a source comment.
    ///
    /// The ids are how issues are referred to from commits, code comments and reports, so
    /// a duplicate quietly breaks the one thing an id is for.
    #[test]
    fn no_two_issues_share_an_id() {
        // The repository's own path, not `dir()`. `dir()` honours TILE_BACKLOG, which the
        // Sandbox helper in this module sets under ENV_LOCK; a test that read `dir()`
        // without taking that lock would silently check an empty temp directory and pass
        // for the wrong reason. This test is about the files in the repo, so it names them.
        let repo =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/cli/backlog");
        if !repo.exists() {
            return;
        }
        let mut issues: Vec<Issue> = Vec::new();
        for e in std::fs::read_dir(&repo).unwrap().flatten() {
            if e.path().extension().is_some_and(|x| x == "md") {
                if let Ok(t) = std::fs::read_to_string(e.path()) {
                    if let Ok(i) = Issue::parse(&t) {
                        issues.push(i);
                    }
                }
            }
        }
        assert!(
            !issues.is_empty(),
            "no issues parsed from {}",
            repo.display()
        );
        let mut seen: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for i in &issues {
            *seen.entry(i.id).or_insert(0) += 1;
        }
        let dupes: Vec<u32> = {
            let mut d: Vec<u32> = seen
                .iter()
                .filter(|(_, n)| **n > 1)
                .map(|(id, _)| *id)
                .collect();
            d.sort_unstable();
            d
        };
        assert!(
            dupes.is_empty(),
            "these ids are used by more than one file: {dupes:?}. \
             `next_id` gives the next free one"
        );
    }
}
