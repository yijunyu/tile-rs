//! What has been tried before, and what it cost.
//!
//! Two stores, deliberately:
//!
//! | store | path | contents | gate |
//! |---|---|---|---|
//! | local | `~/.tile-rs/attempts` | *your* conversions and measurements | none |
//! | knowledge | `corpus.sealed` | the curated cross-target corpus | **license** |
//!
//! Both use the same line format, so there is one storage layer rather than two. The
//! knowledge store is **not SQLite**: `rusqlite` compiles C, which fails requirement (3)
//! and our own CI gate, and encrypting it (sqlcipher) makes that worse. It is exported at
//! release time to this format and sealed.
//!
//! ## Doctrine inherited from kernel-impact, and enforced here
//!
//! * **`headroom` NULL means UNASSESSED; `0.0` means CLOSED.** They are different facts
//!   and the reader never collapses one into the other. A large share with zero headroom
//!   is not an opportunity; a large share with unknown headroom is not a zero.
//! * **Headroom is a measurement.** Nothing in this module invents one, and a record
//!   without a stated basis is rejected on the way in, not silently displayed.
//! * **The corpus is a snapshot**, so every report states its export timestamp. Answering
//!   from month-old measurements without saying so violates the same rule as inventing a
//!   number.

use std::fmt;
use std::path::PathBuf;

/// One recorded attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub kernel: String,
    pub target: String,
    /// implemented | partial | absent | planned
    pub state: String,
    /// `None` is UNASSESSED. `Some(0.0)` is CLOSED. Never conflate them.
    pub headroom: Option<f64>,
    /// Why that number. A headroom without one is not admitted.
    pub basis: String,
    pub updated: String,
}

impl Record {
    /// How the headroom should be shown. The three cases are three different facts.
    pub fn headroom_cell(&self) -> String {
        match self.headroom {
            None => "unassessed".to_string(),
            Some(0.0) => "closed".to_string(),
            Some(h) => format!("{:.2}", h),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Corpus {
    pub records: Vec<Record>,
    /// When this snapshot was exported. Empty for the local store, which is live.
    pub exported: String,
    pub source_revision: String,
}

#[derive(Debug)]
pub enum CorpusError {
    Parse { line: usize, what: String },
    Io(String),
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorpusError::Parse { line, what } => write!(f, "corpus line {line}: {what}"),
            CorpusError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Tab-separated, one record per line, with `#` headers for the snapshot metadata.
/// Chosen because it diffs, greps, and cannot grow a query engine by accident.
pub fn parse(text: &str) -> Result<Corpus, CorpusError> {
    let mut c = Corpus::default();
    for (n, raw) in text.lines().enumerate() {
        let line = n + 1;
        let t = raw.trim_end();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix("#exported ") {
            c.exported = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = t.strip_prefix("#revision ") {
            c.source_revision = rest.trim().to_string();
            continue;
        }
        if t.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = t.split('\t').collect();
        if f.len() != 6 {
            return Err(CorpusError::Parse {
                line,
                what: format!("expected 6 tab-separated fields, found {}", f.len()),
            });
        }
        let headroom = match f[3] {
            "" | "NULL" => None,
            v => Some(v.parse::<f64>().map_err(|e| CorpusError::Parse {
                line,
                what: format!("headroom {v:?}: {e}"),
            })?),
        };
        // A headroom is a MEASUREMENT. One with no stated basis is not admitted, because
        // a number whose provenance nobody recorded is indistinguishable from a guess.
        if headroom.is_some() && f[4].trim().is_empty() {
            return Err(CorpusError::Parse {
                line,
                what: "a headroom without a basis is not a measurement".into(),
            });
        }
        c.records.push(Record {
            kernel: f[0].to_string(),
            target: f[1].to_string(),
            state: f[2].to_string(),
            headroom,
            basis: f[4].to_string(),
            updated: f[5].to_string(),
        });
    }
    Ok(c)
}

pub fn render(c: &Corpus) -> String {
    let mut s = String::new();
    if !c.exported.is_empty() {
        s.push_str(&format!("#exported {}\n", c.exported));
    }
    if !c.source_revision.is_empty() {
        s.push_str(&format!("#revision {}\n", c.source_revision));
    }
    for r in &c.records {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            r.kernel,
            r.target,
            r.state,
            r.headroom.map(|h| h.to_string()).unwrap_or_default(),
            r.basis,
            r.updated
        ));
    }
    s
}

/// The user's own attempts. Never gated, never sealed: they are theirs.
pub fn local_path() -> PathBuf {
    crate::provision::home().join("attempts")
}

pub fn load_local() -> Result<Corpus, CorpusError> {
    let p = local_path();
    if !p.exists() {
        return Ok(Corpus::default());
    }
    let text = std::fs::read_to_string(&p).map_err(|e| CorpusError::Io(e.to_string()))?;
    parse(&text)
}

/// Append one attempt to the local store.
pub fn append_local(r: &Record) -> Result<(), CorpusError> {
    use std::io::Write as _;
    let p = local_path();
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| CorpusError::Io(e.to_string()))?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
        .map_err(|e| CorpusError::Io(e.to_string()))?;
    // ONE write call, not one per field.
    //
    // `writeln!` on a File issues a separate syscall for each format fragment, so two
    // processes appending at once interleave mid-record and leave a line with the wrong
    // number of fields -- after which the whole store fails to parse for everybody. A
    // transient race becomes a permanent fault. Found by running four conversions
    // concurrently: "expected 6 tab-separated fields, found 5".
    //
    // A single write to a file opened with O_APPEND is atomic, which is why the line is
    // built first and handed over whole.
    let line = format!(
        "{}\t{}\t{}\t{}\t{}\t{}\n",
        r.kernel,
        r.target,
        r.state,
        r.headroom.map(|h| h.to_string()).unwrap_or_default(),
        r.basis,
        r.updated
    );
    f.write_all(line.as_bytes())
        .map_err(|e| CorpusError::Io(e.to_string()))
}

/// Records touching any of `kernels`. Only what the arguments reach — never the whole
/// corpus, which is a different question and a much longer answer.
pub fn relevant<'a>(c: &'a Corpus, kernels: &[String]) -> Vec<&'a Record> {
    c.records
        .iter()
        .filter(|r| kernels.iter().any(|k| k == &r.kernel))
        .collect()
}

/// The report `-s` prints.
pub fn report(local: &Corpus, knowledge: Option<&Corpus>, kernels: &[String]) -> String {
    let mut s = String::new();
    let rows_local = relevant(local, kernels);
    let rows_know = knowledge.map(|k| relevant(k, kernels)).unwrap_or_default();

    s.push_str(&format!(
        "prior attempts for {}:\n",
        if kernels.is_empty() {
            "(no kernel identified)".to_string()
        } else {
            kernels.join(", ")
        }
    ));
    if rows_local.is_empty() && rows_know.is_empty() {
        s.push_str("  nothing recorded\n");
    }
    for (label, rows) in [("yours", &rows_local), ("corpus", &rows_know)] {
        for r in rows.iter() {
            s.push_str(&format!(
                "  [{label}] {:<10} {:<8} {:<12} headroom {:<11} {}\n",
                r.kernel,
                r.target,
                r.state,
                r.headroom_cell(),
                r.basis
            ));
        }
    }
    match knowledge {
        Some(k) => {
            // The corpus is a snapshot. Answering from month-old measurements without
            // saying so violates the same rule as inventing a number.
            s.push_str(&format!(
                "  corpus exported {} at revision {}\n",
                if k.exported.is_empty() {
                    "(unknown)"
                } else {
                    &k.exported
                },
                if k.source_revision.is_empty() {
                    "(unknown)"
                } else {
                    &k.source_revision
                }
            ));
        }
        None => {
            s.push_str(
                "  the curated cross-target corpus is not open here: it adds measured \
                 headroom,\n  dispatch shares and model-impact projections for these \
                 kernels. `tile license status`\n  says how to enable it. Everything else \
                 above is yours and is always readable.\n",
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
#exported 2026-09-01T10:00:00Z
#revision abc1234
softmax\tmetal\timplemented\t0.31\t113 TFLOP/s vs 85 on the half-filled tiling\t2026-08-01
argmax\tmetal\timplemented\t\t\t2026-08-02
rope\tcpp\timplemented\t0\tbandwidth-saturated at 1.4 TB/s\t2026-07-15
";

    #[test]
    fn unassessed_and_closed_are_different_facts() {
        // The single most important distinction in this module. NULL is not zero: a
        // large share with unknown headroom is not an opportunity that has been closed.
        let c = parse(SAMPLE).unwrap();
        let argmax = c.records.iter().find(|r| r.kernel == "argmax").unwrap();
        let rope = c.records.iter().find(|r| r.kernel == "rope").unwrap();
        assert_eq!(argmax.headroom, None);
        assert_eq!(argmax.headroom_cell(), "unassessed");
        assert_eq!(rope.headroom, Some(0.0));
        assert_eq!(rope.headroom_cell(), "closed");
        assert_ne!(argmax.headroom_cell(), rope.headroom_cell());
    }

    #[test]
    fn a_headroom_without_a_basis_is_refused() {
        // A number whose provenance nobody recorded is indistinguishable from a guess.
        let e = parse("k\tmetal\timplemented\t0.4\t\t2026-01-01\n").unwrap_err();
        assert!(e.to_string().contains("not a measurement"), "{e}");
    }

    #[test]
    fn a_malformed_row_is_an_error_not_a_skip() {
        let e = parse("k\tmetal\timplemented\n").unwrap_err();
        assert!(e.to_string().contains("6 tab-separated"), "{e}");
    }

    #[test]
    fn the_snapshot_metadata_survives_a_round_trip() {
        let c = parse(SAMPLE).unwrap();
        assert_eq!(c.exported, "2026-09-01T10:00:00Z");
        assert_eq!(c.source_revision, "abc1234");
        let again = parse(&render(&c)).unwrap();
        assert_eq!(again.records, c.records);
        assert_eq!(again.exported, c.exported);
    }

    #[test]
    fn only_the_kernels_in_the_arguments_are_reported() {
        let c = parse(SAMPLE).unwrap();
        let rows = relevant(&c, &["softmax".to_string()]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kernel, "softmax");
    }

    #[test]
    fn an_unlicensed_report_still_shows_your_own_attempts() {
        // A missing license degrades a FEATURE; it never removes a capability.
        let local = parse("softmax\tmetal\tpartial\t\t\t2026-08-30\n").unwrap();
        let out = report(&local, None, &["softmax".to_string()]);
        assert!(out.contains("[yours]"), "{out}");
        assert!(out.contains("not open here"), "{out}");
        assert!(out.contains("license status"), "{out}");
    }

    #[test]
    fn a_licensed_report_states_how_old_the_snapshot_is() {
        let local = Corpus::default();
        let know = parse(SAMPLE).unwrap();
        let out = report(&local, Some(&know), &["softmax".to_string()]);
        assert!(out.contains("[corpus]"), "{out}");
        assert!(out.contains("exported 2026-09-01"), "{out}");
        assert!(out.contains("revision abc1234"), "{out}");
    }

    #[test]
    fn nothing_recorded_says_so_rather_than_printing_an_empty_table() {
        let out = report(&Corpus::default(), None, &["nosuch".to_string()]);
        assert!(out.contains("nothing recorded"), "{out}");
    }
}
