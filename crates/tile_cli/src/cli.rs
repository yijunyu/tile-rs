//! Argument parsing, by hand.
//!
//! The grammar is small and frozen, and this repo's whole dependency graph is 18 crates
//! — `tile_spec` hand-rolled a Gherkin parser rather than take one dependency. A
//! hand-written parser keeps the default binary honest against that budget, and it makes
//! the one deliberate wart explicit: **`-v` is version and `-V` is verbose**, which
//! inverts the GNU convention. Both long forms are unambiguous, and a lone `-V` warns.

use std::fmt;

/// Stable exit codes, grouped by **who can fix it and how** — which is the only
/// grouping a script or an agent can act on.
///
/// The distinction that matters most here is [`UNSUPPORTED`] versus [`UNAVAILABLE`].
/// "tile-rs has no Metal frontend" and "this binary was built without the Ascend
/// targets" are not the same answer: the second is fixed by fetching a different
/// binary, the first is not fixed by anything the caller can do today. Collapsing them
/// tells a caller to retry forever, or to give up on something that is one flag away.
///
/// Documented in `--help` and asserted by `features/10_portability_and_grammar.feature`.
pub mod exit {
    pub const OK: i32 = 0;
    /// The transformation ran and failed.
    pub const TRANSFORM_FAILED: i32 = 1;
    /// The command as typed cannot be satisfied, but a nearby command can. Covers a
    /// malformed flag AND a route that exists but must be asked for by name (`--via`):
    /// in both cases the remedy is to type something else, right now.
    pub const USAGE: i32 = 2;
    /// tile-rs cannot do this **yet**. No command, build or purchase changes it today —
    /// the capability has not been written. NEVER a claim that it is impossible: a
    /// missing frontend is a piece of work nobody has done, not a law of nature, and the
    /// message must carry that tense.
    pub const UNSUPPORTED: i32 = 3;
    /// It exists, but not here. A toolchain to install, or a target this binary was
    /// built without. Acquirable: retrying after acquisition is the right move.
    pub const UNAVAILABLE: i32 = 4;
    pub const LICENSE: i32 = 5;
    /// Real hardware is required and none was detected.
    pub const DEVICE: i32 = 6;
    /// The UI or daemon subsystem could not start.
    pub const SUBSYSTEM: i32 = 7;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cmd {
    /// The default: act on the input files.
    Run,
    Doctor,
    ListForms,
    /// `--list-ops`: which tile_std ops each backend actually lowers, measured.
    ListOps,
    /// `tile install [<id>]` — provision a toolchain ahead of time, or list what the
    /// manifest knows about for this platform.
    Install(Option<String>),
    /// `-d` — MCP over stdio.
    Daemon,
    /// `tile license status`
    License,
    /// `-u` — serve the view on loopback.
    Ui,
    /// `tile backlog [add|close <id>]` — what the tool could not do.
    Backlog(Option<String>),
    Help,
    Version,
    /// Recognised, not yet implemented. Named so the user gets a milestone, not a
    /// "unknown option".
    NotYet(&'static str, &'static str),
}

#[derive(Debug, Clone)]
pub struct Args {
    pub cmd: Cmd,
    pub inputs: Vec<String>,
    pub output: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub opt_level: Option<u8>,
    pub info_only: bool,
    pub keep: bool,
    pub keep_dir: Option<String>,
    pub show_route: bool,
    pub via: Vec<String>,
    pub cross: Option<String>,
    pub force: bool,
    pub json: bool,
    /// `doctor --targets`: the per-target codegen/on-hardware table.
    pub targets: bool,
    /// `--ui [native|web]`. `None` is the default web view.
    pub ui_mode: Option<String>,
    pub verbose: u8,
    pub tile: usize,
    pub stats: bool,
    pub ui_port: u16,
    pub model: Option<String>,
    pub run: bool,
    pub iterations: usize,
    pub offline: bool,
    pub no_install: bool,
    pub yes: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            cmd: Cmd::Run,
            inputs: Vec::new(),
            output: None,
            from: None,
            to: None,
            opt_level: None,
            info_only: false,
            keep: false,
            keep_dir: None,
            show_route: false,
            via: Vec::new(),
            cross: None,
            force: false,
            json: false,
            targets: false,
            ui_mode: None,
            verbose: 0,
            tile: 8192,
            stats: false,
            ui_port: 0,
            model: None,
            run: false,
            iterations: 50,
            offline: false,
            no_install: false,
            yes: false,
        }
    }
}

impl Args {
    /// The level actually used: bare `-O` and the default are both O2 — the highest
    /// level that needs no network, no toolchain, no device and no clock.
    pub fn level(&self) -> u8 {
        self.opt_level.unwrap_or(2)
    }
}

#[derive(Debug)]
pub struct UsageError(pub String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn need_value(
    it: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
    flag: &str,
) -> Result<String, UsageError> {
    it.next()
        .ok_or_else(|| UsageError(format!("{flag} needs a value")))
}

pub fn parse<I: IntoIterator<Item = String>>(argv: I) -> Result<Args, UsageError> {
    let all: Vec<String> = argv.into_iter().collect();
    let mut it = all.into_iter().peekable();
    let mut a = Args::default();
    let mut saw_verbose_alone = false;

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => a.cmd = Cmd::Help,
            "-v" | "--version" => a.cmd = Cmd::Version,
            "-V" | "--verbose" => {
                a.verbose += 1;
                saw_verbose_alone = true;
            }
            "doctor" => a.cmd = Cmd::Doctor,
            "backlog" => a.cmd = Cmd::Backlog(None),
            "install" => {
                // `tile install` with no id lists what this platform can be given.
                a.cmd = Cmd::Install(None);
            }
            "--offline" => a.offline = true,
            "--no-install" => a.no_install = true,
            "--yes" => a.yes = true,
            "--list-ops" => a.cmd = Cmd::ListOps,
            "--list-forms" => a.cmd = Cmd::ListForms,
            "-i" | "--info-only" => a.info_only = true,
            "-k" | "--keep" => a.keep = true,
            "-r" | "--run" => a.run = true,
            "--iterations" => {
                let v = need_value(&mut it, "--iterations")?;
                a.iterations = v
                    .parse()
                    .map_err(|_| UsageError(format!("--iterations wants a number, got {v:?}")))?;
                if a.iterations == 0 {
                    return Err(UsageError("--iterations 0 measures nothing".into()));
                }
            }
            "--force" => a.force = true,
            "--json" => a.json = true,
            "--targets" => a.targets = true,
            "--route" => a.show_route = true,
            "-o" => a.output = Some(need_value(&mut it, "-o")?),
            "-f" | "--from" => a.from = Some(need_value(&mut it, "--from")?),
            "-t" | "--to" => a.to = Some(need_value(&mut it, "--to")?),
            "--keep-dir" => {
                a.keep_dir = Some(need_value(&mut it, "--keep-dir")?);
                a.keep = true;
            }
            "--via" => {
                let v = need_value(&mut it, "--via")?;
                a.via = v.split(',').map(|s| s.trim().to_string()).collect();
            }
            "--cross" => a.cross = Some(need_value(&mut it, "--cross")?),
            "--tile" => {
                let v = need_value(&mut it, "--tile")?;
                a.tile = v
                    .parse()
                    .map_err(|_| UsageError(format!("--tile wants a number, got {v:?}")))?;
            }
            "-O" => a.opt_level = Some(2),
            "-s" | "--stats" => a.stats = true,
            "license" => a.cmd = Cmd::License,
            "-d" | "--daemon" => a.cmd = Cmd::Daemon,
            "--listen" => {
                let addr = need_value(&mut it, "--listen")?;
                // "Binds localhost" is not a security story for a server that will one
                // day hand out license-gated data. A token comes first.
                return Err(UsageError(format!(
                    "--listen {addr} is not available: a TCP daemon serving licensed data \
                     needs an auth token, and there is none yet. Use stdio (`tile -d`), \
                     which is what MCP clients expect."
                )));
            }
            "-u" | "--ui" => {
                a.cmd = Cmd::Ui;
                // The documented grammar is `--ui [native]`, but `--ui` took no value, so
                // `tile k.mlir --ui native` treated "native" as an INPUT FILENAME and
                // then silently served the web UI. The user asked for a window and got an
                // HTTP server with nothing said -- and the command hung, because the
                // server blocks.
                //
                // Consume the mode only when the token is exactly one of the modes AND no
                // file of that name exists, so a kernel someone really did call `native`
                // is still readable.
                if let Some(next) = it.peek() {
                    let n = next.as_str();
                    if (n == "native" || n == "web") && !std::path::Path::new(n).exists() {
                        a.ui_mode = Some(n.to_string());
                        it.next();
                    }
                }
            }
            "--ui-port" => {
                let v = need_value(&mut it, "--ui-port")?;
                a.ui_port = v
                    .parse()
                    .map_err(|_| UsageError(format!("--ui-port wants a number, got {v:?}")))?;
            }
            "-m" | "--model" => a.model = Some(need_value(&mut it, "--model")?),
            "--" => {
                a.inputs.extend(it.by_ref());
                break;
            }
            other => {
                if let Some(rest) = other.strip_prefix("-O") {
                    let n: u8 = rest
                        .parse()
                        .map_err(|_| UsageError(format!("unknown optimization level {other:?}")))?;
                    if n > 4 {
                        return Err(UsageError(format!(
                            "-O{n}: levels are 0 to 4 (see `tile --help`)"
                        )));
                    }
                    a.opt_level = Some(n);
                } else if other == "-" {
                    a.inputs.push("-".to_string());
                } else if other.starts_with("-V") && other[1..].chars().all(|c| c == 'V') {
                    a.verbose += (other.len() - 1) as u8;
                    saw_verbose_alone = true;
                } else if other.starts_with('-') && other.len() > 1 {
                    return Err(UsageError(format!("unknown option {other:?}")));
                } else if matches!(a.cmd, Cmd::Backlog(None)) {
                    a.cmd = Cmd::Backlog(Some(other.to_string()));
                } else if let Cmd::Backlog(Some(sub)) = &a.cmd {
                    // `backlog close 3` — the id rides along with the subcommand.
                    a.cmd = Cmd::Backlog(Some(format!("{sub} {other}")));
                } else if matches!(a.cmd, Cmd::Install(None)) {
                    // The bare word after `install` is the tool id, not an input file.
                    a.cmd = Cmd::Install(Some(other.to_string()));
                } else {
                    a.inputs.push(other.to_string());
                }
            }
        }
    }

    // `-V` alone, with nothing to be verbose about, is almost always someone expecting
    // GNU's `-V` = version. Say so rather than printing nothing.
    if saw_verbose_alone && a.inputs.is_empty() && a.cmd == Cmd::Run {
        return Err(UsageError(
            "-V is VERBOSE in this tool, not version. Did you mean -v/--version?".into(),
        ));
    }

    // Contradictions are usage errors, not silent precedence rules.
    if a.run && a.info_only {
        return Err(UsageError(
            "--run needs a kernel to run, and --info-only produces none".into(),
        ));
    }
    if a.run && a.cross.is_some() {
        // Running means running HERE. A cross-generated kernel targets a device that is
        // by definition not this one.
        return Err(UsageError(
            "--run executes on this machine, which --cross says is not the target".into(),
        ));
    }
    if a.info_only && a.output.is_some() {
        return Err(UsageError("--info-only produces no output; drop -o".into()));
    }
    if a.inputs.len() > 1 && a.output.is_some() {
        return Err(UsageError(
            "-o names one file but several inputs were given; use an output directory".into(),
        ));
    }
    if a.inputs.iter().any(|i| i == "-") && a.from.is_none() {
        return Err(UsageError(
            "stdin has no name to sniff; pass -f <form> to say what it is".into(),
        ));
    }
    if a.cross.is_some() && a.opt_level == Some(4) {
        return Err(UsageError(
            "-O4 measures on hardware, which the native target alone can do; --cross cannot".into(),
        ));
    }
    if a.model.is_some() && a.cmd != Cmd::Daemon {
        // An engine provisioned beside nothing is an engine nobody can reach.
        return Err(UsageError(
            "-m provisions an engine beside the daemon, so it needs -d".into(),
        ));
    }
    if a.offline && a.no_install {
        // Not a contradiction, but saying both hides which one is doing the work when a
        // refusal arrives. --offline already implies --no-install.
        a.no_install = false;
    }
    if a.cmd == Cmd::Run && a.inputs.is_empty() && !a.stats {
        a.cmd = Cmd::Help;
    }
    Ok(a)
}

pub const HELP: &str = "\
tile — the tile-rs kernel swissknife

USAGE
  tile [OPTIONS] <INPUT>...
  tile doctor                      report platform, accelerators and SDKs
  tile install [<tool>]            provision a toolchain ahead of time, or list them
  tile license status              what a license key would enable, and whether one is here
  tile backlog                     what tile could NOT do, grouped by area, with a verdict
  tile backlog add                 record a gap (reads the fields from stdin)
  tile backlog close <id>          mark one fixed
  tile --list-forms                the reader/writer matrix
  tile --list-ops                  which tile_std ops each backend lowers, measured

WHAT IT DOES
  A single input, no -o     profile the kernel, then optimize it for THIS machine
  Input and output differ   lower (down the stack) or lift (up it)
  Input and output match    optimize — for the forms tile-rs can READ (tile, mlir)

OPTIONS
  -o <FILE>          output path; wins over any extension inference
                     \"-\" as an input reads stdin (needs -f); -o - writes stdout
  -f, --from <FORM>  force the input form      -t, --to <FORM>  force the output form
  -O[0-4]            optimization level; bare -O is -O2, and -O2 is the default
                       O0 verbatim · O1 local · O2 +fusion/tiling/hazards
                       O3 +toolchain · O4 +measured on hardware
  -i, --info-only    profile and report; change nothing
  -k, --keep         keep intermediate representations
  -r, --run          run the generated kernel on this machine: check it against a
                     reference (and against PyTorch when available), and time it
      --iterations N samples to take when timing (default 50)
      --keep-dir D   keep them here (implies -k)
      --route        print the candidate routes and exit
      --via a,b      pin the intermediate forms (a lift is only ever taken this way)
      --cross <P>    generate for a platform other than the detected one
      --force        overwrite an existing output (never implicit)
      --tile <N>     tile width to plan against (default 8192)
      --json         machine-readable output where it exists
      --offline      never touch the network; diagnose instead
      --no-install   do not acquire anything; print the command that would
      --yes          assume yes where a choice would otherwise be offered
  -s, --stats        prior attempts for the kernels in the arguments
  -d, --daemon       serve MCP over stdio, for agents
  -m, --model <M>    provision a ds4-rs engine beside the daemon (needs -d)
  -u, --ui           serve the view on loopback and open a browser
      --ui-port <N>  bind this port instead of an ephemeral one
  -V, --verbose      repeatable (-VV, -VVV). NOTE: -V is verbose, -v is version.
  -v, --version      -h, --help

EXIT CODES  (grouped by who can fix it, and how)
  0  ok
  1  the transformation ran and failed
  2  the command as typed cannot be satisfied, but a nearby one can (incl. --via)
  3  tile-rs cannot do this YET — the capability has not been written
  4  it exists, but not here — a toolchain to install, or a target this build lacks
  5  a license is required            6  a device is required and none was detected
  7  the UI or daemon could not start

  3 is never a claim of impossibility. A missing frontend is work nobody has done.

ENVIRONMENT
  TILE_SIMULATE=<absences>   run as if something were missing, for testing and for
                            reproducing a bug report from a barer machine. Comma list of
                            no-emitters, no-toolchains, no-devices, no-vendor-clis,
                            no-network, no-license, or `bare` for all of them. A
                            simulated run ALWAYS says so on stderr; it can only ever
                            remove a capability, never grant one.

No telemetry. Nothing is downloaded except from the pinned manifest, and every fetch is
printed before it starts.
";

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Args, UsageError> {
        parse(s.split_whitespace().map(str::to_string))
    }

    #[test]
    fn version_and_verbose_are_the_specified_way_round() {
        assert_eq!(p("-v").unwrap().cmd, Cmd::Version);
        assert_eq!(p("k.mlir -V").unwrap().verbose, 1);
    }

    #[test]
    fn a_lone_dash_capital_v_explains_the_inversion() {
        let e = p("-V").unwrap_err();
        assert!(e.0.contains("VERBOSE"), "{e}");
        assert!(e.0.contains("-v/--version"), "{e}");
    }

    #[test]
    fn verbosity_is_repeatable_both_ways() {
        assert_eq!(p("k.mlir -VV").unwrap().verbose, 2);
        assert_eq!(p("k.mlir -V -V -V").unwrap().verbose, 3);
    }

    #[test]
    fn bare_o_is_o2_and_so_is_the_default() {
        assert_eq!(p("k.mlir -O").unwrap().level(), 2);
        assert_eq!(p("k.mlir").unwrap().level(), 2);
        assert_eq!(p("k.mlir -O3").unwrap().level(), 3);
        assert_eq!(p("k.mlir -O0").unwrap().level(), 0);
    }

    #[test]
    fn an_out_of_range_level_names_the_range() {
        let e = p("k.mlir -O7").unwrap_err();
        assert!(e.0.contains("0 to 4"), "{e}");
    }

    #[test]
    fn info_only_with_o_is_a_usage_error_not_a_precedence_rule() {
        let e = p("k.mlir -i -o out.metal").unwrap_err();
        assert!(e.0.contains("--info-only produces no output"), "{e}");
    }

    #[test]
    fn several_inputs_with_one_o_is_refused() {
        let e = p("a.mlir b.mlir -o out.metal").unwrap_err();
        assert!(e.0.contains("output directory"), "{e}");
    }

    #[test]
    fn stdin_without_from_is_refused_because_there_is_no_name_to_sniff() {
        let e = p("- -t msl").unwrap_err();
        assert!(e.0.contains("stdin"), "{e}");
        assert!(e.0.contains("-f"), "{e}");
    }

    #[test]
    fn stdin_with_from_is_accepted() {
        let a = p("- -f mlir -t msl -o out.metal").unwrap();
        assert_eq!(a.inputs, vec!["-".to_string()]);
    }

    #[test]
    fn o4_under_cross_is_a_usage_error() {
        let e = p("k.rs --cross ascend -O4 -o out.cce").unwrap_err();
        assert!(e.0.contains("measures on hardware"), "{e}");
    }

    #[test]
    fn keep_dir_implies_keep() {
        let a = p("k.mlir --keep-dir ./ir -t msl").unwrap();
        assert!(a.keep);
        assert_eq!(a.keep_dir.as_deref(), Some("./ir"));
    }

    #[test]
    fn via_is_a_comma_list() {
        let a = p("k.mlir --via mlir,linalg -t rvv").unwrap();
        assert_eq!(a.via, vec!["mlir".to_string(), "linalg".to_string()]);
    }

    #[test]
    fn every_documented_flag_is_implemented_rather_than_a_stub() {
        // This test used to assert that -s, -d, -u and -m named the milestone they were
        // waiting for. They have all landed, so what it asserts now is that NONE of them
        // is a stub — and it will fail again the moment a new one is added as one, which
        // is the point.
        for argv in [
            "k.mlir -s",
            "-d",
            "k.mlir -u",
            "-d -m qwen3",
            "install",
            "license status",
            "doctor",
        ] {
            let a = p(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
            assert!(
                !matches!(a.cmd, Cmd::NotYet(..)),
                "{argv:?} is still a stub: {:?}",
                a.cmd
            );
        }
    }

    #[test]
    fn an_unknown_option_is_refused_by_name() {
        let e = p("k.mlir --wat").unwrap_err();
        assert!(e.0.contains("--wat"), "{e}");
    }

    #[test]
    fn no_arguments_at_all_shows_help() {
        assert_eq!(p("").unwrap().cmd, Cmd::Help);
    }

    #[test]
    fn help_documents_every_exit_code() {
        for code in [
            "0  ok",
            "1  the transformation",
            "2  the command",
            "3  tile-rs cannot",
            "4  it exists",
            "5  a license",
            "6  a device",
            "7  the UI",
        ] {
            assert!(HELP.contains(code), "help is missing {code:?}");
        }
    }

    #[test]
    fn help_says_that_exit_3_is_not_a_claim_of_impossibility() {
        // The whole point of splitting 3 from 4: a caller who reads "unsupported" as
        // "impossible" stops asking for a thing that is merely unwritten.
        assert!(HELP.contains("never a claim of impossibility"));
    }

    #[test]
    fn the_two_refusal_codes_are_distinct() {
        // 3 = nobody has written it; 4 = it exists but not here. Collapsing them tells a
        // caller to retry forever, or to give up one download early.
        assert_ne!(exit::UNSUPPORTED, exit::UNAVAILABLE);
    }

    #[test]
    fn help_states_the_default_optimization_level() {
        assert!(HELP.contains("-O2 is the default"));
    }

    #[test]
    fn help_states_the_telemetry_stance() {
        assert!(HELP.contains("No telemetry"));
    }
}
