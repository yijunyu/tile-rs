//! Running the real binary on machines that are missing things.
//!
//! Almost every interesting failure in this tool is an *absence* — no emitter compiled
//! in, no toolchain, no vendor SDK, no accelerator, no network. None of those paths run
//! on a developer's machine, because a developer's machine has everything. So they rot,
//! and the first person to meet them is a user on a bare box with a deadline.
//!
//! These tests run the actual `tile` binary as a subprocess in a constructed
//! environment, and assert on the exit code and the message. Two mechanisms, used for
//! different things:
//!
//! * **`TILE_SIMULATE`** for what the process cannot otherwise change about itself: a
//!   build feature that IS compiled in, a device that IS present.
//! * **A stripped `PATH` and a redirected `HOME`** for things the OS can genuinely take
//!   away. These are not simulations at all — the tool really cannot find `nvidia-smi`
//!   when `PATH` is empty — so they check the same code path for real.
//!
//! The exit-code taxonomy is the point: 2 (type something else), 3 (nobody wrote it),
//! 4 (exists, not here), 6 (no hardware). A refusal that returns the wrong one sends a
//! script into an infinite retry or gives up a download early, and only an end-to-end
//! run can catch that.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_tile")
}

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/forms")
        .join(name)
}

/// A machine to run on.
struct Machine {
    simulate: Option<&'static str>,
    /// Empty PATH: no vendor CLI can be found, for real.
    strip_path: bool,
    /// HOME pointed at an empty directory: nothing is provisioned, for real.
    empty_home: bool,
}

/// A private temp directory for one test.
///
/// The scratch tests used to sweep `tile-softmax-*` out of the shared temp directory and
/// then assert none came back — which races with every sibling test that also converts
/// that kernel, because cargo runs them in parallel. Isolating is correct where
/// coordinating is merely lucky: the child inherits TMPDIR, so its scratch lands here and
/// nothing else can put anything in it.
struct PrivateTmp(PathBuf);

impl PrivateTmp {
    fn new(tag: &str) -> PrivateTmp {
        let d = std::env::temp_dir().join(format!("tile-tmp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        PrivateTmp(d)
    }
    fn entries(&self, prefix: &str) -> Vec<String> {
        std::fs::read_dir(&self.0)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.starts_with(prefix))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Drop for PrivateTmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl Machine {
    fn real() -> Machine {
        Machine {
            simulate: None,
            strip_path: false,
            empty_home: false,
        }
    }
    fn simulating(spec: &'static str) -> Machine {
        Machine {
            simulate: Some(spec),
            strip_path: false,
            empty_home: false,
        }
    }
    /// The emptiest machine that can still run the tool, built out of real deprivation
    /// rather than a flag.
    fn stripped() -> Machine {
        Machine {
            simulate: None,
            strip_path: true,
            empty_home: true,
        }
    }

    fn run(&self, args: &[&str]) -> Run {
        self.run_in(args, None)
    }

    fn run_in(&self, args: &[&str], tmp: Option<&PrivateTmp>) -> Run {
        let mut c = Command::new(bin());
        c.args(args);
        if let Some(t) = tmp {
            c.env("TMPDIR", &t.0);
        }
        // Start from a clean slate so a developer's shell cannot leak in.
        c.env_remove("TILE_SIMULATE");
        if let Some(spec) = self.simulate {
            c.env("TILE_SIMULATE", spec);
        }
        if self.strip_path {
            c.env("PATH", "");
        }
        let home;
        if self.empty_home {
            home = std::env::temp_dir().join(format!("tile-emptyhome-{}", std::process::id()));
            std::fs::create_dir_all(&home).unwrap();
            c.env("HOME", &home);
        }
        let out: Output = c.output().expect("the tile binary runs");
        Run {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    }
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    #[track_caller]
    fn expect_code(&self, want: i32) -> &Run {
        assert_eq!(
            self.code, want,
            "wanted exit {want}, got {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.code, self.stdout, self.stderr
        );
        self
    }
    #[track_caller]
    fn says(&self, needle: &str) -> &Run {
        let all = format!("{}{}", self.stdout, self.stderr);
        assert!(
            all.contains(needle),
            "output does not contain {needle:?}\n{all}"
        );
        self
    }
    #[track_caller]
    fn does_not_say(&self, needle: &str) -> &Run {
        let all = format!("{}{}", self.stdout, self.stderr);
        assert!(
            !all.contains(needle),
            "output unexpectedly contains {needle:?}\n{all}"
        );
        self
    }
}

fn out_path(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("tile-env-{}-{}.out", tag, std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

// ── The rule that makes the hook safe ─────────────────────────────────────────────

#[test]
fn the_default_build_stays_within_its_dependency_budget() {
    // Measured on the RESOLVED graph, not on Cargo.lock: the lockfile records optional
    // dependencies whether or not a build uses them, so counting it would fail the moment
    // a feature exists. Nobody who wants `tile k.mlir -o k.metal` should pay for an AEAD.
    let out = Command::new(env!("CARGO"))
        .args(["tree", "-e", "normal", "--prefix", "none"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut crates: Vec<&str> = text
        .lines()
        .filter_map(|l| l.split(" v").next())
        .filter(|l| !l.trim().is_empty())
        .collect();
    crates.sort_unstable();
    crates.dedup();
    assert!(
        crates.len() <= 15,
        "the default binary pulls {} crates: {crates:?}",
        crates.len()
    );
}

#[test]
fn the_documented_absences_are_exactly_the_ones_help_names() {
    // A testing hook whose vocabulary drifts from its documentation is a hook nobody can
    // use. `--help` is where a user learns it exists; the table is where it is defined.
    let help = Machine::real().run(&["--help"]);
    for (name, _) in tile_cli::simenv::ABSENCES {
        help.says(name);
    }
}

#[test]
fn a_simulated_run_always_announces_itself() {
    // Without this, someone leaves TILE_SIMULATE set, meets a refusal they cannot
    // explain, and files it against the wrong component.
    Machine::simulating("no-emitters")
        .run(&["doctor"])
        .expect_code(0)
        .says("SIMULATED ENVIRONMENT")
        .says("no-emitters")
        .says("unset TILE_SIMULATE");
}

#[test]
fn a_real_run_says_nothing_about_simulation() {
    Machine::real()
        .run(&["doctor"])
        .expect_code(0)
        .does_not_say("SIMULATED");
}

#[test]
fn an_unknown_absence_is_refused_with_the_vocabulary() {
    Machine::simulating("no-gpus")
        .run(&["doctor"])
        .expect_code(2)
        .says("unknown TILE_SIMULATE term")
        .says("no-devices");
}

// ── Absent build features ─────────────────────────────────────────────────────────

#[test]
fn without_the_emitters_a_conversion_is_unavailable_not_unsupported() {
    // Exit 4, NOT 3. The capability exists — it is one binary away — and telling the
    // caller "unsupported" would send them away from something that works elsewhere.
    let out = out_path("noemit");
    Machine::simulating("no-emitters")
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-o",
            out.to_str().unwrap(),
        ])
        .expect_code(4)
        .says("build feature, not a missing form")
        .says("--features emitters");
    assert!(
        !out.exists(),
        "nothing may be written when the route cannot be taken"
    );
}

#[test]
fn without_the_emitters_the_form_is_still_listed() {
    // The distinction has to survive into --list-forms, or a user concludes the backend
    // was removed rather than not built.
    Machine::simulating("no-emitters")
        .run(&["--list-forms"])
        .expect_code(0)
        .says("msl")
        .says("write-only");
}

#[test]
fn without_the_emitters_identify_and_profile_still_work() {
    // The floor: a build with no emitters is still useful. If -i broke too, the feature
    // split would be pointless.
    Machine::simulating("no-emitters")
        .run(&[corpus("softmax.mlir").to_str().unwrap(), "-i"])
        .expect_code(0)
        .says("form: mlir")
        .says("tile plan:")
        .says("hazards:");
}

// ── A machine with no vendor tooling, for real ────────────────────────────────────

#[test]
fn an_empty_path_is_not_an_error() {
    // No nvidia-smi, no npu-smi, no xcrun, no system_profiler. A missing vendor CLI is
    // the NORMAL case on most machines and must never be a failure.
    Machine::stripped()
        .run(&["doctor"])
        .expect_code(0)
        .says("platform:")
        .says("none detected");
}

#[test]
fn with_no_accelerator_the_default_target_is_still_runnable() {
    // Requirement (5)'s floor: the default never produces an artifact the user cannot
    // execute. With nothing detected that means the CPU linalg bridge.
    Machine::stripped()
        .run(&["doctor"])
        .expect_code(0)
        .says("default output form: linalg")
        .says("always runnable");
}

#[test]
fn a_bare_machine_can_still_convert() {
    // The whole argument for Tier 0: MLIR -> target source needs no toolchain, no
    // vendor SDK and no network, so an air-gapped box with nothing installed still gets
    // the tool's core value.
    let out = out_path("bare");
    Machine::stripped()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-o",
            out.to_str().unwrap(),
        ])
        .expect_code(0);
    let text = std::fs::read_to_string(&out).expect("the kernel was written");
    assert!(text.contains("kernel void"), "expected Metal source");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn version_and_help_work_on_a_machine_with_nothing() {
    Machine::stripped()
        .run(&["--version"])
        .expect_code(0)
        .says("tile ");
    Machine::stripped()
        .run(&["--help"])
        .expect_code(0)
        .says("EXIT CODES");
}

// ── The exit-code taxonomy, end to end ────────────────────────────────────────────

#[test]
fn a_missing_frontend_is_unsupported_not_unavailable() {
    // Exit 3: no build, download or purchase fixes this today.
    Machine::real()
        .run(&[
            corpus("softmax.metal").to_str().unwrap(),
            "-o",
            out_path("lift").to_str().unwrap(),
            "-t",
            "msl",
        ])
        .expect_code(3)
        .says("no msl frontend has")
        .says("missing work, not a missing possibility");
}

#[test]
fn the_two_refusals_return_different_codes_for_the_same_kernel() {
    // The clearest statement of the taxonomy: one input, two absences, two answers.
    let missing_frontend = Machine::real().run(&[
        corpus("softmax.metal").to_str().unwrap(),
        "-t",
        "msl",
        "-o",
        out_path("t1").to_str().unwrap(),
    ]);
    let missing_feature = Machine::simulating("no-emitters").run(&[
        corpus("softmax.mlir").to_str().unwrap(),
        "-t",
        "msl",
        "-o",
        out_path("t2").to_str().unwrap(),
    ]);
    missing_frontend.expect_code(3);
    missing_feature.expect_code(4);
    assert_ne!(missing_frontend.code, missing_feature.code);
}

#[test]
fn a_usage_error_never_borrows_a_capability_code() {
    Machine::real()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-i",
            "-o",
            "x.metal",
        ])
        .expect_code(2)
        .says("--info-only produces no output");
}

// ── Protocol hygiene under simulation ─────────────────────────────────────────────

#[test]
fn the_banner_never_contaminates_stdout() {
    // `-o -` makes stdout a data channel. A diagnostic there corrupts a pipeline, and
    // the simulation banner is exactly the kind of well-meaning message that leaks.
    let r = Machine::simulating("no-network").run(&[
        corpus("softmax.mlir").to_str().unwrap(),
        "-t",
        "msl",
        "-o",
        "-",
    ]);
    r.expect_code(0);
    assert!(
        r.stderr.contains("SIMULATED"),
        "the banner belongs on stderr"
    );
    assert!(
        !r.stdout.contains("SIMULATED"),
        "the banner leaked into stdout"
    );
    assert!(
        r.stdout.contains("kernel void"),
        "stdout must carry the kernel"
    );
    assert!(
        !r.stdout.contains("form: mlir"),
        "the profile leaked into stdout"
    );
}

#[test]
fn nothing_is_written_when_the_output_directory_is_absent() {
    Machine::real()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-o",
            "/tmp/definitely-not-here-xyz/out.metal",
        ])
        .expect_code(2)
        .says("does not exist");
}

#[test]
fn an_existing_output_is_never_clobbered_silently() {
    let out = out_path("clobber");
    std::fs::write(&out, "PRECIOUS").unwrap();
    Machine::real()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-o",
            out.to_str().unwrap(),
        ])
        .expect_code(2)
        .says("--force");
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "PRECIOUS");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn a_scratch_directory_left_by_a_killed_run_is_swept_by_the_next_one() {
    // A SIGKILL mid-conversion cannot run its own cleanup. "No orphans" is therefore not
    // a property of the run that died — it is a property of the run that follows, and
    // this is the only place that can be checked.
    let stale = std::env::temp_dir().join("tile-ghost-999999");
    std::fs::create_dir_all(&stale).unwrap();
    std::fs::write(stale.join("softmax.1.linalg.mlir"), "orphan").unwrap();

    let out = out_path("sweep");
    Machine::real()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-o",
            out.to_str().unwrap(),
        ])
        .expect_code(0);

    assert!(
        !stale.exists(),
        "the next run must sweep a dead run's scratch"
    );
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_dir_all(&stale);
}

#[test]
fn a_default_run_leaves_no_scratch_of_its_own() {
    // Its own TMPDIR, so a sibling test converting the same kernel in parallel cannot
    // put a directory here and make this fail.
    let tmp = PrivateTmp::new("noscratch");
    let out = out_path("noscratch");
    Machine::real()
        .run_in(
            &[
                corpus("softmax.mlir").to_str().unwrap(),
                "-t",
                "msl",
                "-o",
                out.to_str().unwrap(),
            ],
            Some(&tmp),
        )
        .expect_code(0);
    let leaked = tmp.entries("tile-softmax-");
    assert!(
        leaked.is_empty(),
        "a scratch directory survived: {leaked:?}"
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn a_scratch_belonging_to_a_live_process_is_never_swept() {
    // The sweep must not delete the working directory of a concurrent run. Erring the
    // other way costs a stale directory; erring this way corrupts someone's conversion.
    let mine = std::env::temp_dir().join(format!("tile-live-{}", std::process::id()));
    std::fs::create_dir_all(&mine).unwrap();
    let out = out_path("live");
    Machine::real()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-o",
            out.to_str().unwrap(),
        ])
        .expect_code(0);
    assert!(mine.exists(), "the sweep deleted a live process's scratch");
    let _ = std::fs::remove_dir_all(&mine);
    let _ = std::fs::remove_file(&out);
}

// ── The daemon, driven as a client would ──────────────────────────────────────────

/// Send JSON-RPC lines to `tile -d` and collect the replies.
fn mcp(lines: &[&str]) -> Vec<String> {
    use std::io::Write as _;
    let mut c = Command::new(bin());
    c.arg("-d");
    c.env_remove("TILE_SIMULATE");
    c.stdin(std::process::Stdio::piped());
    c.stdout(std::process::Stdio::piped());
    c.stderr(std::process::Stdio::piped());
    let mut child = c.spawn().expect("the daemon starts");
    {
        let mut pipe = child.stdin.take().expect("stdin");
        for l in lines {
            writeln!(pipe, "{l}").unwrap();
        }
    }
    let out = child.wait_with_output().expect("wait");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn the_daemon_speaks_mcp_over_stdio() {
    let replies = mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    ]);
    assert_eq!(replies.len(), 2, "one reply per request: {replies:?}");
    assert!(replies[0].contains("\"protocolVersion\""), "{}", replies[0]);
    for tool in [
        "doctor",
        "list_forms",
        "identify",
        "profile",
        "routes",
        "convert",
        "install",
    ] {
        assert!(replies[1].contains(tool), "tools/list omits {tool}");
    }
}

#[test]
fn a_notification_produces_no_line_at_all() {
    // One reply per REQUEST. A stray line for a notification desynchronises a client
    // that is matching replies to ids.
    let replies = mcp(&[
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
    ]);
    assert_eq!(replies.len(), 1, "{replies:?}");
    assert!(replies[0].contains("\"id\":1"));
}

#[test]
fn a_malformed_line_gets_a_parse_error_and_the_session_continues() {
    // A client that sends one bad frame must not lose the connection.
    let replies = mcp(&[
        "{ this is not json",
        r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#,
    ]);
    assert_eq!(replies.len(), 2, "{replies:?}");
    assert!(replies[0].contains("-32700"), "{}", replies[0]);
    assert!(replies[1].contains("\"id\":7"), "{}", replies[1]);
}

#[test]
fn nothing_but_protocol_reaches_stdout_even_at_maximum_verbosity() {
    use std::io::Write as _;
    let mut c = Command::new(bin());
    c.args(["-d", "-VVV"]);
    c.env_remove("TILE_SIMULATE");
    c.stdin(std::process::Stdio::piped());
    c.stdout(std::process::Stdio::piped());
    c.stderr(std::process::Stdio::piped());
    let mut child = c.spawn().unwrap();
    {
        let mut pipe = child.stdin.take().unwrap();
        writeln!(pipe, r#"{{"jsonrpc":"2.0","id":1,"method":"ping"}}"#).unwrap();
    }
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        assert!(
            line.starts_with('{'),
            "a diagnostic reached stdout and would corrupt the framing: {line}"
        );
    }
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("MCP over stdio"),
        "the diagnostic must still be produced, on stderr"
    );
}

#[test]
fn an_mcp_convert_and_the_cli_produce_the_same_bytes() {
    // "An agent can do exactly what a person can do" has to be checkable, not asserted.
    let src = std::fs::read_to_string(corpus("softmax.mlir")).unwrap();
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"convert","arguments":{{"filename":"softmax.mlir","to":"msl","level":0,"source":{}}}}}}}"#,
        tile_cli::json::Json::s(src.clone())
    );
    let replies = mcp(&[&req]);
    assert_eq!(replies.len(), 1, "{replies:?}");
    let parsed = tile_cli::json::parse(&replies[0]).expect("a JSON reply");
    let text = parsed
        .get("result")
        .and_then(|r| r.get("content"))
        .and_then(|c| match c {
            tile_cli::json::Json::Arr(a) => a.first().cloned(),
            _ => None,
        })
        .and_then(|c| c.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .expect("text content");

    let out = out_path("mcpparity");
    Machine::real()
        .run(&[
            corpus("softmax.mlir").to_str().unwrap(),
            "-t",
            "msl",
            "-O0",
            "-o",
            out.to_str().unwrap(),
        ])
        .expect_code(0);
    let via_cli = std::fs::read_to_string(&out).unwrap();
    assert_eq!(text, via_cli, "MCP and the CLI disagree");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn the_daemon_refuses_a_tcp_listener_rather_than_binding_one() {
    // A TCP server handing out license-gated data with "binds localhost" as its whole
    // security story is not a security story.
    Machine::real()
        .run(&["-d", "--listen", "127.0.0.1:7777"])
        .expect_code(2)
        .says("auth token");
}
