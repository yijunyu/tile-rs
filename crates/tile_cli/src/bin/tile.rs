//! `tile` — the binary. argv in, rendering out; every decision lives in the library so
//! the daemon (M8) and the UI (M10) can make the same ones.
//!
//! Protocol hygiene: results go to stdout, everything else to stderr. `-o -` therefore
//! yields a kernel on stdout and nothing else, which is what makes the tool pipeable.

use std::io::{Read, Write};
use std::process::ExitCode;

use tile_cli::cli::{self, exit, Cmd};
use tile_cli::{
    daemon, default_form_for, derived_output, emit, forms, optimize, platform, profile, provision,
    routes, scratch, simenv, ui, VERSION,
};

fn main() -> ExitCode {
    // Resolve the environment BEFORE anything else, and announce it if it is simulated.
    // A testing hook that can silently change behaviour is a bug generator: someone
    // leaves the variable set and files the resulting refusal against the wrong thing.
    let env = match simenv::Env::from_process() {
        Ok(e) => e,
        Err(e) => {
            eprint!("tile: {e}");
            return ExitCode::from(exit::USAGE as u8);
        }
    };
    if let Some(banner) = env.banner() {
        eprintln!("{banner}");
    }

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match cli::parse(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("tile: {e}");
            eprintln!("try `tile --help`");
            return ExitCode::from(exit::USAGE as u8);
        }
    };
    ExitCode::from(run(args, &env) as u8)
}

fn run(args: cli::Args, env: &simenv::Env) -> i32 {
    match args.cmd {
        Cmd::Help => {
            print!("{}", cli::HELP);
            exit::OK
        }
        Cmd::Version => {
            println!("tile {VERSION}");
            exit::OK
        }
        Cmd::ListForms => {
            print!("{}", routes::list_forms());
            exit::OK
        }
        Cmd::ListOps => {
            // Measured on the spot rather than read from a table: each cell is a real
            // lowering attempt, so the report cannot drift from the emitters.
            print!("{}", tile_cli::emit::coverage_report());
            exit::OK
        }
        Cmd::Doctor => doctor(&args, env),
        Cmd::Install(ref which) => install(&args, which.as_deref()),
        Cmd::Daemon => {
            // Provision the engine beside the daemon, and supervise it.
            //
            // Every failure below is non-fatal by construction: the kernel tools do not
            // depend on the engine, and losing them because a model could not be found
            // would be the wrong trade. That is the clause worth keeping true.
            let mut running = None;
            if let Some(model) = &args.model {
                let plat = platform::detect_in(env);
                let family = plat.primary_family();
                let mut say = |m: &str| eprintln!("tile: {m}");
                match tile_cli::engine::start(family, model, &mut say) {
                    Ok(r) => running = Some(r),
                    Err(e) => {
                        eprintln!("tile: {e}");
                        eprintln!("tile: serving the kernel tools without a model");
                    }
                }
            }
            let code = daemon::serve_stdio(env, args.verbose);
            // The engine must not outlive the daemon it was started beside: an orphaned
            // server holding the GPU is exactly the failure mode that makes people stop
            // trusting a supervisor.
            if let Some(mut r) = running {
                eprintln!("tile: stopping {}", r.engine);
                r.stop();
            }
            code
        }
        Cmd::License => license_status(),
        Cmd::Ui => serve_ui(&args, env),
        Cmd::Backlog(ref sub) => backlog_cmd(sub.as_deref()),
        Cmd::NotYet(flag, milestone) => {
            eprintln!(
                "tile: {flag} is specified but not yet built — it lands in {milestone}.\n\
                 See docs/cli/plan.md for what {milestone} delivers and what it depends on."
            );
            exit::SUBSYSTEM
        }
        Cmd::Run => run_inputs(&args, env),
    }
}

/// The per-target table: what emits, and what anybody has actually run.
///
/// MEASURED is not SUPPORTED, and the difference is the one that matters before you trust
/// a number. Every form here emits; the split says which ones have been checked against a
/// reference on the hardware they name. It is the same distinction `HardwareParams`
/// enforces on the compiler side, where an unmeasured architecture refuses to answer
/// bound queries rather than borrowing another chip's capacities -- so printing the two
/// as one list would quietly undo that at the CLI.
fn print_targets() {
    // #012. These were one list, headed "measured -- emitted AND checked against a
    // reference on that hardware". Nine backends sat under it and this repo holds the
    // evidence for three. The others came from projects that do have the hardware, so the
    // claims may be true; what was wrong was printing them beside claims this repo can
    // show, with nothing to tell them apart.
    let (mut here, mut inherited, mut supported) = (Vec::new(), Vec::new(), Vec::new());
    for f in forms::FORMS {
        if !f.writable || !matches!(f.role, forms::Role::Mlir | forms::Role::Source) {
            continue;
        }
        match f.fidelity {
            forms::Fidelity::Validated(hw, forms::Evidence::Here(w)) => {
                here.push(format!("{} (on {hw}; {w})", f.id))
            }
            forms::Fidelity::Validated(hw, forms::Evidence::Inherited) => {
                inherited.push(format!("{} (on {hw})", f.id))
            }
            forms::Fidelity::Unvalidated => supported.push(f.id.to_string()),
            _ => {}
        }
    }
    println!("targets:");
    println!("  measured HERE — a run recorded in this repo, with where to find it:");
    println!("      {}", here.join(", "));
    println!("  claimed UPSTREAM — validated by a project that has the hardware; this");
    println!("      repo carries no run for these, which is not the same as their being");
    println!("      wrong:");
    println!("      {}", inherited.join(", "));
    println!("  unmeasured — emits correctly; never run on the hardware:");
    println!("      {}", supported.join(", "));
    // The sentence that keeps the table from being read as a capability ranking.
    println!(
        "  unmeasured is not unsupported, and it is not unlimited: the codegen is\n\
         exercised, but nothing here knows that chip's real capacities, so a bound\n\
         derived from another chip's numbers would approve exactly the tilings that fail."
    );
}

fn doctor(args: &cli::Args, env: &simenv::Env) -> i32 {
    if args.targets {
        // Just the table. `doctor` alone answers "what is this machine"; `--targets`
        // answers "what can be trusted", and they are different questions.
        print_targets();
        return exit::OK;
    }
    let p = platform::detect_in(env);
    if args.json {
        // Deliberately hand-rolled: one function is cheaper than a serialization
        // dependency, and the shape is asserted by a golden test.
        println!("{{");
        println!("  \"triple\": \"{}\",", p.triple);
        println!("  \"os\": \"{}\",", p.os);
        println!("  \"arch\": \"{}\",", p.arch);
        println!("  \"primary_family\": \"{}\",", p.primary_family());
        print!("  \"accelerators\": [");
        for (i, a) in p.accels.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!(
                "{{\"family\": \"{}\", \"device\": \"{}\", \"via\": \"{}\"}}",
                a.family, a.device, a.via
            );
        }
        println!("],");
        print!("  \"sdks\": [");
        for (i, s) in p.sdks.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!(
                "{{\"family\": \"{}\", \"name\": \"{}\", \"version\": \"{}\"}}",
                s.family,
                s.name,
                s.version.clone().unwrap_or_default()
            );
        }
        println!("]");
        println!("}}");
        return exit::OK;
    }
    print!("{p}");
    // The SDKs again, as their own section.
    //
    // The accelerator lines above name the SDK that belongs to each DEVICE, which leaves
    // out every SDK for a device that is not here -- and those are exactly the ones that
    // decide whether a cross-generation target can be compiled locally. Reporting them
    // only beside a device made "the toolkit is installed" invisible unless the hardware
    // happened to be present too.
    if p.sdks.is_empty() {
        println!("sdks: none detected");
    } else {
        println!("sdks:");
        for sdk in &p.sdks {
            let ver = sdk
                .version
                .clone()
                .unwrap_or_else(|| "version unknown".into());
            let here = if p.accels.iter().any(|a| a.family == sdk.family) {
                ""
            } else {
                "  (no matching device on this machine)"
            };
            println!("  {} {ver} [{}]{here}", sdk.name, sdk.family);
        }
    }
    for a in p.unsupported_present() {
        println!(
            "note: {} is present but tile-rs has no ROCm/HIP target; it will not be \
             offered the `aie` form (that is the Ryzen AI NPU)",
            a.device
        );
    }
    let fam = p.primary_family();
    let form = default_form_for(fam);
    println!("default output form: {} ({})", form.id, form.display);
    if fam == "none" {
        println!("  (no accelerator detected — the CPU linalg bridge is always runnable)");
    }

    print_targets();
    exit::OK
}

/// Acquisition policy from the flags. `--offline` is the stronger of the two.
/// Whether the source calls the matmul intrinsic, asked before the full `RefOp::detect`
/// so the shape can be read first.
/// The shape the MLIR states, when it states one (#017).
///
/// Every non-matmul shape used to be built with `rows: 1` and `cols` set to the total
/// element count -- literals, never read from the source -- so `base = row * num_elements`
/// was always 0 and row indexing went untested. The reference was handed the same
/// flattened shape and agreed with the kernel about a question the MLIR had not asked.
///
/// `None` when the source does not state constant dimensions; the old flattening is kept
/// for that case rather than guessing, and it is still correct for a single row.
fn shape_from_rows_cols(
    mlir: &str,
    op: tile_cli::run::RefOp,
    kernel: &tile_cli::profile::Kernel,
) -> Option<tile_cli::run::Shape> {
    let (rows, cols) = tile_cli::run::Shape::rows_cols_from_mlir(mlir)?;
    // Trust the source only when it agrees with the element count the profile measured.
    // A disagreement means one of the two is reading the module wrongly, and a shape built
    // on the wrong one sizes buffers the kernel does not write.
    if rows * cols != kernel.numel().max(1) {
        return None;
    }
    if op == tile_cli::run::RefOp::Matvec {
        return Some(tile_cli::run::Shape::Matvec { rows, cols });
    }
    if op.is_two_input_elementwise() {
        return Some(tile_cli::run::Shape::Rows2 { rows, cols });
    }
    if op.is_row_reduction() {
        return Some(tile_cli::run::Shape::RowReduce { rows, cols });
    }
    Some(tile_cli::run::Shape::Rows { rows, cols })
}

fn op_is_matmul(src: &str) -> bool {
    src.contains("__tile_matmul_")
}

fn policy(args: &cli::Args) -> provision::Policy {
    if args.offline {
        provision::Policy::Offline
    } else if args.no_install {
        provision::Policy::NeverInstall
    } else {
        provision::Policy::OnDemand
    }
}

fn install(args: &cli::Args, which: Option<&str>) -> i32 {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let Some(id) = which else {
        // No id: show what this platform can be given, and which of it is already here.
        let rows = provision::listing(os, arch);
        if rows.is_empty() {
            println!("nothing in the manifest for {os}/{arch}");
            return exit::OK;
        }
        println!("tools for {os}/{arch}:");
        for (t, done) in rows {
            let state = if done {
                "installed".to_string()
            } else if let Some(b) = &t.barrier {
                format!("needs you ({b})")
            } else if !t.pinned() {
                "not pinned yet".to_string()
            } else {
                "available".to_string()
            };
            println!("  {:<20} {:<8} {}", t.id, t.version, state);
        }
        println!(
            "\ninstalls go under {} and never anywhere else",
            provision::home().display()
        );
        return exit::OK;
    };

    let mut report = |m: &str| eprintln!("tile: {m}");
    match provision::ensure(id, os, arch, policy(args), &mut report) {
        Ok(provision::Outcome::AlreadyPresent) => {
            println!("{id} is already installed");
            exit::OK
        }
        Ok(provision::Outcome::Installed { bytes, sha256 }) => {
            println!("installed {id} ({bytes} bytes, sha256 {})", &sha256[..16]);
            exit::OK
        }
        Err(e) => {
            eprintln!("tile: {e}");
            exit::UNAVAILABLE
        }
    }
}

/// Today, for the expiry comparison. Derived from the clock rather than hardcoded, and
/// the only place in the tool that reads it.
/// The device name for a measurement's basis. A number with no machine attached is not
/// reproducible by anyone.
fn plat_device(env: &simenv::Env) -> String {
    let p = platform::detect_in(env);
    p.accels
        .first()
        .map(|a| a.device.clone())
        .unwrap_or_else(|| "(no accelerator)".into())
}

fn today() -> String {
    // No date crate: an RFC-3339 date out of the Unix epoch is forty lines of arithmetic
    // and this is its only caller. Civil-from-days, Howard Hinnant's algorithm.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs / 86_400 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// `-u`: build the view, bind loopback, print the URL, and serve until interrupted.
///
/// Headless is the NORMAL case for a compiler tool — ssh, CI, a build box — so a machine
/// with no display gets the URL rather than an error. Exit 7 is for a subsystem asked for
/// by name that could not start, which binding a loopback port almost never is.
fn serve_ui(args: &cli::Args, env: &simenv::Env) -> i32 {
    // `--ui native` is not built, and must say so rather than serving the web view.
    //
    // Before this, "native" was parsed as an input FILENAME and the HTTP server started
    // anyway: the user asked for a window, got a server, was told nothing, and the
    // command hung because the server blocks. Substituting a different subsystem in
    // silence is the same class of fault as an emitter that ignores an intrinsic.
    //
    // Exit 3, not 7: 7 is for a subsystem that exists and could not start. This one has
    // not been written, and the message has to carry that tense.
    if args.ui_mode.as_deref() == Some("native") {
        eprintln!(
            "tile: --ui native is not built yet: there is no eframe window to open.\n\
             \x20 The reason it is not built is that a window cannot be asserted by a\n\
             \x20 headless CI box, and the server half has to exist either way — so the\n\
             \x20 view lives behind `tile --ui`, which serves the same core on loopback\n\
             \x20 and prints its address. Use that."
        );
        return exit::UNSUPPORTED;
    }

    let input = args
        .inputs
        .first()
        .and_then(|p| read_input(p).ok().map(|text| (p.clone(), text)));
    let plat = platform::detect_in(env);
    let family = args
        .cross
        .clone()
        .unwrap_or_else(|| plat.primary_family().to_string());
    let view = ui::View::build(
        env,
        input.as_ref().map(|(p, t)| (p.as_str(), t.as_str())),
        &family,
    );
    let body = ui::page(&view);

    let listener = match ui::bind(args.ui_port) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("tile: could not bind a loopback port: {e}");
            return exit::SUBSYSTEM;
        }
    };
    let addr = match listener.local_addr() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("tile: {e}");
            return exit::SUBSYSTEM;
        }
    };
    let url = format!("http://{addr}/");

    // The URL goes to stdout so it can be piped; everything else to stderr.
    println!("{url}");
    if ui::windowing_available() {
        if !ui::open_browser(&url) {
            eprintln!("tile: could not open a browser; the address above still works");
        }
    } else {
        eprintln!("tile: no windowing subsystem here — open the address above from elsewhere");
    }
    eprintln!("tile: serving until interrupted (ctrl-c)");
    let _ = ui::serve(&listener, &body, None);
    exit::OK
}

/// `tile backlog` — what the tool could not do, and whether that has become a pattern.
///
/// The exit code carries the verdict, so a session can branch on it without parsing
/// prose: 0 healthy, 3 stop and reconsider. That is the mechanism behind "if issues
/// accumulate, stop using the tool and rethink it" — it is a check anything can run,
/// not a judgement someone has to remember to make.
fn backlog_cmd(sub: Option<&str>) -> i32 {
    use tile_cli::backlog;
    let (issues, unreadable) = backlog::load_reporting_problems();
    // Printed before the listing, because a file the loader could not read is a gap that
    // is NOT in the numbers below it -- and the numbers look complete either way.
    for p in &unreadable {
        eprintln!("tile: backlog: cannot read {p}");
    }
    match sub
        .map(|s| s.split_whitespace().collect::<Vec<_>>())
        .as_deref()
    {
        None | Some([]) => {
            print!("{}", backlog::render(&issues));
            match backlog::verdict(&issues) {
                backlog::Verdict::Healthy => exit::OK,
                _ => exit::UNSUPPORTED,
            }
        }
        Some(["add"]) => {
            // Fields on stdin, so an agent can pipe them without quoting a paragraph
            // through a shell.
            let mut text = String::new();
            if std::io::stdin().read_to_string(&mut text).is_err() {
                eprintln!("tile: backlog add reads the entry from stdin");
                return exit::USAGE;
            }
            let field = |name: &str| -> String {
                text.split(&format!("## {name}"))
                    .nth(1)
                    .map(|s| s.split("\n##").next().unwrap_or("").trim().to_string())
                    .unwrap_or_default()
            };
            let head = |name: &str| -> String {
                text.lines()
                    .find_map(|l| l.strip_prefix(&format!("{name}: ")))
                    .unwrap_or("")
                    .trim()
                    .to_string()
            };
            let (area, title) = (head("area"), head("title"));
            if area.is_empty() || title.is_empty() {
                eprintln!(
                    "tile: an entry needs `area:` and `title:` lines, then `## wanted`, \n                       `## got` and `## workaround` sections. The workaround is the point: \n                       an entry without one is a report, not evidence."
                );
                return exit::USAGE;
            }
            let issue = backlog::Issue {
                id: backlog::next_id(&issues),
                area,
                title,
                wanted: field("wanted"),
                got: field("got"),
                workaround: field("workaround"),
                opened: today(),
                closed: None,
            };
            match backlog::save(&issue) {
                Ok(p) => {
                    println!("recorded #{:03} in {}", issue.id, p.display());
                    let v = backlog::verdict(&backlog::load());
                    if !matches!(v, backlog::Verdict::Healthy) {
                        eprintln!("tile: {v}");
                        return exit::UNSUPPORTED;
                    }
                    exit::OK
                }
                Err(e) => {
                    eprintln!("tile: could not record it: {e}");
                    exit::TRANSFORM_FAILED
                }
            }
        }
        Some(["close", id]) => {
            let Ok(id) = id.parse::<u32>() else {
                eprintln!("tile: backlog close wants an issue number");
                return exit::USAGE;
            };
            let Some(mut issue) = issues.iter().find(|i| i.id == id).cloned() else {
                eprintln!("tile: no issue #{id:03}");
                return exit::USAGE;
            };
            if !issue.is_open() {
                println!("#{id:03} was already closed");
                return exit::OK;
            }
            issue.closed = Some(today());
            match backlog::save(&issue) {
                Ok(_) => {
                    println!("closed #{id:03}  {}", issue.title);
                    exit::OK
                }
                Err(e) => {
                    eprintln!("tile: {e}");
                    exit::TRANSFORM_FAILED
                }
            }
        }
        Some(other) => {
            eprintln!(
                "tile: unknown backlog command {:?}; try add or close <id>",
                other.join(" ")
            );
            exit::USAGE
        }
    }
}

fn license_status() -> i32 {
    match tile_cli::license::status(&today()) {
        Ok(l) => {
            println!("licensed to: {}", l.subject);
            println!("expires:     {}", l.expires);
            println!("features:    {}", l.features.join(", "));
            println!("verified offline against the release key; no request was made");
            exit::OK
        }
        Err(e) => {
            eprintln!("tile: {e}");
            // Absent is not a failure of the tool: everything except the corpus works.
            if matches!(e, tile_cli::license::LicenseError::Absent) {
                exit::OK
            } else {
                exit::LICENSE
            }
        }
    }
}

/// `-s`: prior attempts for the kernels in the arguments.
///
/// The local store is always readable. The curated corpus needs a token — and its absence
/// degrades the REPORT, never the conversion, which still runs.
fn stats_for(kernels: &[String]) -> String {
    let local = tile_cli::corpus::load_local().unwrap_or_default();
    let knowledge = open_corpus();
    tile_cli::corpus::report(&local, knowledge.as_ref(), kernels)
}

fn open_corpus() -> Option<tile_cli::corpus::Corpus> {
    let l = tile_cli::license::status(&today()).ok()?;
    let sealed = std::fs::read(provision::home().join("corpus.sealed")).ok()?;
    let plain = tile_cli::license::unseal(&sealed, &l.content_key).ok()?;
    tile_cli::corpus::parse(&String::from_utf8_lossy(&plain)).ok()
}

fn read_input(path: &str) -> Result<String, String> {
    if path == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("stdin: {e}"))?;
        return Ok(s);
    }
    // A directory is a plausible thing to type -- `tile kernels/ -t msl` reads like it
    // should work -- so answer with the command that does, rather than with errno 21.
    // The shell expands the glob, so this is a real instruction and not a suggestion the
    // tool could have followed itself.
    if std::path::Path::new(path).is_dir() {
        let sep = if path.ends_with('/') { "" } else { "/" };
        return Err(format!(
            "{path} is a directory; tile converts files. Try `tile {path}{sep}*.mlir` \
             (or whatever extension is in there) to name them explicitly."
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    String::from_utf8(bytes).map_err(|e| {
        format!(
            "{path}: kernel sources must be UTF-8; first invalid sequence at byte {}",
            e.utf8_error().valid_up_to()
        )
    })
}

fn run_inputs(args: &cli::Args, env: &simenv::Env) -> i32 {
    // A SIGKILL mid-conversion cannot run its own cleanup, so "no orphans" is only true
    // if the NEXT run makes it true.
    let swept = scratch::Scratch::sweep_stale();
    if swept > 0 && args.verbose > 0 {
        eprintln!(
            "tile: swept {swept} scratch director{} left by an earlier run",
            if swept == 1 { "y" } else { "ies" }
        );
    }

    let plat = platform::detect_in(env);
    let family = args
        .cross
        .clone()
        .unwrap_or_else(|| plat.primary_family().to_string());
    let default_out = default_form_for(&family);

    // -O4 is the level DEFINED by measurement, so it must not quietly become -O3.
    //
    // It used to run and report success on a machine with no accelerator at all: the
    // pass list said "not done here: measured-autotuning" and the exit code said 0, so a
    // caller asking for the measured level got an unmeasured artifact and no signal. That
    // is the same substitution `HardwareParams.measured` refuses on the compiler side.
    if args.level() == 4 && plat.accels.is_empty() {
        eprintln!(
            "tile: -O4 measures on hardware and no accelerator was detected here.\n\
                 \x20 -O3 is the highest level available on this machine; it uses a toolchain\n\
                 \x20 but never a clock, so it needs no device."
        );
        return exit::DEVICE;
    }

    let mut worst = exit::OK;
    let mut failed: Vec<&str> = Vec::new();
    for input in &args.inputs {
        let code = run_one(args, env, input, &family, default_out);
        if code != exit::OK {
            // Several inputs: one failure must not lose the others' results, but the
            // process still exits non-zero.
            worst = code;
            failed.push(input);
        }
    }
    // Say WHICH inputs failed, on stderr, at the end.
    //
    // Each input prints its own name as a heading on stdout and its reason on stderr, so
    // with several inputs the two halves of the same fact land on different streams and
    // in different places. Reading stderr alone -- which is what a caller redirecting
    // results does -- you learn that something failed and not what. Only reported for
    // more than one input, where the ambiguity actually exists.
    if args.inputs.len() > 1 && !failed.is_empty() {
        eprintln!(
            "tile: {} of {} inputs failed: {}",
            failed.len(),
            args.inputs.len(),
            failed.join(", ")
        );
    }
    worst
}

fn run_one(
    args: &cli::Args,
    env: &simenv::Env,
    input: &str,
    family: &str,
    default_out: &'static forms::Form,
) -> i32 {
    let text = match read_input(input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tile: {e}");
            return exit::USAGE;
        }
    };

    let resolved = match forms::resolve_input(input, &text, args.from.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("tile: {input}: {e}");
            return exit::USAGE;
        }
    };
    if let Some(other) = resolved.contradicted_by {
        eprintln!(
            "tile: warning: {input} was forced to form \"{}\" but its content looks like \"{}\"",
            resolved.form.id, other.id
        );
    }

    let prof = profile::profile(input, &text, resolved.form, resolved.how, family, args.tile);

    // The profile is information, so it goes to stderr whenever stdout carries a result.
    let profile_to_stdout = args.output.as_deref() != Some("-");
    if profile_to_stdout {
        print!("{}", prof.render());
    } else {
        eprint!("{}", prof.render());
    }

    if args.stats {
        // Names come from the profile, so `-s` answers about the kernels actually in the
        // file rather than about its filename.
        let kernels: Vec<String> = prof.kernels.iter().map(|k| k.name.clone()).collect();
        let report = stats_for(&kernels);
        if profile_to_stdout {
            print!("{report}");
        } else {
            eprint!("{report}");
        }
    }

    if args.info_only {
        return exit::OK;
    }

    // Where are we going?
    let out_form = match forms::resolve_output_from(
        args.output.as_deref(),
        args.to.as_deref(),
        default_out,
        Some(resolved.form),
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("tile: {e}");
            return exit::USAGE;
        }
    };

    // Naming a target that does not run here IS cross generation. Say so.
    //
    // Without this the tool happily wrote CUDA on an Apple laptop and reported nothing
    // unusual, so `--cross` looked like the only way to do it and the artifact's
    // unrunnability here was left for the user to discover. `--cross` remains the way to
    // say it deliberately; this is the same fact observed rather than declared.
    if args.cross.is_none()
        && out_form.family != "*"
        && out_form.family != family
        && family != "none"
    {
        // "Cannot be run here" is a claim about a HARNESS, not about a family name.
        // Comparing families alone said spirv could not be run or measured on this
        // machine, and then it was: Mesa's Vulkan driver sits on the same Apple GPU, so
        // the two families are not disjoint. The note now reports which of the two facts
        // holds instead of inferring the stronger one from the weaker.
        if tile_cli::run::harness_source(out_form).is_ok() {
            eprintln!(
                "tile: note: {} targets {} and this machine is {} — a different API on the \
                 same device. There is a harness for it, so `-r` runs and measures here. \
                 Pass --cross {} to silence this.",
                out_form.id, out_form.family, family, out_form.family
            );
        } else {
            eprintln!(
                "tile: note: {} targets {} but this machine is {} — this is cross \
                 generation, and there is no harness for it here, so the result cannot be \
                 run or measured. Pass --cross {} to say so deliberately (and to silence \
                 this).",
                out_form.id, out_form.family, family, out_form.family
            );
        }
    }

    let route = match routes::plan_with_opt(resolved.form.id, out_form.id, &args.via, args.level())
    {
        Ok(r) => r,
        Err(routes::RouteError::ViaNotOnAnyRoute {
            from,
            to,
            via,
            available,
        }) => {
            // Exit 2: the graph is fine and the command is wrong. `--via` used to be
            // accepted and then ignored whenever the pinned form was not on the route
            // taken -- the user believed they had constrained the run and had not.
            eprintln!(
                "tile: --via {} names a form that no route from \"{from}\" to \"{to}\" \
                 passes through.",
                via.join(",")
            );
            eprintln!("  Routes that do exist:");
            for r in &available {
                eprintln!("    {r}");
            }
            eprintln!("  Drop --via to take the first of these, or pin forms that are on one.");
            return exit::USAGE;
        }
        Err(routes::RouteError::OnlyThroughLift { from, to, via }) => {
            // Exit 2, not 3: the route exists and is takeable right now. What is wrong is
            // the command, and the remedy is one flag — so this must not share a code with
            // "tile-rs cannot do this".
            eprintln!(
                "tile: no direct route from \"{from}\" to \"{to}\".\n\
                 A route exists through a lift ({}), but a lift never composes into a\n\
                 lowering automatically — a synthesised intermediate feeding a lowering is\n\
                 exactly the kind of wrongness that stays invisible until it corrupts a run.\n\
                 Ask for it by name:  tile {} --via {} -t {}\n\
                 The result would be marked [synthesised], and is only as good as the lift.",
                via.join(", "),
                shell_quote(input),
                via.join(","),
                to
            );
            return exit::USAGE;
        }
        Err(routes::RouteError::NeedsFrontend { from, to, would_be }) => {
            // Exit 3. No install, purchase or rebuild changes this — but it is a piece of
            // work nobody has done, NOT a law of nature, and the wording has to carry that
            // tense or a reader concludes the pair is impossible and stops asking.
            eprintln!(
                "tile: cannot go from \"{from}\" to \"{to}\" yet.\n  \
                 tile-rs can emit \"{from}\" but cannot read it — no {from} frontend has\n  \
                 been written. That is missing work, not a missing possibility."
            );
            if let Some(r) = would_be {
                eprintln!(
                    "  With one, the route would be:  {}\n  \
                     and results through it would be marked [synthesised], because a lifted\n  \
                     kernel is a reconstruction, not a recovery.",
                    r.describe()
                );
            }
            eprintln!(
                "  tile-rs reads {} today. Adding a reader is a form-table row plus a\n  \
                 frontend; `tile --list-forms` shows which forms have one.",
                readable_forms().join(", ")
            );
            return exit::UNSUPPORTED;
        }
        Err(routes::RouteError::NoRoute { from, to, nearest }) => {
            eprintln!("tile: no route from \"{from}\" to \"{to}\" has been built yet.");
            if nearest.is_empty() {
                eprintln!(
                    "  Nothing leaves \"{from}\" at all. `tile --list-forms` shows the matrix."
                );
            } else {
                eprintln!("  Nearest routes from \"{from}\":");
                for r in &nearest {
                    eprintln!("    {}", r.describe());
                }
            }
            return exit::UNSUPPORTED;
        }
    };

    if args.show_route {
        // Show the route that would actually be TAKEN, prologue and all. Listing the
        // bare lowering while the run inserts an optimize hop makes `--route` a
        // different answer from `-o`, which is the one thing it must never be.
        let mut all = routes::routes(resolved.form.id, out_form.id, &args.via);
        for r in &mut all {
            *r = routes::with_opt_prologue(r.clone(), args.level());
        }
        println!("routes from {} to {}:", resolved.form.id, out_form.id);
        for r in &all {
            let caps = routes::Caps::from_env(env);
            let missing = r.missing(&caps);
            let need = if missing.is_empty() {
                "available".to_string()
            } else {
                format!(
                    "needs {}",
                    missing
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            println!("  {}  cost {}  ({need})", r.describe(), r.cost());
        }
        return exit::OK;
    }

    // The route exists; can this build take it?
    let caps = routes::Caps::from_env(env);
    let missing = route.missing(&caps);
    if !missing.is_empty() {
        let m = &missing[0];
        eprintln!(
            "tile: the route {} needs {m}, which this build does not have.",
            route.describe()
        );
        match m {
            routes::Need::Feature(f) => {
                // Exit 4 means ACQUIRABLE, and that promise has to be true.
                //
                // `lift` is named in the route graph and declared by no Cargo feature,
                // because no lifter has been written for one to gate. Telling the caller
                // to "rebuild with --features lift" sent them to a command that fails
                // with "the package does not contain this feature" -- a remedy that
                // cannot work is worse than none, and it is exit 3's case, not 4's.
                match routes::how_to_get(f) {
                    routes::HowToGet::Unwritten => {
                        eprintln!(
                            "  \"{f}\" is not a build flag you can turn on: no code exists \
                             behind it yet.\n  This route is written down in the graph and \
                             nobody has implemented it. That is missing work, not a missing \
                             possibility — but no rebuild or download changes it today."
                        );
                        return exit::UNSUPPORTED;
                    }
                    routes::HowToGet::AnotherBuild => {
                        // Acquirable, but NOT by rebuilding this tree: the emitter is not
                        // in this repository. Saying "--features ascend" sent people to a
                        // command that fails.
                        eprintln!(
                            "  \"{f}\" gates code that is not in this repository, so \
                             rebuilding here cannot turn it on.\n  `tile --list-forms` \
                             still lists {}: a build that includes the {f} emitter can do \
                             this conversion.",
                            out_form.id
                        );
                        return exit::UNAVAILABLE;
                    }
                    routes::HowToGet::Rebuild => {}
                }
                // Fixable today, by rebuilding or fetching a binary that has it.
                // Reporting "unsupported" would send the caller away from a capability
                // that is one download from working.
                eprintln!(
                    "  \"{f}\" is a build feature, not a missing form: `tile --list-forms` \
                     still lists {}.\n  Rebuild with `--features {f}`, or use a release \
                     binary that has it.",
                    out_form.id
                );
                return exit::UNAVAILABLE;
            }
            routes::Need::Toolchain(t) => {
                // A tool that is BUILT rather than downloaded has no manifest entry and
                // never will, so the generic "cannot provision it here" is a dead end
                // for it. Say what would actually work instead. This is the exit-4
                // promise -- acquirable means the next line tells you how.
                if *t == "ascendc-to-rs" {
                    eprintln!("  the AscendC lifter is built from source, not downloaded:");
                    eprintln!(
                        "    cargo install --path crates/ascendc_to_rs   (in the ascend-rs checkout)"
                    );
                    eprintln!("  or point tile at one you already have:");
                    eprintln!("    TILE_ASCENDC_TO_RS=/path/to/ascendc-to-rs");
                    return exit::UNAVAILABLE;
                }
                // On demand means HERE: the route needs it, so this is the moment to get
                // it -- not at startup, and not all of them.
                let mut report = |m: &str| eprintln!("tile: {m}");
                match provision::ensure(
                    t,
                    std::env::consts::OS,
                    std::env::consts::ARCH,
                    policy(args),
                    &mut report,
                ) {
                    Ok(_) => {
                        eprintln!("tile: {t} is present; re-run to use it");
                        return exit::UNAVAILABLE;
                    }
                    Err(e) => {
                        eprintln!("tile: {e}");
                        return exit::UNAVAILABLE;
                    }
                }
            }
            routes::Need::Device(d) => {
                eprintln!("  no {d} device was detected");
                return exit::DEVICE;
            }
            routes::Need::Frontend(x) => {
                eprintln!(
                    "  no {x} frontend has been written yet, so no build can take this\n  \
                     route. tile-rs reads {} today.",
                    readable_forms().join(", ")
                );
                return exit::UNSUPPORTED;
            }
            routes::Need::Nothing => unreachable!("Nothing is always satisfied"),
        }
    }

    // Where does it land? `-o` wins; otherwise `<stem>.opt.<ext>` beside the input.
    let out_path = args
        .output
        .clone()
        .unwrap_or_else(|| derived_output(input, out_form));
    // --force destroyed a file and said only "wrote out.metal", which is the same line
    // it prints when it created one. Losing work should never be the quiet case.
    if out_path != "-" && args.force && std::path::Path::new(&out_path).exists() {
        eprintln!("tile: --force: {out_path} existed and was overwritten");
    }
    if out_path != "-" && std::path::Path::new(&out_path).exists() && !args.force {
        eprintln!(
            "tile: {out_path} already exists. Pass --force to replace it; nothing was written."
        );
        return exit::USAGE;
    }
    // Never write THROUGH a symlink. `fs::write` follows one, so `-o link.metal` where
    // `link.metal -> real.metal` silently replaces `real.metal` -- a file at a path the
    // user never named, possibly outside the directory they are working in. --force is
    // permission to replace the file that was named, not to follow a link out of the
    // tree. Verified: this used to clobber the link target's contents.
    if out_path != "-" {
        if let Ok(md) = std::fs::symlink_metadata(&out_path) {
            if md.file_type().is_symlink() {
                let target = std::fs::read_link(&out_path)
                    .map(|t| t.display().to_string())
                    .unwrap_or_else(|_| "somewhere else".into());
                eprintln!(
                    "tile: {out_path} is a symlink to {target}; writing it would replace \
                     that file, not this one. Pass -o {target} if that is what you meant, \
                     or remove the link."
                );
                return exit::USAGE;
            }
        }
    }
    if out_path != "-" {
        if let Some(dir) = std::path::Path::new(&out_path).parent() {
            // A missing output directory is an error, not a silent mkdir: creating
            // directories the user did not ask for is how a typo becomes a new tree.
            if !dir.as_os_str().is_empty() && !dir.exists() {
                eprintln!("tile: {} does not exist", dir.display());
                return exit::USAGE;
            }
        }
    }

    // Walk the route. Every hop is a real artifact; the scratch directory holds the
    // intermediates and is removed unless -k asked for them.
    let scratch = scratch::Scratch::new(input, args.keep, args.keep_dir.as_deref());
    let mut opt_report: Option<optimize::OptReport> = None;
    // The module as it was handed to the last emit, for the coverage check.
    let mut pre_emit = String::new();
    // The original source is kept: -r looks for the intrinsic there, where its name
    // survives. The emitted target source has already been translated into the target's
    // own vocabulary and no longer says what the kernel computes.
    let original = text.clone();
    let mut current = text;
    let mut current_form = resolved.form;
    for (i, hop) in route.hops.iter().enumerate() {
        let to = forms::by_id(hop.to).expect("edge target");
        // -V reports each hop as it runs; -VV adds what it consumed and produced.
        //
        // Both flags parsed and neither changed a single line of output, so `-VV` was
        // indistinguishable from no flag at all. A verbosity control that controls
        // nothing is a promise in the help text and nowhere else.
        if args.verbose > 0 {
            eprintln!(
                "tile: hop {} of {}: {} -> {} [{}]",
                i + 1,
                route.hops.len(),
                hop.from,
                hop.to,
                hop.fidelity
            );
        }
        if args.verbose > 1 {
            eprintln!(
                "tile:   in {} bytes, need {:?}, cost {}",
                current.len(),
                hop.need,
                hop.cost
            );
        }
        if hop.from == hop.to {
            // The optimize hop. It runs the passes its level permits and REPORTS both
            // what fired and what is not written yet, so nobody has to infer from the
            // output whether fusion happened.
            let (next, report) = optimize::optimize(&current, args.level());
            opt_report = Some(report);

            // The tile-rs-specific check, per operation rather than per kernel: does this
            // tiling fit what the target can actually hold? A CHECK, not a rewrite --
            // re-chunking a tile changes what the kernel computes unless it loops, and
            // this layer cannot know that it does.
            if args.level() >= 2 {
                let hp = profile::params_for(family);
                let dtype_bytes = prof
                    .kernels
                    .first()
                    .map(|k| k.widest().bytes())
                    .unwrap_or(4);
                match tile_cli::passes::check_bounds(
                    &tile_cli::mlir::parse(&next),
                    &hp,
                    dtype_bytes,
                ) {
                    Err(why) => {
                        // f10513c's rule reaching the optimizer: another chip's
                        // capacities would approve precisely the tilings that fail here.
                        scratch.finish();
                        eprintln!(
                            "tile: -O{} cannot check this tiling.\n  {why}",
                            args.level()
                        );
                        return exit::UNSUPPORTED;
                    }
                    Ok(v) if !v.is_empty() => {
                        for x in &v {
                            eprintln!("  BOUND: {} on {} — {}", x.rule, x.op, x.detail);
                        }
                        scratch.finish();
                        eprintln!(
                            "tile: the tiling does not fit {}. Emitting it would produce a kernel the hardware cannot run.",
                            hp.npu_arch
                        );
                        return exit::TRANSFORM_FAILED;
                    }
                    Ok(_) => {}
                }
            }
            if i + 1 < route.hops.len() {
                if let Err(e) = scratch.write_hop(i + 1, to, &next) {
                    scratch.finish();
                    eprintln!("tile: could not write intermediate: {e}");
                    return exit::TRANSFORM_FAILED;
                }
            }
            current = next;
            continue;
        }
        // `tile -> mlir` is the one edge that needs a toolchain: the codegen backend
        // compiles the kernel crate and writes the MLIR beside the artifact.
        // The one lift that runs. It consumes the source PATH, not the text already
        // read, because the lifter resolves includes relative to it.
        if hop.from == "cpp" && hop.to == "tile" {
            match tile_cli::lift_cpp::lift(std::path::Path::new(input)) {
                Ok(rs) => {
                    current = rs;
                    current_form = to;
                    continue;
                }
                Err(e) => {
                    scratch.finish();
                    eprintln!("tile: {e}");
                    return match e {
                        tile_cli::lift_cpp::LiftError::NotInstalled => exit::UNAVAILABLE,
                        _ => exit::TRANSFORM_FAILED,
                    };
                }
            }
        }
        if hop.from == "tile" && hop.to == "mlir" {
            let mut report = |m: &str| eprintln!("tile: {m}");
            match tile_cli::lower_rs::lower(&current, out_form, &mut report) {
                Ok(l) => {
                    // The backend emitted the target source on the way past. Taking it
                    // rather than re-emitting from the MLIR keeps one lowering, not two
                    // that could disagree.
                    if let Some(src) = l.target_source {
                        if i + 1 < route.hops.len() {
                            if let Err(e) =
                                scratch.write_hop(i + 1, forms::by_id("mlir").unwrap(), &l.mlir)
                            {
                                scratch.finish();
                                eprintln!("tile: could not write intermediate: {e}");
                                return exit::TRANSFORM_FAILED;
                            }
                        }
                        current = src;
                        break;
                    }
                    current = l.mlir;
                    continue;
                }
                Err(e) => {
                    scratch.finish();
                    eprintln!("tile: {e}");
                    return match e {
                        tile_cli::lower_rs::LowerError::Rejected { .. } => exit::TRANSFORM_FAILED,
                        tile_cli::lower_rs::LowerError::RustflagsSet { .. }
                        | tile_cli::lower_rs::LowerError::NotATarget { .. } => exit::USAGE,
                        _ => exit::UNAVAILABLE,
                    };
                }
            }
        }

        // Did the emitter actually lower what this kernel is made of? Lowering is a
        // partial function written as a total one: an intrinsic with no arm falls through
        // to output that compiles and computes something else. Backlog #005.
        for ig in emit::ignored_intrinsics(to, &current) {
            eprintln!("tile: IGNORED: {ig}");
        }

        pre_emit = current.clone();
        let next = match emit::emit(to, &current) {
            Ok(s) => s,
            Err(emit::EmitError::NotCompiledIn { form, feature }) => {
                scratch.finish();
                eprintln!(
                    "tile: this build has no {form} emitter.\n  \
                     \"{feature}\" is a build feature, not a missing form: `tile --list-forms`\n  \
                     still lists {form}. Rebuild with `--features {feature}`, or use a\n  \
                     release binary that has it."
                );
                return exit::UNAVAILABLE;
            }
            Err(e) => {
                // A failed run must not litter. `-k` still preserves the hops that DID
                // succeed, because a failure is exactly when a user wants to look at them.
                for k in scratch.finish() {
                    eprintln!("tile: kept {k}");
                }
                eprintln!(
                    "tile: hop {} of {} ({} -> {}) failed: {e}",
                    i + 1,
                    route.hops.len(),
                    hop.from,
                    hop.to
                );
                return exit::TRANSFORM_FAILED;
            }
        };
        // Not the last hop? It is an intermediate, and it goes in the scratch.
        if i + 1 < route.hops.len() {
            if let Err(e) = scratch.write_hop(i + 1, to, &next) {
                scratch.finish();
                eprintln!("tile: could not write intermediate: {e}");
                return exit::TRANSFORM_FAILED;
            }
        }
        current = next;
        current_form = to;
    }

    // The bounds gate for a target with real tiles, AFTER lowering. Before the
    // emitter has run, a cube op's block shape is not decided -- it blocks K
    // and N, and caps Kb by M, when the whole shape will not fit L0 -- so the
    // pre-lowering pass defers cube ops rather than judging a tiling nobody
    // chose yet. Here every tile is spelled out with the memory that holds it,
    // so this checks what will actually run instead of predicting it.
    if args.level() >= 2 && current.contains("pto.alloc_tile") {
        let hp = profile::params_for(family);
        match tile_cli::passes::check_pto_tiles(&current, &hp) {
            Err(why) => {
                scratch.finish();
                eprintln!(
                    "tile: -O{} cannot check this tiling.\n  {why}",
                    args.level()
                );
                return exit::UNSUPPORTED;
            }
            Ok(v) if !v.is_empty() => {
                for x in &v {
                    eprintln!("  BOUND: {} on {} — {}", x.rule, x.op, x.detail);
                }
                scratch.finish();
                eprintln!(
                    "tile: the EMITTED tiling does not fit {}. This is the tiling the \
                     emitter chose, not a predicted one.",
                    hp.npu_arch
                );
                return exit::TRANSFORM_FAILED;
            }
            Ok(_) => {}
        }
    }

    if out_path == "-" {
        // stdout carries the result and nothing else, which is what makes this pipeable.
        print!("{current}");
    } else if let Err(e) = std::fs::write(&out_path, &current) {
        scratch.finish();
        eprintln!("tile: could not write {out_path}: {e}");
        return exit::TRANSFORM_FAILED;
    }

    // -O4: measure candidate dispatch configurations on the real device.
    //
    // This is what makes O4 the MEASURED level. It runs AFTER the artifact is written,
    // because the artifact is what gets measured -- and because a sweep that failed must
    // not cost the user the conversion that succeeded.
    let mut tuning_note: Option<String> = None;
    if args.level() == 4 {
        match tile_cli::run::KernelAbi::parse(out_form, &current) {
            // A kernel whose width is compiled in has nothing to sweep, and saying so beats
            // measuring the one width it has and calling it "best ... under the slowest
            // configuration measured" -- a superlative over a set of one.
            Ok(abi) if abi.local_x.is_some() => {
                eprintln!(
                    "  O4: {} fixes its workgroup at {} in the module, so there is no \
                     dispatch width to sweep. The kernel was emitted and checked; nothing \
                     was tuned.",
                    out_form.id,
                    abi.local_x.unwrap_or(0)
                );
            }
            Ok(abi) => {
                // The same shape the `-r` path uses. This built its own with `rows: 1`
                // and `cols` = the total element count -- the flattening #017 removed --
                // so -O4 was tuning a dispatch geometry that did not match the one the
                // kernel is verified under. Two places deciding the same thing is how the
                // first one drifted.
                let op4 = tile_cli::run::RefOp::detect(&original).ok();
                let eps4 = tile_cli::run::Shape::rms_eps_from_mlir(&original).unwrap_or(1e-6);
                let shape = if original.contains("__tile_matmul_") {
                    tile_cli::run::Shape::matmul_from_mlir(&original)
                } else if let (Some(o), Some(k)) = (op4, prof.kernels.first()) {
                    shape_from_rows_cols(&original, o, k).or_else(|| {
                        Some(tile_cli::run::Shape::Rows {
                            rows: 1,
                            cols: k.numel().max(1),
                        })
                    })
                } else {
                    prof.kernels.first().map(|k| tile_cli::run::Shape::Rows {
                        rows: 1,
                        cols: k.numel().max(1),
                    })
                };
                let sh4 = shape;
                match shape.map(|sh| {
                    tile_cli::run::autotune(out_form, &current, &abi, sh, args.iterations)
                }) {
                    Some(Ok(cands)) => {
                        eprintln!("  O4 measured on {}:", plat_device(env));
                        for c in &cands {
                            eprintln!(
                                "      threadgroup {:>5}  median {:.2} us",
                                c.threads, c.median_us
                            );
                        }
                        // A width must be CORRECT before it can be fastest.
                        //
                        // The sweep times each threadgroup width and never reads the
                        // output buffer -- it exits before any value is printed -- so -O4
                        // selected a width purely on speed. That is exactly the wrong test
                        // for this axis: the power-of-two tree fold produced WRONG ANSWERS
                        // at thread counts that were not powers of two, and a sweep that
                        // only times would have recommended one of them for being fastest.
                        //
                        // Each candidate is re-run through the ordinary verified path at
                        // its own width, and any that disagrees is dropped with its reason
                        // stated. `run_at_width` is the same comparison the single run
                        // uses, so the two cannot drift.
                        // An op with no reference cannot be checked at all. That must
                        // not silently remove the recommendation -- it qualifies it. The
                        // widths are still ranked, and the report says the ranking rests
                        // on timing alone.
                        let mut checked: Vec<tile_cli::run::Candidate> = Vec::new();
                        let verifiable = op4.is_some();
                        if !verifiable {
                            eprintln!(
                                "      note: this op has no reference here, so the widths \
                                 below are ranked on TIMING\n            alone -- nothing \
                                 confirmed the kernel still computes the same answer at \
                                 each."
                            );
                        }
                        for c in &cands {
                            if !verifiable {
                                checked.push(*c);
                                continue;
                            }
                            match op4
                                .ok_or_else(|| "no reference for this op".to_string())
                                .and_then(|o: tile_cli::run::RefOp| {
                                    tile_cli::run::verify_at_width(
                                        out_form,
                                        &current,
                                        &abi,
                                        sh4.expect("a shape, or autotune would not have run"),
                                        o,
                                        eps4,
                                        c.threads,
                                    )
                                }) {
                                Ok(true) => checked.push(*c),
                                Ok(false) => eprintln!(
                                    "      threadgroup {}: DROPPED — it computes a \
                                     different answer at this width, so it cannot be \
                                     recommended however fast it is",
                                    c.threads
                                ),
                                Err(e) => eprintln!(
                                    "      threadgroup {}: dropped, could not be verified \
                                     ({e})",
                                    c.threads
                                ),
                            }
                        }
                        if checked.is_empty() {
                            eprintln!(
                                "tile: -O4 measured {} widths and none of them reproduced \
                                 the reference. Not recommending a dispatch width -- a \
                                 fast configuration that computes the wrong answer is \
                                 worth less than no recommendation.",
                                cands.len()
                            );
                            return exit::OK;
                        }
                        if verifiable && checked.len() < cands.len() {
                            eprintln!(
                                "      {} of {} widths verified; the rest are excluded from \
                                 the ranking.",
                                checked.len(),
                                cands.len()
                            );
                        }
                        let cands = checked;
                        let worst = cands.first().expect("non-empty");
                        let best = cands.last().expect("non-empty");
                        // A MEASURED headroom, with both numbers in its basis. This is
                        // the one place a headroom may be written, because it is the one
                        // place something was measured.
                        let gain = if worst.median_us > 0.0 {
                            (worst.median_us - best.median_us) / worst.median_us
                        } else {
                            0.0
                        };
                        eprintln!(
                            "      best: threadgroup {} at {:.2} us — {:.0}% under the \
                             slowest configuration measured",
                            best.threads,
                            best.median_us,
                            gain * 100.0
                        );
                        for c in &cands {
                            // Every candidate, with ITS OWN measurement. headroom stays
                            // None on these: one timing establishes a cost, not a
                            // headroom, and None means UNASSESSED rather than zero.
                            let rec = tile_cli::corpus::Record {
                                kernel: prof
                                    .kernels
                                    .first()
                                    .map(|k| k.name.clone())
                                    .unwrap_or_else(|| "<unnamed>".into()),
                                target: out_form.id.to_string(),
                                state: "implemented".into(),
                                headroom: None,
                                basis: format!(
                                    "O4 candidate: threadgroup {} measured {:.2} us on {}",
                                    c.threads,
                                    c.median_us,
                                    plat_device(env)
                                ),
                                updated: today(),
                            };
                            let _ = tile_cli::corpus::append_local(&rec);
                        }
                        // Say it in the ARTIFACT too, not only in the corpus.
                        //
                        // The measurement is about how the host must LAUNCH this kernel,
                        // and whoever writes that host code reads the .metal file, not
                        // `tile -s`. A measured result nobody can act on is a number in a
                        // drawer. Appended here rather than in the emitter: emit stays a
                        // pure function of its input, and this is the CLI recording what
                        // it observed on a specific machine.
                        tuning_note = Some(format!(
                            "\n// O4 (measured on {}): dispatch this with a threadgroup of \
                             {} threads.\n\
                             // Measured {:.2} us there, against {:.2} us at {} — the \
                             slowest width tried.\n\
                             // This is a measurement of ONE machine, not a portable \
                             constant: re-run `-O4` on the target you deploy to.\n",
                            plat_device(env),
                            best.threads,
                            best.median_us,
                            worst.median_us,
                            worst.threads
                        ));

                        let summary = tile_cli::corpus::Record {
                            kernel: prof
                                .kernels
                                .first()
                                .map(|k| k.name.clone())
                                .unwrap_or_else(|| "<unnamed>".into()),
                            target: out_form.id.to_string(),
                            state: "implemented".into(),
                            headroom: Some(gain),
                            basis: format!(
                                "O4 sweep on {}: best threadgroup {} at {:.2} us vs worst {} at {:.2} us",
                                plat_device(env),
                                best.threads,
                                best.median_us,
                                worst.threads,
                                worst.median_us
                            ),
                            updated: today(),
                        };
                        let _ = tile_cli::corpus::append_local(&summary);
                    }
                    Some(Err(e)) => eprintln!("tile: -O4 could not measure: {e}"),
                    None => eprintln!(
                        "tile: -O4 has no shape to measure for this kernel; nothing was tuned"
                    ),
                }
            }
            Err(why) => eprintln!("tile: -O4 cannot read the emitted signature: {why}"),
        }
    }

    // Append the measured launch note to the artifact that was just written.
    //
    // After the write rather than before it, because the thing measured IS the written
    // artifact -- and because a sweep that fails must not cost the user a conversion that
    // succeeded. Nothing is appended when there is no measurement to report.
    if let Some(note) = &tuning_note {
        if out_path == "-" {
            print!("{note}");
        } else if let Err(e) = std::fs::OpenOptions::new()
            .append(true)
            .open(&out_path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, note.as_bytes()))
        {
            eprintln!("tile: measured the tuning but could not append it to {out_path}: {e}");
        }
    }

    // Record the attempt, so `-s` has something to report.
    //
    // `corpus::append_local` existed and nothing ever called it, so the local store was
    // always empty and `-s` said "nothing recorded" however many conversions had been
    // run. The two-store design was in place; only the writing half was missing.
    //
    // `headroom: None` is deliberate and must stay: None is UNASSESSED and Some(0.0) is
    // CLOSED, and a conversion measures neither. Writing 0.0 here would turn "nobody
    // looked" into "there is nothing to gain" -- silently, and in the direction that
    // stops anyone looking again.
    for k in &prof.kernels {
        let rec = tile_cli::corpus::Record {
            kernel: k.name.clone(),
            target: out_form.id.to_string(),
            state: "implemented".into(),
            headroom: None,
            basis: format!(
                "converted {} at -O{} [{}]",
                route.forms().join(" -> "),
                args.level(),
                route.fidelity()
            ),
            updated: today(),
        };
        if let Err(e) = tile_cli::corpus::append_local(&rec) {
            // Never fatal: the artifact is written and correct. Losing the note is a
            // smaller harm than failing a conversion that succeeded.
            if args.verbose > 0 {
                eprintln!("tile: could not record this attempt: {e}");
            }
        }
    }

    // A kernel whose operation the emitter never looked at is not a lowering. Said after
    // the file is written, because the output is evidence worth keeping — but the exit
    // code has to carry it, or a build system treats this as a success.
    let ignored = emit::ignored_intrinsics(out_form, &pre_emit);
    if !ignored.is_empty() {
        eprintln!(
            "tile: the {} emitter produced output without using {} of this kernel's operations.\n  What it wrote is not a lowering of what you gave it.",
            out_form.id,
            ignored.len()
        );
    }

    let kept = scratch.finish();
    if args.verbose > 0 || !kept.is_empty() {
        for k in &kept {
            eprintln!("tile: kept {k}");
        }
    }
    if out_path != "-" {
        println!("wrote {out_path}");
    }
    if !ignored.is_empty() {
        eprintln!(
            "  Not a failure of your kernel: the emitter has no arm for it and did not refuse.\n  Recorded as backlog #005; `tile backlog` has the evidence."
        );
        return exit::TRANSFORM_FAILED;
    }

    if args.run {
        let code = run_kernel(args, &prof, out_form, &current, &original, resolved.form);
        if code != exit::OK {
            return code;
        }
    }
    if let Some(r) = &opt_report {
        eprintln!("{}", r.render());
    }

    // -O3: ask the target's own compiler what it thinks of what we just wrote. It runs on
    // the OUTPUT, so it happens here rather than inside the optimize hop — and it is
    // never fatal, because refusing a conversion for a check that could not run would be
    // worse than not offering the check.
    if args.level() >= 3 && out_path != "-" {
        let fb = optimize::toolchain_feedback(out_form, &current);
        eprintln!("  O3: {fb}");
        if let tile_cli::toolchain::Feedback::Rejected { .. } = fb {
            eprintln!(
                "  the file was written; the target compiler rejected it, which is a \n                   lowering bug worth reporting rather than a reason to hide the output."
            );
            return exit::TRANSFORM_FAILED;
        }
    }
    // The last line of every run: where it went, and how much to trust it.
    eprintln!(
        "route: {}  -O{}  [{}]",
        route.forms().join(" -> "),
        args.level(),
        route.fidelity()
    );
    let _ = current_form;
    let _ = std::io::stdout().flush();
    exit::OK
}

/// The forms tile-rs can read, counted rather than written down — a refusal that names
/// a stale list is worse than one that names none.
fn readable_forms() -> Vec<&'static str> {
    forms::FORMS
        .iter()
        .filter(|f| f.readable)
        .map(|f| f.id)
        .collect()
}

/// `-r`: put the kernel on the device, check it, and time it.
///
/// Returns OK even when the run could not happen for want of a device or a harness — a
/// conversion that succeeded is not retroactively a failure because a *check* could not
/// run. It is a failure only when the kernel ran and was WRONG.
fn run_kernel(
    args: &cli::Args,
    prof: &profile::Profile,
    out_form: &'static forms::Form,
    emitted: &str,
    original: &str,
    in_form: &'static forms::Form,
) -> i32 {
    use tile_cli::run;
    use tile_cli::torchref;

    let Some(kernel) = prof.kernels.first() else {
        // Not OK: `-r` was asked for and did not happen. See the signature-parse branch
        // below for the argument — a caller reading exit 0 concludes the kernel ran and
        // agreed. TRANSFORM_FAILED rather than UNSUPPORTED because nothing is missing from
        // tile-rs here; the input simply had no kernel in it.
        eprintln!("tile: --run found no kernel to run");
        return exit::TRANSFORM_FAILED;
    };

    let op = match run::RefOp::detect(original) {
        Ok(o) => o,
        Err(hint) => {
            // UNSUPPORTED: the harness has no reference for this operation, which is a
            // piece of work nobody has done rather than a property of the kernel. Returning
            // OK here reported a verification that never ran as a success.
            eprintln!("tile: {}", run::RunError::NoReference { hint });
            return exit::UNSUPPORTED;
        }
    };

    // The dispatch geometry. A matmul's is read out of the MLIR rather than guessed: the
    // three dimensions are operands of the intrinsic call, and inventing them here would
    // let the harness disagree with the kernel about the shape it is multiplying.
    let shape = if op_is_matmul(original) {
        match run::Shape::matmul_from_mlir(original) {
            Some(s) => s,
            None => {
                eprintln!(
                    "tile: --run found a matmul whose dimensions are not constants in the \
                     source, so the harness cannot size its buffers"
                );
                return exit::OK;
            }
        }
    } else if let Some(shape) = shape_from_rows_cols(original, op, kernel) {
        shape
    } else if op == run::RefOp::Matvec {
        // rows x cols matrix against a cols vector. The kernel indexes its matrix row by
        // threadgroup position and strides the columns, so the shape's rows must be the
        // grid and its cols the row length -- driving it as elementwise would size the
        // output buffer rows*cols and read back memory past what the kernel wrote.
        run::Shape::Matvec {
            rows: 4,
            cols: kernel.numel().max(1),
        }
    } else if op.is_two_input_elementwise() {
        // Two tiles in, one out, same extent. Driving this as Shape::Rows would supply one
        // input buffer to a kernel that reads two, and the arity check in `run` refuses
        // that -- correctly, but the shape is the thing that was wrong.
        run::Shape::Rows2 {
            rows: 1,
            cols: kernel.numel().max(1),
        }
    } else if op.is_row_reduction() {
        // One value out per row in. Sizing this as elementwise would make the harness
        // read back a buffer the kernel never wrote past its first element.
        run::Shape::RowReduce {
            rows: 1,
            cols: kernel.numel().max(1),
        }
    } else {
        run::Shape::Rows {
            rows: 1,
            cols: kernel.numel().max(1),
        }
    };

    // What does it compute? Looked for in the ORIGINAL source, where the intrinsic name
    // survives; the emitted target source has already been translated into the target's
    // own vocabulary.

    let abi = match run::KernelAbi::parse(out_form, emitted) {
        Ok(a) => a,
        Err(why) => {
            // Only Metal has a signature parser today. Everything else has no harness at
            // all, so this is reached only when the emitted Metal is shaped unexpectedly.
            //
            // UNSUPPORTED, not OK. The conversion did succeed and the file was written,
            // which is what OK used to be reporting -- but the caller asked for `-r`, and
            // the run did not happen. A CI step reading exit 0 from `tile k.mlir -t msl -r`
            // concludes the kernel ran and agreed with torch. That is the exact shape of
            // false report this tool exists to refuse: the run is not unsupported by the
            // device, it is unwritten in the harness, which is what 3 means.
            eprintln!("tile: --run cannot read the emitted kernel's signature: {why}");
            return exit::UNSUPPORTED;
        }
    };

    let (device, values, target) = match run::run(out_form, emitted, &abi, shape, args.iterations) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tile: {e}");
            return match e {
                run::RunError::Wrong { .. } => exit::TRANSFORM_FAILED,
                run::RunError::NoDevice { .. } => exit::OK,
                _ => exit::OK,
            };
        }
    };

    // eps from the SOURCE, so the reference and the kernel use one number. The emitter
    // reads the same operand and refuses a non-constant, so a kernel that reached here has
    // one; 1e-6 is the fallback for ops that take no eps at all, where it is never read.
    let eps = run::Shape::rms_eps_from_mlir(original).unwrap_or(1e-6);
    // The eps is baked into the kernel at lowering time and passed to the references at
    // run time. If those two numbers differ, the references agree with each other and the
    // kernel is blamed -- so it is checked rather than assumed. See `eps_reaches_the_kernel`.
    if op == run::RefOp::RmsNorm && !run::eps_reaches_the_kernel(emitted, eps) {
        eprintln!(
            "tile: the reference is using eps={eps:e}, and no such value appears in the \n             \x20     emitted kernel. One of the two is reading the MLIR wrongly, and the \n             \x20     accuracy figures below compare two different functions. Fix this \n             \x20     before reading the verdict -- it will name the kernel, which may be \n             \x20     the side that is right."
        );
    }
    let (checked, worst, nan_mismatch) = run::compare_detailed(op, shape, &values, eps, &abi.dtype);
    let ours = run::reference_output(op, shape, eps, &abi.dtype);
    let vs_ours = torchref::Accuracy::of(&values, &ours);
    let mut say = |m: &str| eprintln!("tile: {m}");
    let torch = match torchref::reference(op, shape, eps, &abi.dtype, policy(args), &mut say) {
        Ok((version, t, via)) => Ok((
            match via {
                torchref::Via::SystemPython => version,
                torchref::Via::Uv => format!("{version} (via uv)"),
            },
            torchref::Accuracy::of(&values, &t),
            torchref::Accuracy::of(&ours, &t),
        )),
        Err(status) => Err(status),
    };

    // What did the optimizer buy? Only asked when -O0 would differ, and only reported
    // when it actually did: "1.00x" implies a measurement that was never taken.
    let (unoptimized, identical) = if args.level() == 0 || in_form.id != "mlir" {
        (None, true)
    } else {
        let (o0, _) = optimize::optimize(original, 0);
        match emit::emit(out_form, &o0) {
            Ok(src) if src != emitted => {
                match run::run(out_form, &src, &abi, shape, args.iterations) {
                    Ok((_, _, t)) => (Some(t), false),
                    Err(_) => (None, true),
                }
            }
            _ => (None, true),
        }
    };

    // #013: the budget is computed once and used for the report, the verdict AND the
    // exit code below. There were three separate constants doing this job.
    let budget = run::error_budget(op, shape, &abi.dtype);
    let tol = match budget {
        Some(b) => torchref::Tolerance::Summation(b),
        // The relative bound is never tighter than the buffer type's own roundoff: an
        // f16 result is stored in half, so it cannot be closer than 2^-11 however good the
        // lowering is. Derived from the format, not chosen -- see run::unit_roundoff.
        None => torchref::Tolerance::Fixed {
            abs: 1e-5,
            rel: 1e-4f32.max(run::unit_roundoff(&abi.dtype)),
        },
    };
    let report = run::RunReport {
        device,
        op,
        dtype: abi.dtype.clone(),
        elements: shape.out_size(),
        checked,
        worst_error: worst,
        nan_mismatch,
        tolerance: tol.abs(),
        target,
        reference: run::time_reference(op, shape, args.iterations),
        unoptimized,
        optimizer_changed_nothing: identical,
        budget,
        vs_ours,
        torch,
    };
    eprint!("{}", report.render());

    // Was `worst > 1e-4`: a third constant, an order looser than the one the verdict
    // used, so a kernel could be reported outside tolerance and still exit 0. It reads
    // the same budget as everything else now.
    if worst > tol.abs() {
        eprintln!(
            "tile: the kernel is outside tolerance. The file was written; a lowering that \n               computes the wrong answer is worth reporting, not hiding."
        );
        return exit::TRANSFORM_FAILED;
    }
    exit::OK
}

fn shell_quote(s: &str) -> String {
    if s.contains(' ') {
        format!("'{s}'")
    } else {
        s.to_string()
    }
}
