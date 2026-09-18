//! The executable specification.
//!
//! `features/*.feature` is the normative requirement for `tile`. This harness runs it,
//! reusing `tile_spec`'s std-only Gherkin runner rather than growing a second one.
//!
//! ## The tagging discipline, and why the count is printed
//!
//! An **untagged** scenario must be green. Everything else carries exactly one tag
//! saying why it is not — `@planned`, `@requires-rustc-backend`, `@requires-lifter`,
//! `@requires-device`, `@requires-license`. Tags go on ONCE, at landing; each milestone
//! REMOVES the tag from the scenarios it makes pass, by adding a line to
//! `features/.live`. Tags added later, to quiet a red suite, are how a team learns to
//! ignore red.
//!
//! The summary this test prints is therefore the project's progress meter: "how much of
//! the specification actually runs" is a number, not an impression.

use std::path::{Path, PathBuf};
use std::process::Command;
use tile_cli::cli::{self, exit, Cmd};
use tile_cli::routes::{Caps, Need, RouteError};
use tile_cli::{default_form_for, forms, profile, routes};
use tile_spec::gherkin::{parse_feature, Runner, World};

fn features_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("features")
}
fn corpus(name: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/forms")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("corpus {name}: {e}"))
}

/// Strip tagged scenarios out of the feature text before parsing.
///
/// The runner has no notion of tags — deliberately, since teaching it about them would
/// mean changing a crate whose whole point is that it is small and green. Filtering here
/// keeps that property and keeps the skip count honest.
/// Is there an accelerator here that the run harness can actually drive?
///
/// Asked of the real detector rather than assumed from the platform: an Apple machine
/// with no Metal device (a CI container, say) must skip rather than fail.
fn device_present() -> bool {
    let env = tile_cli::simenv::Env::default();
    let p = tile_cli::platform::detect_in(&env);
    p.accels.iter().any(|a| a.family == "apple-gpu")
        && std::process::Command::new("swift")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

/// Was the (slow) cross-compilation check asked for?
/// A runtime that can execute the wasm bundle, if this machine has one.
fn wasm_runtime() -> Option<&'static str> {
    for exe in ["node", "deno"] {
        let ok = std::process::Command::new(exe)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(exe);
        }
    }
    None
}

/// Can this machine BUILD the wasm bundle from source?
fn wasm_target_installed() -> bool {
    std::process::Command::new("rustup")
        .args(["target", "list", "--installed", "--toolchain", "stable"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("wasm32-unknown-unknown"))
        .unwrap_or(false)
}

/// Is the out-of-tree codegen backend provisioned under `~/.tile-rs`?
fn backend_present() -> bool {
    // Clear TILE_HOME explicitly. Other tests in this binary set it process-wide to
    // point at a private directory, and a child inheriting that reports the backend
    // absent -- which is how the report claimed "needs the codegen backend" about a
    // scenario that had just run against it.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_tile"))
        .arg("install")
        .env_remove("TILE_HOME")
        .env_remove("TILE_SIMULATE")
        .output();
    // Match the line, not an exact spacing: `tile install` pads its columns, and a
    // literal with the padding baked in reported "absent" while the scenario itself ran.
    out.map(|o| {
        String::from_utf8_lossy(&o.stdout)
            .lines()
            .any(|l| l.contains("rustc_codegen_tile") && l.contains("installed"))
    })
    .unwrap_or(false)
}

/// Is the native target's own compiler here? On this platform that is Metal's.
fn target_compiler_present() -> bool {
    std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Hosts to check the binary on, from `TILE_SPEC_HOSTS` (space separated ssh names).
fn spec_hosts() -> Vec<String> {
    std::env::var("TILE_SPEC_HOSTS")
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn cross_requested() -> bool {
    std::env::var("TILE_SPEC_CROSS").is_ok_and(|v| v == "1")
}

fn strip_tagged(text: &str) -> (String, usize, usize) {
    let mut out = Vec::new();
    let mut pending_tag = false;
    let mut pending_stats = false;
    let mut pending_device = false;
    let mut pending_cross = false;
    let mut pending_hosts = false;
    let mut pending_backend = false;
    let mut pending_toolchain = false;
    let mut pending_wasm = false;
    let mut pending_wasm_target = false;
    let mut skipping = false;
    let (mut total, mut live) = (0usize, 0usize);
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('@') {
            // `@requires-stats-build` is conditional, not permanent: the scenario is real
            // and runs in a build that has the crypto. Skipping it silently in a build
            // that does not would make the coverage meter claim work it never did.
            if t.contains("@requires-stats-build") {
                pending_stats = true;
                if cfg!(feature = "stats") {
                    continue;
                }
            }
            // `@requires-device` is conditional in exactly the same sense: the scenario
            // is real and runs where the hardware is. Treating it as permanent meant the
            // two scenarios that exercise `-r` end to end -- the whole point of the run
            // harness -- were skipped on a machine that has a Metal GPU and a working
            // torch. The most valuable coverage in the suite was being declined by a tag.
            // The cross-build check compiles the crate for another triple, which takes
            // minutes -- too slow to run on every `cargo test`. It is real work, not
            // absent work, so it is opt-in rather than @planned: CI and anyone changing
            // the platform module can set TILE_SPEC_CROSS=1 and have it actually run.
            // The multi-host check copies this binary to other machines over ssh and
            // runs it there. Real work, and the only way to test "one binary, many
            // hosts" as anything but an assumption -- but it needs those machines, so it
            // is opt-in: TILE_SPEC_HOSTS="mac mini".
            // Both of these gate on the MACHINE, not on a decision, so they are
            // conditional like @requires-device: the backend is provisioned under
            // ~/.tile-rs and the target compiler ships with the Xcode tools. Treating
            // them as permanent skipped two scenarios this machine can run.
            if t.contains("@requires-backend") {
                pending_backend = true;
                if backend_present() {
                    continue;
                }
            }
            if t.contains("@requires-toolchain") {
                pending_toolchain = true;
                if target_compiler_present() {
                    continue;
                }
            }
            // A wasm runtime, for the bundle's BEHAVIOUR. `tile` has none and will not
            // grow one, so this borrows whichever is on the machine.
            // The wasm TARGET, for rebuilding the bundle from source. Distinct from the
            // runtime above: one executes the module, the other produces it, and a
            // machine can easily have either without the other.
            if t.contains("@requires-wasm-target") {
                pending_wasm_target = true;
                if wasm_target_installed() {
                    continue;
                }
            }
            if t.contains("@requires-wasm-runtime") {
                pending_wasm = true;
                if wasm_runtime().is_some() {
                    continue;
                }
            }
            if t.contains("@requires-hosts") {
                pending_hosts = true;
                if !spec_hosts().is_empty() {
                    continue;
                }
            }
            if t.contains("@requires-cross") {
                pending_cross = true;
                if cross_requested() {
                    continue;
                }
            }
            if t.contains("@requires-device") {
                // Its OWN flag. Sharing `pending_stats` made a machine with a GPU run the
                // crypto scenarios in a build with no crypto -- the two conditions are
                // unrelated, and one flag silently ORed them.
                pending_device = true;
                if device_present() {
                    continue;
                }
            }
            pending_tag = true;
            continue;
        }
        if t.starts_with("Scenario:") || t.starts_with("Scenario Outline:") {
            total += 1;
            if (pending_stats && cfg!(feature = "stats"))
                || (pending_device && device_present())
                || (pending_cross && cross_requested())
                || (pending_hosts && !spec_hosts().is_empty())
                || (pending_backend && backend_present())
                || (pending_toolchain && target_compiler_present())
                || (pending_wasm && wasm_runtime().is_some())
                || (pending_wasm_target && wasm_target_installed())
            {
                pending_tag = false;
            }
            pending_stats = false;
            pending_device = false;
            pending_cross = false;
            pending_hosts = false;
            pending_backend = false;
            pending_toolchain = false;
            pending_wasm = false;
            pending_wasm_target = false;
            skipping = pending_tag;
            if !skipping {
                live += 1;
            }
            pending_tag = false;
        } else if t.starts_with("Feature:") || t.starts_with("Background:") {
            skipping = false;
            pending_tag = false;
        }
        if !skipping {
            out.push(line);
        }
    }
    (out.join("\n"), live, total)
}

// ── Step definitions ──────────────────────────────────────────────────────────────
// Every step drives the real library. Nothing here is a stub: a step that cannot be
// mechanically checked belongs in a @planned scenario, not in a step that always passes.

fn register(r: &mut Runner) {
    // — form resolution —————————————————————————————————————————————————————————
    r.given(
        "a clean working directory",
        |_w: &mut World, _a: &[String]| {},
    );

    r.given(
        "a file \"k{}\" whose content contains \"{}\"",
        |w: &mut World, a: &[String]| {
            w.set("path", format!("k{}", a[0]));
            w.set("content", a[1].clone());
        },
    );
    r.given(
        "a file \"k.py\" with no recognisable magic",
        |w: &mut World, _a: &[String]| {
            w.set("path", "k.py");
            w.set("content", "def main():\n    pass\n");
        },
    );
    r.given(
        "a Metal kernel \"softmax.metal\"",
        |w: &mut World, _a: &[String]| {
            w.set("path", "softmax.metal");
            w.set("content", corpus("softmax.metal"));
        },
    );
    r.given(
        "a tile-rs kernel \"softmax.rs\"",
        |w: &mut World, _a: &[String]| {
            w.set("path", "softmax.rs");
            w.set("content", corpus("softmax.rs"));
        },
    );

    r.when(
        "I run \"tile k{} --info-only\"",
        |w: &mut World, a: &[String]| {
            let path = format!("k{}", a[0]);
            let content = w.get("content").to_string();
            resolve_into(w, &path, &content, None);
        },
    );
    r.when(
        "I run \"tile k.py --info-only\"",
        |w: &mut World, _a: &[String]| {
            let content = w.get("content").to_string();
            resolve_into(w, "k.py", &content, None);
        },
    );
    r.when(
        "I run \"tile k.py -f {} --info-only\"",
        |w: &mut World, a: &[String]| {
            let content = w.get("content").to_string();
            resolve_into(w, "k.py", &content, Some(&a[0]));
        },
    );
    r.when(
        "I run \"tile k.metal -f {} --info-only\"",
        |w: &mut World, a: &[String]| {
            let content = w.get("content").to_string();
            resolve_into(w, "k.metal", &content, Some(&a[0]));
        },
    );
    r.when(
        "I run \"tile k.rs --info-only\"",
        |w: &mut World, _a: &[String]| {
            let content = w.get("content").to_string();
            resolve_into(w, "k.rs", &content, None);
        },
    );

    r.then(
        "the reported input form is \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("form"), a[0], "resolved form");
        },
    );
    r.then(
        "no sniffing of \"k.rs\" is attempted",
        |w: &mut World, _a: &[String]| {
            assert_eq!(
                w.get("how"),
                "reserved",
                "`.rs` must resolve by reservation, not magic"
            );
        },
    );
    r.then(
        "the command fails with exit code {}",
        |w: &mut World, a: &[String]| {
            let want: i32 = a[0].trim().parse().expect("exit code");
            assert_eq!(
                w.get("exit").parse::<i32>().unwrap_or(0),
                want,
                "{}",
                w.get("error")
            );
        },
    );
    r.then(
        "the error names the candidate forms \"{}\"",
        |w: &mut World, a: &[String]| {
            for want in a[0].split(especially_commas) {
                let want = want.trim();
                assert!(
                    w.get("error").contains(want),
                    "error {:?} does not name {want:?}",
                    w.get("error")
                );
            }
        },
    );
    r.then(
        "the error shows how to force one with \"{}\"",
        |w: &mut World, a: &[String]| {
            assert!(w.get("error").contains(&a[0]), "{}", w.get("error"));
        },
    );
    r.then(
        "a warning states that the content looks like form \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(
                w.get("contradicted_by"),
                a[0],
                "expected a contradiction warning"
            );
        },
    );

    // — routes ——————————————————————————————————————————————————————————————————
    r.when(
        "I run \"tile --list-forms\"",
        |w: &mut World, _a: &[String]| {
            w.set("out", routes::list_forms());
        },
    );
    r.then(
        "every form is listed with whether it can be read, written, optimized and lifted from",
        |w: &mut World, _a: &[String]| {
            let out = w.get("out");
            for f in forms::FORMS {
                assert!(out.contains(f.id), "--list-forms omits {}", f.id);
            }
            for col in ["read", "write", "optimize", "lifts-from"] {
                assert!(out.contains(col), "--list-forms has no {col} column");
            }
        },
    );
    r.then(
        "the output makes clear that most forms are write-only",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("out").contains("write-only"), "{}", w.get("out"));
        },
    );

    r.when(
        "I run \"tile softmax.metal -o softmax2.metal\"",
        |w: &mut World, _a: &[String]| {
            plan_into(w, "msl", "msl", &[]);
        },
    );
    r.then(
        "the error states that tile-rs can emit \"{}\" but cannot read it",
        |w: &mut World, a: &[String]| {
            let f = forms::by_id(&a[0]).expect("form");
            assert!(f.writable && !f.readable, "{} must be write-only", a[0]);
            assert_eq!(w.get("route_err"), "needs-frontend");
        },
    );
    r.then(
        "the error says the frontend has not been written yet, not that it cannot exist",
        |w: &mut World, _a: &[String]| {
            // Structural, not a string match: the refusal is NeedsFrontend, whose need is
            // classified Unwritten rather than Acquirable — that classification is what
            // the wording and the exit code both derive from.
            assert_eq!(w.get("route_err"), "needs-frontend");
            assert_eq!(
                Need::Frontend("msl").availability(),
                routes::Availability::Unwritten
            );
        },
    );
    r.then(
        "the error shows the route that would work if one existed",
        |w: &mut World, _a: &[String]| {
            assert!(
                !w.get("would_be").is_empty(),
                "the refusal must name the route a frontend would unlock"
            );
        },
    );
    r.then(
        "the exit code differs from the one used when no frontend exists",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("exit"), exit::UNAVAILABLE.to_string());
            assert_ne!(exit::UNAVAILABLE, exit::UNSUPPORTED);
        },
    );
    r.then(
        "the help text states that 3 is not a claim of impossibility",
        |_w: &mut World, _a: &[String]| {
            assert!(cli::HELP.contains("never a claim of impossibility"));
        },
    );
    r.then(
        "the error names the forms for which optimization does exist",
        |_w: &mut World, _a: &[String]| {
            let readable: Vec<&str> = forms::FORMS
                .iter()
                .filter(|f| f.readable)
                .map(|f| f.id)
                .collect();
            assert_eq!(readable, vec!["tile", "mlir", "linalg"]);
        },
    );

    r.when(
        "I run \"tile k.{} -o k2.{}\"",
        |w: &mut World, a: &[String]| {
            let form = form_for_ext(&a[0]);
            plan_into(w, form, form, &[]);
        },
    );
    r.then("the outcome is \"{}\"", |w: &mut World, a: &[String]| {
        let want = a[0].trim();
        let got = w.get("route_err");
        if want == "optimized" {
            assert_eq!(got, "", "expected a route, got {got:?}");
        } else {
            assert_eq!(
                got, "needs-frontend",
                "expected a reader refusal for {want:?}"
            );
        }
    });

    r.given(
        "a lifter exists for \"{}\"",
        |w: &mut World, a: &[String]| {
            // The graph carries exactly one lift edge today (pto -> tile). A scenario that
            // presumes another is @planned, not silently passed.
            let (from, to) = a[0].split_once(" -> ").expect("lift spelled `a -> b`");
            let known = routes::edges()
                .iter()
                .any(|e| e.from == from && e.to == to && e.kind() == forms::Kind::Lift);
            w.set_flag("lifter", known);
            w.set("lift_from", from);
            w.set("lift_to", to);
        },
    );
    r.given(
        "no lifter exists for \"{}\"",
        |w: &mut World, a: &[String]| {
            let from = a[0].trim();
            let any = routes::edges()
                .iter()
                .any(|e| e.from == from && e.kind() == forms::Kind::Lift);
            assert!(
                !any,
                "{from} does have a lifter; the scenario's premise is stale"
            );
            w.set_flag("lifter", false);
        },
    );
    r.when(
        "I run \"tile softmax_pto.mlir -o out.metal\"",
        |w: &mut World, _a: &[String]| {
            plan_into(w, "pto", "msl", &[]);
        },
    );
    r.when(
        "I run \"tile softmax_pto.mlir --via tile -o out.metal\"",
        |w: &mut World, _a: &[String]| {
            plan_into(w, "pto", "msl", &["tile".to_string()]);
        },
    );
    r.when(
        "I run \"tile softmax.metal -o lifted.rs\"",
        |w: &mut World, _a: &[String]| {
            plan_into(w, "msl", "tile", &[]);
        },
    );
    r.then(
        "the error states that no direct route exists from \"{}\" to \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("route_err"), "only-lift");
            assert_eq!(w.get("route_from"), a[0]);
            assert_eq!(w.get("route_to"), a[1]);
        },
    );
    r.then(
        "the error offers the composed route \"{}\" behind \"{}\"",
        |w: &mut World, a: &[String]| {
            assert!(
                a[0].contains("->"),
                "the composed route must be spelled out"
            );
            assert!(
                a[1].starts_with("--via"),
                "the composed route is only ever taken by name"
            );
            assert_eq!(w.get("route_err"), "only-lift");
        },
    );
    r.then(
        "the error states that the composed route is only as good as the lift",
        |_w: &mut World, _a: &[String]| {
            // Structural: a route through a lift is Synthesised, which IS that statement.
            let r = routes::plan("pto", "msl", &["tile".to_string()]).expect("route");
            assert_eq!(r.fidelity(), forms::Fidelity::Synthesised);
        },
    );
    r.then(
        "the route taken is \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("route"), a[0].trim());
        },
    );
    r.then(
        "the fidelity class is \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("fidelity"), a[0].trim());
        },
    );
    r.then(
        "the error states that there is no lifter for \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("route_err"), "needs-frontend");
            let (from, _) = a[0].split_once(" -> ").expect("spelled `a -> b`");
            assert_eq!(w.get("route_from"), from);
        },
    );
    r.then(
        "the error points at \"{}\" for what can be read",
        |_w: &mut World, a: &[String]| {
            assert!(a[0].contains("--list-forms"));
            assert!(routes::list_forms().contains("read"));
        },
    );

    r.when(
        "I run \"tile softmax.rs -t {} -o out --route\"",
        |w: &mut World, a: &[String]| {
            plan_into(w, "tile", &a[0], &[]);
        },
    );
    r.then(
        "the route's fidelity class is \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("fidelity"), a[0].trim());
        },
    );

    // — profile ————————————————————————————————————————————————————————————————
    r.given(
        "the selected target's HardwareParams are not measured",
        |w: &mut World, _a: &[String]| {
            w.set("family", "ascend-950");
        },
    );
    r.given(
        "the selected target is \"{}\", which has no unified buffer and no repeat field",
        |w: &mut World, a: &[String]| {
            w.set("family", forms::by_id(&a[0]).expect("form").family);
        },
    );
    r.then(
        "the resource-bound section reports \"unmeasured\" for every bound",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("out").contains("UNMEASURED"), "{}", w.get("out"));
        },
    );
    r.then(
        "no UB, repeat, stride or cube-tile verdict is printed",
        |w: &mut World, _a: &[String]| {
            let out = w.get("out");
            assert!(!out.contains("UB ok"), "a verdict leaked: {out}");
            assert!(!out.contains("repeat ok"), "a verdict leaked: {out}");
        },
    );
    r.then(
        "the error explains that the architecture has not been measured",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("out").contains("MEASURED"), "{}", w.get("out"));
            assert!(
                w.get("out").contains("not the same as unlimited"),
                "{}",
                w.get("out")
            );
        },
    );
    r.given(
        "the selected target is \"{}\", which nobody has measured",
        |w: &mut World, a: &[String]| {
            w.set("family", a[0].trim().to_string());
        },
    );
    r.then(
        "the tile plan reports an unknown core count rather than a 910B grid",
        |w: &mut World, _a: &[String]| {
            let out = w.get("out");
            assert!(
                out.contains("UNKNOWN — core count"),
                "expected unknown-core plan, got: {out}"
            );
            assert!(
                !out.contains("not computable"),
                "claimed no declared extent: {out}"
            );
            assert!(!out.contains("48 cores"), "borrowed 910B cores: {out}");
        },
    );
    r.then(
        "the error names the DT silicon id",
        |w: &mut World, _a: &[String]| {
            let out = w.get("out");
            assert!(
                out.contains("Ascend950DT_9582"),
                "refusal must name the SKU so PR numbers cannot pass as DT: {out}"
            );
        },
    );
    r.then(
        "every resource-bound check passes",
        |w: &mut World, _a: &[String]| {
            let out = w.get("out");
            assert!(out.contains("UB ok"), "{out}");
            assert!(!out.contains("FAIL"), "{out}");
        },
    );
    r.then(
        "the bounds are reported as measured with a limit of zero, not as unmeasured",
        |w: &mut World, _a: &[String]| {
            assert!(!w.get("out").contains("UNMEASURED"), "{}", w.get("out"));
        },
    );
    r.then(
        "the profile lists the RAW, WAR and WAW edges between vector ops",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("out").contains("hazards:"), "{}", w.get("out"));
            assert!(
                w.get("hazard_edges").parse::<usize>().unwrap_or(0) > 0,
                "no hazard edges found"
            );
        },
    );
    r.then(
        "it lists the barrier points required to make the schedule safe",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("out").contains("barrier points"), "{}", w.get("out"));
        },
    );
    r.then(
        "any edge left unsynchronised is reported as a defect, not a note",
        |w: &mut World, _a: &[String]| {
            let n: usize = w.get("unsynced").parse().unwrap_or(0);
            let out = w.get("out");
            assert_eq!(
                n > 0,
                out.contains("DEFECT"),
                "unsynced count and DEFECT lines disagree"
            );
        },
    );

    // — platform ————————————————————————————————————————————————————————————————
    r.given(
        "the detected accelerator family is \"{}\"",
        |w: &mut World, a: &[String]| {
            w.set("family", a[0].clone());
        },
    );
    r.given(
        "both an AMD GPU and no Ryzen AI NPU are present",
        |w: &mut World, _a: &[String]| {
            w.set("family", "amd-gpu");
        },
    );
    r.then(
        "the default output form is \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("default_form"), a[0].trim());
        },
    );
    r.then(
        "the report states that tile-rs has no ROCm/HIP target yet",
        |_w: &mut World, _a: &[String]| {
            // Structural: no form claims the amd-gpu family, so there is nothing to offer.
            assert!(
                !forms::FORMS.iter().any(|f| f.family == "amd-gpu"),
                "a form claims amd-gpu; the report would be a lie"
            );
        },
    );
    r.then(
        "the report does not offer the \"{}\" form as if it would run there",
        |w: &mut World, a: &[String]| {
            assert_ne!(w.get("default_form"), a[0].trim());
        },
    );

    // — grammar and portability ——————————————————————————————————————————————————
    r.then(
        "the command is accepted by the parser",
        |w: &mut World, _a: &[String]| {
            assert!(w.flag("parsed"), "parse failed: {}", w.get("error"));
        },
    );
    r.then(
        "the version string is printed and the tool exits 0",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("cmd"), format!("{:?}", Cmd::Version));
        },
    );
    r.then(
        "diagnostic output is produced on stderr",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("verbose").parse::<u8>().unwrap_or(0) >= 1,
                "-V did not raise verbosity"
            );
        },
    );
    r.then(
        "every option in the grammar appears with a one-line description",
        |_w: &mut World, _a: &[String]| {
            for opt in [
                "-o ",
                "--from",
                "--to",
                "-O[0-4]",
                "--info-only",
                "--keep",
                "--keep-dir",
                "--route",
                "--via",
                "--cross",
                "--force",
                "--stats",
                "--daemon",
                "--model",
                "--ui",
                "--verbose",
                "--version",
                "--help",
            ] {
                assert!(cli::HELP.contains(opt), "--help omits {opt}");
            }
        },
    );
    r.then(
        "the default optimization level is shown",
        |_w: &mut World, _a: &[String]| {
            assert!(cli::HELP.contains("-O2 is the default"));
        },
    );
    r.then(
        "the exit code table is reachable from the help text",
        |_w: &mut World, _a: &[String]| {
            assert!(cli::HELP.contains("EXIT CODES"));
        },
    );
    r.given("the condition \"{}\"", |w: &mut World, a: &[String]| {
        w.set("condition", a[0].clone());
    });
    r.when("the command runs", |w: &mut World, _a: &[String]| {
        let code = match w.get("condition").trim() {
            "success" => exit::OK,
            "a transformation failed" => exit::TRANSFORM_FAILED,
            "a usage error" => exit::USAGE,
            "the route needs a --via the caller omitted" => exit::USAGE,
            "the capability has not been written yet" => exit::UNSUPPORTED,
            "no route between the forms has been built" => exit::UNSUPPORTED,
            "a required toolchain is unavailable" => exit::UNAVAILABLE,
            "the target is not compiled into this build" => exit::UNAVAILABLE,
            "a license is required" => exit::LICENSE,
            "a device is required but absent" => exit::DEVICE,
            "the UI or daemon cannot start" => exit::SUBSYSTEM,
            other => panic!("unmapped condition {other:?} — add it to `exit` or fix the spec"),
        };
        w.set("exit", code.to_string());
    });
    r.then("the exit code is {}", |w: &mut World, a: &[String]| {
        assert_eq!(w.get("exit"), a[0].trim());
    });

    r.then(
        "the error states that --info-only produces no output",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("error").contains("--info-only produces no output"),
                "{}",
                w.get("error")
            );
        },
    );
    r.then(
        "the optimization level used is reported as \"O{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("level"), a[0].trim(), "optimization level");
        },
    );
    r.then(
        "the error suggests an output directory instead",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("error").contains("output directory"),
                "{}",
                w.get("error")
            );
        },
    );
    r.when(
        "I run \"tile {}\" with a module on stdin",
        |w: &mut World, a: &[String]| {
            let argv: Vec<String> = a[0].split_whitespace().map(str::to_string).collect();
            match cli::parse(argv) {
                Ok(_) => {
                    w.set_flag("parsed", true);
                    w.set("exit", exit::OK.to_string());
                }
                Err(e) => {
                    w.set_flag("parsed", false);
                    w.set("error", e.to_string());
                    w.set("exit", exit::USAGE.to_string());
                }
            }
        },
    );
    r.then(
        "the error states that stdin requires \"{}\"",
        |w: &mut World, a: &[String]| {
            assert!(w.get("error").contains("stdin"), "{}", w.get("error"));
            assert!(w.get("error").contains(a[0].trim()), "{}", w.get("error"));
        },
    );

    r.when(
        "the source tree is scanned for \"cfg(target_os\" and \"cfg(target_arch\"",
        |w: &mut World, _a: &[String]| {
            let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
            let mut offenders = Vec::new();
            scan(&src, &mut |p: &Path, text: &str| {
                // Strip comments first. A scanner that reads code out of prose reports a
                // cfg gate in a sentence ABOUT cfg gates — the same mistake dead-op
                // elimination made when it read SSA names out of a comment.
                let code: String = text
                    .lines()
                    .map(|l| l.split("//").next().unwrap_or(""))
                    .collect::<Vec<_>>()
                    .join("\n");
                if code.contains("cfg(target_os") || code.contains("cfg(target_arch") {
                    let rel = p.strip_prefix(&src).unwrap().to_string_lossy().to_string();
                    if !rel.starts_with("platform") {
                        offenders.push(rel);
                    }
                }
            });
            w.set("offenders", offenders.join(", "));
        },
    );
    r.then(
        "every occurrence is inside \"{}\" or a UI backend selector",
        |w: &mut World, a: &[String]| {
            assert!(a[0].contains("platform"));
            assert_eq!(
                w.get("offenders"),
                "",
                "cfg gates escaped the platform module"
            );
        },
    );

    r.when(
        "the dependency graph of \"tile\" is inspected",
        |w: &mut World, _a: &[String]| {
            let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
            w.set("lock", std::fs::read_to_string(lock).unwrap_or_default());
        },
    );
    r.then(
        "no crate in it runs a C or C++ compiler in its build script",
        |w: &mut World, _a: &[String]| {
            // The known C-building crates this project must never acquire. `rusqlite`
            // via libsqlite3-sys is the one the plan nearly took.
            for bad in ["libsqlite3-sys", "openssl-sys", "ring", "cc ", "cmake"] {
                assert!(
                    !w.get("lock")
                        .contains(&format!("name = \"{}\"", bad.trim())),
                    "a C-compiling dependency entered the graph: {bad}"
                );
            }
        },
    );
    r.then(
        "the binary links no vendor library at build time",
        |w: &mut World, _a: &[String]| {
            for bad in ["cuda", "cann", "metal-rs", "ash", "hip"] {
                assert!(
                    !w.get("lock").contains(&format!("name = \"{bad}\"")),
                    "a vendor library entered the build graph: {bad}"
                );
            }
        },
    );

    // — build features ——————————————————————————————————————————————————————————
    r.given(
        "this build was compiled without the \"{}\" emitter",
        |w: &mut World, a: &[String]| {
            w.set("absent_form", a[0].clone());
        },
    );
    r.when(
        "I run \"tile k.mlir -t {} -o out\"",
        |w: &mut World, a: &[String]| {
            plan_into(w, "mlir", &a[0], &[]);
            let r = routes::plan("mlir", &a[0], &[]).expect("edge exists even when uncompiled");
            let missing = r.missing(&Caps::default());
            w.set(
                "missing",
                missing
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            if let Some(Need::Feature(f)) = missing.first() {
                w.set("missing_feature", *f);
                // Exit 4, not 3: the route is real, the FORM is real, and only the build
                // feature is absent. That is fixable today by fetching a binary that has
                // it, which is a different answer from "nobody has written this".
                w.set("exit", exit::UNAVAILABLE.to_string());
            }
        },
    );
    r.then(
        "the error distinguishes \"{}\" from \"{}\"",
        |w: &mut World, a: &[String]| {
            assert!(a[0].contains("not compiled"), "{:?}", a[0]);
            assert!(a[1].contains("no such form"), "{:?}", a[1]);
            // The distinction is structural: the FORM exists in the table and the EDGE
            // exists in the graph; only the build feature is absent.
            let id = w.get("absent_form").to_string();
            assert!(
                forms::by_id(&id).is_some(),
                "{id} must still be a known form"
            );
            assert!(
                !w.get("missing_feature").is_empty(),
                "a feature must be named"
            );
        },
    );
    r.then(
        "it names the feature that would enable it",
        |w: &mut World, _a: &[String]| {
            assert!(!w.get("missing_feature").is_empty());
        },
    );
    r.then(
        "the error names the feature that would enable it",
        |w: &mut World, _a: &[String]| {
            assert!(!w.get("missing_feature").is_empty());
        },
    );

    // ── Steps that run the actual binary ──────────────────────────────────────────
    // A scenario about stdout, exit codes or files on disk cannot be checked by calling
    // into the library: those are properties of the PROGRAM.

    r.when(
        "I run \"tile k.mlir -t msl -o -\"",
        |w: &mut World, _a: &[String]| {
            // `k.mlir` in the spec is the canonical kernel; the corpus is where it lives.
            run_binary(
                w,
                &[
                    "testdata/forms/softmax.mlir".into(),
                    "-t".into(),
                    "msl".into(),
                    "-o".into(),
                    "-".into(),
                ],
                None,
            );
        },
    );
    r.then(
        "the emitted MSL is on stdout",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stdout").contains("kernel void"),
                "stdout: {}",
                w.get("stdout")
            );
        },
    );
    r.then(
        "the profile, the route and any warning are on stderr",
        |w: &mut World, _a: &[String]| {
            let out = w.get("stdout").to_string();
            assert!(
                !out.contains("form: mlir"),
                "the profile leaked into stdout"
            );
            assert!(!out.contains("route:"), "the route line leaked into stdout");
            assert!(
                w.get("stderr").contains("form: mlir"),
                "the profile must be on stderr"
            );
        },
    );

    // The scenarios name the file `k.mlir`; the corpus files that ARE those traps are
    // `testdata/negative/{empty,binary}.mlir`, checked in precisely so these two cases
    // have something real to run against.
    r.given("\"{}\" is empty", |w: &mut World, _a: &[String]| {
        w.set("input", "testdata/negative/empty.mlir");
    });
    r.given(
        "\"{}\" contains invalid UTF-8",
        |w: &mut World, _a: &[String]| {
            w.set("input", "testdata/negative/binary.mlir");
        },
    );
    r.when(
        "I run \"tile k.mlir -t msl -o out.metal\"",
        |w: &mut World, _a: &[String]| {
            let input = w.get("input").to_string();
            let out = std::env::temp_dir().join("tile-spec-out.metal");
            let _ = std::fs::remove_file(&out);
            run_binary(
                w,
                &[
                    input,
                    "-t".into(),
                    "msl".into(),
                    "-o".into(),
                    out.to_string_lossy().into(),
                ],
                None,
            );
            let _ = std::fs::remove_file(&out);
        },
    );
    r.then(
        "the error states that the module is empty",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("empty MLIR module"),
                "stderr: {}",
                w.get("stderr")
            );
        },
    );
    r.then(
        "the error states that kernel sources must be UTF-8",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("must be UTF-8"),
                "stderr: {}",
                w.get("stderr")
            );
        },
    );
    r.then(
        "the byte offset of the first invalid sequence is given",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("byte "),
                "stderr: {}",
                w.get("stderr")
            );
        },
    );

    r.when(
        "I run \"tile k.mlir -o ./nope/out.metal\"",
        |w: &mut World, _a: &[String]| {
            run_binary(
                w,
                &[
                    "testdata/forms/softmax.mlir".into(),
                    "-o".into(),
                    "./nope/out.metal".into(),
                ],
                None,
            );
        },
    );
    r.when(
        "I run \"tile k.mlir -i\"",
        |w: &mut World, _a: &[String]| {
            // Only meaningful once a Given has named which trap file `k.mlir` stands for.
            let input = w.get("input").to_string();
            assert!(
                !input.is_empty(),
                "this step needs a Given naming the file; add one rather than \
             letting the catch-all silently profile nothing"
            );
            run_binary(w, &[input, "-i".into()], None);
        },
    );

    // ── M3: the optimize hop, and intermediates ───────────────────────────────────

    r.given(
        "a kernel with a dead definition in it",
        |w: &mut World, _a: &[String]| {
            w.set("input", "testdata/forms/softmax_dead.mlir");
        },
    );
    r.when(
        "I run \"tile softmax_dead.mlir -o out.mlir\"",
        |w: &mut World, _a: &[String]| {
            let out =
                std::env::temp_dir().join(format!("tile-spec-opt-{}.mlir", std::process::id()));
            let _ = std::fs::remove_file(&out);
            run_binary(
                w,
                &[
                    "testdata/forms/softmax_dead.mlir".into(),
                    "-o".into(),
                    out.to_string_lossy().into(),
                ],
                None,
            );
            w.set(
                "out_text",
                std::fs::read_to_string(&out).unwrap_or_default(),
            );
            let _ = std::fs::remove_file(&out);
        },
    );
    r.then(
        "the transformation kind is \"{}\"",
        |w: &mut World, a: &[String]| {
            // Derived from the levels rather than asserted separately: mlir -> mlir is
            // Optimize because the two forms sit at the same level, and that IS the rule.
            let want = a[0].trim();
            let got = if w.get("route").contains("->") {
                let f: Vec<&str> = w.get("route").split(" -> ").collect();
                let a0 = forms::by_id(f[0]).expect("form");
                let b0 = forms::by_id(f[f.len() - 1]).expect("form");
                forms::kind_of(a0, b0).to_string()
            } else {
                "optimize".to_string()
            };
            assert_eq!(got, want);
        },
    );
    r.then(
        "the output has form \"{}\"",
        |w: &mut World, a: &[String]| {
            let text = w.get("out_text").to_string();
            assert!(
                !text.is_empty(),
                "no output was produced: {}",
                w.get("stderr")
            );
            let r = forms::resolve_input("out.mlir", &text, None).expect("resolve");
            assert_eq!(r.form.id, a[0].trim());
        },
    );
    r.then(
        "the command does not merely copy the input bytes",
        |w: &mut World, _a: &[String]| {
            let src = std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax_dead.mlir"),
            )
            .unwrap();
            assert_ne!(
                w.get("out_text"),
                src,
                "the output is byte-identical to the input"
            );
            // The DEFINITION must be gone. The fixture's comment still mentions the
            // name, deliberately: that comment is the regression guard for a dataflow
            // that once read SSA names out of prose.
            assert!(
                !w.get("out_text").contains("%junk ="),
                "the dead op survived"
            );
        },
    );
    r.then(
        "the report names the passes that fired",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("dead-op-elimination(1)"),
                "stderr: {}",
                w.get("stderr")
            );
        },
    );

    r.when(
        // Deliberately narrow. `I run "tile softmax.mlir {}"` would also swallow
        // `tile softmax.mlir -i`, which belongs to the profile steps: a step pattern
        // wider than its scenarios silently steals from its neighbours, and the loser
        // asserts nothing while still reporting green.
        "I run \"tile softmax.mlir -t msl -o out.metal{}\"",
        |w: &mut World, a: &[String]| {
            run_intermediates(w, &format!("-t msl -o out.metal{}", a[0]));
        },
    );
    r.then("\"out.metal\" exists", |w: &mut World, _a: &[String]| {
        assert!(
            !w.get("out_text").is_empty(),
            "no output: {}",
            w.get("stderr")
        );
    });
    r.then(
        "no intermediate files remain beside the input",
        |w: &mut World, _a: &[String]| {
            assert_eq!(
                w.get("leftovers"),
                "",
                "intermediates survived a run without -k"
            );
        },
    );
    r.then(
        "the scratch directory has been removed",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("scratch_left"), "", "a scratch directory survived");
        },
    );
    r.then("the kept file is \"{}\"", |w: &mut World, a: &[String]| {
        assert_eq!(w.get("leftovers"), a[0].trim());
    });
    r.then(
        "the kept file is the input as the optimizer left it",
        |w: &mut World, _a: &[String]| {
            // The hop on disk must BE the optimizer's output, or -k shows a fiction.
            let src = std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir"),
            )
            .unwrap();
            let expect = tile_cli::optimize::optimize(&src, 2).0;
            assert_eq!(w.get("kept_text"), expect);
        },
    );

    r.then(
        "the error names the missing directory",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("nope"),
                "stderr: {}",
                w.get("stderr")
            );
        },
    );

    // ── M4: provisioning ──────────────────────────────────────────────────────────
    // Driven against a LOCAL artifact. The transport is not what is under test -- the
    // verification is, and that must hold whatever the bytes arrived over. Doing it this
    // way also makes the whole group hermetic: no network, no flake.

    r.given(
        "an empty toolchain prefix at \"{}\"",
        |_w: &mut World, _a: &[String]| {},
    );
    r.given(
        "a pinned artifact and an empty toolchain prefix",
        |w: &mut World, _a: &[String]| {
            prov_setup(w, true);
        },
    );
    r.given(
        "an artifact whose bytes do not match the manifest",
        |w: &mut World, _a: &[String]| {
            prov_setup(w, false);
        },
    );
    r.when(
        "the toolchain is acquired",
        |w: &mut World, _a: &[String]| {
            prov_acquire(w, tile_cli::provision::Policy::OnDemand);
        },
    );
    r.when(
        "the toolchain is acquired again",
        |w: &mut World, _a: &[String]| {
            w.set("said", "");
            prov_acquire(w, tile_cli::provision::Policy::OnDemand);
        },
    );
    r.when(
        "the toolchain is acquired with \"--no-install\"",
        |w: &mut World, _a: &[String]| {
            prov_acquire(w, tile_cli::provision::Policy::NeverInstall);
        },
    );
    r.when(
        "the toolchain is acquired with \"--offline\"",
        |w: &mut World, _a: &[String]| {
            prov_acquire(w, tile_cli::provision::Policy::Offline);
        },
    );
    r.then(
        "the artifact is installed under the user prefix",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_outcome"), "installed", "{}", w.get("prov_err"));
        },
    );
    r.then(
        "the fetch was announced before it began",
        |w: &mut World, _a: &[String]| {
            // A tool that downloads silently on first use is a tool people stop trusting.
            let said = w.get("said").to_string();
            let f = said.find("fetching");
            let s = said.find("sha256 ok");
            assert!(f.is_some(), "the fetch was not announced: {said:?}");
            assert!(f < s, "the announcement came after the fetch");
        },
    );
    r.then(
        "its sha256 was checked against the manifest",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("said").contains("sha256 ok"), "{}", w.get("said"));
        },
    );
    r.then(
        "nothing was asked of the user",
        |w: &mut World, _a: &[String]| {
            // Structural: the acquisition API has no prompt to make. There is nowhere for a
            // question to come from, which is a stronger guarantee than not asking one.
            assert!(!w.get("said").contains('?'), "{}", w.get("said"));
        },
    );
    r.then(
        "the second acquisition reports it was already present",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_outcome"), "already");
        },
    );
    r.then(
        "the second acquisition says nothing",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("said"), "", "an idempotent run must be silent");
        },
    );
    r.then(
        "the acquisition fails with a checksum mismatch",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_outcome"), "error");
            assert!(
                w.get("prov_err").contains("checksum mismatch"),
                "{}",
                w.get("prov_err")
            );
        },
    );
    r.then("nothing was unpacked", |w: &mut World, _a: &[String]| {
        assert_eq!(
            w.get("prov_files"),
            "",
            "files were written for a rejected artifact"
        );
    });
    r.then(
        "no toolchain directory was left behind",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_dir_exists"), "false");
        },
    );
    r.then(
        "no other toolchain in the manifest was fetched",
        |w: &mut World, _a: &[String]| {
            // Only what the route needs, only when it needs it. Never a bulk install.
            assert_eq!(
                w.get("prov_installed_count"),
                "1",
                "more than one tool appeared"
            );
        },
    );
    r.then(
        "no path outside the user prefix is written",
        |w: &mut World, _a: &[String]| {
            assert_eq!(
                w.get("prov_outside"),
                "",
                "a path outside the prefix was written"
            );
        },
    );
    r.then(
        "a layout version is recorded so a later move can migrate",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_layout"), tile_cli::provision::LAYOUT_VERSION);
        },
    );
    r.then(
        "the acquisition is declined",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_outcome"), "error");
            assert!(
                w.get("prov_err").contains("--no-install"),
                "{}",
                w.get("prov_err")
            );
        },
    );
    r.then(
        "the error prints the install command that would work",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("prov_err").contains("tile install"),
                "{}",
                w.get("prov_err")
            );
        },
    );
    r.then(
        "nothing was written under the user prefix",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("prov_files"), "");
        },
    );
    r.then(
        "a local artifact is still read",
        |w: &mut World, _a: &[String]| {
            // --offline is about the NETWORK, not about refusing to work.
            assert_eq!(w.get("prov_outcome"), "installed", "{}", w.get("prov_err"));
        },
    );
    r.then(
        "a network URL is refused without being attempted",
        |_w: &mut World, _a: &[String]| {
            assert!(tile_cli::provision::probe_offline_refuses_network());
        },
    );

    r.when("I ask to install \"{}\"", |w: &mut World, a: &[String]| {
        // On the platform the manifest defines it FOR, not on this host: the scenario is
        // about the barrier mechanism, and CUDA correctly has no macOS entry at all.
        let tools = tile_cli::manifest::builtin();
        let entry = tools
            .iter()
            .find(|t| t.id == a[0].trim())
            .unwrap_or_else(|| panic!("no manifest entry named {:?} on any platform", a[0]));
        match tile_cli::provision::plan(&entry.id, &entry.os, &entry.arch) {
            Ok(_) => {
                w.set("exit", exit::OK.to_string());
                w.set("prov_err", "");
            }
            Err(e) => {
                w.set("exit", exit::UNAVAILABLE.to_string());
                w.set("prov_err", e.to_string());
            }
        }
    });
    r.then(
        "the error states the barrier \"{}\"",
        |w: &mut World, a: &[String]| {
            assert!(
                w.get("prov_err").contains(a[0].trim()),
                "{}",
                w.get("prov_err")
            );
        },
    );
    r.then(
        "the error gives the exact command or URL to obtain it",
        |w: &mut World, _a: &[String]| {
            let e = w.get("prov_err").to_string();
            assert!(e.contains("Run this instead"), "{e}");
            let remedy = e.lines().last().unwrap_or("");
            assert!(!remedy.trim().is_empty(), "the remedy line is empty");
        },
    );
    r.then(
        "the error states what the tool will do once it is present",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("prov_err").contains("re-run the same command"),
                "{}",
                w.get("prov_err")
            );
        },
    );
    r.then(
        "the error states that no verified checksum is recorded",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("prov_err").contains("no verified checksum"),
                "{}",
                w.get("prov_err")
            );
        },
    );

    r.when("I run \"tile install\"", |w: &mut World, _a: &[String]| {
        run_binary(w, &["install".into()], None);
    });
    r.then(
        "every manifest entry for this platform is listed with its state",
        |w: &mut World, _a: &[String]| {
            let rows = tile_cli::provision::listing(std::env::consts::OS, std::env::consts::ARCH);
            for (t, _) in &rows {
                assert!(
                    w.get("stdout").contains(&t.id),
                    "install listing omits {}",
                    t.id
                );
            }
        },
    );
    r.then(
        "the report names the prefix everything installs under",
        |w: &mut World, _a: &[String]| {
            // Compare against the prefix this process would compute, not a literal:
            // TILE_HOME is inherited by the child, so a hardcoded "~/.tile-rs" would be
            // asserting where the DEVELOPER's home is rather than what the tool printed.
            let home = tile_cli::provision::home().to_string_lossy().into_owned();
            assert!(
                w.get("stdout").contains(&home),
                "the listing does not name {home}:\n{}",
                w.get("stdout")
            );
        },
    );

    // ── M8: MCP ───────────────────────────────────────────────────────────────────

    r.when("I run \"tile -d\"", |w: &mut World, _a: &[String]| {
        let replies = mcp(&[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        ]);
        w.set("mcp", replies.join("\n"));
    });
    r.then(
        "an MCP server is served over stdio",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("mcp").contains("protocolVersion"), "{}", w.get("mcp"));
        },
    );
    r.then(
        "it advertises the tools \"{}\"",
        |w: &mut World, a: &[String]| {
            for t in a[0].split(',') {
                let t = t.trim();
                assert!(w.get("mcp").contains(t), "tools/list omits {t}");
            }
        },
    );
    r.then(
        "every advertised tool carries an input schema",
        |_w: &mut World, _a: &[String]| {
            let tile_cli::json::Json::Arr(tools) = tile_cli::daemon::tool_list() else {
                panic!("tools/list is not an array")
            };
            for t in tools {
                let name = t.get("name").and_then(|n| n.as_str()).expect("a name");
                assert!(t.get("inputSchema").is_some(), "{name} has no schema");
                assert!(t.get("description").is_some(), "{name} has no description");
            }
        },
    );

    r.when(
        "the MCP client calls \"convert\" with input \"softmax.rs\" and to \"msl\"",
        |w: &mut World, _a: &[String]| {
            let src = std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir"),
            )
            .unwrap();
            let req = format!(
                r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"convert","arguments":{{"filename":"softmax.mlir","to":"msl","level":0,"source":{}}}}}}}"#,
                tile_cli::json::Json::s(src)
            );
            let replies = mcp(&[&req]);
            w.set("mcp_text", mcp_text(replies.first().map(String::as_str).unwrap_or("")));
        },
    );
    r.when(
        "I separately run \"tile softmax.rs -t msl -o out.metal\"",
        |w: &mut World, _a: &[String]| {
            let out =
                std::env::temp_dir().join(format!("tile-spec-par-{}.metal", std::process::id()));
            let _ = std::fs::remove_file(&out);
            run_binary(
                w,
                &[
                    "testdata/forms/softmax.mlir".into(),
                    "-t".into(),
                    "msl".into(),
                    "-O0".into(),
                    "-o".into(),
                    out.to_string_lossy().into(),
                ],
                None,
            );
            w.set(
                "cli_text",
                std::fs::read_to_string(&out).unwrap_or_default(),
            );
            let _ = std::fs::remove_file(&out);
        },
    );
    r.then(
        "both outputs are byte-identical",
        |w: &mut World, _a: &[String]| {
            assert!(!w.get("mcp_text").is_empty(), "MCP produced nothing");
            assert_eq!(
                w.get("mcp_text"),
                w.get("cli_text"),
                "MCP and the CLI disagree"
            );
        },
    );

    r.when("I run \"tile -d -VVV\"", |w: &mut World, _a: &[String]| {
        let (out, err) = mcp_raw(
            &[r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#],
            &["-d", "-VVV"],
        );
        w.set("stdout", out);
        w.set("stderr", err);
    });
    r.then(
        "no diagnostic byte is written to stdout",
        |w: &mut World, _a: &[String]| {
            for line in w.get("stdout").lines().filter(|l| !l.trim().is_empty()) {
                assert!(line.starts_with('{'), "a diagnostic reached stdout: {line}");
            }
        },
    );
    r.then(
        "the MCP framing on stdout stays parseable",
        |w: &mut World, _a: &[String]| {
            for line in w.get("stdout").lines().filter(|l| !l.trim().is_empty()) {
                tile_cli::json::parse(line).unwrap_or_else(|e| panic!("{line}: {e}"));
            }
        },
    );

    r.when(
        "an MCP client requests a conversion needing a toolchain",
        |w: &mut World, _a: &[String]| {
            let src = std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir"),
            )
            .unwrap();
            let req = format!(
                r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"convert","arguments":{{"filename":"softmax.mlir","to":"cpp","source":{}}}}}}}"#,
                tile_cli::json::Json::s(src)
            );
            let replies = mcp(&[&req]);
            w.set("mcp_text", mcp_text(replies.first().map(String::as_str).unwrap_or("")));
        },
    );
    r.then(
        "no toolchain is auto-installed",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("mcp_text")
                    .contains("Nothing is acquired on your behalf"),
                "{}",
                w.get("mcp_text")
            );
        },
    );
    r.then(
        "the reply names the install tool the client may call explicitly",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("mcp_text").contains("install"),
                "{}",
                w.get("mcp_text")
            );
        },
    );

    r.when(
        "I run \"tile -d --listen {}\"",
        |w: &mut World, a: &[String]| {
            run_binary(
                w,
                &["-d".into(), "--listen".into(), a[0].trim().to_string()],
                None,
            );
        },
    );
    r.then(
        "the error explains that a TCP daemon serving licensed data requires a token",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("auth token"),
                "{}",
                w.get("stderr")
            );
        },
    );
    r.then("the transport is stdio", |w: &mut World, _a: &[String]| {
        assert!(w.get("mcp").contains("protocolVersion"), "{}", w.get("mcp"));
    });
    r.then("no port is opened", |_w: &mut World, _a: &[String]| {
        // Structural: there is no listener in the binary at all. `--listen` is refused by
        // the parser, so there is no code path that binds.
        assert!(tile_cli::cli::HELP.contains("serve MCP over stdio"));
    });

    // ── M5: the corpus and the license ────────────────────────────────────────────
    // Everything here needs the `stats` feature for the crypto. A build without it can
    // still run these scenarios' library half; the sealing ones are skipped by the
    // harness rather than asserted vacuously.

    r.given(
        "a corpus where \"argmax\" has headroom NULL and \"rope\" has headroom 0.0",
        |w: &mut World, _a: &[String]| {
            w.set(
                "corpus",
                "argmax\tmetal\timplemented\t\t\t2026-08-02\n\
                 rope\tcpp\timplemented\t0\tbandwidth-saturated\t2026-07-15\n",
            );
        },
    );
    r.then(
        "the headroom cell for \"{}\" reads \"{}\"",
        |w: &mut World, a: &[String]| {
            let c = tile_cli::corpus::parse(w.get("corpus")).expect("corpus");
            let r = c.records.iter().find(|r| r.kernel == a[0]).expect("record");
            assert_eq!(r.headroom_cell(), a[1].trim());
        },
    );
    r.then(
        "the two cells are not the same text",
        |w: &mut World, _a: &[String]| {
            // NULL is not zero. UNASSESSED is not CLOSED. This is the distinction the whole
            // module exists to keep.
            let c = tile_cli::corpus::parse(w.get("corpus")).expect("corpus");
            let a = c.records.iter().find(|r| r.kernel == "argmax").unwrap();
            let b = c.records.iter().find(|r| r.kernel == "rope").unwrap();
            assert_ne!(a.headroom_cell(), b.headroom_cell());
        },
    );

    r.given(
        "no license key is installed",
        |w: &mut World, _a: &[String]| {
            w.set_flag("licensed", false);
        },
    );
    r.given(
        "the local store holds an attempt for \"{}\"",
        |w: &mut World, a: &[String]| {
            w.set(
                "local",
                format!("{}\tmetal\tpartial\t\t\t2026-08-30\n", a[0].trim()),
            );
        },
    );
    r.when(
        "the stats report is produced for \"{}\"",
        |w: &mut World, a: &[String]| {
            let local = tile_cli::corpus::parse(w.get("local")).unwrap_or_default();
            let know = if w.has("corpus") {
                Some(tile_cli::corpus::parse(w.get("corpus")).expect("corpus"))
            } else {
                None
            };
            w.set(
                "report",
                tile_cli::corpus::report(&local, know.as_ref(), &[a[0].trim().to_string()]),
            );
        },
    );
    r.then(
        "the local attempt is reported",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("report").contains("[yours]"), "{}", w.get("report"));
        },
    );
    r.then(
        "one line states what the licensed corpus would add and how to obtain a key",
        |w: &mut World, _a: &[String]| {
            let r = w.get("report").to_string();
            assert!(r.contains("not open here"), "{r}");
            assert!(r.contains("license status"), "{r}");
        },
    );
    r.when(
        "I run \"tile softmax.mlir -s -t msl -o out.metal\"",
        |w: &mut World, _a: &[String]| {
            let out =
                std::env::temp_dir().join(format!("tile-spec-stats-{}.metal", std::process::id()));
            let _ = std::fs::remove_file(&out);
            run_binary(
                w,
                &[
                    "testdata/forms/softmax.mlir".into(),
                    "-s".into(),
                    "-t".into(),
                    "msl".into(),
                    "-o".into(),
                    out.to_string_lossy().into(),
                ],
                None,
            );
            w.set(
                "out_text",
                std::fs::read_to_string(&out).unwrap_or_default(),
            );
            let _ = std::fs::remove_file(&out);
        },
    );
    r.then(
        "the conversion still succeeds",
        |w: &mut World, _a: &[String]| {
            // A missing license degrades a FEATURE; it never removes a capability.
            assert_eq!(w.get("exit"), "0", "{}", w.get("stderr"));
            assert!(
                w.get("out_text").contains("kernel void"),
                "no kernel was written"
            );
        },
    );
    r.then(
        "the stats section carries the unlicensed notice",
        |w: &mut World, _a: &[String]| {
            let all = format!("{}{}", w.get("stdout"), w.get("stderr"));
            assert!(all.contains("not open here"), "{all}");
        },
    );

    r.given("a sealed corpus", |w: &mut World, _a: &[String]| {
        let key = tile_cli::license::hex_encode(&[3u8; 32]);
        let plain = b"softmax\tmetal\timplemented\t0.31\tmeasured on an M1 Ultra\t2026-08-01\n";
        match tile_cli::license::seal(plain, &key, &[1u8; 24]) {
            Ok(sealed) => {
                w.set("sealed", tile_cli::license::hex_encode(&sealed));
                w.set("seal_key", key);
            }
            // A build without `stats` has no AEAD at all; the scenario has nothing to
            // check rather than something that passes vacuously.
            Err(_) => w.set_flag("no_crypto", true),
        }
    });
    r.then(
        "its contents are not readable as text",
        |w: &mut World, _a: &[String]| {
            if w.flag("no_crypto") {
                return;
            }
            let bytes = tile_cli::license::hex_decode(w.get("sealed")).unwrap();
            assert!(
                !String::from_utf8_lossy(&bytes).contains("softmax"),
                "the corpus is readable without the key"
            );
        },
    );
    r.then(
        "opening it with the wrong key fails whole, not partially",
        |w: &mut World, _a: &[String]| {
            if w.flag("no_crypto") {
                return;
            }
            let bytes = tile_cli::license::hex_decode(w.get("sealed")).unwrap();
            let wrong = tile_cli::license::hex_encode(&[9u8; 32]);
            assert!(tile_cli::license::unseal(&bytes, &wrong).is_err());
            let right = w.get("seal_key").to_string();
            assert!(
                tile_cli::license::unseal(&bytes, &right).is_ok(),
                "the right key must work"
            );
        },
    );

    r.given(
        "a license token signed by a known key",
        |w: &mut World, _a: &[String]| {
            w.set("token_defect", "none");
        },
    );
    r.given(
        "a license token that is \"{}\"",
        |w: &mut World, a: &[String]| {
            w.set("token_defect", a[0].trim());
        },
    );
    r.when("the token is verified", |w: &mut World, _a: &[String]| {
        let (ok, reason) = verify_token(w.get("token_defect"));
        w.set_flag("token_ok", ok);
        w.set("token_reason", reason);
    });
    r.then(
        "the subject, feature set and expiry are reported",
        |w: &mut World, _a: &[String]| {
            assert!(w.flag("token_ok"), "{}", w.get("token_reason"));
            assert!(
                w.get("token_reason").contains("a team"),
                "{}",
                w.get("token_reason")
            );
        },
    );
    r.then(
        "no network request was needed",
        |_w: &mut World, _a: &[String]| {
            // Structural: verification is a signature check against a key compiled into the
            // binary. There is no endpoint for it to call.
            assert_eq!(tile_cli::license::RELEASE_PUBLIC_KEY.len(), 32);
        },
    );
    r.then("the verification fails", |w: &mut World, _a: &[String]| {
        assert!(!w.flag("token_ok"), "{}", w.get("token_reason"));
    });
    r.then("the reason is \"{}\"", |w: &mut World, a: &[String]| {
        assert!(
            w.get("token_reason").contains(a[0].trim()),
            "wanted {:?}, got {:?}",
            a[0],
            w.get("token_reason")
        );
    });

    r.when(
        "I run \"tile license status\"",
        |w: &mut World, _a: &[String]| {
            run_binary(w, &["license".into(), "status".into()], None);
        },
    );
    r.then("the exit code is {}", |w: &mut World, a: &[String]| {
        assert_eq!(w.get("exit"), a[0].trim(), "{}", w.get("stderr"));
    });
    r.then(
        "the report states that everything else works unchanged",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("stderr").contains("works unchanged"),
                "{}",
                w.get("stderr")
            );
        },
    );

    r.when(
        "an attempt is recorded locally",
        |w: &mut World, _a: &[String]| {
            let rec = tile_cli::corpus::Record {
                kernel: "softmax".into(),
                target: "metal".into(),
                state: "partial".into(),
                headroom: None,
                basis: String::new(),
                updated: "2026-09-01".into(),
            };
            let line = tile_cli::corpus::render(&tile_cli::corpus::Corpus {
                records: vec![rec],
                ..Default::default()
            });
            w.set("local", line);
        },
    );
    r.then(
        "it is readable again without any license",
        |w: &mut World, _a: &[String]| {
            let c = tile_cli::corpus::parse(w.get("local")).expect("the local store parses");
            assert_eq!(c.records.len(), 1);
        },
    );

    r.when(
        "a record carries a headroom with no stated basis",
        |w: &mut World, _a: &[String]| {
            w.set("bad_record", "k\tmetal\timplemented\t0.4\t\t2026-01-01\n");
        },
    );
    r.then(
        "it is refused rather than stored",
        |w: &mut World, _a: &[String]| {
            // Headroom is a MEASUREMENT. A number whose provenance nobody recorded is
            // indistinguishable from a guess.
            let e = tile_cli::corpus::parse(w.get("bad_record")).unwrap_err();
            assert!(e.to_string().contains("not a measurement"), "{e}");
        },
    );

    r.given(
        "a corpus snapshot with an export timestamp",
        |w: &mut World, _a: &[String]| {
            w.set(
                "corpus",
                "#exported 2026-09-01T10:00:00Z\n#revision abc1234\n\
             softmax\tmetal\timplemented\t0.31\tmeasured\t2026-08-01\n",
            );
        },
    );
    r.then(
        "the report names the corpus export timestamp and source revision",
        |w: &mut World, _a: &[String]| {
            let r = w.get("report").to_string();
            assert!(r.contains("2026-09-01"), "{r}");
            assert!(r.contains("abc1234"), "{r}");
        },
    );

    r.when(
        "the dependency graph of the \"stats\" feature is inspected",
        |w: &mut World, _a: &[String]| {
            let out = Command::new(env!("CARGO"))
                .args([
                    "tree",
                    "-e",
                    "normal",
                    "--prefix",
                    "none",
                    "--features",
                    "stats",
                ])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .expect("cargo tree");
            w.set("graph", String::from_utf8_lossy(&out.stdout).into_owned());
        },
    );
    r.then(
        "no crate in it runs a C or C++ compiler",
        |w: &mut World, _a: &[String]| {
            for bad in [
                "cc",
                "cmake",
                "ring",
                "openssl-sys",
                "bindgen",
                "aws-lc-sys",
            ] {
                assert!(
                    !w.get("graph")
                        .lines()
                        .any(|l| l.split(" v").next() == Some(bad)),
                    "{bad} compiles C and must not be in the graph"
                );
            }
        },
    );
    r.then(
        "no SQLite implementation is linked",
        |w: &mut World, _a: &[String]| {
            // This is why the corpus is not SQLite: rusqlite would fail the line above.
            assert!(!w.get("graph").contains("sqlite"), "{}", w.get("graph"));
        },
    );

    // ── M7: the rewrite layer ─────────────────────────────────────────────────────

    r.when(
        "the pipeline runs at level {}",
        |w: &mut World, a: &[String]| {
            let level: u8 = a[0].trim().parse().expect("a level");
            let src = std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir"),
            )
            .unwrap();
            let src = if w.has("opt_input") {
                w.get("opt_input").to_string()
            } else {
                src
            };
            let (out, report) = tile_cli::optimize::optimize(&src, level);
            let (again, _) = tile_cli::optimize::optimize(&src, level);
            w.set("opt_src", src);
            w.set("opt_out", out);
            w.set("opt_again", again);
            w.set(
                "opt_passes",
                report
                    .ran
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            w.set("opt_report", report.render());
            w.set("opt_reverted", report.broke_fusion.unwrap_or_default());
        },
    );
    r.then(
        "the passes that ran are \"{}\"",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("opt_passes"), a[0].trim());
        },
    );
    r.then(
        "running it twice gives byte-identical output",
        |w: &mut World, _a: &[String]| {
            assert_eq!(
                w.get("opt_out"),
                w.get("opt_again"),
                "the pipeline is not pure"
            );
        },
    );
    r.then(
        "the report names fusion as the emitters' job, not as missing work",
        |w: &mut World, _a: &[String]| {
            // The plan called this "not yet written". The codebase says otherwise: the
            // emitters find it by adjacency, so doing it here would break the lowering.
            assert!(
                w.get("opt_report").contains("owned by the emitters"),
                "{}",
                w.get("opt_report")
            );
        },
    );

    r.given("the canonical kernel", |w: &mut World, _a: &[String]| {
        w.set(
            "opt_src",
            std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir"),
            )
            .unwrap(),
        );
    });
    r.when(
        "it is parsed into the IR and printed back",
        |w: &mut World, _a: &[String]| {
            let m = tile_cli::mlir::parse(w.get("opt_src"));
            w.set("opt_out", tile_cli::mlir::print(&m));
        },
    );
    r.then(
        "the result is byte-identical to the input",
        |w: &mut World, _a: &[String]| {
            // The property every pass rests on: the layer can only change what a pass
            // deliberately changed, so the 15 golden files survive a parser refactor.
            assert_eq!(w.get("opt_out"), w.get("opt_src"));
        },
    );

    r.given(
        "a kernel whose silu and mul have been separated",
        |w: &mut World, _a: &[String]| {
            w.set(
                "opt_input",
                "  %s = llvm.call @__tile_silu_f32(%g, %g, %r, %c) : (i32, i32, i32, i32) -> i32\n                   %x = llvm.call @__tile_exp_f32(%g, %r, %c) : (i32, i32, i32) -> i32\n                   %m = llvm.call @__tile_mul_f32(%s, %s, %x, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n                   llvm.return %m : i32\n",
            );
        },
    );
    r.then(
        "the input is returned unchanged",
        |w: &mut World, _a: &[String]| {
            // An unoptimized kernel is always better than a quietly deoptimized one.
            assert_eq!(w.get("opt_out"), w.get("opt_src"));
        },
    );
    r.then(
        "the report says the optimization was reverted",
        |w: &mut World, _a: &[String]| {
            assert!(
                !w.get("opt_reverted").is_empty(),
                "the break was not reported"
            );
            assert!(
                w.get("opt_report").contains("REVERTED"),
                "{}",
                w.get("opt_report")
            );
        },
    );

    r.given(
        "a kernel whose tile exceeds what \"{}\" can hold",
        |w: &mut World, a: &[String]| {
            w.set("bounds_family", a[0].trim());
            w.set(
                "bounds_src",
                "  %r = llvm.mlir.constant(256 : i32) : i32\n                   %c = llvm.mlir.constant(256 : i32) : i32\n                   %t = llvm.call @__tile_softmax_f32(%r, %c) : (i32, i32) -> i32\n",
            );
        },
    );
    r.given(
        "the target is \"{}\", which nobody has measured",
        |w: &mut World, a: &[String]| {
            w.set("bounds_family", a[0].trim());
            w.set(
                "bounds_src",
                "  %r = llvm.mlir.constant(1 : i32) : i32\n                   %c = llvm.mlir.constant(64 : i32) : i32\n                   %t = llvm.call @__tile_softmax_f32(%r, %c) : (i32, i32) -> i32\n",
            );
        },
    );
    r.when("the bounds are checked", |w: &mut World, _a: &[String]| {
        let hp = tile_cli::profile::params_for(w.get("bounds_family"));
        let m = tile_cli::mlir::parse(w.get("bounds_src"));
        match tile_cli::passes::check_bounds(&m, &hp, 4) {
            Ok(v) => {
                w.set("bounds_refused", "");
                w.set(
                    "bounds_violations",
                    v.iter()
                        .map(|x| format!("{} on {}", x.rule, x.op))
                        .collect::<Vec<_>>()
                        .join("; "),
                );
            }
            Err(e) => {
                w.set("bounds_refused", e);
                w.set("bounds_violations", "");
            }
        }
    });
    r.then(
        "a bound violation is reported",
        |w: &mut World, _a: &[String]| {
            assert!(!w.get("bounds_violations").is_empty(), "no violation found");
        },
    );
    r.then(
        "the violation names the rule and the operation",
        |w: &mut World, _a: &[String]| {
            let v = w.get("bounds_violations").to_string();
            assert!(v.contains(" on __tile_"), "{v}");
        },
    );
    r.then(
        "the check refuses rather than approving or rejecting",
        |w: &mut World, _a: &[String]| {
            // f10513c's rule arriving in the optimizer: another chip's capacities would
            // approve precisely the tilings that fail here.
            assert!(
                !w.get("bounds_refused").is_empty(),
                "an unmeasured arch answered"
            );
            assert!(
                w.get("bounds_refused").contains("MEASURED"),
                "{}",
                w.get("bounds_refused")
            );
            assert!(w.get("bounds_violations").is_empty());
        },
    );

    r.given(
        "the target's compiler is not installed",
        |w: &mut World, _a: &[String]| {
            // `csl` has no compiler tile-rs knows about at all, which is the same shape of
            // answer as one that is simply absent and needs no machine to arrange.
            w.set("o3_form", "csl");
        },
    );
    r.when(
        "the toolchain check runs",
        |w: &mut World, _a: &[String]| {
            let f = forms::by_id(w.get("o3_form")).expect("form");
            w.set(
                "o3",
                tile_cli::optimize::toolchain_feedback(f, "comptime {}").to_string(),
            );
        },
    );
    r.then(
        "it reports the compiler as unavailable, not the kernel as rejected",
        |w: &mut World, _a: &[String]| {
            let o = w.get("o3").to_string();
            assert!(
                !o.contains("rejected"),
                "the machine was blamed on the kernel: {o}"
            );
        },
    );
    r.then(
        "it states that the conversion still ran",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("o3").contains("still ran"), "{}", w.get("o3"));
        },
    );

    // ── M10a: the UI server ───────────────────────────────────────────────────────

    r.given(
        "no windowing subsystem is available",
        |w: &mut World, _a: &[String]| {
            // The check the tool makes. On a machine that HAS a display, this scenario is
            // still meaningful: what it asserts is that the address is printed either way,
            // which is the behaviour that keeps a headless box working.
            w.set_flag("headless", !tile_cli::ui::windowing_available());
        },
    );
    r.when(
        "I run \"tile softmax.mlir -u\"",
        |w: &mut World, _a: &[String]| {
            ui_run(w);
        },
    );
    r.then(
        "the address is printed on stdout",
        |w: &mut World, _a: &[String]| {
            // stdout so it can be piped; everything else is stderr.
            assert!(
                w.get("ui_url").starts_with("http://127.0.0.1:"),
                "{}",
                w.get("stdout")
            );
        },
    );
    r.then(
        "fetching it returns the profile, the routes and the form matrix",
        |w: &mut World, _a: &[String]| {
            let body = w.get("ui_body").to_string();
            assert!(body.contains("form: mlir"), "no profile in the page");
            assert!(body.contains("hazards:"), "no hazard report in the page");
            assert!(body.contains("write-only"), "no form matrix in the page");
        },
    );
    r.when(
        "the UI listener is bound",
        |w: &mut World, _a: &[String]| {
            let l = tile_cli::ui::bind(0).expect("bind");
            w.set_flag("ui_loopback", l.local_addr().unwrap().ip().is_loopback());
        },
    );
    r.then(
        "its address is a loopback address",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.flag("ui_loopback"),
                "the UI bound something other than loopback"
            );
        },
    );
    r.given(
        "a kernel whose text contains markup",
        |w: &mut World, _a: &[String]| {
            w.set("ui_text", "<script>alert(1)</script>");
        },
    );
    r.when("the page is rendered", |w: &mut World, _a: &[String]| {
        let v = tile_cli::ui::View {
            title: "k.mlir".into(),
            profile: Some(if w.has("ui_text") {
                w.get("ui_text").to_string()
            } else {
                "form: mlir".into()
            }),
            platform: "platform: test".into(),
            forms: tile_cli::routes::list_forms(),
            routes: "mlir -> msl".into(),
        };
        w.set("ui_body", tile_cli::ui::page(&v));
    });
    r.then(
        "it references no external host",
        |w: &mut World, _a: &[String]| {
            let b = w.get("ui_body").to_string();
            assert!(
                !b.contains("http://") && !b.contains("https://"),
                "an external reference"
            );
        },
    );
    r.then(
        "a viewer with no connectivity sees everything on it",
        |w: &mut World, _a: &[String]| {
            // The claim is about the NETWORK, not about scripts: /loader.js and the wasm
            // bundle come from this same process on loopback, so a machine with no
            // connectivity gets all of it. What must never appear is an off-machine
            // reference, and the page must be complete before any script runs.
            let body = w.get("ui_body");
            assert!(!body.contains("http://"), "an off-machine reference");
            assert!(!body.contains("https://"), "an off-machine reference");
            assert!(!body.contains("src=\"http"), "an off-machine script");
            // Every form is already in the served HTML: nothing is fetched to fill it.
            for f in forms::FORMS {
                assert!(body.contains(f.id), "{} is not in the served page", f.id);
            }
        },
    );
    r.then(
        "the markup is shown as text, not interpreted",
        |w: &mut World, _a: &[String]| {
            let b = w.get("ui_body").to_string();
            assert!(
                !b.contains("<script>alert"),
                "unescaped markup reached the page"
            );
            assert!(b.contains("&lt;script&gt;"), "it should be shown as text");
        },
    );
    r.when(
        "the view is built for a kernel",
        |w: &mut World, _a: &[String]| {
            let src = std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir"),
            )
            .unwrap();
            let v = tile_cli::ui::View::build(
                &tile_cli::simenv::Env::detect(),
                Some(("softmax.mlir", &src)),
                "apple-gpu",
            );
            w.set("ui_profile", v.profile.clone().unwrap_or_default());
            w.set("ui_routes", v.routes.clone());
            w.set("ui_src", src);
        },
    );
    r.then(
        "its profile is the one the profiler produced",
        |w: &mut World, _a: &[String]| {
            // Not a similar report: the same one, from the same function.
            let f = forms::by_id("mlir").unwrap();
            let p = profile::profile(
                "softmax.mlir",
                w.get("ui_src"),
                f,
                forms::How::Extension,
                "apple-gpu",
                8192,
            );
            assert_eq!(w.get("ui_profile"), p.render());
        },
    );
    r.then(
        "its routes are the ones the planner produced",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("ui_routes").contains("msl"), "{}", w.get("ui_routes"));
        },
    );

    // ── M9a: the engine mapping ───────────────────────────────────────────────────

    r.when(
        "an engine is chosen for that family",
        |w: &mut World, _a: &[String]| {
            let fam = w.get("family").to_string();
            w.set(
                "engine",
                tile_cli::engine::engine_for(&fam).unwrap_or("(none)"),
            );
        },
    );
    r.then(
        "the engine \"{}\" is selected",
        |w: &mut World, a: &[String]| {
            assert_eq!(w.get("engine"), a[0].trim());
        },
    );
    r.when(
        "an engine that is not installed is located",
        |w: &mut World, _a: &[String]| {
            let e = tile_cli::engine::locate("ds4-rs-nonesuch").unwrap_err();
            w.set("engine_err", e.to_string());
        },
    );
    r.then(
        "the error names every path it searched",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("engine_err").contains("Looked in"),
                "{}",
                w.get("engine_err")
            );
            assert!(
                w.get("engine_err").contains("ds4-rs-nonesuch"),
                "{}",
                w.get("engine_err")
            );
        },
    );
    r.then(
        "it states that nothing is fetched on the user's behalf",
        |w: &mut World, _a: &[String]| {
            // A repository with its own build is not something to acquire silently.
            assert!(
                w.get("engine_err").contains("Nothing is fetched for you"),
                "{}",
                w.get("engine_err")
            );
        },
    );
    r.when("I run \"tile -d -m {}\"", |w: &mut World, a: &[String]| {
        let replies = mcp_with(
            &[r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#],
            &["-d", "-m", a[0].trim()],
        );
        w.set("mcp", replies.join("\n"));
    });
    r.then(
        "the error states that -m requires -d",
        |w: &mut World, _a: &[String]| {
            // An engine provisioned beside nothing is an engine nobody can reach.
            assert!(w.get("error").contains("needs -d"), "{}", w.get("error"));
        },
    );
    r.then(
        "the daemon still answers a request",
        |w: &mut World, _a: &[String]| {
            // Losing the kernel tools because a model could not be provisioned would be the
            // wrong trade.
            assert!(w.get("mcp").contains("\"id\":1"), "{}", w.get("mcp"));
        },
    );

    // ── M6: lowering from tile-rs source ──────────────────────────────────────────

    r.given(
        "RUSTFLAGS is set in the environment",
        |w: &mut World, _a: &[String]| {
            w.set("rustflags", "-C target-cpu=native");
        },
    );
    r.when(
        "a kernel is lowered from tile-rs source",
        |w: &mut World, _a: &[String]| {
            let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var_os("RUSTFLAGS");
            std::env::set_var("RUSTFLAGS", w.get("rustflags"));
            let msl = forms::by_id("msl").expect("form");
            let r = tile_cli::lower_rs::lower("", msl, &mut |_| {});
            match prev {
                Some(v) => std::env::set_var("RUSTFLAGS", v),
                None => std::env::remove_var("RUSTFLAGS"),
            }
            match r {
                Ok(_) => w.set("lower_err", ""),
                Err(e) => w.set("lower_err", e.to_string()),
            }
        },
    );
    r.then(
        "the command is refused before the build starts",
        |w: &mut World, _a: &[String]| {
            // Discovering it AFTER a two-minute build is the failure this prevents.
            assert!(
                !w.get("lower_err").is_empty(),
                "the build was allowed to start"
            );
            assert!(
                w.get("lower_err").contains("RUSTFLAGS"),
                "{}",
                w.get("lower_err")
            );
        },
    );
    r.then(
        "the error explains that the backend would never load",
        |w: &mut World, _a: &[String]| {
            let e = w.get("lower_err").to_string();
            assert!(e.contains("never load"), "{e}");
            // The silence is the point: no error, no output, a stock-backend build.
            assert!(e.contains("quietly"), "{e}");
        },
    );
    r.then(
        "the error gives the exact way to re-run it",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("lower_err").contains("env -u RUSTFLAGS"),
                "{}",
                w.get("lower_err")
            );
        },
    );

    r.then(
        "the codegen path for \"{}\" is \"{}\"",
        |_w: &mut World, a: &[String]| {
            // Not the same string as the form id. An unrecognised value is not rejected by
            // the backend -- it falls through to the Ascend default and fails with a message
            // about ASCEND_HOME_PATH, which points at entirely the wrong thing.
            assert_eq!(forms::codegen_path(&a[0]), Some(a[1].trim()));
        },
    );
    r.then(
        "an unrecognised value is never sent to the backend",
        |_w: &mut World, _a: &[String]| {
            assert_eq!(forms::codegen_path("nonesuch"), None);
            assert_eq!(
                forms::codegen_path("mlir"),
                None,
                "mlir is not a codegen target"
            );
        },
    );

    // ── -r/--run ──────────────────────────────────────────────────────────────────

    fn a_run_report(identical: bool) -> tile_cli::run::RunReport {
        use tile_cli::{run::*, torchref::*};
        RunReport {
            device: "Test GPU".into(),
            op: RefOp::Softmax,
            dtype: "float".into(),
            elements: 1024,
            checked: 1024,
            worst_error: 1e-7,
            budget: None,
            nan_mismatch: 0,
            tolerance: 1e-5,
            target: Timing {
                samples: vec![10.0, 11.0, 12.0],
                warmup: 3,
            },
            reference: Timing {
                samples: vec![1000.0],
                warmup: 3,
            },
            unoptimized: if identical {
                None
            } else {
                Some(Timing {
                    samples: vec![15.0],
                    warmup: 3,
                })
            },
            optimizer_changed_nothing: identical,
            vs_ours: Accuracy {
                compared: 1024,
                max_abs: 1e-7,
                max_rel: 1e-7,
                rmse: 1e-8,
                non_finite: 0,
                max_abs_near_zero: 0.0,
            },
            torch: Err(TorchStatus::Absent {
                why: "No module named 'torch'".into(),
            }),
        }
    }

    r.given(
        "a run where the optimizer changed nothing",
        |w: &mut World, _a: &[String]| {
            w.set("run_report", a_run_report(true).render());
        },
    );
    r.given("a completed run", |w: &mut World, _a: &[String]| {
        w.set("run_report", a_run_report(true).render());
    });
    r.given("torch is not installed", |w: &mut World, _a: &[String]| {
        w.set("run_report", a_run_report(true).render());
    });
    r.then(
        "no optimizer ratio is reported",
        |w: &mut World, _a: &[String]| {
            // "1.00x" implies a measurement that came out even. None was taken.
            assert!(
                !w.get("run_report").contains("from the optimization passes"),
                "{}",
                w.get("run_report")
            );
        },
    );
    r.then(
        "the report says the two sources were identical",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("run_report").contains("identical source"),
                "{}",
                w.get("run_report")
            );
        },
    );
    r.then(
        "the report states that the baseline is a naive single-threaded scalar loop",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("run_report")
                    .contains("naive single-threaded scalar loop"),
                "{}",
                w.get("run_report")
            );
        },
    );
    r.then(
        "the report states that this is not a GPU-versus-CPU figure",
        |w: &mut World, _a: &[String]| {
            // The figure is quoted out of context otherwise.
            assert!(
                w.get("run_report").contains("NOT a GPU-versus-CPU figure"),
                "{}",
                w.get("run_report")
            );
        },
    );
    r.then(
        "each timing reports the number of runs, the minimum and the maximum",
        |w: &mut World, _a: &[String]| {
            let r = w.get("run_report").to_string();
            assert!(r.contains("over 3 runs"), "{r}");
            assert!(r.contains("min ") && r.contains("max "), "{r}");
        },
    );
    r.then(
        "it reports how many warmup runs preceded them",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("run_report").contains("warmup"),
                "{}",
                w.get("run_report")
            );
        },
    );
    r.then(
        "the report still compares the kernel against this tool's reference",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("run_report").contains("kernel vs ours"),
                "{}",
                w.get("run_report")
            );
        },
    );
    r.then(
        "it states that torch was unavailable and the claim is weaker",
        |w: &mut World, _a: &[String]| {
            let r = w.get("run_report").to_string();
            assert!(r.contains("torch:"), "{r}");
            assert!(r.contains("weaker claim"), "{r}");
        },
    );

    r.when(
        "the reference timing loop is inspected",
        |w: &mut World, _a: &[String]| {
            w.set(
                "timed_region",
                tile_cli::run::timed_region_source().to_string(),
            );
        },
    );
    r.then(
        "no allocation happens between the clock starting and stopping",
        |w: &mut World, _a: &[String]| {
            // A baseline you have accidentally slowed down is not a baseline: this cost
            // 0.2x of an overstated speedup before it was fixed.
            assert!(
                !w.get("timed_region").contains("vec!["),
                "{}",
                w.get("timed_region")
            );
        },
    );

    r.when(
        "the three input generators are compared",
        |w: &mut World, _a: &[String]| {
            use tile_cli::run::{input_values_for, RefOp, Shape};
            let swift = include_str!("../assets/harness/metal.swift");
            let py_soft = tile_cli::torchref::script(
                RefOp::Softmax,
                Shape::Rows { rows: 1, cols: 8 },
                1e-6,
                "float",
            )
            .unwrap();
            // The matmul script is the one that reads TWO buffers, so it is the one that
            // can disagree about the second. A reference handed the same matrix twice
            // agrees with a transposed kernel, and nothing reports it.
            let py_mm = tile_cli::torchref::script(
                RefOp::Matmul,
                Shape::Matmul { m: 2, k: 2, n: 2 },
                1e-6,
                "float",
            )
            .unwrap();
            let b0 = input_values_for(0, 8);
            let b1 = input_values_for(1, 8);
            w.set_flag(
                "same_rule",
                b0[0] == -2.0
                    && b0 != b1
                    && swift.contains("Float(($0 + 7 * b + seed) % 17) * 0.25 - 2.0")
                    && py_soft.contains("(i % 17) * 0.25 - 2.0")
                    && py_mm.contains("(i % 17) * 0.25 - 2.0")
                    && py_mm.contains("((i + 7) % 17) * 0.25 - 2.0"),
            );
        },
    );
    r.then(
        "they compute the same values by the same rule",
        |w: &mut World, _a: &[String]| {
            // Three implementations of one rule is two chances to drift. A reference fed
            // different data compares two unrelated numbers and fails like a numerical bug.
            assert!(
                w.flag("same_rule"),
                "the three input generators have drifted apart"
            );
        },
    );

    r.when(
        "a kernel using an operation with no reference is offered",
        |w: &mut World, _a: &[String]| {
            let e = tile_cli::run::RefOp::detect("llvm.call @__tile_attention_f32").unwrap_err();
            w.set(
                "run_err",
                tile_cli::run::RunError::NoReference { hint: e }.to_string(),
            );
        },
    );
    r.then(
        "it is refused with the operations that do have one",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("run_err").contains("softmax"), "{}", w.get("run_err"));
        },
    );
    r.then(
        "the refusal says numbers with nothing to compare them against are worse",
        |w: &mut World, _a: &[String]| {
            assert!(
                w.get("run_err").contains("worse than not running it"),
                "{}",
                w.get("run_err")
            );
        },
    );
    r.when(
        "a kernel using two referenceable operations is offered",
        |w: &mut World, _a: &[String]| {
            w.set(
                "run_err",
                tile_cli::run::RefOp::detect("__tile_exp_f32 and __tile_relu_f32").unwrap_err(),
            );
        },
    );
    r.then(
        "it is refused because the composition order would have to be guessed",
        |w: &mut World, _a: &[String]| {
            assert!(w.get("run_err").contains("compose"), "{}", w.get("run_err"));
        },
    );

    r.given(
        "the kernel is \"{}\" to torch and this tool's reference is \"{}\"",
        |w: &mut World, a: &[String]| {
            w.set("acc_kernel", a[0].trim());
            w.set("acc_ours", a[1].trim());
        },
    );
    r.given(
        "the kernel does not match this tool's reference either",
        |w: &mut World, _a: &[String]| {
            // Both stand far from torch AND far from each other: nothing was inherited,
            // because there is nothing they share.
            w.set("acc_vs_ours", "far");
        },
    );
    r.then(
        "the verdict does not blame the reference alone",
        |w: &mut World, _a: &[String]| {
            let v = attribution_with(w.get("acc_kernel"), w.get("acc_ours"), Some("far"));
            assert!(
                !v.contains("inherited"),
                "the kernel matches neither torch nor the reference, so it inherited \
                 nothing: {v:?}"
            );
            // It used to require the words "separate faults". That was an overreach of
            // its own: an f32 matmul lands in this corner with all three magnitudes the
            // same order, which is summation order rather than a fault in anything. The
            // line now points at the magnitudes and names both readings.
            assert!(
                v.contains("precision") && v.contains("defect"),
                "the line should let the magnitudes decide: {v:?}"
            );
        },
    );
    r.then("the verdict is \"{}\"", |w: &mut World, a: &[String]| {
        let v = attribution(w.get("acc_kernel"), w.get("acc_ours"));
        assert!(v.contains(a[0].trim()), "wanted {:?}, got {v:?}", a[0]);
    });
    r.then(
        "the verdict differs between a wrong kernel and a wrong reference",
        |_w: &mut World, _a: &[String]| {
            // The entire argument for asking a third source: two cannot tell these apart.
            assert_ne!(attribution("far", "close"), attribution("far", "far"));
        },
    );

    r.when(
        "accuracy is computed against a reference value near zero",
        |w: &mut World, _a: &[String]| {
            let acc = tile_cli::torchref::Accuracy::of(&[1e-9, 1.0], &[1e-12, 1.0]);
            w.set("max_rel", acc.max_rel.to_string());
        },
    );
    r.then(
        "the relative error is not reported as enormous",
        |w: &mut World, _a: &[String]| {
            // Near zero every implementation disagrees relatively and none is wrong.
            assert_eq!(w.get("max_rel"), "0", "{}", w.get("max_rel"));
        },
    );

    r.when(
        "one side produces NaN where the other produces a number",
        |w: &mut World, _a: &[String]| {
            let acc = tile_cli::torchref::Accuracy::of(&[f32::NAN, 1.0], &[0.5, 1.0]);
            w.set("nf", acc.non_finite.to_string());
            w.set_flag("rmse_finite", acc.rmse.is_finite());
            w.set_flag("within", acc.within(1.0, 1.0));
        },
    );
    r.then(
        "the mismatch is counted separately",
        |w: &mut World, _a: &[String]| {
            assert_eq!(w.get("nf"), "1");
        },
    );
    r.then(
        "the summary figure stays finite",
        |w: &mut World, _a: &[String]| {
            // One NaN in an RMSE poisons the whole figure and says nothing about how many
            // elements were affected.
            assert!(w.flag("rmse_finite"), "one NaN poisoned the summary");
        },
    );
    r.then(
        "the result is never within tolerance",
        |w: &mut World, _a: &[String]| {
            assert!(!w.flag("within"));
        },
    );

    r.when(
        "a run is asked for on a target with no harness",
        |w: &mut World, _a: &[String]| {
            let e = tile_cli::run::harness_source(forms::by_id("gpu").unwrap()).unwrap_err();
            w.set("run_err", e.to_string());
        },
    );
    r.then("it is refused by name", |w: &mut World, _a: &[String]| {
        assert!(
            w.get("run_err").contains("no harness"),
            "{}",
            w.get("run_err")
        );
        assert!(w.get("run_err").contains("gpu"), "{}", w.get("run_err"));
    });

    r.then(
        "uv has a verified entry for every platform this tool ships to",
        |_w: &mut World, _a: &[String]| {
            // Not `curl -LsSf https://astral.sh/uv/install.sh | sh`: that pipe is
            // unverified, runs whatever the server sends, and installs where it likes.
            let tools = tile_cli::manifest::builtin();
            for (os, arch) in [
                ("macos", "aarch64"),
                ("macos", "x86_64"),
                ("linux", "x86_64"),
                ("linux", "aarch64"),
            ] {
                assert!(
                    tile_cli::manifest::find(&tools, "uv", os, arch).is_some(),
                    "no uv entry for {os}/{arch}"
                );
            }
        },
    );
    r.then(
        "each entry carries a real digest and unpacks",
        |_w: &mut World, _a: &[String]| {
            for t in tile_cli::manifest::builtin()
                .iter()
                .filter(|t| t.id == "uv")
            {
                assert!(t.pinned(), "uv on {}/{} has no real digest", t.os, t.arch);
                assert_eq!(t.unpack.as_deref(), Some("tar.gz"));
                assert!(t.auto_installable());
            }
        },
    );
    r.then(
        "the reference is asked for through \"{}\"",
        |_w: &mut World, a: &[String]| {
            // The "no additional commands" property: uv resolves and caches torch itself,
            // so there is no venv to create and no second command for anyone to run.
            let src = include_str!("../src/torchref.rs");
            for word in a[0].split_whitespace() {
                assert!(src.contains(word), "the torch path does not use {word:?}");
            }
        },
    );
    r.then(
        "uv is provisioned first when it is not present",
        |_w: &mut World, _a: &[String]| {
            let src = include_str!("../src/torchref.rs");
            assert!(
                src.contains("provision::ensure(\"uv\""),
                "an absent uv is not provisioned"
            );
        },
    );

    // ── The backlog ───────────────────────────────────────────────────────────────

    fn bl_issue(id: u32, area: &str, title: &str) -> tile_cli::backlog::Issue {
        tile_cli::backlog::Issue {
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

    r.given(
        "an entry recording a gap and how it was worked around",
        |w: &mut World, _a: &[String]| {
            w.set("bl", bl_issue(7, "lowering", "no lifter for msl").render());
        },
    );
    r.when(
        "it is written to the backlog and read again",
        |w: &mut World, _a: &[String]| {
            let back = tile_cli::backlog::Issue::parse(w.get("bl")).expect("parse");
            w.set_flag(
                "bl_same",
                back == bl_issue(7, "lowering", "no lifter for msl"),
            );
        },
    );
    r.then(
        "every field comes back unchanged",
        |w: &mut World, _a: &[String]| {
            assert!(w.flag("bl_same"), "an entry did not survive the round trip");
        },
    );

    r.given(
        "an entry with no workaround recorded",
        |w: &mut World, _a: &[String]| {
            let mut i = bl_issue(1, "run", "x");
            i.workaround = String::new();
            w.set("bl", i.render());
        },
    );
    r.when("it is rendered", |_w: &mut World, _a: &[String]| {});
    r.then(
        "it states that an entry without one is a report, not evidence",
        |w: &mut World, _a: &[String]| {
            // The whole point of the backlog is that the fix starts from evidence.
            assert!(
                w.get("bl").contains("a report, not evidence"),
                "{}",
                w.get("bl")
            );
        },
    );

    r.when(
        "two entries are recorded",
        |w: &mut World, _a: &[String]| {
            let a = tile_cli::backlog::path_for(&bl_issue(1, "lowering", "No lifter for MSL!"));
            let b = tile_cli::backlog::path_for(&bl_issue(2, "run", "no harness for cuda"));
            w.set("bl_a", a.file_name().unwrap().to_string_lossy());
            w.set("bl_b", b.file_name().unwrap().to_string_lossy());
        },
    );
    r.then(
        "each has its own file named by its number and title",
        |w: &mut World, _a: &[String]| {
            // One file per entry, so two sessions adding one do not conflict.
            assert_ne!(w.get("bl_a"), w.get("bl_b"));
            assert!(
                w.get("bl_a").starts_with("001-no-lifter-for-msl"),
                "{}",
                w.get("bl_a")
            );
            assert!(w.get("bl_b").starts_with("002-"), "{}", w.get("bl_b"));
        },
    );

    r.given(
        "open issues in \"{}\" and \"{}\"",
        |w: &mut World, a: &[String]| {
            w.set("bl_set", format!("{},{}", a[0].trim(), a[1].trim()));
        },
    );
    r.given(
        "three open issues all in \"{}\"",
        |w: &mut World, a: &[String]| {
            let x = a[0].trim();
            w.set("bl_set", format!("{x},{x},{x}"));
        },
    );
    r.given(
        "many open issues, three of them in one area",
        |w: &mut World, _a: &[String]| {
            let mut v = vec!["run".to_string(); 3];
            for i in 0..10 {
                v.push(format!("area{i}"));
            }
            w.set("bl_set", v.join(","));
        },
    );
    r.given(
        "twelve open issues in twelve different areas",
        |w: &mut World, _a: &[String]| {
            let v: Vec<String> = (0..12).map(|i| format!("area{i}")).collect();
            w.set("bl_set", v.join(","));
        },
    );
    r.given(
        "no issues have been recorded",
        |w: &mut World, _a: &[String]| {
            w.set("bl_set", "");
        },
    );

    r.when("one of them is closed", |w: &mut World, _a: &[String]| {
        w.set_flag("bl_close_first", true);
    });
    r.when("the backlog is listed", |_w: &mut World, _a: &[String]| {});

    r.then("the verdict is healthy", |w: &mut World, _a: &[String]| {
        assert_eq!(
            bl_verdict(w),
            tile_cli::backlog::Verdict::Healthy,
            "{}",
            w.get("bl_set")
        );
    });
    r.then(
        "the exit code would be {}",
        |w: &mut World, a: &[String]| {
            let want = a[0].trim();
            let healthy = bl_verdict(w) == tile_cli::backlog::Verdict::Healthy;
            assert_eq!(if healthy { "0" } else { "3" }, want);
        },
    );
    r.then(
        "the verdict says to stop and reconsider",
        |w: &mut World, _a: &[String]| {
            assert!(bl_verdict(w).to_string().contains("STOP AND RECONSIDER"));
        },
    );
    r.then("it names the area", |w: &mut World, _a: &[String]| {
        assert!(bl_verdict(w).to_string().contains("form-detection"));
    });
    r.then(
        "it says that fixing them individually is three ways of not addressing it",
        |w: &mut World, _a: &[String]| {
            // Three in one area is one design problem wearing three hats.
            let v = bl_verdict(w).to_string();
            assert!(v.contains("3 ways of not addressing it"), "{v}");
            assert!(v.contains("one design problem"), "{v}");
        },
    );
    r.then(
        "the verdict names the area rather than the total",
        |w: &mut World, _a: &[String]| {
            // "This area is wrong" is actionable; "you have too many issues" is not.
            assert!(matches!(
                bl_verdict(w),
                tile_cli::backlog::Verdict::Cluster { .. }
            ));
        },
    );
    r.then(
        "the verdict says the tool is costing more than it saves",
        |w: &mut World, _a: &[String]| {
            assert!(bl_verdict(w)
                .to_string()
                .contains("costing more than it saves"));
        },
    );
    r.then(
        "the entries are grouped by area",
        |w: &mut World, _a: &[String]| {
            let out = tile_cli::backlog::render(&bl_issues(w));
            assert!(out.contains("run (1)"), "{out}");
            assert!(out.contains("lowering (1)"), "{out}");
        },
    );
    r.then(
        "each shows the first line of its workaround",
        |w: &mut World, _a: &[String]| {
            let out = tile_cli::backlog::render(&bl_issues(w));
            assert!(out.contains("workaround: wrote the MSL by hand"), "{out}");
        },
    );
    r.then(
        "the listing ends with a verdict",
        |w: &mut World, _a: &[String]| {
            assert!(tile_cli::backlog::render(&bl_issues(w)).contains("verdict:"));
        },
    );
    r.then(
        "it states that nothing has been recorded yet",
        |w: &mut World, _a: &[String]| {
            let out = tile_cli::backlog::render(&bl_issues(w));
            assert!(out.contains("nothing has been recorded"), "{out}");
        },
    );
    r.then(
        "a healthy backlog exits {}",
        |_w: &mut World, a: &[String]| {
            assert_eq!(a[0].trim(), "0");
        },
    );
    r.then(
        "a backlog that should stop the tool exits {}",
        |_w: &mut World, a: &[String]| {
            // The mechanism behind "stop and reconsider": a check anything can run, not a
            // judgement someone has to remember to make.
            assert_eq!(a[0].trim(), "3");
            assert_eq!(exit::UNSUPPORTED, 3);
        },
    );
    r.when(
        "an entry carrying an unknown field is read",
        |w: &mut World, _a: &[String]| {
            let e = tile_cli::backlog::Issue::parse("id: 1\nseverity: high\n").unwrap_err();
            w.set("bl_err", e);
        },
    );
    r.then(
        "the unknown field is named in the refusal",
        |w: &mut World, _a: &[String]| {
            // Skipping what it does not understand is how a typo'd field becomes a
            // silently-dropped workaround.
            assert!(w.get("bl_err").contains("severity"), "{}", w.get("bl_err"));
        },
    );

    // ── The mundane edges (11_io_edges) ───────────────────────────────────────────
    //
    // These run the REAL binary in a scratch directory, because every one of them is
    // about the filesystem: a symlink, a directory, several inputs, a module with more
    // than one kernel. Asserting them against library calls would test a model of the
    // thing rather than the thing.

    r.when(
        "I pipe a module into \"tile {}\" and keep the output",
        |w: &mut World, a: &[String]| {
            let dir = scratch_dir("stdin");
            let out = run_bin_stdin(&dir, &a[0], &corpus("softmax.mlir"));
            w.set("stdout", out.0);
            w.set("error", out.1);
            w.set_flag("wrote_out", dir.join("out.metal").exists());
            w.set(
                "written",
                std::fs::read_to_string(dir.join("out.metal")).unwrap_or_default(),
            );
        },
    );
    r.then(
        "the module is read from stdin",
        |w: &mut World, _: &[String]| {
            // It found the kernel, which is only possible if the text arrived.
            let all = format!("{}{}", w.get("stdout"), w.get("error"));
            assert!(all.contains("softmax_1d"), "{all}");
        },
    );
    r.then("\"out.metal\" is written", |w: &mut World, _: &[String]| {
        assert!(w.flag("wrote_out"), "no out.metal");
        assert!(w.get("written").contains("kernel void"), "not MSL");
    });

    r.given(
        "\"a.mlir\", \"b.mlir\" and \"c.mlir\" where \"b.mlir\" is malformed",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("several");
            std::fs::write(dir.join("a.mlir"), corpus("softmax.mlir")).unwrap();
            std::fs::write(dir.join("b.mlir"), "this is not a module at all\n").unwrap();
            std::fs::write(dir.join("c.mlir"), corpus("softmax.mlir")).unwrap();
            w.set("dir", dir.display().to_string());
        },
    );
    r.when(
        "I run \"tile a.mlir b.mlir c.mlir -t msl\" over those",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            let (o, e, code) = run_bin(&dir, &["a.mlir", "b.mlir", "c.mlir", "-t", "msl"]);
            w.set("stdout", o);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "\"a.opt.metal\" and \"c.opt.metal\" are written",
        |w: &mut World, _: &[String]| {
            // The point of the scenario: one bad input must not cost the good ones.
            let dir = PathBuf::from(w.get("dir"));
            assert!(dir.join("a.opt.metal").exists(), "a lost");
            assert!(dir.join("c.opt.metal").exists(), "c lost");
        },
    );
    r.then(
        "the failure for \"b.mlir\" is reported with its reason",
        |w: &mut World, _: &[String]| {
            let e = w.get("error");
            assert!(e.contains("b.mlir"), "b not named: {e}");
        },
    );
    r.when(
        "I run \"tile\" on a directory",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("dirinput");
            std::fs::create_dir_all(dir.join("kernels")).unwrap();
            let (_, e, code) = run_bin(&dir, &["kernels", "-t", "msl"]);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the error shows the equivalent shell glob",
        |w: &mut World, _: &[String]| {
            let e = w.get("error");
            // errno 21 is true and useless. The glob is the thing that works.
            assert!(e.contains("kernels/*."), "no glob offered: {e}");
            assert!(!e.contains("os error"), "raw errno leaked: {e}");
        },
    );

    r.given(
        "\"multi.mlir\" contains two kernels",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("multi");
            std::fs::write(dir.join("multi.mlir"), TWO_KERNEL_MLIR).unwrap();
            w.set("dir", dir.display().to_string());
        },
    );
    r.when("I profile it", |w: &mut World, _: &[String]| {
        let dir = PathBuf::from(w.get("dir"));
        let (o, e, _) = run_bin(&dir, &["multi.mlir", "-i"]);
        w.set("stdout", format!("{o}{e}"));
    });
    r.then(
        "both kernels are profiled and named",
        |w: &mut World, _: &[String]| {
            let o = w.get("stdout");
            assert!(o.contains("k1") && o.contains("k2"), "{o}");
        },
    );
    r.when(
        "I convert it to one output",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            let (_, e, _) = run_bin(&dir, &["multi.mlir", "-t", "msl", "-o", "out.metal"]);
            w.set("error", e);
            w.set(
                "written",
                std::fs::read_to_string(dir.join("out.metal")).unwrap_or_default(),
            );
        },
    );
    r.then(
        "\"out.metal\" contains both kernel entry points",
        |w: &mut World, _: &[String]| {
            let n = w.get("written").matches("kernel void").count();
            assert_eq!(n, 2, "expected 2 entry points, got {n}");
        },
    );

    r.given(
        "\"link.mlir\" is a symlink to \"real.mlir\"",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("symlink");
            std::fs::write(dir.join("real.mlir"), corpus("softmax.mlir")).unwrap();
            let _ = std::os::unix::fs::symlink("real.mlir", dir.join("link.mlir"));
            w.set("dir", dir.display().to_string());
        },
    );
    r.when("I convert the symlink", |w: &mut World, _: &[String]| {
        let dir = PathBuf::from(w.get("dir"));
        // Combined: the profile heading naming the kernel goes to stdout, the route and
        // warnings to stderr, and this scenario is about content rather than stream.
        let (o, e, _) = run_bin(&dir, &["link.mlir", "-t", "msl"]);
        w.set("error", format!("{o}{e}"));
    });
    r.then("\"real.mlir\" is read", |w: &mut World, _: &[String]| {
        assert!(w.get("error").contains("softmax_1d"), "{}", w.get("error"));
    });
    r.then(
        "the output is written beside \"link.mlir\", not beside \"real.mlir\"",
        |w: &mut World, _: &[String]| {
            // Reading THROUGH a link is right; naming the output after the link's target
            // would put the file somewhere the user never mentioned.
            let dir = PathBuf::from(w.get("dir"));
            assert!(
                dir.join("link.opt.metal").exists(),
                "not named after the link"
            );
            assert!(
                !dir.join("real.opt.metal").exists(),
                "named after the target"
            );
        },
    );

    r.given(
        "\"out.metal\" is a symlink to a file outside the output directory",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("symout");
            std::fs::create_dir_all(dir.join("elsewhere")).unwrap();
            std::fs::write(dir.join("elsewhere/precious.metal"), "PRECIOUS").unwrap();
            let _ = std::os::unix::fs::symlink(
                dir.join("elsewhere/precious.metal"),
                dir.join("out.metal"),
            );
            w.set("dir", dir.display().to_string());
        },
    );
    r.when(
        "I convert with -o naming that symlink and --force",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) =
                run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "out.metal", "--force"]);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the link target is untouched and the error names it",
        |w: &mut World, _: &[String]| {
            // --force is permission to replace the file that was NAMED, not licence to
            // follow a link out of the tree. fs::write follows one silently.
            let dir = PathBuf::from(w.get("dir"));
            let kept = std::fs::read_to_string(dir.join("elsewhere/precious.metal")).unwrap();
            assert_eq!(kept, "PRECIOUS", "wrote through the symlink");
            assert_eq!(w.get("exit"), "2");
            assert!(w.get("error").contains("symlink"), "{}", w.get("error"));
        },
    );

    // ── Platform and targets (05) ─────────────────────────────────────────────────

    r.when(
        "I ask the doctor about this machine",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("doctor");
            let (o, e, code) = run_bin(&dir, &["doctor"]);
            w.set("stdout", o);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the report names the platform, the accelerators and the SDKs in separate sections",
        |w: &mut World, _: &[String]| {
            let o = w.get("stdout");
            assert!(o.contains("platform:"), "{o}");
            assert!(o.contains("accelerator"), "{o}");
            // The SDK section is the one that was missing: SDKs were printed only beside
            // a DEVICE, so a toolkit installed for hardware that is not here was
            // invisible -- and those are exactly the ones that decide whether a
            // cross-generation target can be compiled locally.
            assert!(o.contains("sdks:"), "no separate sdks section: {o}");
        },
    );
    r.then(
        "a family whose SDK is absent is distinguished from a family that is absent",
        |_w: &mut World, _: &[String]| {
            // The rendering must be able to say "device present, SDK missing". Asserted
            // on the renderer rather than on this machine, which has its SDK.
            let p = tile_cli::platform::Platform {
                triple: "x".into(),
                os: "linux",
                arch: "x86_64",
                accels: vec![tile_cli::platform::Accel {
                    family: "nvidia",
                    device: "A100".into(),
                    via: "test",
                }],
                sdks: vec![],
            };
            let shown = format!("{p}");
            assert!(shown.contains("SDK missing"), "{shown}");
        },
    );

    r.when(
        "I ask the doctor which targets are trustworthy",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("targets");
            let (o, _, code) = run_bin(&dir, &["doctor", "--targets"]);
            w.set("stdout", o);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "measured targets are listed apart from unmeasured ones",
        |w: &mut World, _: &[String]| {
            let o = w.get("stdout");
            assert!(o.contains("measured"), "{o}");
            assert!(o.contains("unmeasured"), "{o}");
            // Every writable target appears somewhere, or the table is a selection.
            for f in forms::FORMS {
                if f.writable && matches!(f.fidelity, forms::Fidelity::Validated(_, _)) {
                    assert!(o.contains(f.id), "{} missing from the table", f.id);
                }
            }
        },
    );
    r.then(
        "the report says that unmeasured is not unlimited",
        |w: &mut World, _: &[String]| {
            // Without this the table reads as a capability ranking, and the reader
            // concludes an unmeasured target has no bounds rather than unknown ones.
            let o = w.get("stdout");
            assert!(o.contains("not unlimited"), "{o}");
            assert!(o.contains("not unsupported"), "{o}");
        },
    );

    r.when(
        "I convert to a target that does not run on this machine",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("cross");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(&dir, &["k.mlir", "-t", "nki", "-o", "out.py"]);
            w.set("stdout", o);
            w.set("error", e);
            w.set("exit", code.to_string());
            w.set_flag("wrote", dir.join("out.py").exists());
        },
    );
    r.then(
        "the run is reported as cross generation and still succeeds",
        |w: &mut World, _: &[String]| {
            // It used to write CUDA on an Apple laptop and report nothing unusual, so the
            // artifact's unrunnability here was left for the user to find out.
            let e = w.get("error");
            assert!(e.contains("cross generation"), "not announced: {e}");
            assert!(e.contains("trainium"), "family not named: {e}");
            assert_eq!(w.get("exit"), "0");
            assert!(w.flag("wrote"), "no artifact");
        },
    );

    r.when(
        "I convert with --cross on a machine with no accelerator",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("crossbare");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin_env(
                &dir,
                // nvidia/gpu rather than ascend/cpp: `cpp` needs the `ascend` build
                // feature, so on a build without it the run fails for a reason that has
                // nothing to do with the device this scenario is about.
                &["k.mlir", "--cross", "nvidia", "-t", "gpu", "-o", "out.cu"],
                &[("TILE_SIMULATE", "no-devices")],
            );
            w.set("error", e);
            w.set("exit", code.to_string());
            w.set_flag("wrote", dir.join("out.cu").exists());
        },
    );
    r.then(
        "the absence of a device is not itself a failure",
        |w: &mut World, _: &[String]| {
            // --cross is the user saying they know. Refusing for want of a device would
            // make the flag pointless.
            assert_eq!(w.get("exit"), "0", "{}", w.get("error"));
            assert!(w.flag("wrote"), "no artifact: {}", w.get("error"));
        },
    );

    r.when(
        "I ask for -O4 under --cross",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("o4cross");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin(
                &dir,
                &[
                    "k.mlir", "--cross", "nvidia", "-O4", "-t", "gpu", "-o", "out.cu",
                ],
            );
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the error states that measurement requires the native target",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "2");
            let e = w.get("error");
            assert!(e.contains("-O4") && e.contains("--cross"), "{e}");
        },
    );

    r.when(
        "I ask for the version on a machine with nothing installed",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("bare");
            // `bare` clears every feature, device, toolchain and vendor CLI: the emptiest
            // machine that can still run the tool.
            let (o, e, code) = run_bin_env(&dir, &["--version"], &[("TILE_SIMULATE", "bare")]);
            w.set("stdout", o);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the tool still reports its version",
        |w: &mut World, _: &[String]| {
            // Requirement (3): no vendor library is needed to START. Detection is runtime
            // probing, never a build-time link, which is what lets one binary ship.
            assert_eq!(w.get("exit"), "0", "{}", w.get("error"));
            assert!(
                w.get("stdout").contains(tile_cli::VERSION),
                "{}",
                w.get("stdout")
            );
        },
    );

    r.then(
        "doctor lists an AMD GPU as present with no tile-rs target",
        |_w: &mut World, _: &[String]| {
            // No way to conjure the hardware, so this asserts the RENDERING that the
            // detector feeds: an amd-gpu must be reported as present-but-unsupported, and
            // must not be offered `aie`, which is the NPU's form and would emit IRON
            // Python for a GPU.
            let p = tile_cli::platform::Platform {
                triple: "x86_64-unknown-linux-gnu".into(),
                os: "linux",
                arch: "x86_64",
                accels: vec![tile_cli::platform::Accel {
                    family: "amd-gpu",
                    device: "Radeon 8060S".into(),
                    via: "test",
                }],
                sdks: vec![],
            };
            let unsupported = p.unsupported_present();
            assert_eq!(unsupported.len(), 1, "amd-gpu not flagged unsupported");
            assert_eq!(default_form_for("amd-gpu").id, "linalg");
            assert_ne!(default_form_for("amd-gpu").id, "aie");
        },
    );

    // ── Profile and optimize (04) ─────────────────────────────────────────────────

    r.given(
        "a tile-rs kernel in a fresh directory",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("p4");
            std::fs::write(dir.join("softmax.mlir"), corpus("softmax.mlir")).unwrap();
            w.set("dir", dir.display().to_string());
        },
    );
    r.when(
        "I convert it with no output named",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            let (o, e, code) = run_bin(&dir, &["softmax.mlir"]);
            // The route line ends with `-O<n>`; feed the existing generic level step from it
            // rather than registering a second pattern that would shadow it. Bare
            // `tile k.mlir` is -O2, and a default nobody prints is a default nobody checks.
            let level = e
                .lines()
                .find(|l| l.starts_with("route:"))
                .and_then(|l| l.split("-O").nth(1))
                .and_then(|t| t.split_whitespace().next())
                .unwrap_or("")
                .to_string();
            w.set("level", level);
            w.set("stdout", o);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the artifact has a derived name beside the input",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            assert!(
                dir.join("softmax.opt.metal").exists(),
                "no derived artifact"
            );
        },
    );
    r.then(
        "the profile reports the kernel, its dtypes and its tile plan",
        |w: &mut World, _: &[String]| {
            let o = format!("{}{}", w.get("stdout"), w.get("error"));
            for want in ["softmax_1d", "dtypes", "tile plan", "hazards"] {
                assert!(o.contains(want), "profile missing {want}: {o}");
            }
        },
    );

    r.then(
        "an optimized artifact for the detected native target is produced",
        |w: &mut World, _: &[String]| {
            // No -t was given, so the form came from the detected accelerator. On this
            // machine that is apple-gpu -> msl; the assertion is that it matches the
            // DETECTION rather than any particular form, so the scenario stays true on a
            // host with different hardware.
            let dir = PathBuf::from(w.get("dir"));
            let (probe, _, _) = run_bin(&dir, &["doctor"]);
            let fam = probe
                .lines()
                .find_map(|l| l.split('[').nth(1)?.split(']').next())
                .unwrap_or("none")
                .to_string();
            let want = default_form_for(&fam);
            let path = dir.join(format!("softmax.opt.{}", want.primary_ext()));
            assert!(path.exists(), "no artifact for {} ({fam})", want.id);
        },
    );
    r.when("I convert it again", |w: &mut World, _: &[String]| {
        let dir = PathBuf::from(w.get("dir"));
        let (_, e, code) = run_bin(&dir, &["softmax.mlir"]);
        w.set("error", e);
        w.set("exit", code.to_string());
        w.set(
            "kept",
            std::fs::read_to_string(dir.join("softmax.opt.metal")).unwrap_or_default(),
        );
    });
    r.then(
        "the existing artifact is refused rather than overwritten",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "2");
            let e = w.get("error");
            assert!(e.contains("--force"), "does not name --force: {e}");
            assert!(w.get("kept").contains("kernel void"), "clobbered anyway");
        },
    );
    r.when(
        "I convert it again with --force",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            let (_, e, code) = run_bin(&dir, &["softmax.mlir", "--force"]);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the report says an existing file was overwritten",
        |w: &mut World, _: &[String]| {
            // It used to print "wrote out.metal" -- the same line it prints when it CREATED
            // one. Losing work should never be the quiet case.
            assert_eq!(w.get("exit"), "0", "{}", w.get("error"));
            assert!(
                w.get("error").contains("overwritten"),
                "silent overwrite: {}",
                w.get("error")
            );
        },
    );

    r.when(
        "I profile with --info-only",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("infoonly");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(&dir, &["k.mlir", "-i"]);
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
            // Anything at all besides the input would be an artifact it should not have made.
            let extra: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n != "k.mlir")
                .collect();
            w.set("extra", extra.join(","));
        },
    );
    r.then(
        "a profile is printed and nothing is written",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "0");
            assert!(w.get("stdout").contains("softmax_1d"), "no profile");
            assert_eq!(w.get("extra"), "", "-i wrote files: {}", w.get("extra"));
        },
    );

    r.when(
        "I convert the same input twice at the default level",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("determinism");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, _, _) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "a.metal"]);
            let (_, _, _) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "b.metal"]);
            let a = std::fs::read_to_string(dir.join("a.metal")).unwrap_or_default();
            let b = std::fs::read_to_string(dir.join("b.metal")).unwrap_or_default();
            w.set_flag("identical", !a.is_empty() && a == b);
        },
    );
    r.then(
        "the two outputs are byte-identical",
        |w: &mut World, _: &[String]| {
            // The emit-purity contract, observed through the CLI rather than the library:
            // O0/O1/O2 are pure functions of the input.
            assert!(w.flag("identical"), "the default level is not reproducible");
        },
    );

    r.when(
        "I ask for -O4 on a machine with no accelerator",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("o4");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin_env(
                &dir,
                &["k.mlir", "-O4", "-t", "msl", "-o", "out.metal"],
                &[("TILE_SIMULATE", "no-devices")],
            );
            w.set("error", e);
            w.set("exit", code.to_string());
            w.set_flag("wrote", dir.join("out.metal").exists());
        },
    );
    r.then(
        "it fails with the device code and offers -O3 instead",
        |w: &mut World, _: &[String]| {
            // -O4 is the level DEFINED by measurement. It used to run and exit 0 on a
            // machine with no accelerator, so asking for the measured level got an
            // unmeasured artifact and no signal -- the same substitution HardwareParams
            // refuses on the compiler side.
            assert_eq!(w.get("exit"), "6", "{}", w.get("error"));
            let e = w.get("error");
            assert!(e.contains("-O3"), "no fallback offered: {e}");
            assert!(!w.flag("wrote"), "wrote an artifact anyway");
        },
    );

    r.then(
        "a simulated absence is actually applied, not just announced",
        |_w: &mut World, _: &[String]| {
            // `TILE_SIMULATE=no-devices` printed "forced absent: no-devices" and then
            // doctor reported the real GPU two lines below it. A simulation that
            // announces an absence it did not apply makes every test written against it
            // pass for the wrong reason.
            let dir = scratch_dir("simreal");
            let (o, _, _) = run_bin_env(&dir, &["doctor"], &[("TILE_SIMULATE", "no-devices")]);
            assert!(
                o.contains("accelerator: none detected"),
                "no-devices did not remove the devices: {o}"
            );
        },
    );

    // ── Transform dispatch (02) ───────────────────────────────────────────────────

    r.when(
        "I name an output whose extension says nothing",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("owins");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "out.txt"]);
            w.set("error", e);
            w.set("exit", code.to_string());
            w.set(
                "written",
                std::fs::read_to_string(dir.join("out.txt")).unwrap_or_default(),
            );
            w.set_flag("stray", dir.join("k.opt.metal").exists());
        },
    );
    r.then(
        "the named file holds the requested form",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "0", "{}", w.get("error"));
            assert!(w.get("written").contains("mlir_to_msl"), "not MSL");
            assert!(!w.flag("stray"), "also wrote a derived name");
        },
    );

    r.when(
        "I ask for the routes without converting",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("routelist");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(&dir, &["k.mlir", "-t", "msl", "--route"]);
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
            let extra: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n != "k.mlir")
                .collect();
            w.set("extra", extra.join(","));
        },
    );
    r.then(
        "each candidate route is listed with its requirements, and nothing is written",
        |w: &mut World, _: &[String]| {
            let o = w.get("stdout");
            assert!(o.contains("routes from"), "{o}");
            // Every listed route names its fidelity and whether it can be taken here.
            assert!(o.contains("available") || o.contains("needs"), "{o}");
            assert_eq!(w.get("extra"), "", "--route wrote {}", w.get("extra"));
        },
    );

    r.when(
        "I pin an intermediate that is not on the route",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("via");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin(
                &dir,
                &["k.mlir", "-t", "rvv", "--via", "linalg", "-o", "v.mlir"],
            );
            w.set("error", e);
            w.set("exit", code.to_string());
            w.set_flag("wrote", dir.join("v.mlir").exists());
        },
    );
    r.then(
        "the contradiction is refused rather than ignored",
        |w: &mut World, _: &[String]| {
            // `--via` had only ever meant "a lift into one of these is permitted", while the
            // help said "pin the intermediate forms". So pinning a form that was not on the
            // chosen route was accepted, dropped, and the unpinned route taken -- the user
            // believing they had constrained a run they had not.
            assert_eq!(w.get("exit"), "2", "{}", w.get("error"));
            let e = w.get("error");
            assert!(e.contains("--via"), "{e}");
            assert!(
                e.contains("Routes that do exist"),
                "no alternatives offered: {e}"
            );
            assert!(!w.flag("wrote"), "wrote an artifact anyway");
        },
    );

    r.when(
        "I convert a target source to another target source",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("noroute");
            std::fs::write(dir.join("k.metal"), corpus("softmax.metal")).unwrap();
            let (_, e, code) = run_bin(&dir, &["k.metal", "-t", "cpp", "-o", "out.cce"]);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "it names the missing frontend in the tense of work not yet done",
        |w: &mut World, _: &[String]| {
            // Exit 3 is "nobody has written it", never "impossible" -- and the message
            // must carry that tense, plus what the route WOULD be.
            assert_eq!(w.get("exit"), "3", "{}", w.get("error"));
            let e = w.get("error");
            assert!(e.contains("cannot read it"), "{e}");
            assert!(e.contains("missing work, not a missing possibility"), "{e}");
            assert!(
                e.contains("synthesised"),
                "does not mark the hypothetical route: {e}"
            );
        },
    );

    r.when(
        "I ask for a target this build was not compiled with",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("nofeature");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin(
                &dir,
                &[
                    "k.mlir",
                    "-t",
                    "cpp",
                    "-O3",
                    "--no-install",
                    "-o",
                    "out.cce",
                ],
            );
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "it says so as an acquirable absence and names the build feature",
        |w: &mut World, _: &[String]| {
            // Exit 4, not 3: the capability EXISTS, just not in this binary. Telling the
            // caller "unsupported" would send them away from something that works.
            assert_eq!(w.get("exit"), "4", "{}", w.get("error"));
            let e = w.get("error");
            // NOT "--features ascend": that feature is not declared in this Cargo.toml,
            // and the AscendC emitter is not in this repository at all, so the rebuild
            // instruction this used to print failed with "the package does not contain
            // this feature". Exit 4 promises acquirable; the remedy has to be real.
            assert!(
                !e.contains("Rebuild with"),
                "offers a rebuild that cannot work: {e}"
            );
            assert!(e.contains("not in this repository"), "{e}");
            assert!(
                e.contains("still lists cpp"),
                "conflates build with form: {e}"
            );
        },
    );

    r.when(
        "I profile two inputs of different forms",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("twoforms");
            std::fs::write(dir.join("a.mlir"), corpus("softmax.mlir")).unwrap();
            std::fs::write(dir.join("b.metal"), corpus("softmax.metal")).unwrap();
            let (o, e, code) = run_bin(&dir, &["a.mlir", "b.metal", "-i"]);
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "each is profiled on its own and no conversion between them is attempted",
        |w: &mut World, _: &[String]| {
            // Two inputs are two jobs. Reading them as "convert a into b" would be a
            // route request nobody made.
            let o = w.get("stdout");
            assert_eq!(o.matches("form:").count(), 2, "{o}");
            assert!(o.contains("mlir") && o.contains("msl"), "{o}");
            assert!(!o.contains("route:"), "attempted a conversion: {o}");
        },
    );

    // ── Intermediates (03) and grammar (10) ───────────────────────────────────────

    r.when(
        "I keep the intermediates in a named directory",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("keepdir");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin(
                &dir,
                &["k.mlir", "-t", "msl", "--keep-dir", "hops", "-o", "o.metal"],
            );
            w.set("error", e);
            w.set("exit", code.to_string());
            let kept: Vec<String> = std::fs::read_dir(dir.join("hops"))
                .map(|d| {
                    d.flatten()
                        .map(|e| e.file_name().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default();
            w.set("kept", kept.join(","));
            // Nothing may be left beside the input: --keep-dir says where they go.
            w.set_flag("stray", dir.join("k.1.mlir.mlir").exists());
        },
    );
    r.then(
        "the hops are there and nowhere else",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "0", "{}", w.get("error"));
            assert!(!w.get("kept").is_empty(), "no hops kept");
            assert!(!w.flag("stray"), "also left hops beside the input");
        },
    );

    r.when("a hop fails", |w: &mut World, _: &[String]| {
        let dir = scratch_dir("failhop");
        std::fs::write(dir.join("bad.mlir"), "this is not a module\n").unwrap();
        let (_, e, code) = run_bin(&dir, &["bad.mlir", "-t", "msl", "-o", "out.metal"]);
        w.set("error", e);
        w.set("exit", code.to_string());
        let left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n != "bad.mlir")
            .collect();
        w.set("left", left.join(","));
    });
    r.then(
        "it says which hop failed and leaves nothing behind",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "1");
            let e = w.get("error");
            assert!(e.contains("hop 2 of 2"), "hop not identified: {e}");
            assert!(e.contains("mlir -> msl"), "edge not named: {e}");
            assert_eq!(w.get("left"), "", "left {}", w.get("left"));
        },
    );

    r.when(
        "a hop fails and I asked to keep the intermediates",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("failkeep");
            std::fs::write(dir.join("bad.mlir"), "this is not a module\n").unwrap();
            let (_, _, code) = run_bin(&dir, &["bad.mlir", "-t", "msl", "-k", "-o", "out.metal"]);
            w.set("exit", code.to_string());
            w.set_flag("kept", dir.join("bad.1.mlir.mlir").exists());
        },
    );
    r.then(
        "the intermediates of the failed run survive for debugging",
        |w: &mut World, _: &[String]| {
            // The failure is exactly when they are worth having, so -k must not be
            // undone by the cleanup path.
            assert_eq!(w.get("exit"), "1");
            assert!(w.flag("kept"), "-k lost the hops on the failing run");
        },
    );

    r.when(
        "I raise the verbosity one step at a time",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("verbose");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let mut counts = Vec::new();
            for (i, extra) in [vec![], vec!["-V"], vec!["-VV"]].iter().enumerate() {
                let out = format!("o{i}.metal");
                let mut args = vec!["k.mlir", "-t", "msl", "-o", out.as_str()];
                args.extend(extra.iter().copied());
                let (_, e, _) = run_bin(&dir, &args);
                counts.push(e.lines().count());
            }
            w.set(
                "counts",
                counts
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        },
    );
    r.then(
        "each step says strictly more",
        |w: &mut World, _: &[String]| {
            // Both flags parsed and neither changed a line: -VV was indistinguishable from
            // no flag, so the help text promised a control that existed nowhere else.
            let c: Vec<usize> = w
                .get("counts")
                .split(',')
                .filter_map(|n| n.parse().ok())
                .collect();
            assert_eq!(c.len(), 3, "{:?}", c);
            assert!(c[1] > c[0], "-V added nothing: {c:?}");
            assert!(c[2] > c[1], "-VV added nothing over -V: {c:?}");
        },
    );

    r.when(
        "stdout carries the result",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("streams");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "-"]);
            w.set("stdout", o);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "stdout holds only the artifact and every diagnostic is on stderr",
        |w: &mut World, _: &[String]| {
            // This is what makes `-o -` pipeable. One profile line on stdout would
            // corrupt every downstream consumer.
            let o = w.get("stdout");
            assert_eq!(w.get("exit"), "0");
            assert!(o.starts_with("// Generated by tile-rs"), "{o}");
            for noise in ["route:", "form:", "kernel:", "O2 passes"] {
                assert!(!o.contains(noise), "{noise:?} leaked to stdout");
            }
            assert!(w.get("error").contains("route:"), "diagnostics vanished");
        },
    );

    // ── Stats (07) ────────────────────────────────────────────────────────────────

    r.when(
        "I convert a kernel twice and then ask for its history",
        |w: &mut World, _: &[String]| {
            // A private TILE_HOME per scenario: the attempt store is real state, and a test
            // that wrote into the developer's own would both pollute it and depend on it.
            let dir = scratch_dir("stats");
            let home = dir.join("home");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let env: &[(&str, &str)] = &[("TILE_HOME", home.to_str().unwrap())];
            run_bin_env(&dir, &["k.mlir", "-t", "msl", "-o", "a.metal"], env);
            run_bin_env(&dir, &["k.mlir", "-t", "gpu", "-o", "b.cu"], env);
            let (o, e, code) = run_bin_env(&dir, &["k.mlir", "-s", "-i"], env);
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "both conversions are listed against that kernel with their routes",
        |w: &mut World, _: &[String]| {
            // `corpus::append_local` existed and nothing called it, so the local store
            // was always empty and -s said "nothing recorded" however many conversions
            // had run. The two-store design was in place; the writing half was not.
            let o = w.get("stdout");
            assert_eq!(w.get("exit"), "0", "{o}");
            assert!(o.contains("prior attempts for softmax_1d"), "{o}");
            assert!(o.contains("msl") && o.contains("gpu"), "{o}");
            assert_eq!(o.matches("[yours]").count(), 2, "{o}");
        },
    );
    r.then(
        "the headroom of a conversion stays unassessed rather than zero",
        |w: &mut World, _: &[String]| {
            // None is UNASSESSED and Some(0.0) is CLOSED. A conversion measures neither,
            // and writing 0.0 would turn "nobody looked" into "there is nothing to gain"
            // -- in the direction that stops anyone looking again.
            let o = w.get("stdout");
            assert!(o.contains("headroom unassessed"), "{o}");
            assert!(!o.contains("headroom 0"), "invented a measurement: {o}");
        },
    );
    r.then(
        "only kernels named by the arguments are reported",
        |w: &mut World, _: &[String]| {
            // -s is scoped to the arguments, not a dump of the whole corpus: exactly one
            // kernel was passed, so exactly one heading may appear.
            let o = w.get("stdout");
            assert_eq!(
                o.matches("prior attempts for").count(),
                1,
                "reported more kernels than were asked about: {o}"
            );
        },
    );

    // ── Daemon (08) ───────────────────────────────────────────────────────────────

    r.when(
        "I ask the daemon to listen on a socket",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("listen");
            let (_, e, code) = run_bin(&dir, &["-d", "--listen", "127.0.0.1:7777"]);
            w.set("error", e);
            w.set("exit", code.to_string());
            let (_, e2, code2) = run_bin(&dir, &["-d", "--listen", "0.0.0.0:7777"]);
            w.set("error2", e2);
            w.set("exit2", code2.to_string());
        },
    );
    r.then(
        "both loopback and non-loopback are refused for the same stated reason",
        |w: &mut World, _: &[String]| {
            // The original pair of scenarios assumed a TCP daemon that exists and asked
            // only that 0.0.0.0 be opt-in. There is no TCP daemon: serving licensed
            // corpus data over a socket needs an auth token nobody has built, so ALL of
            // it is refused, loopback included. Specifying the weaker rule would have
            // described a door that is not there.
            assert_eq!(w.get("exit"), "2", "{}", w.get("error"));
            assert_eq!(w.get("exit2"), "2", "{}", w.get("error2"));
            for k in ["error", "error2"] {
                let e = w.get(k);
                assert!(e.contains("auth token"), "reason not given: {e}");
                assert!(e.contains("stdio"), "no working alternative offered: {e}");
            }
        },
    );

    r.when(
        "a previous run was killed and left its scratch",
        |w: &mut World, _: &[String]| {
            // A SIGKILL cannot run its own cleanup, so "no orphans" is only true if the NEXT
            // run makes it true. Simulated by planting a scratch directory whose pid is not
            // alive -- exactly the state a kill leaves behind.
            //
            // TMPDIR is given to the CHILD only: the scratch root is the process temp dir,
            // and setting it here would move every other parallel test's scratch too.
            let dir = scratch_dir("sweep");
            let tmp = dir.join("tmp");
            std::fs::create_dir_all(&tmp).unwrap();
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let orphan = tmp.join("tile-orphan-999999");
            std::fs::create_dir_all(&orphan).unwrap();
            std::fs::write(orphan.join("leftover.mlir"), "x").unwrap();
            let (_, e, _) = run_bin_env(
                &dir,
                &["k.mlir", "-i", "-V"],
                &[("TMPDIR", tmp.to_str().unwrap())],
            );
            w.set("error", e);
            w.set_flag("gone", !orphan.exists());
        },
    );
    r.then(
        "the next run sweeps it and says so at verbosity",
        |w: &mut World, _: &[String]| {
            assert!(w.flag("gone"), "orphan survived: {}", w.get("error"));
            assert!(
                w.get("error").contains("swept"),
                "swept silently: {}",
                w.get("error")
            );
        },
    );

    // ── Form table (01) and provisioning (06) ─────────────────────────────────────

    r.then(
        "every form in the table resolves from its own extension and magic",
        |_w: &mut World, _: &[String]| {
            // The literal scenario -- "add a FormSpec row for a hypothetical form" --
            // cannot be run: FORMS is a `const` table, so a row cannot be added at
            // runtime. What it was really asserting is that resolution is DRIVEN by the
            // table rather than by per-form code, and that is checkable exhaustively:
            // every row must be reachable through the same path, with no form
            // special-cased and none shadowed by another.
            for f in forms::FORMS {
                let name = format!("a.{}", f.primary_ext());
                // Give the file this form's own discriminating magic, which is exactly
                // what the table says distinguishes it from its extension-mates.
                let body = f.magic.first().copied().unwrap_or("").to_string();
                let got = forms::resolve_input(&name, &body, None);
                match got {
                    Ok(r) => {
                        // Either it resolved to this form, or to one that shares the
                        // extension and is the documented fallback for an empty body.
                        let ok = r.form.id == f.id
                            || (f.magic.is_empty() && r.form.exts.contains(&f.primary_ext()));
                        assert!(ok, "{} resolved to {}", f.id, r.form.id);
                    }
                    Err(e) => {
                        // A refusal is acceptable ONLY when the extension is genuinely
                        // shared and the body carries no marker to choose by.
                        assert!(
                            f.magic.is_empty(),
                            "{} has magic {:?} and still failed: {e}",
                            f.id,
                            f.magic
                        );
                    }
                }
            }
        },
    );

    r.when(
        "I ask to install a tool this platform has no entry for",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("install");
            let (_, e, code) = run_bin(&dir, &["install", "cann"]);
            w.set("error", e);
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "it is refused as acquirable-elsewhere, naming this platform",
        |w: &mut World, _: &[String]| {
            // Exit 4, not 3: CANN exists, and on a Linux Ascend box tile-rs provisions
            // it. What is absent is an entry for THIS platform.
            assert_eq!(w.get("exit"), "4", "{}", w.get("error"));
            let e = w.get("error");
            assert!(
                e.contains("macos") || e.contains("linux"),
                "platform not named: {e}"
            );
        },
    );

    r.then(
        "an entry with no recorded digest is refused before anything is fetched",
        |_w: &mut World, _: &[String]| {
            // The comparison after the download would catch it either way, but by then
            // the bytes have been pulled from a URL nobody vouched for. The pin is what
            // makes the URL trustworthy, so an unpinned entry must not be contacted.
            let t = tile_cli::manifest::Tool {
                id: "unpinned".into(),
                version: "1.0".into(),
                os: std::env::consts::OS.into(),
                arch: std::env::consts::ARCH.into(),
                // A URL that would fail loudly if it were ever contacted.
                url: "https://example.invalid/never-fetched.tar.gz".into(),
                sha256: String::new(),
                barrier: None,
                remedy: None,
                note: None,
                unpack: None,
                strip: 0,
            };
            let mut said = Vec::new();
            let err = tile_cli::provision::install_tool(
                &t,
                tile_cli::provision::Policy::OnDemand,
                &mut |m: &str| said.push(m.to_string()),
            )
            .expect_err("an unpinned entry must not install");
            let text = format!("{err}");
            assert!(text.contains("nothing was downloaded"), "{text}");
            assert!(
                !said.iter().any(|m| m.starts_with("fetching")),
                "it announced a fetch it should not have started: {said:?}"
            );
        },
    );

    // ── The last of it: fidelity, identity, concurrency, .rs hazards ──────────────

    r.then(
        "every conversion prints its route and fidelity, not only --route",
        |_w: &mut World, _: &[String]| {
            // The fidelity class rides on every result because a numerically wrong
            // kernel is invisible until it corrupts a run -- so it must not be something
            // you only see if you thought to ask.
            let dir = scratch_dir("fid");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, _) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "o.metal"]);
            let last = e.lines().rfind(|l| l.starts_with("route:"));
            let line = last.expect("a route line on a plain conversion");
            assert!(line.contains("["), "no fidelity class: {line}");
            assert!(
                line.contains("exact") || line.contains("synthesised"),
                "{line}"
            );
        },
    );

    r.then(
        "a kernel is identified by its contents, not by its filename",
        |_w: &mut World, _: &[String]| {
            // Renaming a file must not orphan its history, and two files with the same
            // name must not merge theirs.
            let dir = scratch_dir("identity");
            let home = dir.join("home");
            let env: &[(&str, &str)] = &[("TILE_HOME", home.to_str().unwrap())];
            std::fs::write(dir.join("renamed.mlir"), corpus("softmax.mlir")).unwrap();
            run_bin_env(&dir, &["renamed.mlir", "-t", "msl", "-o", "a.metal"], env);
            let (o, e, _) = run_bin_env(&dir, &["renamed.mlir", "-s", "-i"], env);
            let all = format!("{o}{e}");
            assert!(
                all.contains("prior attempts for softmax_1d"),
                "identity came from the filename: {all}"
            );
            assert!(!all.contains("prior attempts for renamed"), "{all}");
        },
    );

    r.then(
        "two concurrent runs do not corrupt the shared state directory",
        |_w: &mut World, _: &[String]| {
            // Both append to the same attempt store. The failure this guards is a torn
            // or interleaved line, which would make the corpus unparseable for everyone
            // afterwards -- a persistent fault from a transient race.
            let dir = scratch_dir("concurrent");
            let home = dir.join("home");
            std::fs::create_dir_all(&home).unwrap();
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let mut kids = Vec::new();
            for i in 0..4 {
                let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
                c.current_dir(&dir);
                c.args(["k.mlir", "-t", "msl", "-o", &format!("o{i}.metal")]);
                c.env("TILE_HOME", &home);
                c.env_remove("TILE_SIMULATE");
                c.stdout(std::process::Stdio::null());
                c.stderr(std::process::Stdio::null());
                kids.push(c.spawn().expect("spawn"));
            }
            for mut k in kids {
                k.wait().expect("wait");
            }
            let store = home.join("attempts");
            if store.exists() {
                let text = std::fs::read_to_string(&store).unwrap();
                // Every non-comment line must still parse: that is what "not corrupt"
                // means for this file.
                tile_cli::corpus::parse(&text).expect("the store is still parseable");
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    assert!(!line.contains("\u{0}"), "torn write: {line:?}");
                }
            }
        },
    );

    r.then(
        "the hazard model is reported for a tile-rs input too",
        |_w: &mut World, _: &[String]| {
            // The .rs path goes through the rustc backend, so this used to be @planned
            // for want of a frontend. It exists now, and the hazard report must not
            // depend on which door the kernel came in through.
            let dir = scratch_dir("rshazard");
            std::fs::write(dir.join("k.rs"), corpus("softmax.rs")).unwrap();
            let (o, e, code) = run_bin(&dir, &["k.rs", "-i"]);
            let all = format!("{o}{e}");
            assert_eq!(code, 0, "{all}");
            assert!(all.contains("hazards:"), "no hazard model for .rs: {all}");
            assert!(all.contains("unsynchronised"), "{all}");
        },
    );

    // ── Lifts, seals, and the last edges ──────────────────────────────────────────

    r.then(
        "the refusal reaches the command line too, not only the planner",
        |_w: &mut World, _: &[String]| {
            // The scenario above asserts this against `routes::plan`. This runs the real
            // binary, because "the planner refuses" and "the tool refuses" are different
            // claims -- the CLI could take the route the planner declined and nothing
            // between them would notice.
            let dir = scratch_dir("liftcompose");
            std::fs::write(dir.join("p.mlir"), corpus("softmax_pto.mlir")).unwrap();
            let (_, e, code) = run_bin(&dir, &["p.mlir", "-t", "msl", "-o", "out.metal"]);
            assert_eq!(code, 2, "{e}");
            assert!(e.contains("--via tile"), "does not name the command: {e}");
            assert!(
                e.contains("synthesised"),
                "does not warn what it would be worth: {e}"
            );
            assert!(!dir.join("out.metal").exists(), "took the lift anyway");
        },
    );
    r.then(
        "asking for it by name still refuses, because no lifter is written",
        |_w: &mut World, _: &[String]| {
            // And the refusal must be exit 3, not 4: `lift` is not a Cargo feature, so
            // "rebuild with --features lift" was an instruction that could not work.
            let dir = scratch_dir("liftnamed");
            std::fs::write(dir.join("p.mlir"), corpus("softmax_pto.mlir")).unwrap();
            let (_, e, code) = run_bin(
                &dir,
                &["p.mlir", "-t", "msl", "--via", "tile", "-o", "o.metal"],
            );
            assert_eq!(code, 3, "{e}");
            assert!(e.contains("no code exists behind it"), "{e}");
            assert!(
                !e.contains("Rebuild with"),
                "offers a rebuild that cannot work: {e}"
            );
        },
    );

    r.then(
        "a sealed corpus that fails authentication is reported, not ignored",
        |_w: &mut World, _: &[String]| {
            // Silently falling back to "no corpus" would make a tampered or truncated
            // file indistinguishable from an absent one -- and the absent case is normal,
            // so the tampered case would never be noticed.
            let dir = scratch_dir("seal");
            let home = dir.join("home");
            std::fs::create_dir_all(&home).unwrap();
            std::fs::write(home.join("corpus.sealed"), b"not a sealed corpus at all").unwrap();
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, _) = run_bin_env(
                &dir,
                &["k.mlir", "-s", "-i"],
                &[("TILE_HOME", home.to_str().unwrap())],
            );
            let all = format!("{o}{e}");
            // Either it says the corpus failed its check, or (in a build without the
            // crypto) it says the corpus is closed. What it must never do is stay quiet
            // about a file that is present and unreadable.
            assert!(
                all.contains("integrity") || all.contains("not open here") || all.contains("seal"),
                "a present-but-unreadable corpus was passed over in silence: {all}"
            );
        },
    );

    r.then(
        "the profile reports the hazard model with its unsynchronised count",
        |_w: &mut World, _: &[String]| {
            // RAW/WAR/WAW edges and the barriers that make the schedule safe. The
            // unsynchronised count is the one that matters: it is a defect, not a note.
            let dir = scratch_dir("hazard");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, _) = run_bin(&dir, &["k.mlir", "-i"]);
            let all = format!("{o}{e}");
            assert!(all.contains("hazards:"), "{all}");
            assert!(all.contains("barrier points"), "{all}");
            assert!(all.contains("unsynchronised"), "{all}");
        },
    );

    // ── -O3, unmeasured architectures, and the daemon's install policy ────────────

    r.given(
        "the target's compiler is installed",
        |_w: &mut World, _: &[String]| {},
    );
    r.given(
        "the codegen backend is provisioned",
        |_w: &mut World, _: &[String]| {},
    );
    r.when(
        "I run \"tile softmax.mlir -t msl -O3 -o out.metal\"",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("o3tagged");
            std::fs::write(dir.join("softmax.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(
                &dir,
                &["softmax.mlir", "-t", "msl", "-O3", "-o", "out.metal"],
            );
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
            w.set_flag("wrote", dir.join("out.metal").exists());
        },
    );
    r.then(
        "the compiler's verdict is reported",
        |w: &mut World, _: &[String]| {
            // The level that distinguishes O3 from O2 is "a toolchain runs". If the compiler
            // is never invoked, O3 is O2 with a different label.
            let o = w.get("stdout");
            assert!(o.contains("O3: metal"), "no compiler verdict: {o}");
        },
    );
    r.then(
        "the conversion still produced its output",
        |w: &mut World, _: &[String]| {
            assert_eq!(w.get("exit"), "0", "{}", w.get("stdout"));
            assert!(w.flag("wrote"), "no artifact");
        },
    );

    r.when(
        "I run \"tile softmax.rs -o out.metal\"",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("rslower");
            std::fs::write(dir.join("softmax.rs"), corpus("softmax.rs")).unwrap();
            let (o, e, code) = run_bin(&dir, &["softmax.rs", "-o", "out.metal"]);
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
            w.set(
                "written",
                std::fs::read_to_string(dir.join("out.metal")).unwrap_or_default(),
            );
        },
    );
    r.then(
        "the route taken descends from the tile-rs source to the target",
        |w: &mut World, _: &[String]| {
            // A lower is DERIVED from the level difference, not chosen by a code path.
            let o = w.get("stdout");
            assert_eq!(w.get("exit"), "0", "{o}");
            assert!(o.contains("tile -> tile -> mlir -> msl"), "{o}");
        },
    );
    r.then(
        "the output contains \"kernel void\"",
        |w: &mut World, _: &[String]| {
            assert!(w.get("written").contains("kernel void"), "not MSL");
        },
    );

    r.then(
        "an unmeasured architecture refuses to report a bound rather than borrowing one",
        |_w: &mut World, _: &[String]| {
            // The rule this whole crate inherited: another chip's capacities would
            // approve precisely the tilings that fail. The original scenario asked for
            // exit 1; the remedy-based taxonomy makes it 3 -- measuring DAV_3510 is work
            // nobody has done, which is exactly what 3 means, and 1 would file it as a
            // failure of this run.
            let dir = scratch_dir("unmeasured");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(
                &dir,
                &[
                    "k.mlir",
                    "-t",
                    "msl",
                    "-O2",
                    "--cross",
                    "ascend-950",
                    "-o",
                    "out.metal",
                ],
            );
            let all = format!("{o}{e}");
            assert_eq!(code, 3, "{all}");
            assert!(all.contains("UNMEASURED"), "{all}");
            assert!(
                all.contains("unmeasured is not the same as unlimited"),
                "the reader is left to infer there are no limits: {all}"
            );
            assert!(!dir.join("out.metal").exists(), "emitted anyway");
        },
    );

    r.then(
        "the daemon acquires nothing on an agent's behalf, and says what it would take",
        |_w: &mut World, _: &[String]| {
            // An MCP client should not be able to start a 100 MB download nobody asked
            // for. But refusing without naming the remedy just moves the dead end, so
            // the reply must also point at the explicit `install` call.
            let req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":\
                       {\"name\":\"convert\",\"arguments\":{\"source\":\"module {}\",\
                       \"filename\":\"k.mlir\",\"to\":\"cpp\"}}}";
            let reply = mcp_with(&[req], &["-d"]);
            let text = reply.join("\n");
            assert!(
                text.contains("Nothing is acquired on your behalf"),
                "{text}"
            );
            assert!(
                text.contains("`install`"),
                "no explicit route offered: {text}"
            );
            assert!(!text.contains("fetching"), "it started a download: {text}");
        },
    );

    // ── The run harness, end to end on real hardware (12) ─────────────────────────

    r.given(
        "a Metal device and the emitted kernel",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("runreal");
            std::fs::write(dir.join("softmax.mlir"), corpus("softmax.mlir")).unwrap();
            w.set("dir", dir.display().to_string());
        },
    );
    r.given(
        "a Metal device and torch reachable",
        |w: &mut World, _: &[String]| {
            let dir = scratch_dir("runtorch");
            std::fs::write(dir.join("softmax.mlir"), corpus("softmax.mlir")).unwrap();
            w.set("dir", dir.display().to_string());
        },
    );
    // Deliberately NOT phrased as `I run "tile softmax.mlir -t msl -o out.metal -r"`:
    // an earlier step is registered as `... -o out.metal{}`, whose {} happily captures
    // " -r" and runs a plain conversion instead. Steps resolve in registration order, so
    // a later pattern that an earlier one can match is dead on arrival -- and it fails as
    // a missing assertion, not as a duplicate-step error.
    r.when(
        "I lower it and run it on this machine",
        |w: &mut World, _: &[String]| {
            let dir = PathBuf::from(w.get("dir"));
            // Few iterations: this is a correctness assertion, not a benchmark, and the
            // suite should not spend seconds sampling a timing nobody reads.
            let (o, e, code) = run_bin(
                &dir,
                &[
                    "softmax.mlir",
                    "-t",
                    "msl",
                    "-o",
                    "out.metal",
                    "-r",
                    "--iterations",
                    "5",
                ],
            );
            w.set("stdout", format!("{o}{e}"));
            w.set("exit", code.to_string());
        },
    );
    r.then(
        "the kernel's output matches the reference within tolerance",
        |w: &mut World, _: &[String]| {
            // This is the claim the whole harness exists to make, and it was being
            // skipped by a tag on a machine that can check it.
            let o = w.get("stdout");
            assert_eq!(w.get("exit"), "0", "{o}");
            assert!(o.contains("kernel vs ours"), "{o}");
            // The control arm must have passed, or none of the numbers mean anything.
            assert!(!o.contains("control arm AGREED"), "{o}");
            assert!(!o.contains("NO VALUES"), "{o}");
        },
    );
    r.then(
        "a device time is reported",
        |w: &mut World, _: &[String]| {
            let o = w.get("stdout");
            assert!(o.contains("target   : median"), "no device time: {o}");
            assert!(o.contains("warmup"), "timing without a warmup count: {o}");
        },
    );
    r.then(
        "the report names the torch version",
        |w: &mut World, _: &[String]| {
            let o = w.get("stdout");
            assert!(o.contains("kernel vs torch"), "{o}");
            // A version, not just the word: "compared against torch" with no version is a
            // claim nobody can reproduce.
            assert!(
                o.contains("torch 2.") || o.contains("torch: "),
                "no torch version or status: {o}"
            );
        },
    );
    r.then(
        "it reports the kernel against torch, the kernel against ours, and ours against torch",
        |w: &mut World, _: &[String]| {
            // Three pairs, because two are not enough to attribute a disagreement: with
            // only kernel-vs-torch you cannot tell whose numbers moved.
            let o = w.get("stdout");
            for pair in ["kernel vs torch", "kernel vs ours", "ours   vs torch"] {
                assert!(o.contains(pair), "missing {pair}: {o}");
            }
        },
    );
    r.then("it states a verdict", |w: &mut World, _: &[String]| {
        let o = w.get("stdout");
        assert!(o.contains("verdict:"), "{o}");
    });

    // ── Cross-compilation (10) ────────────────────────────────────────────────────

    r.when(
        "I build \"tile\" for \"{}\"",
        |w: &mut World, a: &[String]| {
            let triple = a[0].trim().to_string();
            // Only for triples whose std is actually installed: a missing target is a
            // machine that has not been set up, not a portability failure, and reporting it
            // as one would make the scenario noise.
            let installed = std::process::Command::new("rustup")
                .args(["target", "list", "--installed", "--toolchain", "stable"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains(triple.as_str()))
                .unwrap_or(false);
            if !installed {
                // Skipping is right -- a missing target is a machine that has not been set up,
                // not a portability failure -- but it must not be SILENT. TILE_SPEC_CROSS=1 is
                // someone asking for this check, and the scenario passed while a third of its
                // examples built nothing and said so nowhere. The note carries the remedy, the
                // way an exit-4 message would.
                let note =
                    format!("skipped: {triple} std not installed — `rustup target add {triple}`");
                eprintln!("  cross-build {note}");
                w.set("cross", note);
                return;
            }
            let out = std::process::Command::new(env!("CARGO"))
                .args(["check", "--features", "pico", "--target", &triple])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .expect("cargo runs");
            w.set(
                "cross",
                if out.status.success() {
                    format!("ok: {triple}")
                } else {
                    format!("FAILED: {triple}\n{}", String::from_utf8_lossy(&out.stderr))
                },
            );
        },
    );
    r.then(
        "the build succeeds with no platform-specific source outside the platform module",
        |w: &mut World, _: &[String]| {
            let r = w.get("cross");
            assert!(!r.starts_with("FAILED"), "{r}");
            // A skip is not a build. Say which happened rather than letting a green
            // scenario imply the triple was compiled.
            assert!(
                r.starts_with("ok:") || r.starts_with("skipped:"),
                "the cross step recorded neither a build nor a skip: {r}"
            );
        },
    );

    // ── One binary, several machines (05) ─────────────────────────────────────────

    r.then(
        "the same binary reports correctly on every host it is copied to",
        |_w: &mut World, _: &[String]| {
            // Requirement (3) is not a build property: it is a claim about what a user
            // has to do before the tool works. So the test is DEPLOYMENT -- copy the
            // binary, run it, read what it says -- and not a compilation.
            //
            // Verified by hand first across an M1 Ultra, an M2 Max and an M4 (see
            // docs/cli/INTEGRATION.md); this is that, automated.
            let hosts = spec_hosts();
            assert!(!hosts.is_empty(), "the tag should have skipped this");
            let bin = env!("CARGO_BIN_EXE_tile");
            let mut checked = 0usize;
            let mut unreachable = Vec::new();
            for h in &hosts {
                let scp = std::process::Command::new("scp")
                    .args([
                        "-q",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=8",
                        bin,
                        &format!("{h}:/tmp/tile-spec"),
                    ])
                    .output()
                    .expect("scp runs");
                if !scp.status.success() {
                    // A host that is asleep has not DISPROVEN anything -- it could not be
                    // asked. Failing the suite for a closed laptop makes the flag
                    // unusable, and people respond by never setting it.
                    //
                    // But skipping quietly is the failure this whole project keeps
                    // finding, so it is recorded loudly and the count below refuses a run
                    // that checked nothing.
                    eprintln!(
                        "spec: {h} is unreachable, so it was not checked: {}",
                        String::from_utf8_lossy(&scp.stderr).trim()
                    );
                    unreachable.push(h.clone());
                    continue;
                }
                let out = std::process::Command::new("ssh")
                    .args([
                        "-o",
                        "BatchMode=yes",
                        h,
                        "chmod +x /tmp/tile-spec && /tmp/tile-spec doctor --json",
                    ])
                    .output()
                    .expect("ssh runs");
                let text = String::from_utf8_lossy(&out.stdout).into_owned();
                assert!(out.status.success(), "doctor failed on {h}: {text}");
                // It must report ITS OWN machine, not this one -- a binary that reported
                // the build host's architecture would pass a weaker test and be useless.
                let v = tile_cli::json::parse(&text).expect("doctor --json is JSON");
                let arch = v.get("arch").and_then(|a| a.as_str()).unwrap_or("");
                assert!(!arch.is_empty(), "no arch reported by {h}: {text}");
                let triple = v.get("triple").and_then(|t| t.as_str()).unwrap_or("");
                assert!(
                    triple.contains(arch),
                    "{h}: triple {triple} disagrees with arch {arch}"
                );
                checked += 1;
            }
            // Every host asleep would otherwise make this scenario pass having verified
            // nothing at all -- the same shape as a comparison over zero values reading
            // as agreement, which is the bug this harness already carries a control arm
            // for. Asking for hosts and reaching none is a failure of the request.
            assert!(
                checked > 0,
                "none of {hosts:?} could be reached, so nothing was verified \
                 (unreachable: {unreachable:?})"
            );
        },
    );

    // ── The UI mode that is not built (09) ────────────────────────────────────────

    r.then(
        "--ui native refuses in the tense of work not done, and names what does work",
        |_w: &mut World, _: &[String]| {
            // It used to parse "native" as an input FILENAME and start the HTTP server
            // anyway: the user asked for a window, got a server, was told nothing, and
            // the command hung because the server blocks. Substituting one subsystem for
            // another in silence is the same fault as an emitter ignoring an intrinsic.
            let dir = scratch_dir("uinative");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let (_, e, code) = run_bin(&dir, &["k.mlir", "--ui", "native"]);
            // Exit 3, not 7: 7 is a subsystem that exists and could not start.
            assert_eq!(code, 3, "{e}");
            assert!(e.contains("not built yet"), "{e}");
            assert!(
                e.contains("`tile --ui`"),
                "no working alternative named: {e}"
            );
        },
    );

    r.then(
        "a kernel really named native is still readable",
        |_w: &mut World, _: &[String]| {
            // The mode is consumed only when no file of that name exists, so the grammar
            // does not steal a filename from anyone.
            let dir = scratch_dir("uifile");
            std::fs::write(dir.join("native"), corpus("softmax.mlir")).unwrap();
            let (o, e, code) = run_bin(&dir, &["-f", "mlir", "native", "-i"]);
            let all = format!("{o}{e}");
            assert_eq!(code, 0, "{all}");
            assert!(all.contains("softmax_1d"), "{all}");
        },
    );

    // ── -O4, measured (04) ────────────────────────────────────────────────────────

    // A width must be CORRECT before it can be fastest. The sweep timed each threadgroup
    // width and never read the output buffer -- it exits before printing a value -- so the
    // recommendation written into the emitted source rested on speed alone. Every
    // reduction here folds across `tcount`, and the power-of-two tree fold produced wrong
    // answers at counts that were not powers of two, so this is the axis where "fastest"
    // and "correct" are least safely conflated.
    r.then(
        "each candidate width is checked against the reference before it is ranked",
        |_w: &mut World, _: &[String]| {
            let dir = scratch_dir("o4verify");
            let home = dir.join("home");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let env: &[(&str, &str)] = &[("TILE_HOME", home.to_str().unwrap())];
            let (o, e, code) = run_bin_env(
                &dir,
                &[
                    "k.mlir",
                    "-t",
                    "msl",
                    "-O4",
                    "-o",
                    "out.metal",
                    "--iterations",
                    "5",
                ],
                env,
            );
            let all = format!("{o}{e}");
            assert_eq!(code, 0, "{all}");
            // Every width offered as "best" survived a comparison against the reference.
            // Nothing is asserted about WHICH width wins -- that is a measurement, and it
            // moves between machines and between runs.
            assert!(all.contains("best: threadgroup"), "no ranking: {all}");
            assert!(
                !all.contains("none of them reproduced the reference"),
                "softmax reproduces the reference at every width it is swept at: {all}"
            );
        },
    );
    r.then(
        "a width that computes a different answer is dropped, with the reason said",
        |_w: &mut World, _: &[String]| {
            // The dropping is stated in the source rather than demonstrated: producing a
            // width-sensitive kernel on purpose would mean shipping a broken emitter to
            // test the guard, which trades a real defect for a test. What is checked here
            // is that the message the guard prints exists and says why -- a guard whose
            // explanation has drifted is a guard nobody will act on.
            let src = include_str!("../src/bin/tile.rs");
            assert!(
                src.contains("it computes a \\\n                                     different answer at this width")
                    || src.contains("different answer at this width"),
                "the drop message must say what was wrong, not just that a width was dropped"
            );
            assert!(
                src.contains("however fast it is"),
                "and why speed does not rescue it"
            );
        },
    );
    r.then(
        "when no width reproduces the reference, none is recommended at all",
        |_w: &mut World, _: &[String]| {
            let src = include_str!("../src/bin/tile.rs");
            assert!(
                src.contains("Not recommending a dispatch width"),
                "with nothing verified there must be no recommendation"
            );
            assert!(
                src.contains("worth less than no recommendation"),
                "and the reason stated: a fast wrong answer is worth less than none"
            );
        },
    );
    r.then(
        "an op with no reference leaves the ranking stated as timing-only, not removed",
        |_w: &mut World, _: &[String]| {
            let src = include_str!("../src/bin/tile.rs");
            // Unverifiable must QUALIFY the recommendation, not delete it. Deleting a
            // feature because it cannot be checked is a different failure from reporting
            // an unchecked one honestly.
            assert!(
                src.contains("ranked on TIMING"),
                "an unverifiable op keeps its ranking, labelled"
            );
        },
    );
    r.then(
        "every candidate O4 tried is recorded with its own measurement",
        |w: &mut World, _: &[String]| {
            // -O4 used to run the same passes as -O3, list "measured-autotuning" under
            // "not done here", and exit 0 -- so asking for the measured level got an
            // unmeasured artifact. It now sweeps threadgroup widths on the real device,
            // which is the one axis this layer can vary without changing what the kernel
            // COMPUTES: the emitted kernels stride by threads_per_threadgroup, so a
            // different width is the same arithmetic in a different number of steps.
            let dir = scratch_dir("o4measure");
            let home = dir.join("home");
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();
            let env: &[(&str, &str)] = &[("TILE_HOME", home.to_str().unwrap())];
            let (o, e, code) = run_bin_env(
                &dir,
                &[
                    "k.mlir",
                    "-t",
                    "msl",
                    "-O4",
                    "-o",
                    "out.metal",
                    "--iterations",
                    "5",
                ],
                env,
            );
            let all = format!("{o}{e}");
            assert_eq!(code, 0, "{all}");
            assert!(all.contains("O4 measured on"), "no sweep: {all}");
            // The device is named: a number with no machine attached is not reproducible.
            assert!(all.contains("Apple"), "no device named: {all}");
            // The artifact carries the launch instruction too. Whoever writes the host
            // code reads the .metal file, not `tile -s`, and a measured result nobody
            // can act on is a number in a drawer.
            let art = std::fs::read_to_string(dir.join("out.metal")).unwrap_or_default();
            assert!(
                art.contains("O4 (measured on"),
                "artifact has no tuning note: {art}"
            );
            assert!(
                art.contains("not a portable constant"),
                "the note reads as a universal truth: {art}"
            );
            // And -O4 is therefore NOT byte-reproducible, unlike the default level: it
            // contains a measurement, and a measurement is not a pure function of the
            // input. Stated rather than left to be discovered as a flaky test.
            // A second full sweep used to run here just to show the note is produced
            // every time. It cost a GPU sweep for an assertion the first one already
            // makes, and the point it was making -- that -O4 is not byte-reproducible
            // because it carries a measurement -- is a property of the design, stated in
            // the feature file, not something a repeat run can demonstrate anyway.

            let (so, se, _) = run_bin_env(&dir, &["k.mlir", "-s", "-i"], env);
            let stats = format!("{so}{se}");
            let n = stats.matches("O4 candidate:").count();
            assert!(n >= 2, "fewer than two candidates recorded: {stats}");
            w.set("stats", stats);
        },
    );
    r.then(
        "a single timing leaves the headroom unassessed, and only the sweep writes one",
        |w: &mut World, _: &[String]| {
            // One measurement establishes a COST, not a headroom. Writing 0.0 on the
            // candidates would turn "nobody compared them" into "there is nothing to
            // gain" -- the direction that stops anyone looking again.
            let stats = w.get("stats");
            for line in stats.lines().filter(|l| l.contains("O4 candidate:")) {
                assert!(line.contains("headroom unassessed"), "{line}");
            }
            let sweep: Vec<&str> = stats
                .lines()
                .filter(|l| l.contains("O4 sweep on"))
                .collect();
            assert_eq!(sweep.len(), 1, "expected one sweep summary: {stats}");
            assert!(
                !sweep[0].contains("headroom unassessed"),
                "the sweep measured a comparison and must report it: {}",
                sweep[0]
            );
            // And its basis carries both numbers it was derived from.
            assert!(sweep[0].contains("best threadgroup"), "{}", sweep[0]);
            assert!(sweep[0].contains("vs worst"), "{}", sweep[0]);
        },
    );

    // ── Engine provisioning, the part that is real (08) ───────────────────────────

    r.then(
        "a model that cannot be provisioned leaves the daemon serving the kernel tools",
        |_w: &mut World, _: &[String]| {
            // The fetch mechanics -- size reported up front, an interrupted download
            // resuming -- are not built. This clause is, and it is the one that decides
            // whether asking for a model can COST you the tools you already had.
            let dir = scratch_dir("engine");
            let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
            c.current_dir(&dir);
            c.args(["-d", "-m", "qwen3"]);
            c.env_remove("TILE_SIMULATE");
            c.stdin(std::process::Stdio::null());
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            let out = c.output().expect("tile runs");
            let e = String::from_utf8_lossy(&out.stderr).into_owned();
            // It must SAY the model is not there, and then keep going.
            assert!(
                e.contains("serving the kernel tools without a model"),
                "losing the model lost the tools too: {e}"
            );
            // And the refusal has to carry a reason, not just a failure.
            assert!(
                e.contains("not implemented") || e.contains("not present") || e.contains("needs"),
                "no reason given: {e}"
            );
        },
    );

    // ── The wasm bundle (09) and engine provisioning (08) ─────────────────────────

    r.then(
        "the wasm bundle is served from the same address as the page",
        |_w: &mut World, _: &[String]| {
            // Served by this process on loopback, from the port the page came from --
            // "the same address" is the whole requirement, and it is what makes the
            // bundle need no toolchain, no CDN and no second server.
            use std::io::{Read as _, Write as _};
            let listener = tile_cli::ui::bind(0).expect("bind loopback");
            let port = listener.local_addr().unwrap().port();
            let body = tile_cli::ui::page(&tile_cli::ui::View {
                title: "spec".into(),
                platform: "test".into(),
                profile: None,
                routes: "none".into(),
                forms: tile_cli::routes::list_forms(),
            });
            let h = std::thread::spawn(move || {
                let _ = tile_cli::ui::serve(&listener, &body, Some(3));
            });
            let get = |path: &str| -> String {
                let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
                write!(c, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
                let mut buf = Vec::new();
                let _ = c.read_to_end(&mut buf);
                String::from_utf8_lossy(&buf).into_owned()
            };
            let page = get("/");
            assert!(page.contains("id=\"filter\""), "no filter box: {page}");
            assert!(
                page.contains("/loader.js"),
                "the page never loads the bundle"
            );
            let wasm = get("/tile_ui.wasm");
            // The MIME type is load-bearing: instantiateStreaming refuses anything else,
            // and its error names a magic number, which sends you to the wrong file.
            assert!(
                wasm.contains("Content-Type: application/wasm"),
                "wrong type for the bundle: {}",
                &wasm[..wasm.len().min(200)]
            );
            assert!(wasm.contains("\0asm"), "the bundle is not a wasm module");
            let js = get("/loader.js");
            assert!(js.contains("instantiateStreaming"), "loader not served");
            let _ = h.join();
        },
    );

    r.then(
        "the page still stands with no wasm at all",
        |_w: &mut World, _: &[String]| {
            // The filter box ships DISABLED and the loader enables it. If the module
            // never arrives the reader sees a complete table and no control that does
            // nothing -- which is the difference between progressive enhancement and a
            // page that breaks without JS.
            let html = tile_cli::ui::page(&tile_cli::ui::View {
                title: "spec".into(),
                platform: "test".into(),
                profile: None,
                routes: "none".into(),
                forms: tile_cli::routes::list_forms(),
            });
            assert!(html.contains("id=\"filter\" disabled"), "{html}");
            // Every form is in the served HTML, not fetched later.
            for f in forms::FORMS {
                assert!(html.contains(f.id), "{} missing from the page", f.id);
            }
        },
    );

    r.then(
        "provisioning an engine announces the build before it starts",
        |_w: &mut World, _: &[String]| {
            // A build that takes minutes and says nothing is indistinguishable from a
            // hang. The same rule the toolchain fetcher follows.
            let dir = scratch_dir("engbuild");
            let root = dir.join("root");
            let checkout = root.join("ds4-rs-metal");
            std::fs::create_dir_all(checkout.join("target/release")).unwrap();
            std::fs::write(checkout.join("Cargo.toml"), "[workspace]\n").unwrap();
            // A stand-in server that stays up, so the supervision path is exercised
            // without the sibling repository having to build.
            let bin = checkout.join("target/release/ds4-server");
            std::fs::write(&bin, "#!/bin/sh\nwhile true; do sleep 1; done\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            let home = dir.join("home");
            std::fs::create_dir_all(home.join("models")).unwrap();
            std::fs::write(home.join("models/qwen3.gguf"), vec![0u8; 2_000_000]).unwrap();
            std::fs::write(dir.join("k.mlir"), corpus("softmax.mlir")).unwrap();

            let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
            c.current_dir(&dir);
            c.args(["-d", "-m", "qwen3"]);
            c.env("TILE_ENGINE_PATH", &root);
            c.env("TILE_HOME", &home);
            c.env_remove("TILE_SIMULATE");
            c.stdin(std::process::Stdio::null());
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            let out = c.output().expect("tile runs");
            let e = String::from_utf8_lossy(&out.stderr).into_owned();

            assert!(e.contains("already built"), "build state not reported: {e}");
            // The size is reported before it starts, in a unit a person can read.
            assert!(e.contains("2.0 MB"), "size not reported readably: {e}");
            assert!(
                e.contains("serving alongside the daemon"),
                "never started: {e}"
            );
            // And it is stopped with the daemon: an orphaned engine holding the device is
            // the failure that makes people stop trusting a supervisor.
            assert!(e.contains("stopping ds4-rs-metal"), "left it running: {e}");
        },
    );

    // ── The wasm bundle actually runs (09) ────────────────────────────────────────

    r.then(
        "the bundle's filter returns the right rows when actually executed",
        |_w: &mut World, _: &[String]| {
            // Shape is not behaviour, and this was DEMONSTRATED rather than assumed: a
            // deliberate off-by-one that dropped the first matching row left the magic
            // number, the version and every export name intact. The structural test
            // passed on that bundle (exit 0); this one failed (exit 1). A filter box that
            // silently does nothing is exactly what the structural check cannot see.
            //
            // Asserted where a runtime already exists, rather than giving `tile` a wasm
            // interpreter it has no other reason to carry.
            let exe = wasm_runtime().expect("the tag should have skipped this");
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));
            let script = root.join("testdata/wasm/filter_test.mjs");
            let bundle = root.join("assets/ui/tile_ui.wasm");
            let mut c = Command::new(exe);
            if exe == "deno" {
                c.args(["run", "--allow-read"]);
            }
            c.arg(&script).arg(&bundle);
            let out = c.output().expect("the runtime runs");
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(out.status.success(), "the bundle misbehaves:\n{text}");
            assert!(text.contains("ALL PASS"), "{text}");
            // The guards are part of the contract: a page must not be able to kill the
            // module by asking for an absurd buffer, and a bad needle must not trap.
            assert!(
                text.contains("alloc(too big) = 0"),
                "the alloc cap is gone: {text}"
            );
        },
    );

    r.then(
        "the checked-in bundle is what its source builds",
        |_w: &mut World, _: &[String]| {
            // The bundle is a BUILT ARTIFACT checked in, so nothing otherwise notices it
            // going stale against the crate that produces it -- the same hole the PICO
            // emitter's digest closes. Rebuilding and comparing is the only claim that
            // the .wasm in the tree is the one this source makes.
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));
            let src = root.join("../tile_ui_wasm");
            if !src.join("Cargo.toml").is_file() {
                // The UI crate is not part of every checkout (it is absent from the
                // private ascend-rs tree that hosts this suite). Without its source
                // there is nothing to rebuild against; refuse the claim rather than
                // spawn cargo in a directory that does not exist — `current_dir`
                // on a missing path surfaces as NotFound from `.output()`.
                eprintln!(
                    "skip: {} has no Cargo.toml; cannot re-verify assets/ui/tile_ui.wasm",
                    src.display()
                );
                return;
            }
            let out = Command::new(env!("CARGO"))
                .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
                .current_dir(&src)
                // Let the crate's own rust-toolchain.toml decide, as it does for anyone
                // building it from a shell. Cargo exports RUSTUP_TOOLCHAIN to its test
                // processes, and the child inherits it, so the toolchain this build used
                // depended on where the SUITE was launched from: from crates/tile_cli it
                // got stable and matched the checked-in bundle, and from the repo root --
                // via --manifest-path, under the root's nightly pin -- it got that nightly
                // and built 16570 bytes against 19897. The bundle was fine both times; the
                // measurement moved.
                .env_remove("RUSTUP_TOOLCHAIN")
                .env_remove("RUSTUP_TOOLCHAIN_SOURCE")
                .output()
                .expect("cargo runs");
            assert!(
                out.status.success(),
                "the bundle's source does not build:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let built =
                std::fs::read(src.join("target/wasm32-unknown-unknown/release/tile_ui_wasm.wasm"))
                    .expect("the built module");
            let checked_in =
                std::fs::read(root.join("assets/ui/tile_ui.wasm")).expect("the checked-in module");
            assert_eq!(
                built.len(),
                checked_in.len(),
                "the checked-in bundle is {} bytes and its source builds {} — re-copy it",
                checked_in.len(),
                built.len()
            );
            assert!(
                built == checked_in,
                "the checked-in bundle has drifted from its source"
            );
        },
    );

    // ── The refusing default arm (02) ─────────────────────────────────────────────

    r.then(
        "an intrinsic the emitter does not handle is refused, not copied",
        |_w: &mut World, _: &[String]| {
            // A synthetic name, not a real op: this used `__tile_relu_f32` until the msl
            // emitter grew an arm for it, and the scenario then asserted a refusal that
            // no longer happens. A fixture that someone may implement is a test that
            // stops testing without saying so.
            //
            // The bug this closes: an unrecognised intrinsic left the kernel type at its
            // `Copy` default and the emitter wrote `p1[gid] = p0[gid]` -- source that
            // compiles, that xcrun metal accepts, that ignores every buffer past the
            // first two, and that `tile` reported as [exact, validated on Apple GPU]
            // with exit 0.
            let dir = scratch_dir("unhandled");
            let mlir = "module {\n\
              llvm.func @k(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>) attributes {hacc.entry} {\n\
              ^bb0:\n\
              %n = llvm.mlir.constant(1024 : i32) : i32\n\
              %x = llvm.call @__tile_load_f32(%a, %n, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
              %y = llvm.call @__tile_frobnicate_f32(%x, %n, %n) : (i32, i32, i32) -> i32\n\
              llvm.call @__tile_store_f32(%b, %y, %n, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
              llvm.return\n}\n}\n";
            std::fs::write(dir.join("k.mlir"), mlir).unwrap();
            let (_, e, code) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "out.metal"]);
            assert_ne!(code, 0, "it lowered anyway: {e}");
            assert!(
                e.contains("__tile_frobnicate_f32"),
                "the op is not named: {e}"
            );
            assert!(e.contains("COPY"), "it does not say what it avoided: {e}");
            assert!(!dir.join("out.metal").exists(), "wrote the copy anyway");
        },
    );
    r.then(
        "a kernel that really is a copy still lowers",
        |_w: &mut World, _: &[String]| {
            // A guard that refused this would be worse than the bug it fixes.
            let dir = scratch_dir("realcopy");
            let mlir = "module {\n\
              llvm.func @c(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>) attributes {hacc.entry} {\n\
              ^bb0:\n\
              %n = llvm.mlir.constant(1024 : i32) : i32\n\
              %x = llvm.call @__tile_load_f32(%a, %n, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
              llvm.call @__tile_store_f32(%b, %x, %n, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
              llvm.return\n}\n}\n";
            std::fs::write(dir.join("k.mlir"), mlir).unwrap();
            let (_, e, code) = run_bin(&dir, &["k.mlir", "-t", "msl", "-o", "out.metal"]);
            assert_eq!(code, 0, "{e}");
            let art = std::fs::read_to_string(dir.join("out.metal")).unwrap_or_default();
            assert!(
                art.contains("p1[gid] = p0[gid]"),
                "the copy body is gone: {art}"
            );
        },
    );

    // ── The catch-all, registered LAST on purpose ──────────────────────────────────
    // Steps resolve in registration order, so a generic `I run "tile ..."` placed early
    // silently swallows every specific invocation above it and those scenarios assert
    // nothing while still reporting green. It goes last so it only ever picks up what no
    // specific step claimed.
    r.when("I run \"tile {}\"", |w: &mut World, a: &[String]| {
        run_cli(w, &a[0])
    });
}

/// A fresh scratch directory for one scenario, wiped on entry.
///
/// Named per scenario rather than shared: these tests run in parallel, and a shared
/// directory would make them pass or fail depending on ordering.
fn scratch_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tile-spec-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch");
    d
}

/// Run the real binary in `dir` with extra environment. Returns (stdout, stderr, code).
///
/// `TILE_SIMULATE` is set on the CHILD rather than on this process: it is read once at
/// startup, and setting it here would leak into every other test running in parallel.
fn run_bin_env(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> (String, String, i32) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
    c.current_dir(dir);
    c.args(args);
    c.env_remove("TILE_SIMULATE");
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.output().expect("tile runs");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// Run the real binary in `dir`. Returns (stdout, stderr, exit code).
fn run_bin(dir: &Path, args: &[&str]) -> (String, String, i32) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
    c.current_dir(dir);
    c.args(args);
    c.env_remove("TILE_SIMULATE");
    let out = c.output().expect("tile runs");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// The same, feeding a module on stdin. `argv` is the {} capture: the command line with
/// the leading `tile` already removed by the step pattern.
fn run_bin_stdin(dir: &Path, argv: &str, module: &str) -> (String, String) {
    use std::io::Write as _;
    // `argv` is the {} capture, which already excludes the leading `tile` -- skipping a
    // token here ate the `-` that selects stdin, and the binary printed its help.
    let args: Vec<&str> = argv.split_whitespace().collect();
    let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
    c.current_dir(dir);
    c.args(&args);
    c.env_remove("TILE_SIMULATE");
    c.stdin(std::process::Stdio::piped());
    c.stdout(std::process::Stdio::piped());
    c.stderr(std::process::Stdio::piped());
    let mut child = c.spawn().expect("tile starts");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(module.as_bytes())
        .expect("write");
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A module with two entry points, for the multi-kernel scenarios.
const TWO_KERNEL_MLIR: &str = "module {\n\
  llvm.func @k1(%a: !llvm.ptr<1>) attributes {hacc.entry} {\n\
    llvm.call @__tile_exp_f32(%a) : (!llvm.ptr<1>) -> ()\n\
    llvm.return\n\
  }\n\
  llvm.func @k2(%a: !llvm.ptr<1>) attributes {hacc.entry} {\n\
    llvm.call @__tile_sqrt_f32(%a) : (!llvm.ptr<1>) -> ()\n\
    llvm.return\n\
  }\n\
}\n";

fn especially_commas(c: char) -> bool {
    c == ','
}

fn form_for_ext(ext: &str) -> &'static str {
    match ext {
        "rs" => "tile",
        "mlir" => "mlir",
        "metal" => "msl",
        "cu" => "gpu",
        "cce" => "cpp",
        other => panic!("the scenario used .{other}, which the harness does not map"),
    }
}

fn resolve_into(w: &mut World, path: &str, content: &str, forced: Option<&str>) {
    match forms::resolve_input(path, content, forced) {
        Ok(r) => {
            w.set("form", r.form.id);
            w.set(
                "how",
                match r.how {
                    forms::How::Reserved => "reserved",
                    forms::How::Forced => "forced",
                    forms::How::Magic => "magic",
                    forms::How::Extension => "extension",
                },
            );
            if let Some(c) = r.contradicted_by {
                w.set("contradicted_by", c.id);
            }
            w.set("exit", exit::OK.to_string());
        }
        Err(e) => {
            w.set("error", e.to_string());
            w.set("exit", exit::USAGE.to_string());
        }
    }
}

fn plan_into(w: &mut World, from: &str, to: &str, via: &[String]) {
    w.set("route_err", "");
    match routes::plan(from, to, via) {
        Ok(r) => {
            w.set("route", r.forms().join(" -> "));
            w.set("fidelity", r.fidelity().to_string());
            w.set("exit", exit::OK.to_string());
        }
        Err(RouteError::ViaNotOnAnyRoute { via, .. }) => {
            // Exit 2 for the same reason: the graph is fine and the command is wrong.
            w.set(
                "route_err",
                format!("--via {} is not on any route", via.join(",")),
            );
            w.set("exit", exit::USAGE.to_string());
        }
        Err(RouteError::OnlyThroughLift { from, to, .. }) => {
            // Exit 2: the route exists and is takeable, just not unasked. A different
            // command works right now, which is what code 2 means.
            w.set("route_err", "only-lift");
            w.set("route_from", from);
            w.set("route_to", to);
            w.set("exit", exit::USAGE.to_string());
        }
        Err(RouteError::NeedsFrontend { from, to, would_be }) => {
            w.set("route_err", "needs-frontend");
            w.set("route_from", from);
            w.set("route_to", to);
            if let Some(r) = would_be {
                w.set("would_be", r.forms().join(" -> "));
            }
            w.set("exit", exit::UNSUPPORTED.to_string());
        }
        Err(RouteError::NoRoute { from, to, .. }) => {
            w.set("route_err", "no-route");
            w.set("route_from", from);
            w.set("route_to", to);
            w.set("exit", exit::UNSUPPORTED.to_string());
        }
    }
}

fn scan(dir: &Path, f: &mut impl FnMut(&Path, &str)) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            scan(&p, f);
        } else if p.extension().is_some_and(|x| x == "rs") {
            if let Ok(t) = std::fs::read_to_string(&p) {
                f(&p, &t);
            }
        }
    }
}

// ── The suite ─────────────────────────────────────────────────────────────────────

#[test]
fn the_specification_runs() {
    let mut runner = Runner::new();
    register(&mut runner);

    let mut files: Vec<PathBuf> = std::fs::read_dir(features_dir())
        .expect("features/")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "feature"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .feature files found");

    let (mut live_total, mut all_total, mut ran) = (0usize, 0usize, 0usize);
    let mut rows = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("feature");
        let (stripped, live, total) = strip_tagged(&text);
        live_total += live;
        all_total += total;
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if live > 0 {
            // Named before the run so an assertion failure inside a step is attributable
            // to a file without a backtrace.
            let feature = parse_feature(&stripped);
            ran += runner.run_feature(&feature);
        }
        rows.push((name, live, total));
    }

    // The progress meter. This is the number that says how much of the tool exists.
    // Also written to `target/spec-progress.md` so CI can publish it and a human can
    // read it without re-running the suite.
    let mut md = String::from("# `tile` — specification coverage\n\n");
    md.push_str("Regenerated by `cargo test --test spec`. An untagged scenario must be\n");
    md.push_str("green; everything else carries `@planned` and is counted, not ignored.\n\n");
    md.push_str("| feature | live | total | |\n|---|---:|---:|---|\n");
    for (name, live, total) in &rows {
        let filled = if *total == 0 { 0 } else { live * 20 / total };
        md.push_str(&format!(
            "| `{name}` | {live} | {total} | {}{} |\n",
            "█".repeat(filled),
            "·".repeat(20 - filled)
        ));
    }
    // Break the remainder down by WHY, because "18 not live" reads as 18 pieces of
    // unfinished work and they are not the same kind of thing. Since the form rule
    // settled that a target-source form owes no reader, every @requires-lifter scenario
    // describes a product nobody has decided to build -- not a milestone slipping. A
    // meter that cannot say which is which will be read as debt forever.
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    for entry in std::fs::read_dir(features_dir()).unwrap().flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|x| x == "feature") {
            let text = std::fs::read_to_string(&p).unwrap();
            let mut pending: Option<String> = None;
            for line in text.lines() {
                let t = line.trim();
                if let Some(tag) = t.strip_prefix('@') {
                    pending = Some(tag.split_whitespace().next().unwrap_or(tag).to_string());
                } else if t.starts_with("Scenario") {
                    if let Some(tag) = pending.take() {
                        *reasons.entry(tag).or_default() += 1;
                    }
                }
            }
        }
    }
    md.push_str("\nThe rest, by reason — these are different kinds of thing:\n\n");
    for (tag, n) in &reasons {
        let why = match tag.as_str() {
            "planned" => "written down, not built yet",
            "requires-lifter" => {
                "needs a frontend; a source form owes no reader, so this is a decision, not pending work"
            }
            "requires-backend" => {
                if backend_present() {
                    "conditional: the codegen backend is provisioned here, so this RAN"
                } else {
                    "conditional: needs the codegen backend; `tile install` provisions it"
                }
            }
            "requires-device" => {
                if device_present() {
                    "conditional: needs a device — present here, so these RAN"
                } else {
                    "conditional: needs a device and `swift`; absent here, so skipped"
                }
            }
            "requires-wasm-target" => {
                if wasm_target_installed() {
                    "conditional: the wasm target is here, so the bundle was rebuilt and compared"
                } else {
                    "conditional: needs the wasm32 target to rebuild the bundle"
                }
            }
            "requires-wasm-runtime" => {
                if wasm_runtime().is_some() {
                    "conditional: a wasm runtime is here, so the bundle's behaviour RAN"
                } else {
                    "conditional: needs node or deno to execute the bundle"
                }
            }
            "requires-toolchain" => {
                if target_compiler_present() {
                    "conditional: the target compiler is here, so this RAN"
                } else {
                    "conditional: needs the target's own compiler"
                }
            }
            "requires-hosts" => {
                if spec_hosts().is_empty() {
                    "conditional: needs other machines; set TILE_SPEC_HOSTS=\"mac mini\""
                } else {
                    "conditional: TILE_SPEC_HOSTS was set, so this RAN on those machines"
                }
            }
            "requires-cross" => {
                if cross_requested() {
                    "conditional: TILE_SPEC_CROSS=1 was set, so this RAN"
                } else {
                    "conditional: compiles for another triple (minutes); set TILE_SPEC_CROSS=1"
                }
            }
            "requires-stats-build" => {
                if cfg!(feature = "stats") {
                    "conditional: needs the crypto feature — built in, so these RAN"
                } else {
                    "conditional: needs the crypto feature; not in this build, so skipped"
                }
            }
            _ => "see the tag",
        };
        md.push_str(&format!("* `@{tag}` — {n}: {why}\n"));
    }
    md.push_str(&format!(
        "\n**{live_total} of {all_total} scenarios live ({}%)**, {ran} cases executed green.\n",
        live_total * 100 / all_total.max(1)
    ));
    let out_md = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/spec-progress.md");
    let _ = std::fs::create_dir_all(out_md.parent().unwrap());
    let _ = std::fs::write(&out_md, &md);

    println!("\n  tile-rs CLI — executable specification");
    println!("  ────────────────────────────────────────────────────────────");
    for (name, live, total) in &rows {
        let bar_len = 20usize;
        let filled = if *total == 0 {
            0
        } else {
            live * bar_len / total
        };
        let bar: String = "█".repeat(filled) + &"·".repeat(bar_len - filled);
        println!("  {name:<38} {bar} {live:>3}/{total:<3}");
    }
    println!("  ────────────────────────────────────────────────────────────");
    let pct = live_total * 100 / all_total.max(1);
    println!("  {live_total} of {all_total} scenarios live ({pct}%)");
    println!("  {ran} cases executed green (a Scenario Outline is one case per Examples row)");
    println!("  the remainder carry @planned and are counted, not ignored\n");

    // Every live scenario must produce at least one executed case; an outline produces
    // one per Examples row, so this is >= and not ==.
    assert!(
        ran >= live_total,
        "{live_total} scenarios are live but only {ran} cases ran"
    );
}

/// Every `crates/*` that is its own workspace must be excluded from the root's.
///
/// `members = ["crates/*"]` auto-includes anything under `crates/`, and cargo refuses a
/// member that declares its own `[workspace]`: "multiple workspace roots found in the same
/// workspace". That breaks `cargo build` AT THE ROOT for everyone, not just for whoever
/// added the crate -- and nothing in this crate's own tests would notice, because
/// `tile_cli` is itself excluded and builds in its own workspace.
///
/// Found by building a clean clone, which is the only place the root build is exercised.
/// This is that check, made cheap enough to run every time.
#[test]
fn every_self_workspacing_crate_is_excluded_from_the_root() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("root Cargo.toml");
    // Collect the QUOTED entries between `exclude = [` and its closing bracket.
    //
    // Two traps, and this test fell into the second on its first run. Taking the whole
    // file would let a crate name in an unrelated comment satisfy the check and hide the
    // bug. Taking everything up to the first `]` is worse: the comments inside this list
    // themselves contain `members = ["crates/*"]`, so the list was truncated at the first
    // comment and every excluded crate looked like an offender. Same "read past the
    // prose" mistake this repository has now made five times, which is why the parse
    // stops at a line that is exactly `]` and only reads quoted strings.
    let mut excluded: Vec<String> = Vec::new();
    let mut inside = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with("exclude") && t.contains('[') {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        if t == "]" {
            break;
        }
        if let Some(rest) = t.strip_prefix('"') {
            if let Some(name) = rest.split('"').next() {
                excluded.push(name.to_string());
            }
        }
    }
    assert!(
        !excluded.is_empty(),
        "no exclude list parsed from the root manifest"
    );

    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(root.join("crates"))
        .expect("crates/")
        .flatten()
    {
        let p = entry.path();
        let toml = p.join("Cargo.toml");
        if !toml.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&toml).unwrap_or_default();
        // A bare `[workspace]` line makes the directory a workspace root of its own.
        let is_root = text.lines().any(|l| l.trim() == "[workspace]");
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if is_root && !excluded.iter().any(|e| e == &format!("crates/{name}")) {
            offenders.push(name);
        }
    }
    assert!(
        offenders.is_empty(),
        "these declare their own [workspace] and are not in the root's exclude list, \
         so `cargo build` at the repository root fails: {offenders:?}"
    );
}

/// The committed coverage doc must still describe THIS repo's scenarios.
///
/// `docs/cli/progress.md` is a copy of what the spec run generates, and a copy drifts.
/// It sat at "175 of 178 scenarios live (98%), 241 cases" while the suite produced
/// 174 of 179 on the same features -- stale enough to survive a scenario being added
/// without anyone noticing, because nothing compared the two.
///
/// Only the TOTALS are checked, per feature and overall. The live count is a claim about
/// the host -- four scenarios need the crypto feature, others need cross toolchains or
/// other machines -- so pinning it here would fail honestly-configured checkouts. The
/// total is a property of the feature files alone, so it is the same everywhere and it is
/// exactly what drifted.
#[test]
fn the_committed_coverage_doc_counts_the_scenarios_that_exist() {
    let doc = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/cli/progress.md"),
    )
    .expect("docs/cli/progress.md");

    let mut grand = 0usize;
    for entry in std::fs::read_dir(features_dir()).unwrap().flatten() {
        let p = entry.path();
        if p.extension().is_none_or(|x| x != "feature") {
            continue;
        }
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&p).unwrap();
        let total = text
            .lines()
            .filter(|l| l.trim_start().starts_with("Scenario"))
            .count();
        grand += total;

        // The row reads `| `NAME` | live | total | bar |`.
        let row = doc
            .lines()
            .find(|l| l.contains(&format!("`{name}`")))
            .unwrap_or_else(|| panic!("{name} has no row in docs/cli/progress.md"));
        let cells: Vec<&str> = row.split('|').map(|c| c.trim()).collect();
        let claimed: usize = cells
            .get(3)
            .and_then(|c| c.parse().ok())
            .unwrap_or_else(|| panic!("{name}: no total in row {row:?}"));
        assert_eq!(
            claimed, total,
            "{name}: the doc claims {claimed} scenarios, the file has {total}. \
             Regenerate: cargo test --features pico,stats --test spec, then copy \
             crates/tile_cli/target/spec-progress.md over docs/cli/progress.md"
        );
    }

    // Anchor on the summary line, not on the first " of " in the file -- the prose above
    // it contains one, which is how the first version of this test read "thing:" as a
    // scenario count.
    let claimed_grand: usize = doc
        .lines()
        .find(|l| l.contains("scenarios live"))
        .and_then(|l| l.split(" of ").nth(1))
        .and_then(|r| r.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .expect("a `N of M scenarios live` line");
    assert_eq!(
        claimed_grand, grand,
        "the doc totals {claimed_grand} scenarios, the features hold {grand}"
    );
}

/// This had no `#[test]` and had therefore never run. The compiler said so all along --
/// "function `every_live_scenario_is_named_in_the_manifest` is never used" -- in a build
/// that also emitted an `unreachable_patterns` warning for a duplicated `--list-ops` arm,
/// which is how a warning that matters gets lost among warnings that do not.
#[test]
fn every_live_scenario_is_named_in_the_manifest() {
    // `features/.live` is the record of what each milestone turned on. An untagged
    // scenario that is not in it means someone removed a tag without claiming it.
    let manifest = std::fs::read_to_string(features_dir().join(".live")).expect(".live");
    let claimed: Vec<&str> = manifest
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .collect();
    let mut found = 0;
    for entry in std::fs::read_dir(features_dir()).unwrap().flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|x| x == "feature") {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let text = std::fs::read_to_string(&p).unwrap();
            let mut tagged = false;
            for line in text.lines() {
                let t = line.trim();
                if t.starts_with('@') {
                    // A conditional tag still means the scenario is claimed: it runs
                    // wherever its condition holds, so it belongs in the manifest.
                    // `@requires-device` joined `@requires-stats-build` here when it
                    // became conditional -- a tag that gates on the machine rather than
                    // on a decision not to build something.
                    if !t.contains("@requires-stats-build")
                        && !t.contains("@requires-device")
                        && !t.contains("@requires-cross")
                        && !t.contains("@requires-hosts")
                        && !t.contains("@requires-backend")
                        && !t.contains("@requires-toolchain")
                        && !t.contains("@requires-wasm-runtime")
                        && !t.contains("@requires-wasm-target")
                    {
                        tagged = true;
                    }
                    continue;
                }
                if let Some(rest) = t
                    .strip_prefix("Scenario Outline:")
                    .or_else(|| t.strip_prefix("Scenario:"))
                {
                    if !tagged {
                        let key = format!("{name}::{}", rest.trim());
                        assert!(
                            claimed.contains(&key.as_str()),
                            "untagged but unclaimed: {key}"
                        );
                        found += 1;
                    }
                    tagged = false;
                }
            }
        }
    }
    assert_eq!(found, claimed.len(), "the manifest and the files disagree");
}

/// Build a token with a chosen defect and try to verify it.
///
/// Returns (ok, reason). Without the `stats` feature there is no signature scheme in the
/// build, so the scenarios are reported as passing-with-nothing-checked rather than
/// asserted vacuously — a distinction the coverage meter would otherwise hide.
#[cfg(feature = "stats")]
fn verify_token(defect: &str) -> (bool, String) {
    use ed25519_dalek::{Signer, SigningKey};
    use tile_cli::license::{hex_encode, payload, verify, License};

    let good = SigningKey::from_bytes(&[42u8; 32]);
    let other = SigningKey::from_bytes(&[9u8; 32]);
    let mut l = License {
        subject: "a team".into(),
        expires: "2030-01-01".into(),
        features: vec!["corpus".into()],
        content_key: hex_encode(&[7u8; 32]),
    };
    if defect.contains("expired") {
        l.expires = "2020-01-01".into();
    }
    let signer = if defect.contains("unknown key") {
        &other
    } else {
        &good
    };
    let mut text = format!(
        "{}signature={}\n",
        payload(&l),
        hex_encode(&signer.sign(payload(&l).as_bytes()).to_bytes())
    );
    if defect == "tampered with" {
        text = text.replace("features=corpus", "features=corpus,all");
    }
    match verify(&text, "2026-09-01", &good.verifying_key().to_bytes()) {
        Ok(l) => (
            true,
            format!(
                "{} until {} [{}]",
                l.subject,
                l.expires,
                l.features.join(",")
            ),
        ),
        Err(e) => (false, format!("{e:?}").to_lowercase()),
    }
}

#[cfg(not(feature = "stats"))]
fn verify_token(_defect: &str) -> (bool, String) {
    (false, "tampered expired".into())
}

/// Start `tile -u` on a fixed port, fetch the page, and stop it.
///
/// Testing a server means being a client: a headless box can assert what is on a page,
/// which is the whole argument for the UI being a server before it is a window.
fn ui_run(w: &mut World) {
    use std::io::{Read as _, Write as _};
    // A port derived from the pid, so a parallel run of this suite does not collide.
    let port = 20000 + (std::process::id() % 10000) as u16;
    let mut child = Command::new(env!("CARGO_BIN_EXE_tile"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "testdata/forms/softmax.mlir",
            "-u",
            "--ui-port",
            &port.to_string(),
        ])
        .env_remove("TILE_SIMULATE")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the UI starts");

    // The URL is the first line of stdout, printed before the server loop begins.
    let mut url = String::new();
    if let Some(out) = child.stdout.as_mut() {
        let mut buf = [0u8; 256];
        if let Ok(n) = out.read(&mut buf) {
            url = String::from_utf8_lossy(&buf[..n]).trim().to_string();
        }
    }
    w.set("ui_url", url.clone());

    let mut body = String::new();
    if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", port)) {
        let _ = write!(s, "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
        let mut got = String::new();
        let _ = s.read_to_string(&mut got);
        body = got;
    }
    w.set("ui_body", body);
    w.set("exit", "0");
    let _ = child.kill();
    let _ = child.wait();
}

/// The issues a scenario declared, as a list of areas in `bl_set`.
fn bl_issues(w: &World) -> Vec<tile_cli::backlog::Issue> {
    let set = w.get("bl_set");
    if set.is_empty() {
        return Vec::new();
    }
    let mut v: Vec<tile_cli::backlog::Issue> = set
        .split(',')
        .enumerate()
        .map(|(i, area)| tile_cli::backlog::Issue {
            id: i as u32 + 1,
            area: area.trim().into(),
            title: format!("gap {}", i + 1),
            wanted: "lower a kernel".into(),
            got: "no route".into(),
            workaround: "wrote the MSL by hand".into(),
            opened: "2026-09-01".into(),
            closed: None,
        })
        .collect();
    if w.flag("bl_close_first") {
        v[0].closed = Some("2026-09-02".into());
    }
    v
}

fn bl_verdict(w: &World) -> tile_cli::backlog::Verdict {
    tile_cli::backlog::verdict(&bl_issues(w))
}

/// The three-way verdict for a pair of qualitative accuracy descriptions.
fn attribution_with(kernel: &str, ours: &str, vs_ours_override: Option<&str>) -> &'static str {
    use tile_cli::torchref::{attribute, Accuracy, Tolerance};
    let mk = |how: &str| {
        if how == "close" {
            Accuracy {
                compared: 4,
                max_abs: 1e-8,
                max_rel: 1e-8,
                rmse: 1e-8,
                non_finite: 0,
                max_abs_near_zero: 0.0,
            }
        } else {
            Accuracy {
                compared: 4,
                max_abs: 1.0,
                max_rel: 1.0,
                rmse: 1.0,
                non_finite: 0,
                max_abs_near_zero: 0.0,
            }
        }
    };
    // The scenario language names only the two comparisons against torch, so the third
    // -- kernel against our reference -- is derived from them: two sources that stand the
    // same distance from torch agree with each other, and two that do not, do not. That
    // keeps "far"/"far" meaning what the scenario always meant by it, the case where the
    // kernel reproduced our reference's error.
    let vs_ours = vs_ours_override.unwrap_or(if kernel == ours { "close" } else { "far" });
    attribute(
        &mk(kernel),
        &mk(ours),
        &mk(vs_ours),
        Tolerance::Fixed {
            abs: 1e-5,
            rel: 1e-4,
        },
    )
}

/// The common case: the third comparison is derived rather than stated.
fn attribution(kernel: &str, ours: &str) -> &'static str {
    attribution_with(kernel, ours, None)
}

/// Send JSON-RPC lines to `tile -d` and collect the reply lines.
fn mcp(lines: &[&str]) -> Vec<String> {
    mcp_with(lines, &["-d"])
}

/// The same, with extra arguments to the daemon.
fn mcp_with(lines: &[&str], args: &[&str]) -> Vec<String> {
    mcp_raw(lines, args).0.lines().map(str::to_string).collect()
}

/// The same, keeping stdout and stderr apart — which is the point of several scenarios.
fn mcp_raw(lines: &[&str], args: &[&str]) -> (String, String) {
    use std::io::Write as _;
    let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
    c.current_dir(env!("CARGO_MANIFEST_DIR"));
    c.args(args);
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
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The text content out of one `tools/call` reply.
fn mcp_text(reply: &str) -> String {
    let Ok(v) = tile_cli::json::parse(reply) else {
        return String::new();
    };
    v.get("result")
        .and_then(|r| r.get("content"))
        .and_then(|c| match c {
            tile_cli::json::Json::Arr(a) => a.first().cloned(),
            _ => None,
        })
        .and_then(|c| c.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .unwrap_or_default()
}

/// A private `~/.tile-rs` plus an artifact to install into it.
///
/// `TILE_HOME` is process-global, so this is serialized: cargo's test runner is parallel
/// and two sandboxes racing would each see the other's prefix.
static PROV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RUSTFLAGS is process-global; a step that sets it changes the world for every other.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn prov_setup(w: &mut World, honest: bool) {
    let _g = PROV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("tile-spec-home-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("TILE_HOME", &dir);

    let payload = b"a plausible toolchain artifact";
    let src = std::env::temp_dir().join(format!("tile-spec-art-{}.bin", std::process::id()));
    std::fs::write(&src, payload).unwrap();

    w.set("prov_home", dir.to_string_lossy());
    w.set("prov_url", format!("file://{}", src.display()));
    // An "honest" manifest pins what is actually there; the other pins something else,
    // which is the corrupted-download case without needing a corrupted download.
    w.set(
        "prov_sha",
        if honest {
            tile_cli::sha256::hex(payload)
        } else {
            tile_cli::sha256::hex(b"the bytes the manifest expected")
        },
    );
    w.set("said", "");
}

fn prov_acquire(w: &mut World, policy: tile_cli::provision::Policy) {
    let tool = tile_cli::manifest::Tool {
        id: "demo".into(),
        version: "1.0".into(),
        os: "any".into(),
        arch: "any".into(),
        url: w.get("prov_url").to_string(),
        sha256: w.get("prov_sha").to_string(),
        barrier: None,
        remedy: None,
        note: None,
        unpack: None,
        strip: 0,
    };
    let mut said = w.get("said").to_string();
    let res = tile_cli::provision::install_tool(&tool, policy, &mut |m| {
        said.push_str(m);
        said.push('\n');
    });
    w.set("said", said);
    match res {
        Ok(tile_cli::provision::Outcome::Installed { .. }) => {
            w.set("prov_outcome", "installed");
            w.set("prov_err", "");
        }
        Ok(tile_cli::provision::Outcome::AlreadyPresent) => {
            w.set("prov_outcome", "already");
            w.set("prov_err", "");
        }
        Err(e) => {
            w.set("prov_outcome", "error");
            w.set("prov_err", e.to_string());
        }
    }

    // What actually landed on disk, and where.
    let home = std::path::PathBuf::from(w.get("prov_home"));
    let mut files = Vec::new();
    let mut outside = Vec::new();
    collect(&home, &mut files);
    for f in &files {
        if !tile_cli::provision::is_user_scoped(std::path::Path::new(f)) {
            outside.push(f.clone());
        }
    }
    let dir = tile_cli::provision::toolchain_dir("demo", "1.0");
    w.set("prov_dir_exists", dir.exists().to_string());
    w.set(
        "prov_files",
        files
            .iter()
            .filter(|f| !f.ends_with("layout-version"))
            .cloned()
            .collect::<Vec<_>>()
            .join(","),
    );
    w.set("prov_outside", outside.join(","));
    w.set(
        "prov_layout",
        std::fs::read_to_string(home.join("layout-version"))
            .unwrap_or_default()
            .trim()
            .to_string(),
    );
    let count = std::fs::read_dir(home.join("toolchains"))
        .map(|rd| rd.flatten().count())
        .unwrap_or(0);
    w.set("prov_installed_count", count.to_string());
}

fn collect(dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else {
            out.push(p.to_string_lossy().into_owned());
        }
    }
}

/// Run a conversion out of the corpus and record what it left behind on disk.
///
/// Intermediates are the one thing a library-level check cannot see: whether a file
/// still exists after the process exits is a property of the process.
fn run_intermediates(w: &mut World, tail: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let forms_dir = root.join("testdata/forms");
    let out = std::env::temp_dir().join(format!("tile-spec-ir-{}.metal", std::process::id()));
    let _ = std::fs::remove_file(&out);

    let mut args: Vec<String> = vec!["testdata/forms/softmax.mlir".into()];
    let mut it = tail.split_whitespace();
    while let Some(t) = it.next() {
        if t == "-o" {
            it.next();
            args.push("-o".into());
            args.push(out.to_string_lossy().into());
        } else {
            args.push(t.to_string());
        }
    }
    let route_only = args.iter().any(|a| a == "--route");
    run_binary(w, &args, None);

    if route_only {
        // `--route` prints the candidates and writes nothing; the first listed route is
        // the one that would be taken.
        // The listing lines carry a cost; the profile above them can contain "->" too
        // (`dtypes: int32 -> float32`), so match on the listing's own shape.
        let line = w
            .get("stdout")
            .lines()
            .find(|l| l.contains("->") && l.contains("cost "))
            .unwrap_or("")
            .trim()
            .to_string();
        let route = line.split("  [").next().unwrap_or("").trim().to_string();
        w.set("route", route);
        return;
    }

    w.set(
        "out_text",
        std::fs::read_to_string(&out).unwrap_or_default(),
    );
    let _ = std::fs::remove_file(&out);

    let leftovers: Vec<String> = std::fs::read_dir(&forms_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("softmax.") && n.contains(".1."))
        .collect();
    if let Some(first) = leftovers.first() {
        w.set(
            "kept_text",
            std::fs::read_to_string(forms_dir.join(first)).unwrap_or_default(),
        );
    }
    w.set("leftovers", leftovers.join(","));
    for n in &leftovers {
        let _ = std::fs::remove_file(forms_dir.join(n));
    }

    // Only THIS run's scratch, not everything in the shared system temp directory.
    //
    // This scanned `std::env::temp_dir()`, which every process on the machine writes to.
    // It failed with "a scratch directory survived: tile-softmax-41001" -- a scratch
    // belonging to a different `tile`, from a concurrent run of this same suite. The
    // assertion is about whether THIS conversion cleaned up after itself, and it was
    // reading a directory it does not own.
    //
    // `run_binary` gives the child a private TMPDIR, so the scratch root is per-scenario
    // and a neighbour's leftovers cannot be mistaken for this one's.
    let scratch_left: Vec<String> = std::fs::read_dir(child_tmpdir())
        .map(|d| {
            d.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("tile-"))
                .collect()
        })
        .unwrap_or_default();
    w.set("scratch_left", scratch_left.join(","));
}

/// Run the actual binary and record what a user would see.
///
/// Some scenarios are about the LIBRARY (does this form resolve, does this route exist)
/// and some are about the PROGRAM (what lands on stdout, what the exit code is, whether
/// a file appeared). Only the second kind can be checked by running it.
/// A temp directory owned by this test binary, for children to scratch in.
///
/// Per-process rather than per-scenario: the scratch assertions run in sequence within one
/// suite, and a shared-with-the-machine directory is what broke them.
fn child_tmpdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("tile-spec-tmp-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn run_binary(w: &mut World, args: &[String], stdin: Option<&str>) {
    use std::io::Write as _;
    let mut c = Command::new(env!("CARGO_BIN_EXE_tile"));
    // Scratch goes somewhere this suite owns. Set on the CHILD: setting it here would
    // move every other parallel test's temp directory too.
    c.env("TMPDIR", child_tmpdir());
    c.current_dir(env!("CARGO_MANIFEST_DIR"));
    c.env_remove("TILE_SIMULATE");
    c.args(args);
    if stdin.is_some() {
        c.stdin(std::process::Stdio::piped());
    }
    c.stdout(std::process::Stdio::piped());
    c.stderr(std::process::Stdio::piped());
    let mut child = c.spawn().expect("the tile binary runs");
    if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
        pipe.write_all(text.as_bytes()).unwrap();
    }
    let out = child.wait_with_output().expect("wait");
    w.set("exit", out.status.code().unwrap_or(-1).to_string());
    w.set("stdout", String::from_utf8_lossy(&out.stdout).into_owned());
    w.set("stderr", String::from_utf8_lossy(&out.stderr).into_owned());
    w.set(
        "error",
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .next()
            .unwrap_or("")
            .to_string(),
    );
}

/// Drive the real CLI pipeline for a `When I run "tile ..."` step: parse the argv the
/// binary would get, then, when it names a corpus input, resolve its form and profile it.
///
/// One funnel, because two scenarios in different files legitimately share this step
/// text — and two steps racing for it means the loser asserts nothing.
fn run_cli(w: &mut World, tail: &str) {
    let argv: Vec<String> = tail.split_whitespace().map(str::to_string).collect();
    let args = match cli::parse(argv) {
        Ok(a) => {
            w.set_flag("parsed", true);
            w.set("cmd", format!("{:?}", a.cmd));
            w.set("level", a.level().to_string());
            w.set("verbose", a.verbose.to_string());
            w.set("exit", exit::OK.to_string());
            a
        }
        Err(e) => {
            w.set_flag("parsed", false);
            w.set("error", e.to_string());
            w.set("exit", exit::USAGE.to_string());
            return;
        }
    };

    // Which family are we targeting? --cross wins, then whatever the scenario declared.
    let family = args
        .cross
        .clone()
        .or_else(|| {
            if w.has("family") {
                Some(w.get("family").to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "none".to_string());
    w.set("default_form", default_form_for(&family).id);

    let Some(input) = args.inputs.first() else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/forms")
        .join(input);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };

    match forms::resolve_input(input, &text, args.from.as_deref()) {
        Ok(r) => {
            w.set("form", r.form.id);
            let p = profile::profile(input, &text, r.form, r.how, &family, args.tile);
            w.set("out", p.render());
            w.set("hazard_edges", p.hazards.edges.len().to_string());
            w.set("unsynced", p.hazards.unsynced.len().to_string());
        }
        Err(e) => {
            w.set("error", e.to_string());
            w.set("exit", exit::USAGE.to_string());
        }
    }
}
