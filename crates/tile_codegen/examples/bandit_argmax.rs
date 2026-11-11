//! Drive the KernelBand loop over arg_max tiling arms on a real Ascend box.
//!
//! This is the wiring the driver module was written to accept: `drive`'s `run`
//! closure shells out to `bench/arm_probe.py`, which measures ONE arm in ONE
//! process and classifies the outcome. Nothing about the bandit changes here --
//! the loop, the persistence and the failure taxonomy are the tested ones.
//!
//!     PROBE_CMD='... {ARM} ...' BANDIT_TILE_ARMS=256,512,1024 \
//!         cargo run --example bandit_argmax
//!
//! `BANDIT_KERNEL` must identify the SHAPE CLASS and not merely the operator
//! (`arg_max@300x2048`, not `arg_max`): a partition count that wins at one row
//! count can lose at another, so a per-operator key silently applies one
//! shape's measurements to the next shape's decision.
//!
//! `{ARM}`, `{ITERS}` and `{REPEAT}` are substituted into the command. **The
//! template owns the quoting**: write `'{ARM}'`, never a bare `{ARM}`, because
//! the default arm substitutes to the EMPTY string and an unquoted empty value
//! silently shifts every later argument (this cost a baseline measurement: a
//! stub taking the arm positionally read `{ITERS}` as its arm and returned the
//! wrong time, which is exactly the mis-attribution the driver exists to
//! avoid). Without `PROBE_CMD` the ops invocation is used and already quotes
//! it.
//!
//! Two passes, deliberately:
//!   * an APPLICABILITY pass (`iters=1`) that drops arms the kernel refuses.
//!     A refusal is a static property of the shape, not a measurement, so it
//!     must not enter the state as a zero reward -- that would teach the bandit
//!     that a perfectly good arm is slow.
//!   * the BANDIT pass over the survivors.

use std::path::PathBuf;
use std::process::Command;

use tile_codegen::driver::{DriveReport, RunOutcome, drive, load_or_new, save};
use tile_codegen::schedule::Schedule;

#[derive(Debug, Clone, Default)]
struct ProbeResult {
    /// False only when the transport itself failed and no verdict was read.
    transport_ok: bool,
    verified: bool,
    race: bool,
    refused: bool,
    fault: bool,
    us_median: f64,
    /// Set only by a paired probe: the baseline median measured in the SAME
    /// process, so the ratio is robust to drift.
    us_base_median: Option<f64>,
    /// The probe's own reward when it measured paired.
    reward: Option<f64>,
    reason: String,
}

#[derive(Debug, Clone)]
struct Cfg {
    cmd: String,
    arch: String,
    kernel: String,
    budget: usize,
    arms: Vec<Schedule>,
    state_dir: PathBuf,
}

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn default_cmd() -> String {
    let tree = env_or("TREE", "/root/tilers-argmax/resume-argmax");
    format!(
        "ssh -o ConnectTimeout=15 ops \"/root/indock.sh 'cd {tree}; \
         ARGMAX_SCHEDULE=\\\"{{ARM}}\\\" ARGMAX_FORCE_PORTABLE=1 \
         PROBE_O=${{PROBE_O:-300}} PROBE_I=${{PROBE_I:-4096}} \
         PROBE_DTYPE=${{PROBE_DTYPE:-float32}} PROBE_ITERS=${{PROBE_ITERS:-20}} \
         PROBE_REPEAT=${{PROBE_REPEAT:-3}} python3 arm_probe.py'\""
    )
}

/// Run one arm and classify it. A `{ARM}` of `""` is the DEFAULT schedule.
///
/// Transport failures are retried, because over the ssh tunnel a dropped
/// connection and an unhealthy device look identical from here -- and treating
/// the former as a fault would abort a twenty-minute run on one blip, while
/// treating it as a reward-0 failure would poison the state with a measurement
/// that never happened. Neither is acceptable, so it is retried and only then
/// called a fault.
fn probe(cfg: &Cfg, arm: &Schedule, iters: u32, repeat: u32) -> ProbeResult {
    for attempt in 0..3 {
        let r = probe_once(cfg, arm, iters, repeat);
        if r.transport_ok {
            return r;
        }
        if attempt < 2 {
            std::thread::sleep(std::time::Duration::from_secs(20));
        }
    }
    probe_once(cfg, arm, iters, repeat)
}

fn probe_once(cfg: &Cfg, arm: &Schedule, iters: u32, repeat: u32) -> ProbeResult {
    let armkey = if arm.is_default() { String::new() } else { arm.arm_key() };
    let cmd = cfg
        .cmd
        .replace("{ARM}", &armkey)
        .replace("{ITERS}", &iters.to_string())
        .replace("{REPEAT}", &repeat.to_string());
    let out = Command::new("sh").arg("-c").arg(&cmd).output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            return ProbeResult { fault: true, reason: format!("spawn: {e}"), ..Default::default() };
        }
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The probe prints one JSON object; take the last line that parses.
    let mut res = ProbeResult { reason: "no JSON from probe".into(), ..Default::default() };
    res.transport_ok = false;
    for line in stdout.lines().rev() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        if let Some(r) = parse_probe(line) {
            res = ProbeResult { transport_ok: true, ..r };
            break;
        }
    }
    // Exit codes are the probe's own contract, and outrank the JSON text. A
    // 7 or 8 IS a verdict, so it is not retried even when the probe said it
    // without printing JSON.
    match out.status.code() {
        Some(7) => { res.fault = true; res.transport_ok = true; }
        Some(8) => { res.refused = true; res.transport_ok = true; }
        _ => {}
    }
    if res.reason.is_empty() {
        res.reason = String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("").to_string();
    }
    res
}

/// Minimal field extraction -- the probe emits a flat object of numbers,
/// booleans and one string, so a parser dependency is not warranted.
fn parse_probe(line: &str) -> Option<ProbeResult> {
    let mut r = ProbeResult::default();
    let mut ok = false;
    for (k, v) in [
        ("verified", Field::Bool),
        ("race", Field::Bool),
        ("refused", Field::Bool),
        ("fault", Field::Bool),
        ("us_median", Field::Num),
        ("us_base_median", Field::Num),
        ("reward", Field::Num),
        ("reason", Field::Str),
    ] {
        let needle = format!("\"{k}\":");
        let Some(i) = line.find(&needle) else { continue };
        let rest = line[i + needle.len()..].trim_start();
        match v {
            Field::Bool => {
                if rest.starts_with("true") {
                    ok = true;
                    match k {
                        "verified" => r.verified = true,
                        "race" => r.race = true,
                        "refused" => r.refused = true,
                        "fault" => r.fault = true,
                        _ => {}
                    }
                } else if rest.starts_with("false") {
                    ok = true;
                }
            }
            Field::Num => {
                let n: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-').collect();
                if let Ok(f) = n.parse::<f64>() {
                    match k {
                        "us_median" => r.us_median = f,
                        "us_base_median" => r.us_base_median = Some(f),
                        // The probe's own paired reward, if it took one.
                        "reward" => r.reward = Some(f),
                        _ => {}
                    }
                    ok = true;
                }
            }
            Field::Str => {
                if let Some(s) = rest.strip_prefix('"') {
                    r.reason = s.split('"').next().unwrap_or("").to_string();
                    ok = true;
                }
            }
        }
    }
    ok.then_some(r)
}

enum Field {
    Bool,
    Num,
    Str,
}

fn main() {
    let cfg = Cfg {
        cmd: std::env::var("PROBE_CMD").unwrap_or_else(|_| default_cmd()),
        arch: env_or("BANDIT_ARCH", "dav-c220"),
        kernel: env_or("BANDIT_KERNEL", "arg_max"),
        budget: env_or("BANDIT_BUDGET", "20").parse().unwrap_or(20),
        arms: {
            let mut v = vec![Schedule::DEFAULT];
            for t in env_or("BANDIT_TILE_ARMS", "").split(',').filter_map(|s| s.trim().parse::<u32>().ok()) {
                v.push(Schedule { tile: Some(t), ..Schedule::DEFAULT });
            }
            for p in env_or("BANDIT_SPLIT_ARMS", "").split(',').filter_map(|s| s.trim().parse::<u32>().ok()) {
                v.push(Schedule { split_p: Some(p), ..Schedule::DEFAULT });
            }
            if v.len() == 1 {
                v.push(Schedule { tile: Some(1024), ..Schedule::DEFAULT });
            }
            v
        },
        state_dir: PathBuf::from(env_or(
            "BANDIT_STATE_DIR",
            &format!("{}/.tile-rs/bandit", std::env::var("HOME").unwrap_or_default()),
        )),
    };

    // 1. Baseline: the DEFAULT schedule, which is what a reward is relative to.
    eprintln!("measuring baseline (default schedule)");
    let base = probe(&cfg, &Schedule::DEFAULT, 20, 3);
    if base.fault || base.refused {
        eprintln!("baseline is unusable: {base:?}");
        std::process::exit(7);
    }
    if !base.verified {
        eprintln!("baseline did not verify ({}), refusing to normalise against it", base.reason);
        std::process::exit(1);
    }
    eprintln!("baseline {:.2} us", base.us_median);

    // 2. Applicability: drop what the kernel refuses, WITHOUT observing it.
    let mut applicable = Vec::new();
    for arm in &cfg.arms {
        if arm.is_default() {
            applicable.push(arm.clone());
            continue;
        }
        let r = probe(&cfg, arm, 1, 1);
        if r.fault {
            eprintln!("FAULT probing {}: {} -- device must be recovered, stopping", arm.arm_key(), r.reason);
            std::process::exit(7);
        }
        if r.refused {
            eprintln!("  {} not applicable here: {}", arm.arm_key(), r.reason);
            continue;
        }
        applicable.push(arm.clone());
    }
    if applicable.len() < 2 {
        eprintln!("no applicable non-default arms for this shape; nothing to tune");
        std::process::exit(2);
    }
    eprintln!("applicable arms: {}", applicable.iter().map(|a| a.arm_key()).collect::<Vec<_>>().join(", "));

    // 3. The bandit pass.
    let mut state = load_or_new(&cfg.state_dir, &cfg.arch, &cfg.kernel);
    let arms = applicable.clone();
    let base_us = base.us_median;
    let report: DriveReport = drive(&mut state, &arms, cfg.budget, |arm| {
        let r = probe(&cfg, arm, 20, 3);
        if r.fault {
            return RunOutcome::Faulted { reason: r.reason };
        }
        if !r.verified {
            return RunOutcome::Failed;
        }
        // Prefer the probe's paired reward: it compares the arm against a
        // baseline measured in the same process, whereas `base_us` was measured
        // in a different one and drifts (the same arm moved 86.8-97.8 us
        // between pulls while the baseline moved 79.4-91.1 us).
        let reward = match (r.reward, r.us_base_median) {
            (Some(x), _) => x,
            (None, Some(b)) if b > 0.0 => (b - r.us_median) / b,
            _ => (base_us - r.us_median) / base_us,
        }
        .clamp(0.0, 1.0);
        RunOutcome::Verified { reward }
    });

    // 4. Report.
    println!("\n{:>4}  {:<16} {:>9}  {}", "pull", "arm", "reward", "outcome");
    for (i, p) in report.pulls.iter().enumerate() {
        let (r, o) = match &p.outcome {
            RunOutcome::Verified { reward } => (format!("{reward:.4}"), "verified".to_string()),
            RunOutcome::Failed => ("0".into(), "failed".into()),
            RunOutcome::Faulted { reason } => ("-".into(), format!("FAULT {reason}")),
        };
        println!("{:>4}  {:<16} {:>9}  {}", i + 1, p.arm, r, o);
    }
    if let Some(f) = &report.stopped_on_fault {
        println!("\nstopped on fault: {f}");
    }
    println!("\nstate:");
    for (k, st) in &state.arms {
        println!("  {k:<16} n={:<3} mean={:.4}", st.n, st.mean());
    }
    // Best among THIS RUN's arms, not among every arm the state has ever seen
    // for this shape. `report.best_arm` iterates the whole state, so a previous
    // run's `split_p=12` won the line while not being measured here at all --
    // correct history, wrong report.
    let best = arms
        .iter()
        .filter_map(|a| {
            state.arms.get(&a.arm_key()).map(|st| (a.arm_key(), st.mean(), st.n))
        })
        .filter(|(_, _, n)| *n > 0)
        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    match best {
        Some((k, m, n)) => println!("\nbest arm in this run: {k}  (mean {m:.4} over {n} pulls)"),
        None => println!("\nno arm produced a verified reward in this run"),
    }
    if let Err(e) = save(&cfg.state_dir, &state) {
        eprintln!("warning: could not save state: {e}");
    } else {
        println!("state saved under {}", cfg.state_dir.display());
    }
}
