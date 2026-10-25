//! The KernelBand loop at the codegen seam: select an arm, run it, gate on
//! verify, observe only what verified, persist.
//!
//! This is the driver the paper's Algorithm 1 describes, with two deviations
//! that our hardware forces and that are therefore in the types rather than in
//! a comment:
//!
//! 1. **A fault is not a reward of zero.** On Ascend a vector-core timeout
//!    (`507034`) leaves the device unusable: every later measurement on it is
//!    wrong. The default suite reported 2/72 failures *after* such a fault that
//!    had been 0/72 before it, which is the fault being charged to arms that
//!    never ran. So [`RunOutcome`] separates "ran and was wrong" from "the
//!    measurement is invalid", and a fault stops the loop instead of feeding
//!    the next arm a poisoned device.
//! 2. **Nothing unverified-and-valid is ever observed.** A *failed* run is a
//!    measurement and is observed as reward 0 (the paper's rule, and the only
//!    way to stop an arm with no statistic from scoring UCB +inf forever); a
//!    *faulted* run is not a measurement at all and is never observed.
//!
//! The loop itself is hardware-free: `run` is a closure, so the bench's
//! build-gate-measure pipeline drops in without this module knowing about it.

use std::fs;
use std::path::Path;

use crate::bandit::BanditState;
use crate::schedule::Schedule;

/// What one arm's run produced.
#[derive(Clone, Debug, PartialEq)]
pub enum RunOutcome {
    /// Ran and passed the oracle. `reward` is the normalised speed-up
    /// `clip((t_before - t_after) / t_before, 0, 1)`; the paper's reward.
    Verified { reward: f64 },
    /// Ran, but the oracle rejected it or it was not faster. This IS a
    /// measurement, and the paper counts it as reward 0 -- so it is observed.
    /// Leaving it unobserved is not the stricter option, it is a livelock:
    /// an arm with no statistic scores UCB +inf forever, so the bandit would
    /// re-pull the same failing arm for the whole budget and never reach the
    /// arms it has not tried. What must never be cached is an INVALID
    /// measurement ([`RunOutcome::Faulted`]), not an unfavourable one.
    Failed,
    /// The measurement is invalid: a device fault, not an answer. The caller
    /// must recover the device before any further pull.
    Faulted { reason: String },
}

/// One row of the pull-by-pull trace, which is what makes the sample-efficiency
/// claim checkable rather than asserted.
#[derive(Clone, Debug, PartialEq)]
pub struct Pull {
    pub arm: String,
    pub n_after: u32,
    pub outcome: RunOutcome,
}

/// The trace plus why the loop stopped early, if it did.
#[derive(Clone, Debug, PartialEq)]
pub struct DriveReport {
    pub pulls: Vec<Pull>,
    pub stopped_on_fault: Option<String>,
}

impl DriveReport {
    /// Arms that produced a verified reward, in pull order.
    pub fn verified_arms(&self) -> Vec<&str> {
        self.pulls
            .iter()
            .filter(|p| matches!(p.outcome, RunOutcome::Verified { .. }))
            .map(|p| p.arm.as_str())
            .collect()
    }

    /// The arm with the best mean reward in `state` at the end of the run.
    pub fn best_arm(&self, state: &BanditState) -> Option<String> {
        state
            .arms
            .iter()
            .filter(|(_, st)| st.n > 0)
            .max_by(|a, b| {
                a.1.mean()
                    .partial_cmp(&b.1.mean())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // deterministic on a tie: lexicographically smaller key wins
                    .then_with(|| b.0.cmp(a.0))
            })
            .map(|(k, _)| k.clone())
    }
}

/// Run the bandit for at most `budget` pulls over `arms`, calling `run` to
/// execute each selected arm.
///
/// Stops early on a fault, and stops when `select_arm` refuses (an empty legal
/// set), because inventing an arm is exactly what that refusal exists to
/// prevent.
pub fn drive<F>(state: &mut BanditState, arms: &[Schedule], budget: usize, run: F) -> DriveReport
where
    F: FnMut(&Schedule) -> RunOutcome,
{
    drive_with(state, arms, budget, |st, a| st.select_arm(a), run)
}

/// The same loop with KernelBand's pruning mask applied at every pull.
///
/// The mask is re-consulted each pull rather than applied once, because the
/// signature is a measurement: re-profiling after a few pulls is exactly how
/// the paper's `tau = 10` re-clustering is meant to change what is admissible.
pub fn drive_masked<F>(
    state: &mut BanditState,
    arms: &[Schedule],
    sig: &crate::hardware::HwSignature,
    theta: f64,
    budget: usize,
    run: F,
) -> DriveReport
where
    F: FnMut(&Schedule) -> RunOutcome,
{
    drive_with(state, arms, budget, |st, a| st.select_arm_masked(a, sig, theta), run)
}

fn drive_with<F, S>(state: &mut BanditState, arms: &[Schedule], budget: usize, mut select: S, mut run: F) -> DriveReport
where
    F: FnMut(&Schedule) -> RunOutcome,
    S: FnMut(&BanditState, &[Schedule]) -> Result<Schedule, String>,
{
    let mut pulls = Vec::new();
    for _ in 0..budget {
        let Ok(arm) = select(state, arms) else {
            break;
        };
        match run(&arm) {
            RunOutcome::Verified { reward } => {
                let r = reward.clamp(0.0, 1.0);
                let outcome = if r.is_finite() {
                    state.observe(&arm, r);
                    RunOutcome::Verified { reward: r }
                } else {
                    // observe() silently drops a non-finite reward; surface it
                    // as a failure instead of logging a pull that did nothing.
                    RunOutcome::Failed
                };
                let n_after = state.arms.get(&arm.arm_key()).map_or(0, |s| s.n);
                pulls.push(Pull { arm: arm.arm_key(), n_after, outcome });
            }
            RunOutcome::Failed => {
                state.observe(&arm, 0.0);
                let n_after = state.arms.get(&arm.arm_key()).map_or(0, |s| s.n);
                pulls.push(Pull { arm: arm.arm_key(), n_after, outcome: RunOutcome::Failed });
            }
            RunOutcome::Faulted { reason } => {
                let n_after = state.arms.get(&arm.arm_key()).map_or(0, |s| s.n);
                pulls.push(Pull {
                    arm: arm.arm_key(),
                    n_after,
                    outcome: RunOutcome::Faulted { reason: reason.clone() },
                });
                return DriveReport { pulls, stopped_on_fault: Some(reason) };
            }
        }
    }
    DriveReport { pulls, stopped_on_fault: None }
}

/// Load the state for `(arch, kernel)` from `dir`, or start empty.
///
/// A missing file is an empty state, not an error: the first pull of a new
/// operator is normal. An unreadable file is also empty rather than fatal,
/// because losing bandit history must never lose a run.
pub fn load_or_new(dir: &Path, arch: &str, kernel: &str) -> BanditState {
    let mut fresh = BanditState::new(arch, kernel);
    let path = fresh.path_under(dir);
    match fs::read_to_string(&path) {
        Ok(text) => {
            let loaded = BanditState::from_json(&text);
            // A file that parsed but lost its identity would merge two chips'
            // statistics, which is the one thing the key exists to prevent.
            if loaded.arch == arch && loaded.kernel == kernel {
                fresh = loaded;
            }
        }
        Err(_) => {}
    }
    fresh
}

/// Write the state back. Best-effort: a failed save must not fail the run.
pub fn save(dir: &Path, state: &BanditState) -> std::io::Result<()> {
    let path = state.path_under(dir);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, state.to_json())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms() -> Vec<Schedule> {
        Schedule::DEFAULT.legal_arms()
    }

    fn key(n: u32) -> String {
        format!("k_unroll={n}")
    }

    #[test]
    fn first_pass_walks_every_legal_arm_once() {
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let report = drive(&mut st, &legal, legal.len(), |_| RunOutcome::Failed);
        assert_eq!(report.pulls.len(), legal.len());
        // One pull per arm: no arm was examined twice before all were seen once.
        let mut seen: Vec<String> = report.pulls.iter().map(|p| p.arm.clone()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), legal.len());
    }

    #[test]
    fn a_failed_run_is_observed_as_zero_reward_so_the_arm_is_not_re_pulled() {
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let report = drive(&mut st, &legal, legal.len(), |_| RunOutcome::Failed);
        // Every arm was tried exactly once and carries a zero mean -- if a
        // failure left no statistic, the UCB +inf would re-pull one arm for the
        // whole budget (see the doc comment on RunOutcome::Failed).
        assert_eq!(st.arms.len(), legal.len());
        assert!(st.arms.values().all(|s| s.n == 1 && s.mean() == 0.0));
        assert_eq!(report.pulls.len(), legal.len());
        // and none of them counts as verified
        assert!(report.verified_arms().is_empty());
    }

    #[test]
    fn a_fault_stops_the_loop_and_is_not_charged_to_the_next_arm() {
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let mut n_calls = 0;
        let report = drive(&mut st, &legal, 40, |_| {
            n_calls += 1;
            if n_calls == 3 {
                RunOutcome::Faulted { reason: "vector core timeout 507034".into() }
            } else {
                RunOutcome::Verified { reward: 0.5 }
            }
        });
        assert_eq!(n_calls, 3, "must not run again on a poisoned device");
        assert_eq!(report.stopped_on_fault.as_deref(), Some("vector core timeout 507034"));
        assert_eq!(report.pulls.last().unwrap().n_after, 0, "the faulted arm got no statistic");
        // Two verified pulls, two statistics -- and the faulted arm is absent.
        assert_eq!(st.total_n(), 2);
        assert_eq!(st.arms.len(), 2);
    }

    #[test]
    fn reward_is_clamped_and_non_finite_is_a_failure() {
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let report = drive(&mut st, &legal, 3, |a| {
            if a.k_unroll == Some(64) {
                RunOutcome::Verified { reward: 7.5 } // out of range
            } else if a.k_unroll == Some(32) {
                RunOutcome::Verified { reward: f64::NAN }
            } else {
                RunOutcome::Verified { reward: -1.0 }
            }
        });
        for p in &report.pulls {
            if let RunOutcome::Verified { reward } = p.outcome {
                assert!((0.0..=1.0).contains(&reward), "reward {reward} not clamped");
            }
        }
        // NaN became a Failed pull and left no statistic.
        assert!(st.arms.get("k_unroll=32").is_none());
    }

    #[test]
    fn the_best_arm_wins_after_equal_exploration() {
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let report = drive(&mut st, &legal, 4 * legal.len(), |a| {
            let r = match a.k_unroll {
                Some(16) => 0.9,
                Some(8) => 0.4,
                _ => 0.1,
            };
            RunOutcome::Verified { reward: r }
        });
        assert_eq!(report.best_arm(&st).as_deref(), Some(key(16).as_str()));
        // and it is the most-pulled arm once exploration is paid for
        let n16 = st.arms.get(&key(16)).unwrap().n;
        assert!(st.arms.values().all(|s| s.n <= n16), "{:?}", st.arms);
    }

    #[test]
    fn verified_arms_lists_only_observed_ones() {
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let report = drive(&mut st, &legal, legal.len(), |a| {
            if a.k_unroll == Some(2) || a.k_unroll == Some(4) {
                RunOutcome::Failed
            } else {
                RunOutcome::Verified { reward: 0.2 }
            }
        });
        let verified = report.verified_arms();
        assert!(!verified.contains(&key(2).as_str()));
        assert!(!verified.contains(&key(4).as_str()));
        assert_eq!(verified.len(), legal.len() - 2);
    }

    #[test]
    fn persistence_round_trips_and_a_foreign_file_is_ignored() {
        let dir = std::env::temp_dir().join(format!("tile-rs-driver-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut st = BanditState::new("c220", "arg_max");
        st.observe(&Schedule { k_unroll: Some(16), ..Schedule::DEFAULT }, 0.75);
        save(&dir, &st).unwrap();
        let back = load_or_new(&dir, "c220", "arg_max");
        assert_eq!(back.arch, "c220");
        assert_eq!(back.kernel, "arg_max");
        assert_eq!(back.arms.get("k_unroll=16").map(|s| s.n), Some(1));
        // another kernel does not see it
        let other = load_or_new(&dir, "c220", "softmax");
        assert!(other.arms.is_empty());
        // and a c310 file can never be read as c220's
        let foreign = load_or_new(&dir, "c310", "arg_max");
        assert!(foreign.arms.is_empty(), "arch must not share statistics");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_masked_loop_never_runs_an_arm_whose_resource_is_pinned() {
        use crate::hardware::{HwSignature, SATURATION};
        let legal = arms();
        let mut st = BanditState::new("c220", "arg_max");
        let sig = HwSignature { ub: 0.95, ..Default::default() };
        let mut ran = Vec::new();
        let report = drive_masked(&mut st, &legal, &sig, SATURATION, 30, |a| {
            ran.push(a.arm_key());
            RunOutcome::Verified { reward: 0.1 }
        });
        assert!(!report.pulls.is_empty());
        for k in &ran {
            assert!(!k.starts_with("tile="), "a UB-targeting arm reached the device: {k}");
        }
        // and the other families are still explored
        assert!(ran.iter().any(|k| k.starts_with("k_unroll=")), "{ran:?}");
    }

    #[test]
    fn an_empty_legal_set_stops_instead_of_inventing_an_arm() {
        let mut st = BanditState::new("c220", "arg_max");
        let report = drive(&mut st, &[], 10, |_| RunOutcome::Verified { reward: 1.0 });
        assert!(report.pulls.is_empty());
    }
}
