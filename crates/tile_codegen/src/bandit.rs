//! UCB1 bandit over legal [`Schedule`] arms — KernelBand at the codegen seam.
//!
//! Arms are schedule choices (not threadgroup widths). Rewards come from a
//! **verified** measured run only: `observe` is never called for a candidate
//! that failed verify, so the state file cannot encode “fast and wrong”.
//!
//! Gating: selection is opt-in (`TILE_SCHEDULE_BANDIT=1`). With the gate off,
//! callers keep their existing exhaustive/default path — identical behavior.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::hardware::HwSignature;
use crate::schedule::Schedule;

/// One arm's running statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArmStat {
    pub n: u32,
    pub sum_reward: f64,
}

impl ArmStat {
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum_reward / f64::from(self.n)
        }
    }
}

/// Persistent UCB state, keyed by hardware identity + kernel id.
#[derive(Clone, Debug, PartialEq)]
pub struct BanditState {
    /// Hardware id (`HardwareParams::npu_arch` / chip). Never shared across chips.
    pub arch: String,
    /// Kernel/form id so one form's stats do not pollute another's.
    pub kernel: String,
    /// arm_key → stats
    pub arms: BTreeMap<String, ArmStat>,
    /// UCB exploration constant (KernelBand default 2.0).
    pub c: f64,
}

impl BanditState {
    pub fn new(arch: impl Into<String>, kernel: impl Into<String>) -> Self {
        Self {
            arch: arch.into(),
            kernel: kernel.into(),
            arms: BTreeMap::new(),
            c: 2.0,
        }
    }

    /// Total pulls across arms (for UCB normalisation).
    pub fn total_n(&self) -> u64 {
        self.arms.values().map(|a| u64::from(a.n)).sum()
    }

    /// UCB1 score for an arm; un-pulled arms score `+∞` so they are tried first.
    pub fn ucb(&self, key: &str) -> f64 {
        match self.arms.get(key) {
            None => f64::INFINITY,
            Some(st) if st.n == 0 => f64::INFINITY,
            Some(st) => {
                let t = f64::from(self.total_n().max(1) as u32);
                st.mean() + self.c * (t.ln() / f64::from(st.n)).sqrt()
            }
        }
    }

    /// Pick the arm with the highest UCB among `legal`. Deterministic tie-break
    /// by arm key so two empty states select the same arm.
    ///
    /// Refuses an empty legal set rather than inventing an arm.
    pub fn select_arm(&self, legal: &[Schedule]) -> Result<Schedule, String> {
        if legal.is_empty() {
            return Err("bandit: no legal arms to select".into());
        }
        let mut best: Option<(&Schedule, f64)> = None;
        for s in legal {
            let key = s.arm_key();
            let score = self.ucb(&key);
            match best {
                None => best = Some((s, score)),
                Some((_, bs)) if score > bs => best = Some((s, score)),
                Some((bs_sch, bs)) if score == bs => {
                    // deterministic: lexicographically smaller arm key wins
                    if s.arm_key() < bs_sch.arm_key() {
                        best = Some((s, score));
                    }
                }
                _ => {}
            }
        }
        Ok(best.expect("legal non-empty").0.clone())
    }

    /// UCB over the arms whose target resource still has headroom -- the
    /// paper's masked index, with [`HwSignature`] as `h(k)` and `theta` as
    /// `theta_sat`.
    ///
    /// This can only NARROW the choice: the DEFAULT arm asks for nothing and so
    /// is always admissible, and an empty admissible set falls back to the
    /// unmasked list rather than inventing an arm. A caller with no counters
    /// passes [`HwSignature::default()`], which prunes nothing and reproduces
    /// `select_arm` exactly.
    pub fn select_arm_masked(
        &self,
        legal: &[Schedule],
        sig: &HwSignature,
        theta: f64,
    ) -> Result<Schedule, String> {
        let admissible: Vec<Schedule> = legal
            .iter()
            .filter(|a| a.is_default() || !a.mask_blocked(sig, theta))
            .cloned()
            .collect();
        if admissible.is_empty() {
            return self.select_arm(legal);
        }
        self.select_arm(&admissible)
    }

    /// Record a **verified** reward for `arm`. Callers that have not run
    /// verify must not call this — the API has no “trust me” flag.
    pub fn observe(&mut self, arm: &Schedule, reward: f64) {
        if !reward.is_finite() {
            return;
        }
        let st = self.arms.entry(arm.arm_key()).or_default();
        st.n += 1;
        st.sum_reward += reward;
    }

    /// Same-path JSON (no serde): enough for `~/.tile-rs/bandit/{arch}__{kernel}.json`.
    pub fn to_json(&self) -> String {
        let mut arms = String::new();
        for (k, st) in &self.arms {
            if !arms.is_empty() {
                arms.push(',');
            }
            arms.push_str(&format!(
                "\"{}\":{{\"n\":{},\"sum_reward\":{}}}",
                k.escape_debug(),
                st.n,
                st.sum_reward
            ));
        }
        format!(
            "{{\"arch\":\"{}\",\"kernel\":\"{}\",\"c\":{},\"arms\":{{{}}}}}",
            self.arch.escape_debug(),
            self.kernel.escape_debug(),
            self.c,
            arms
        )
    }

    /// Minimal parse of [`Self::to_json`]. Unknown fields ignored; missing → empty state.
    pub fn from_json(s: &str) -> BanditState {
        let mut st = BanditState::new("", "");
        if let Some(v) = json_str(s, "arch") {
            st.arch = v;
        }
        if let Some(v) = json_str(s, "kernel") {
            st.kernel = v;
        }
        if let Some(v) = json_num(s, "c") {
            st.c = v;
        }
        if let Some(arms_start) = s.find("\"arms\"") {
            let rest = &s[arms_start..];
            if let Some(brace) = rest.find('{') {
                // Arms body runs to the matching close brace (first `}` is
                // the end of arm 0's stats object, not the map).
                let body_start = brace + 1;
                let mut depth = 1i32;
                let mut end = rest.len();
                for (idx, ch) in rest[body_start..].char_indices() {
                    match ch {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = body_start + idx;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let body = &rest[body_start..end];
                // "key":{"n":1,"sum_reward":2.0} — walk every quoted key;
                // nested "n"/"sum_reward" keys parse with n=0 and are skipped.
                let mut i = 0;
                let b = body.as_bytes();
                while i < b.len() {
                    while i < b.len() && b[i] != b'"' {
                        i += 1;
                    }
                    if i >= b.len() {
                        break;
                    }
                    let ks = i + 1;
                    let ke = match body[ks..].find('"') {
                        Some(p) => ks + p,
                        None => break,
                    };
                    let key = body[ks..ke].to_string();
                    i = ke + 1;

                    // An ARM key is followed by `:{`. A nested statistics key is
                    // followed by `:` and a number. Without this test the nested
                    // `"n"` and `"sum_reward"` were read as arms -- and because
                    // the value below used to run to the end of the whole map,
                    // they even slurped counts from the NEXT arm's object. That
                    // produced phantom arms, inflated `total_n` (which feeds the
                    // UCB exploration term), and would corrupt a real arm's
                    // statistics if a phantom key ever collided with one.
                    let after = body[i..].trim_start();
                    let Some(rest) = after.strip_prefix(':') else { continue };
                    let rest = rest.trim_start();
                    if !rest.starts_with('{') {
                        continue;
                    }
                    // Bound the value to THIS object, so a missing field cannot
                    // be satisfied by the next arm's copy of it.
                    let mut depth = 0i32;
                    let mut endv = rest.len();
                    for (idx, ch) in rest.char_indices() {
                        match ch {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    endv = idx + 1;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    let value = &rest[..endv];
                    let n = json_num(value, "n").unwrap_or(0.0) as u32;
                    let sum = json_num(value, "sum_reward").unwrap_or(0.0);
                    if n > 0 {
                        st.arms.insert(key, ArmStat { n, sum_reward: sum });
                    }
                }
            }
        }
        st
    }

    /// Path for this (arch, kernel) pair under `dir`.
    pub fn path_under(&self, dir: &Path) -> PathBuf {
        let safe_arch: String = self
            .arch
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let safe_k: String = self
            .kernel
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        dir.join(format!("{safe_arch}__{safe_k}.json"))
    }
}

fn json_str(s: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = s.find(&pat)?;
    let rest = &s[i + pat.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn json_num(s: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = s.find(&pat)?;
    let rest = &s[i + pat.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E' || *c == '+')
        .collect();
    digits.parse().ok()
}

/// Env gate: unset / empty / `0` = off (callers keep exhaustive/default path).
pub fn bandit_enabled() -> bool {
    match std::env::var("TILE_SCHEDULE_BANDIT") {
        Ok(v) => !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HwSignature;
use crate::schedule::Schedule;

    fn arms() -> Vec<Schedule> {
        Schedule::DEFAULT.legal_arms()
    }

    #[test]
    fn unpulled_arms_score_infinity_and_are_tried_first() {
        let st = BanditState::new("metal", "softmax");
        assert_eq!(st.ucb("default"), f64::INFINITY);
        let a = st.select_arm(&arms()).unwrap();
        // first selection: unexplored arms all tie at ∞ → lexicographic key
        // "default" < "k_unroll=..." so default wins the first pull
        assert_eq!(a.arm_key(), "default");
    }

    #[test]
    fn every_arm_is_pulled_once_before_any_is_pulled_twice() {
        let mut st = BanditState::new("metal", "matmul");
        let legal = arms();
        // ∞-tie break is lexicographic arm_key, not legal_arms declaration order.
        let mut expected_order = legal.clone();
        expected_order.sort_by_key(|a| a.arm_key());
        for expected in &expected_order {
            let picked = st.select_arm(&legal).unwrap();
            assert_eq!(
                picked.arm_key(),
                expected.arm_key(),
                "first pass must walk arms in select order"
            );
            st.observe(&picked, 1.0);
        }
        // all pulled once → next select is a scored arm, not ∞-only
        let next = st.select_arm(&legal).unwrap();
        assert!(st.ucb(&next.arm_key()).is_finite());
    }

    #[test]
    fn a_higher_reward_arm_eventually_wins_after_equal_exploration() {
        let mut st = BanditState::new("metal", "matmul");
        let fast = Schedule { k_unroll: Some(16), ..Schedule::DEFAULT };
        let slow = Schedule { k_unroll: Some(1), ..Schedule::DEFAULT };
        // equal pull counts, different means
        for _ in 0..20 {
            st.observe(&fast, 10.0);
            st.observe(&slow, 1.0);
        }
        let legal = vec![fast.clone(), slow.clone()];
        let pick = st.select_arm(&legal).unwrap();
        assert_eq!(pick.k_unroll, Some(16));
    }

    #[test]
    fn a_saturated_resource_prunes_the_arms_that_target_it() {
        use crate::hardware::{HwSignature, SATURATION};
        let legal = Schedule::DEFAULT.legal_arms();
        let mut st = BanditState::new("c220", "arg_max");
        let sig = HwSignature { ub: 0.95, ..Default::default() };
        let mut seen = std::collections::BTreeSet::new();
        // NOTE: an unobserved arm keeps UCB = +inf, so selection without
        // recording a pull returns the SAME arm every time (the inf tie-break is
        // lexicographic). Advancing the state here is what makes this a test of
        // the mask rather than a test of the tie-break.
        for _ in 0..60 {
            let a = st.select_arm_masked(&legal, &sig, SATURATION).unwrap();
            assert!(a.tile.is_none(), "tile targets UB, which is saturated: {a:?}");
            seen.insert(a.arm_key());
            st.observe(&a, 0.1);
        }
        // ...and the families targeting other resources are still reachable.
        assert!(seen.iter().any(|k| k.starts_with("k_unroll=")), "{seen:?}");
    }

    #[test]
    fn un_saturating_re_admits_the_arm() {
        use crate::hardware::{HwSignature, SATURATION};
        let legal = Schedule::DEFAULT.legal_arms();
        let hot = HwSignature { ub: 0.95, ..Default::default() };
        let cool = HwSignature { ub: 0.40, ..Default::default() };
        let sweep = |sig: &HwSignature| {
            let mut st = BanditState::new("c220", "arg_max");
            let mut seen = std::collections::BTreeSet::new();
            for _ in 0..60 {
                let a = st.select_arm_masked(&legal, sig, SATURATION).unwrap();
                seen.insert(a.arm_key());
                st.observe(&a, 0.1);
            }
            seen
        };
        let blocked = sweep(&hot);
        let admitted = sweep(&cool);
        assert!(!blocked.iter().any(|k| k.starts_with("tile=")), "{blocked:?}");
        assert!(admitted.iter().any(|k| k.starts_with("tile=")), "{admitted:?}");
    }

    #[test]
    fn everything_saturated_still_returns_the_default_not_an_error() {
        use crate::hardware::{HwSignature, SATURATION};
        let legal = Schedule::DEFAULT.legal_arms();
        let st = BanditState::new("c220", "arg_max");
        let pinned = HwSignature { dram: 0.99, mte2: 0.99, vector: 0.99, ub: 0.99, latency: 0.99 };
        let a = st.select_arm_masked(&legal, &pinned, SATURATION).unwrap();
        assert!(a.is_default(), "the do-nothing arm is always admissible, got {a:?}");
    }

    #[test]
    fn an_unmeasured_signature_matches_unmasked_selection() {
        use crate::hardware::{HwSignature, SATURATION};
        let legal = Schedule::DEFAULT.legal_arms();
        let a = BanditState::new("c220", "arg_max");
        let b = BanditState::new("c220", "arg_max");
        for _ in 0..25 {
            let plain = a.select_arm(&legal).unwrap();
            let masked = b.select_arm_masked(&legal, &HwSignature::default(), SATURATION).unwrap();
            assert_eq!(plain.arm_key(), masked.arm_key());
        }
    }

    #[test]
    fn observe_rejects_non_finite_rewards() {
        let mut st = BanditState::new("metal", "k");
        let a = Schedule::DEFAULT;
        st.observe(&a, f64::NAN);
        st.observe(&a, f64::INFINITY);
        assert!(st.arms.is_empty(), "NaN/Inf must not pollute the state");
    }

    #[test]
    fn a_multi_arm_roundtrip_produces_no_phantom_arms() {
        // The nested `"n"` / `"sum_reward"` keys must never become arms, and a
        // missing field must not be satisfied by the next arm's copy. With one
        // arm this bug is invisible; with several it is not.
        let mut st = BanditState::new("dav-c220", "arg_max");
        st.observe(&Schedule { k_unroll: Some(2), ..Schedule::DEFAULT }, 0.25);
        st.observe(&Schedule { k_unroll: Some(2), ..Schedule::DEFAULT }, 0.75);
        st.observe(&Schedule { tile: Some(512), ..Schedule::DEFAULT }, 0.5);
        let back = BanditState::from_json(&st.to_json());
        assert_eq!(back.arms.len(), st.arms.len(), "arms: {:?}", back.arms.keys().collect::<Vec<_>>());
        assert!(!back.arms.contains_key("n"), "nested `n` became an arm");
        assert!(!back.arms.contains_key("sum_reward"), "nested `sum_reward` became an arm");
        assert_eq!(back.arms.get("k_unroll=2").map(|a| a.n), Some(2));
        assert!((back.arms.get("k_unroll=2").unwrap().sum_reward - 1.0).abs() < 1e-9);
        assert_eq!(back.total_n(), st.total_n());
    }

    #[test]
    fn json_roundtrip_preserves_arch_kernel_and_stats() {
        let mut st = BanditState::new("DAV_2201", "gemm/k=256");
        st.observe(&Schedule::DEFAULT, 3.5);
        st.observe(&Schedule { k_unroll: Some(4), ..Schedule::DEFAULT }, 1.25);
        let j = st.to_json();
        let back = BanditState::from_json(&j);
        assert_eq!(back.arch, st.arch);
        assert_eq!(back.kernel, st.kernel);
        assert_eq!(back.arms.get("default").unwrap().n, 1);
        assert!((back.arms.get("default").unwrap().sum_reward - 3.5).abs() < 1e-9);
        assert_eq!(back.arms.get("k_unroll=4").unwrap().n, 1);
    }

    #[test]
    fn state_paths_do_not_collide_across_arches_or_kernels() {
        let a = BanditState::new("DAV_2201", "softmax").path_under(Path::new("/tmp/b"));
        let b = BanditState::new("DAV_3510", "softmax").path_under(Path::new("/tmp/b"));
        let c = BanditState::new("DAV_2201", "matmul").path_under(Path::new("/tmp/b"));
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert!(a.to_string_lossy().ends_with(".json"));
    }

    #[test]
    fn select_refuses_an_empty_legal_set() {
        let st = BanditState::new("m", "k");
        assert!(st.select_arm(&[]).is_err());
    }

    #[test]
    fn bandit_is_off_unless_explicitly_enabled() {
        // tests must not inherit a developer's shell export as a silent on-switch
        // (set_var/remove_var are unsafe on edition 2024)
        unsafe {
            std::env::remove_var("TILE_SCHEDULE_BANDIT");
        }
        assert!(!bandit_enabled());
        unsafe {
            std::env::set_var("TILE_SCHEDULE_BANDIT", "1");
        }
        assert!(bandit_enabled());
        unsafe {
            std::env::set_var("TILE_SCHEDULE_BANDIT", "0");
        }
        assert!(!bandit_enabled());
        unsafe {
            std::env::remove_var("TILE_SCHEDULE_BANDIT");
        }
    }
}
