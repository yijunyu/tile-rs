//! The candidate frontier `P_t` and the per-strategy transitions.
//!
//! KernelBand's action is a PAIR `(k, s)`: a candidate kernel `k` from the
//! frontier, and a strategy `s` from Table 6 of the paper, applied to it by a
//! generator to produce `k'`. Without a frontier there is nothing for a strategy
//! to be applied *to*, which is why the loop is not simply a knob sweep.
//!
//! Two kinds of strategy, and they are kept apart deliberately:
//!
//! * **Parameter strategies** — Tiling, Vectorization, Pipeline. These set a
//!   value that lowering honours, so they are [`Schedule`] fields and the
//!   existing safety contract (sound for every arm, `default` = today) covers
//!   them. `Schedule::legal_arms` already enumerates them.
//! * **Code-shape strategies** — Fusion, Reordering, Access & Layout. There is
//!   no value to set: Fusion removes a launch or a pass, Reordering swaps one
//!   loop body for another, Access & Layout changes an address expression. Each
//!   is a DIFFERENT KERNEL, so it is a flag on a [`Candidate`] and never a
//!   `Schedule` field -- putting one in `Schedule` would break the property that
//!   lowering is sound for every arm.
//!
//! Transitions are declared as data ([`TRANSFORMS`]) rather than performed here.
//! A declared transform names the flag a materializer must set (for the AscendC
//! kernels in this project, an environment variable beside `ARGMAX_SCHEDULE`),
//! and whether the tree already contains both variants. `applied` is honest
//! about that: a transform marked `false` has a NAME and no body, and the
//! frontier will still propose it -- as a candidate that must not be measured
//! until someone writes it.

use std::collections::{BTreeMap, BTreeSet};

use crate::hardware::Resource;
use crate::schedule::{Schedule, TargetKnobs};

/// The paper's six optimization strategies (Table 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strategy {
    Tiling,
    Vectorization,
    Fusion,
    Pipeline,
    Reordering,
    AccessLayout,
}

impl Strategy {
    pub const ALL: [Strategy; 6] = [
        Strategy::Tiling,
        Strategy::Vectorization,
        Strategy::Fusion,
        Strategy::Pipeline,
        Strategy::Reordering,
        Strategy::AccessLayout,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Strategy::Tiling => "tiling",
            Strategy::Vectorization => "vectorization",
            Strategy::Fusion => "fusion",
            Strategy::Pipeline => "pipeline",
            Strategy::Reordering => "reordering",
            Strategy::AccessLayout => "access_layout",
        }
    }

    /// True when the strategy is a value the schedule can carry.
    pub fn is_parameter(&self) -> bool {
        matches!(self, Strategy::Tiling | Strategy::Vectorization | Strategy::Pipeline)
    }

    /// The schedule knobs this strategy owns. A transition may only move THESE:
    /// a Tiling transition that returned a `k_unroll` arm would be a different
    /// strategy's action wearing its name, and the reward would be attributed to
    /// the wrong arm.
    pub fn knobs(&self) -> &'static [&'static str] {
        match self {
            Strategy::Tiling => &["tile", "split_p", "row_block", "k_unroll"],
            Strategy::Vectorization => &["split_chunk", "desc_count"],
            Strategy::Pipeline => &["buffer_num"],
            _ => &[],
        }
    }

    /// The resource this strategy is trying to recover, for the pruning mask.
    /// `None` for a strategy whose gain is not a saturation play. These are the
    /// paper's `Target(s)`; CALIBRATE (see `hardware.rs`).
    pub fn target(&self) -> Option<Resource> {
        match self {
            Strategy::Tiling => Some(Resource::Ub),
            Strategy::Vectorization => Some(Resource::Mte2),
            Strategy::Fusion => Some(Resource::Dram),
            Strategy::Pipeline => Some(Resource::Latency),
            Strategy::Reordering => Some(Resource::Vector),
            Strategy::AccessLayout => Some(Resource::Dram),
        }
    }
}

/// Code-shape flags a materializer must honour. Named, not valued.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Flags(pub BTreeMap<String, String>);

impl Flags {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn with(mut self, k: &str, v: &str) -> Self {
        self.0.insert(k.to_string(), v.to_string());
        self
    }

    pub fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// One kernel implementation in the frontier.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// Stable identity, unique in the frontier.
    pub id: String,
    /// Parameter knobs this candidate carries.
    pub schedule: Schedule,
    /// Code-shape flags. Empty for a purely parameter candidate.
    pub flags: Flags,
    /// Which candidate this was derived from, and by which strategy — the
    /// provenance that makes a reward attributable to an arm.
    pub parent: Option<String>,
    pub derived_by: Option<Strategy>,
}

impl Candidate {
    pub fn seed() -> Self {
        Self {
            id: "default".into(),
            schedule: Schedule::DEFAULT,
            flags: Flags::new(),
            parent: None,
            derived_by: None,
        }
    }

    pub fn with_schedule(id: &str, schedule: Schedule, parent: &str, s: Strategy) -> Self {
        Self {
            id: id.into(),
            schedule,
            flags: Flags::new(),
            parent: Some(parent.into()),
            derived_by: Some(s),
        }
    }

    /// A flag-only step. `schedule` is passed through from the parent: applying
    /// a code-shape strategy to a candidate keeps its knob choices and changes
    /// its shape. Defaulting it here silently discarded the parent's tiling,
    /// which is the one thing a chain must not do.
    pub fn with_flags(id: &str, schedule: Schedule, flags: Flags, parent: &str, s: Strategy) -> Self {
        Self {
            id: id.into(),
            schedule,
            flags,
            parent: Some(parent.into()),
            derived_by: Some(s),
        }
    }
}

/// A declared code-shape transform: a name, the strategy it belongs to, the
/// flag a materializer must set, and whether the tree already has the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transform {
    pub name: &'static str,
    pub strategy: Strategy,
    pub flag: &'static str,
    /// `true` only when BOTH variants already exist in the tree, so measuring
    /// this candidate costs a run and not a rewrite.
    pub applied: bool,
    /// Where the two variants live, so `applied` can be checked rather than
    /// believed. One file, two paths.
    pub evidence: &'static str,
}

/// Code-shape transforms per operator. Only `arg_max` is populated: the others
/// get a table when someone has actually read their two variants, and a
/// fabricated entry here would propose a candidate that cannot be materialized.
pub const TRANSFORMS_ARG_MAX: &[Transform] = &[
    Transform { name: "drop-prep-zero-fill", strategy: Strategy::Fusion, flag: "ARGMAX_FUSE=drop_prep",
                applied: true, evidence: "arg_max_kernel.cpp: the _prep zero-fill launch; the P>1 path writes every slot before the merge reads it" },
    Transform { name: "one-launch-vs-two", strategy: Strategy::Fusion, flag: "ARGMAX_FUSE=one_launch",
                applied: false, evidence: "Last1D already merges in a second launch; the single-launch form is not written" },
    Transform { name: "dense-vs-interleaved-split", strategy: Strategy::Reordering, flag: "ARGMAX_ORDER=interleaved",
                applied: true, evidence: "arg_max_kernel.cpp: ComputeBatch vs ComputeBatchInterleaved, gated on kInterleaveChunks" },
    Transform { name: "one-descriptor-per-tile", strategy: Strategy::AccessLayout, flag: "ARGMAX_DESC=one",
                applied: true, evidence: "arg_max_kernel.cpp IssueChunk: 'one descriptor, always' (commit 6eff92d)" },
    Transform { name: "fractal-nz-layout", strategy: Strategy::AccessLayout, flag: "ARGMAX_LAYOUT=nz",
                applied: false, evidence: "no NZ path in arg_max; the cube targets have one" },
];

pub fn transforms_for(operator: &str) -> &'static [Transform] {
    match operator {
        "arg_max" => TRANSFORMS_ARG_MAX,
        _ => &[],
    }
}

/// The frontier of candidate implementations.
#[derive(Clone, Debug, Default)]
pub struct Frontier {
    candidates: Vec<Candidate>,
}

impl Frontier {
    /// A frontier seeded with the current implementation.
    pub fn seed() -> Self {
        Self { candidates: vec![Candidate::seed()] }
    }

    pub fn get(&self, id: &str) -> Option<&Candidate> {
        self.candidates.iter().find(|c| c.id == id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Candidate> {
        self.candidates.iter()
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Add a candidate. A duplicate id is refused rather than merged: two
    /// different kernels under one identity would make every reward already
    /// recorded for that id a lie.
    pub fn add(&mut self, c: Candidate) -> Result<(), String> {
        if self.candidates.iter().any(|x| x.id == c.id) {
            return Err(format!("frontier already holds a candidate named `{}`", c.id));
        }
        self.candidates.push(c);
        Ok(())
    }

    /// Apply `s` to the candidate `from`, producing the candidates that
    /// transition yields — the `P_LLM(. | k, s, H)` analogue.
    ///
    /// Parameter strategies delegate to the schedule's own arm set, so a
    /// transition can never invent a value `Schedule::validate` would reject.
    /// Code-shape strategies come from the declared table, and an unapplied
    /// transform is still proposed: it is a candidate to WRITE, and marking it
    /// `applied: false` is how the caller knows not to measure it yet.
    pub fn transition(
        &self,
        operator: &str,
        from: &str,
        s: Strategy,
        honored: &TargetKnobs,
    ) -> Result<Vec<Candidate>, String> {
        let base = self
            .get(from)
            .ok_or_else(|| format!("frontier has no candidate `{from}`"))?
            .clone();
        let mut out = Vec::new();
        if s.is_parameter() {
            let own = s.knobs();
            for arm in base.schedule.legal_arms() {
                if arm == base.schedule {
                    continue; // the do-nothing arm is the candidate we started from
                }
                // only this strategy's knobs; `legal_arms` is a one-knob
                // neighbourhood, so this is a filter and not a cross-product
                if !arm.set_values().iter().all(|(n, _)| own.contains(n)) {
                    continue;
                }
                // ...and only knobs THIS target honors. `legal_arms` speaks for
                // the tile-rs emitters; a hand-written kernel honors a subset,
                // and an arm it ignores is a reward for nothing.
                if !arm.set_values().iter().all(|(n, _)| honored.honors(n)) {
                    continue;
                }
                let id = format!("{from}+{}/{}", s.name(), arm.arm_key());
                out.push(Candidate::with_schedule(&id, arm, from, s));
            }
        } else {
            for t in transforms_for(operator).iter().filter(|t| t.strategy == s) {
                let mut flags = base.flags.clone();
                let (k, v) = t.flag.split_once('=').unwrap_or((t.flag, "1"));
                flags.0.insert(k.to_string(), v.to_string());
                let id = format!("{from}+{}", t.name);
                out.push(Candidate::with_flags(&id, base.schedule.clone(), flags, from, s));
            }
        }
        if out.is_empty() {
            return Err(format!(
                "no `{}` transform for `{operator}` from `{from}`: a strategy with no body \
                 cannot be proposed, and inventing one would be an unmeasurable arm",
                s.name()
            ));
        }
        Ok(out)
    }

    /// Every `(candidate, strategy)` pair the bandit may act on.
    ///
    /// This is the paper's expanding action space `A_t = P_t x S`, enumerated so
    /// a caller can size the pull budget instead of discovering it by running.
    pub fn expandable(&self, operator: &str, honored: &TargetKnobs) -> Vec<(String, Strategy)> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for c in &self.candidates {
            for s in Strategy::ALL {
                if self.transition(operator, &c.id, s, honored).is_ok() && seen.insert((c.id.clone(), s)) {
                    out.push((c.id.clone(), s));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::ARG_MAX_TARGET_KNOBS;

    /// The emitter knob set -- what `legal_arms` speaks for.
    fn emitters() -> TargetKnobs {
        TargetKnobs::emitters()
    }

    #[test]
    fn the_seeded_frontier_is_the_do_nothing_candidate() {
        let f = Frontier::seed();
        assert_eq!(f.len(), 1);
        assert!(f.get("default").unwrap().schedule.is_default());
        assert!(f.get("default").unwrap().flags.is_empty());
    }

    #[test]
    fn a_duplicate_candidate_id_is_refused_not_merged() {
        let mut f = Frontier::seed();
        let e = f.add(Candidate::seed()).unwrap_err();
        assert!(e.contains("already holds"), "{e}");
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn a_parameter_strategy_yields_only_its_own_knobs() {
        let f = Frontier::seed();
        let out = f.transition("arg_max", "default", Strategy::Tiling, &emitters()).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|c| !c.schedule.is_default() && c.flags.is_empty()));
        assert!(out.iter().all(|c| c.derived_by == Some(Strategy::Tiling)));
        assert!(out.iter().all(|c| c.parent.as_deref() == Some("default")));
        // a Tiling arm must not be a `buffer_num` or `split_chunk` change
        for c in &out {
            for (name, _) in c.schedule.set_values() {
                assert!(Strategy::Tiling.knobs().contains(&name), "tiling proposed {name}");
            }
        }
    }

    #[test]
    fn a_parameter_strategy_whose_knobs_are_not_honored_yet_is_an_error_not_a_silent_empty() {
        // Item #11: split_chunk / desc_count / buffer_num are DECLARED but no
        // target honors them, so legal_arms excludes them -- and the frontier
        // must say so rather than propose an arm that would measure nothing.
        let f = Frontier::seed();
        for s in [Strategy::Vectorization, Strategy::Pipeline] {
            let e = f.transition("arg_max", "default", s, &emitters()).unwrap_err();
            assert!(e.contains("unmeasurable"), "{}: {e}", s.name());
        }
        // while the strategies with honored knobs or real bodies do act
        for s in [Strategy::Tiling, Strategy::Fusion, Strategy::Reordering, Strategy::AccessLayout] {
            assert!(f.transition("arg_max", "default", s, &emitters()).is_ok(), "{} should act", s.name());
        }
    }

    #[test]
    fn a_code_shape_strategy_yields_flags_and_never_a_schedule_field() {
        let f = Frontier::seed();
        let fused = f.transition("arg_max", "default", Strategy::Fusion, &emitters()).unwrap();
        assert!(!fused.is_empty());
        for c in &fused {
            assert!(!c.flags.is_empty(), "fusion must produce a flag: {c:?}");
            assert!(c.schedule.is_default(), "fusion is not a knob: {c:?}");
        }
        let reordered = f.transition("arg_max", "default", Strategy::Reordering, &emitters()).unwrap();
        assert!(reordered.iter().all(|c| c.flags.get("ARGMAX_ORDER").is_some()));
    }

    #[test]
    fn an_unapplied_transform_is_still_proposed_and_says_so() {
        let f = Frontier::seed();
        let out = f.transition("arg_max", "default", Strategy::AccessLayout, &emitters()).unwrap();
        let nz = out.iter().find(|c| c.id.ends_with("fractal-nz-layout")).expect("nz proposed");
        let t = TRANSFORMS_ARG_MAX.iter().find(|t| t.name == "fractal-nz-layout").unwrap();
        assert!(!t.applied, "the NZ body does not exist in arg_max");
        assert!(!t.evidence.is_empty(), "an unapplied transform must say why");
        assert_eq!(nz.flags.get("ARGMAX_LAYOUT"), Some("nz"));
    }

    #[test]
    fn a_strategy_with_no_body_for_this_operator_is_an_error_not_an_empty_arm() {
        let f = Frontier::seed();
        // softmax has no declared transforms at all
        let e = f.transition("softmax", "default", Strategy::Fusion, &emitters()).unwrap_err();
        assert!(e.contains("no `fusion` transform"), "{e}");
        assert!(e.contains("unmeasurable"), "{e}");
    }

    #[test]
    fn transitions_never_propose_a_value_validate_rejects() {
        let f = Frontier::seed();
        for s in Strategy::ALL {
            if let Ok(out) = f.transition("arg_max", "default", s, &emitters()) {
                for c in out {
                    assert!(c.schedule.validate().is_ok(), "{:?}", c.schedule);
                    assert!(c.schedule.require_honored().is_ok(), "{:?}", c.schedule);
                }
            }
        }
    }

    #[test]
    fn the_expandable_space_is_frontier_times_the_strategies_that_can_act() {
        let f = Frontier::seed();
        let pairs = f.expandable("arg_max", &emitters());
        let strs: BTreeSet<Strategy> = pairs.iter().map(|(_, s)| *s).collect();
        // Four, not six: Vectorization and Pipeline have no honored knob yet
        // (#11), so they have no action. An action space that claimed six would
        // be spending pulls on arms that cannot change anything.
        assert_eq!(strs.len(), 4, "{strs:?}");
        assert!(!strs.contains(&Strategy::Vectorization));
        assert!(!strs.contains(&Strategy::Pipeline));
        assert!(pairs.iter().all(|(id, _)| id == "default"));
    }

    #[test]
    fn a_target_that_ignores_a_knob_is_never_proposed_it() {
        // arg_max's hand-written kernel does not read k_unroll at all, so a
        // k_unroll arm there would be a reward for a change that never
        // happened. `legal_arms` cannot see that -- it speaks for the emitters.
        let f = Frontier::seed();
        let arg_max = TargetKnobs::new(ARG_MAX_TARGET_KNOBS).unwrap();
        let tiling = f.transition("arg_max", "default", Strategy::Tiling, &arg_max).unwrap();
        assert!(!tiling.is_empty());
        for c in &tiling {
            assert!(c.schedule.k_unroll.is_none(), "k_unroll is not honored here: {:?}", c.schedule);
        }
        // ...while the emitter set does propose them
        let emitters = f.transition("arg_max", "default", Strategy::Tiling, &emitters()).unwrap();
        assert!(emitters.iter().any(|c| c.schedule.k_unroll.is_some()));
    }

    #[test]
    fn an_unknown_knob_name_is_refused_rather_than_shrinking_the_arm_space() {
        let e = TargetKnobs::new(&["tile", "typo_knob"]).unwrap_err();
        assert!(e.contains("typo_knob"), "{e}");
        assert!(TargetKnobs::new(&["tile", "split_p"]).is_ok());
        assert!(ARG_MAX_TARGET_KNOBS.contains(&"split_p"));
    }

    #[test]
    fn every_strategy_declares_a_target_so_the_mask_can_use_it() {
        for s in Strategy::ALL {
            if s.is_parameter() {
                assert!(s.target().is_some(), "{} has no Target(s)", s.name());
            }
        }
        // the code-shape ones are saturation plays too, where a resource applies
        assert_eq!(Strategy::Fusion.target(), Some(Resource::Dram));
        assert_eq!(Strategy::AccessLayout.target(), Some(Resource::Dram));
    }

    #[test]
    fn a_chain_records_its_provenance() {
        let mut f = Frontier::seed();
        let arms = f.transition("arg_max", "default", Strategy::Tiling, &emitters()).unwrap();
        // pick the arm that actually sets `tile`: `legal_arms` also carries the
        // k_unroll arms, which are the same strategy but a different knob.
        let first = arms
            .iter()
            .find(|c| c.schedule.tile.is_some())
            .expect("a tile arm exists")
            .clone();
        let id = first.id.clone();
        f.add(first).unwrap();
        // A mixed chain: a parameter step, then a code-shape step on top of it.
        // (Not Tiling -> Pipeline: Pipeline's only knob is `buffer_num`, which
        // nothing honors yet, so that hop is refused by design.)
        let next = f.transition("arg_max", &id, Strategy::Reordering, &emitters()).unwrap();
        assert!(next.iter().all(|c| c.parent.as_deref() == Some(id.as_str())));
        assert!(next.iter().all(|c| c.derived_by == Some(Strategy::Reordering)));
        // the second hop keeps the first hop's parameter choice AND adds a flag
        assert!(next.iter().all(|c| c.schedule.tile.is_some()));
        assert!(next.iter().all(|c| c.flags.get("ARGMAX_ORDER").is_some()));
    }
}
