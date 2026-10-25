//! Portable schedule knobs as DATA — the KernelBand / P4 surface.
//!
//! A schedule is *separate from the algorithm*: it says how a loop is
//! structured (unroll factor, tile sizes, pipeline depth, layout), never what
//! is computed. Lowering must therefore be **sound for every arm in
//! [`Schedule::legal_arms`]**: same mathematical result, only a different legal
//! machine form.
//!
//! ## Safety contract
//!
//! 1. `Schedule::default()` is exactly today's behavior — no attribute means
//!    no change.
//! 2. An arm is only legal if every target that will honor it preserves
//!    semantics. `k_unroll` qualifies because
//!    `emit_unrolled_k_accumulation` visits `k` in strictly increasing order
//!    with one accumulator for every factor ≥ 1 (proven in `mlir_parse`);
//!    `tile` qualifies because `plan_tiles` takes the tile as an input and
//!    tiling a pointwise extent does not reorder any arithmetic.
//! 3. Illegal schedules are **rejected**, never silently rewritten.
//! 4. A knob that no target honors yet is **declared here but refused at parse
//!    time** ([`Schedule::require_honored`]) — never accepted and ignored. An
//!    ignored knob makes the bandit's reward a lie.
//! 5. The bandit may only *select among legal arms after a verified run* —
//!    never invent a cost, never cache an unmeasured answer.
//!
//! ## What is deliberately NOT here
//!
//! Fusion ("one launch instead of two") and Reordering (dense vs interleaved)
//! are *code-shape* changes, not values: they remove code or swap a whole
//! function body. They cannot be fields of a schedule without breaking
//! contract 2, so they belong to the candidate frontier, not to this type.
//! See `NOTES-tile-search-knobs.md` in the bench tree.

use crate::hardware::{HwSignature, Resource};

/// What honoring a knob does to the result.
///
/// Ordering is by severity so [`Schedule::value_effect`] can take a maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValueEffect {
    /// Same bits, every time. No precision argument needed.
    BitExact,
    /// Float reductions may sum in a different order. Legal where the
    /// benchmark grades on tolerance (CANN Bench does), fatal where the oracle
    /// demands an exact match.
    ReassociatesFp,
    /// Results change, not merely their association. Never selectable.
    ChangesValues,
}

/// Where a knob takes effect. Mirrors the two axes every tunable in this
/// project is classified on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnobSite {
    /// An emitter reads it while lowering (the value changes emitted code).
    Emitter,
    /// The host wrapper reads it to pick tiling parameters (legal by
    /// construction: the platform explicitly permits shape-keyed tiling).
    Dispatch,
}

/// One tunable this crate knows about, whether or not it is wired up yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Knob {
    pub name: &'static str,
    pub site: KnobSite,
    pub effect: ValueEffect,
    /// Which resource the strategy behind this knob is trying to recover.
    /// `None` = the knob is not a saturation play (a correctness/layout choice
    /// with no throughput target), so it can never be masked out. CALIBRATE
    /// against measurements -- see `hardware.rs`; a wrong target only prunes
    /// the wrong arm, it cannot corrupt a run.
    pub target: Option<Resource>,
    /// `Some(values)` = the discrete arm set the bandit may propose.
    /// `None` = declared, but no arm set until it is honored.
    pub arms: Option<&'static [u32]>,
    /// True only when at least one target in this build actually honors it.
    /// A knob with `honored: false` is refused by [`Schedule::parse_kv`].
    pub honored: bool,
    /// Domain bound, enforced by [`Schedule::validate`].
    pub max: u32,
}

/// Upper bound on a declared `k_unroll`. Soundness allows any factor ≥ 1;
/// this is a code-size / I-cache bound so a typo (`k_unroll = 100000`) is
/// rejected rather than expanded into a million loads.
pub const MAX_K_UNROLL: u32 = 64;

/// Powers-of-two arms the bandit proposes for `k_unroll`. Non-pow2 factors
/// are *sound* (same sum order) but not worth arms until a measurement says
/// so; `legal_arms` is the pruning surface, not a soundness whitelist beyond
/// the `validate` bounds.
pub const K_UNROLL_ARMS: &[u32] = &[1, 2, 4, 8, 16, 32, 64];

/// Tile-size arms. Multiples of the 32 B datablock on Ascend; kept a power of
/// two here because every consumer rounds there anyway.
pub const TILE_ARMS: &[u32] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// Partition arms for the reduced axis. Divisors of the common row counts, so
/// the split lands evenly; `validate` bounds them and the Dispatch site refuses
/// a value that does not divide R.
pub const SPLIT_P_ARMS: &[u32] = &[2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32];

/// Ceiling on a declared tile. Beyond this the UB budget cannot hold one
/// 4-byte element per lane of a repeat, so an arm there is a lie, not a knob.
pub const MAX_TILE: u32 = 65536;

/// The registry. Every knob the project has classified, with the two facts the
/// bandit needs: what it does to the result, and whether it can be turned at
/// all in this build.
pub const KNOBS: &[Knob] = &[
    Knob { name: "k_unroll", site: KnobSite::Emitter, effect: ValueEffect::BitExact,
           arms: Some(K_UNROLL_ARMS), honored: true, target: Some(Resource::Vector), max: MAX_K_UNROLL },
    Knob { name: "tile", site: KnobSite::Dispatch, effect: ValueEffect::BitExact,
           arms: Some(TILE_ARMS), honored: true, target: Some(Resource::Ub), max: MAX_TILE },
    // Declared from the inventory (NOTES-tile-search-knobs.md) and refused
    // until a target honors them. Splitting a reduction axis reassociates
    // float sums, so it is the one non-BitExact knob in this table.
    // Honored by the arg_max Dispatch override (ARGMAX_SCHEDULE=split_p=N,
    // arg_max_plugin.cpp): it partitions the reduced axis across cores and the
    // Combine merges in stream order. Split of the arm_max OUTER schedule only.
    Knob { name: "split_p", site: KnobSite::Dispatch, effect: ValueEffect::ReassociatesFp,
           arms: Some(SPLIT_P_ARMS), honored: true, target: Some(Resource::Ub), max: 64 },
    Knob { name: "row_block", site: KnobSite::Dispatch, effect: ValueEffect::BitExact,
           arms: None, honored: false, target: Some(Resource::Ub), max: 64 },
    Knob { name: "buffer_num", site: KnobSite::Emitter, effect: ValueEffect::BitExact,
           arms: None, honored: false, target: Some(Resource::Latency), max: 4 },
    Knob { name: "split_chunk", site: KnobSite::Emitter, effect: ValueEffect::BitExact,
           arms: None, honored: false, target: Some(Resource::Mte2), max: 512 },
    Knob { name: "desc_count", site: KnobSite::Emitter, effect: ValueEffect::BitExact,
           arms: None, honored: false, target: Some(Resource::Dram), max: 8 },
];

/// The knob set ONE target honors.
///
/// `KNOBS[i].honored` is the library-wide default, and it means "a tile-rs
/// emitter honors this". A hand-written kernel is a different target: arg_max's
/// Dispatch override reads `tile` and `split_p` and ignores `k_unroll`
/// completely (it is a tile-rs emitter knob, consumed through `#[tile::k_unroll]`
/// and the emitters, not by AscendC code). Proposing a `k_unroll` arm to it
/// would measure a change that never happened AND observe a reward for it,
/// which is the failure contract 4 exists to prevent -- one level up, so the
/// `honored` flag alone cannot catch it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetKnobs(Vec<&'static str>);

impl TargetKnobs {
    /// The knobs the tile-rs emitters honor, taken from the registry.
    pub fn emitters() -> Self {
        Self(KNOBS.iter().filter(|k| k.honored).map(|k| k.name).collect())
    }

    /// A target's own set, by name. An unknown name is refused: a typo here
    /// would silently shrink the arm space.
    pub fn new(names: &[&'static str]) -> Result<Self, String> {
        for n in names {
            if knob(n).is_none() {
                return Err(format!(
                    "TargetKnobs: `{n}` is not a registered knob; known: {}",
                    KNOBS.iter().map(|x| x.name).collect::<Vec<_>>().join(", ")
                ));
            }
        }
        Ok(Self(names.to_vec()))
    }

    pub fn honors(&self, knob: &str) -> bool {
        self.0.contains(&knob)
    }

    pub fn names(&self) -> &[&'static str] {
        &self.0
    }
}

/// The hand-written arg_max kernel's target set: the Dispatch override in
/// `arg_max_plugin.cpp` reads `tile` and `split_p`. `k_unroll` is absent on
/// purpose -- nothing on that path consumes it.
pub const ARG_MAX_TARGET_KNOBS: &[&str] = &["tile", "split_p"];

fn knob(name: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.name == name)
}

/// One concrete schedule choice. `None` fields mean “backend default”.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    /// K-loop unroll factor for matmul-style reductions. `None` = the
    /// backend's receipted default (MSL 16, GPU `DEFAULT_K_UNROLL`).
    pub k_unroll: Option<u32>,
    /// Pointwise/extent tile for targets that take one (`plan_tiles`).
    /// `None` = the target's own default.
    pub tile: Option<u32>,
    /// Partitions of a reduced axis (arg_max `splitP`). Reassociates float
    /// sums, so it is only legal where the oracle grades on tolerance.
    pub split_p: Option<u32>,
    /// Rows per DMA block for a strided reduction.
    pub row_block: Option<u32>,
    /// Queue / software-pipeline depth.
    pub buffer_num: Option<u32>,
    /// Lanes per split (GatherMask) call in the int64 emulation.
    pub split_chunk: Option<u32>,
    /// How many DMA descriptors carry one tile (1 = the packed form).
    pub desc_count: Option<u32>,
}

impl Schedule {
    pub const DEFAULT: Schedule = Schedule {
        k_unroll: None,
        tile: None,
        split_p: None,
        row_block: None,
        buffer_num: None,
        split_chunk: None,
        desc_count: None,
    };

    /// True when no field is set — byte-for-byte today's behavior.
    pub fn is_default(&self) -> bool {
        *self == Schedule::DEFAULT
    }

    /// Worst effect over the knobs this schedule actually sets.
    ///
    /// A caller that needs an exact oracle consults this before running: a
    /// `ReassociatesFp` schedule is not wrong, it is *unprovable* under an
    /// exact-match reference.
    pub fn value_effect(&self) -> ValueEffect {
        let mut worst = ValueEffect::BitExact;
        for (name, set) in self.set_fields() {
            if set {
                if let Some(k) = knob(name) {
                    worst = worst.max(k.effect);
                }
            }
        }
        worst
    }

    /// Knobs this schedule sets that no target honors in this build.
    pub fn unhonored_knobs(&self) -> Vec<&'static str> {
        self.set_fields()
            .into_iter()
            .filter(|(n, set)| *set && knob(n).map(|k| !k.honored).unwrap_or(true))
            .map(|(n, _)| n)
            .collect()
    }

    /// Refuse a schedule that would be accepted and then ignored.
    ///
    /// Contract 4: a knob the codegen cannot honor must fail loudly here rather
    /// than produce a reward for a change that never happened.
    pub fn require_honored(&self) -> Result<(), String> {
        let missing = self.unhonored_knobs();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "schedule sets {} but no target in this build honors {}; \
                 declared for the arm registry, refused rather than silently \
                 ignored (see NOTES-tile-search-knobs.md)",
                missing.join(", "),
                if missing.len() == 1 { "it" } else { "them" }
            ))
        }
    }

    /// Reject a schedule that no lowering could honor soundly (or at all).
    ///
    /// Zero is rejected rather than clamped: a schedule that “means 1” after
    /// a silent max() is how a knob becomes a lie. Oversized factors are
    /// rejected for code size, not for numerics.
    ///
    /// Only *per-knob domains* are enforced here. Cross-knob cliffs (a deep
    /// queue needs a smaller row block) are owned by the per-kernel tile
    /// planner, which is the only place that knows the UB budget; encoding
    /// them here would be guessing.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in self.set_values() {
            let k = knob(name).ok_or_else(|| format!("unknown schedule field `{name}`"))?;
            if value == 0 {
                return Err(format!(
                    "schedule {name} = 0: zero is not a factor, it is a missing \
                     value; omit {name} for the backend default"
                ));
            }
            if value > k.max {
                return Err(format!(
                    "schedule {name} = {value} exceeds MAX for {name} = {}; \
                     out-of-domain values are a code-size/UB hazard, not a \
                     tuning choice",
                    k.max
                ));
            }
        }
        Ok(())
    }

    /// Stable arm id for bandit state keys, in fixed field order so the
    /// lexicographic tie-break in `BanditState::select_arm` is deterministic.
    /// `"default"`, `"k_unroll=16"`, `"k_unroll=16,tile=1024"`.
    pub fn arm_key(&self) -> String {
        let set: Vec<String> = self
            .set_values()
            .into_iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect();
        if set.is_empty() {
            "default".into()
        } else {
            set.join(",")
        }
    }

    /// Parse `k_unroll=16,tile=1024` (spaces and a trailing comma allowed).
    ///
    /// Accepts every field in the registry, then refuses anything the build
    /// cannot honor, so a caller can never quietly get a no-op arm.
    pub fn parse_kv(s: &str) -> Result<Schedule, String> {
        let mut sch = Schedule::DEFAULT;
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| format!("schedule field `{part}` needs key=value"))?;
            let (k, v) = (k.trim(), v.trim());
            if knob(k).is_none() {
                return Err(format!(
                    "unknown schedule field `{k}`; known fields: {}",
                    KNOBS.iter().map(|x| x.name).collect::<Vec<_>>().join(", ")
                ));
            }
            let n: u32 = v
                .parse()
                .map_err(|_| format!("schedule {k} `{v}` is not an integer"))?;
            sch = sch.with(k, n);
        }
        sch.validate()?;
        sch.require_honored()?;
        Ok(sch)
    }

    /// Set one field by name. Unknown names are a programming error upstream
    /// (`parse_kv` is the checked entry point), so this is infallible.
    fn with(mut self, name: &str, value: u32) -> Schedule {
        match name {
            "k_unroll" => self.k_unroll = Some(value),
            "tile" => self.tile = Some(value),
            "split_p" => self.split_p = Some(value),
            "row_block" => self.row_block = Some(value),
            "buffer_num" => self.buffer_num = Some(value),
            "split_chunk" => self.split_chunk = Some(value),
            "desc_count" => self.desc_count = Some(value),
            _ => {}
        }
        self
    }

    fn get(&self, name: &str) -> Option<u32> {
        match name {
            "k_unroll" => self.k_unroll,
            "tile" => self.tile,
            "split_p" => self.split_p,
            "row_block" => self.row_block,
            "buffer_num" => self.buffer_num,
            "split_chunk" => self.split_chunk,
            "desc_count" => self.desc_count,
            _ => None,
        }
    }

    /// `(name, is_set)` for every registered field, in registry order.
    pub fn set_fields(&self) -> Vec<(&'static str, bool)> {
        KNOBS.iter().map(|k| (k.name, self.get(k.name).is_some())).collect()
    }

    /// `(name, value)` for every field this schedule sets, in registry order.
    pub fn set_values(&self) -> Vec<(&'static str, u32)> {
        KNOBS
            .iter()
            .filter_map(|k| self.get(k.name).map(|v| (k.name, v)))
            .collect()
    }

    /// The tile to hand a consumer that takes one, or `default` when unset.
    pub fn tile_or(&self, default: usize) -> Result<usize, String> {
        match self.tile {
            None => Ok(default),
            Some(t) => {
                let t = t as usize;
                if t == 0 {
                    return Err("schedule tile = 0: use the target default instead".into());
                }
                Ok(t)
            }
        }
    }

    /// True when this arm is trying to recover a resource the kernel no longer
    /// has headroom in (KernelBand's pruning mask).
    ///
    /// Blocked only when EVERY knob the arm sets targets a saturated resource:
    /// a retile plus a deeper pipe can still win when only one of the two is
    /// pinned, and blocking it would lose a real candidate. The DEFAULT arm sets
    /// nothing, so it can never be blocked -- which is what keeps the masked
    /// selection from ever coming up empty on its own.
    pub fn mask_blocked(&self, sig: &HwSignature, theta: f64) -> bool {
        let mut targeted = false;
        for (name, _) in self.set_values() {
            let Some(target) = knob(name).and_then(|k| k.target) else { continue };
            targeted = true;
            if !sig.saturated(target, theta) {
                return false;
            }
        }
        targeted
    }

    /// The arm set a bandit may propose for this base schedule.
    ///
    /// One-knob neighborhood: the default arm plus, for every **honored** knob
    /// this schedule leaves unset, each legal value with only that knob set.
    /// This mirrors the paper's action — apply *one* strategy to *one*
    /// candidate — instead of enumerating a cross-product no measurement
    /// budget can cover.
    ///
    /// A knob left unset on purpose can be pinned by setting it; it is then no
    /// longer expanded.
    pub fn legal_arms(&self) -> Vec<Schedule> {
        let mut arms = vec![self.clone()];
        for k in KNOBS {
            if !k.honored || self.get(k.name).is_some() {
                continue;
            }
            let Some(values) = k.arms else { continue };
            for &v in values {
                let arm = self.clone().with(k.name, v);
                if arm.validate().is_ok() {
                    arms.push(arm);
                }
            }
        }
        arms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty_and_equals_no_override() {
        assert_eq!(Schedule::DEFAULT.k_unroll, None);
        assert_eq!(Schedule::default(), Schedule::DEFAULT);
        assert_eq!(Schedule::DEFAULT.arm_key(), "default");
        assert!(Schedule::DEFAULT.is_default());
        assert_eq!(Schedule::DEFAULT.value_effect(), ValueEffect::BitExact);
    }

    #[test]
    fn zero_k_unroll_is_rejected_not_clamped() {
        let e = Schedule { k_unroll: Some(0), ..Schedule::DEFAULT }.validate().unwrap_err();
        assert!(e.contains("k_unroll = 0"), "{e}");
        assert!(Schedule::parse_kv("k_unroll=0").is_err());
    }

    #[test]
    fn oversized_k_unroll_is_rejected() {
        let s = Schedule { k_unroll: Some(MAX_K_UNROLL + 1), ..Schedule::DEFAULT };
        let e = s.validate().unwrap_err();
        assert!(e.contains("MAX"), "{e}");
    }

    #[test]
    fn unknown_schedule_fields_are_refused_rather_than_ignored() {
        let e = Schedule::parse_kv("tile_siz=64").unwrap_err();
        assert!(e.contains("unknown schedule field"), "{e}");
        assert!(e.contains("k_unroll"), "{e}");
    }

    #[test]
    fn declared_but_unhonored_knobs_are_refused_at_parse_time() {
        // Contract 4: a knob no target honors must not be accepted-and-ignored,
        // or the bandit would record a reward for a change that never happened.
        // (`split_p` used to be the example here; it is now honored by the
        // arg_max Dispatch override, so the example moved to the knobs that are
        // still declared-only.)
        let e = Schedule::parse_kv("buffer_num=2").unwrap_err();
        assert!(e.contains("buffer_num"), "{e}");
        assert!(e.contains("honors"), "{e}");
        // ...but the registry still knows it, with its effect.
        let k = knob("buffer_num").unwrap();
        assert_eq!(k.effect, ValueEffect::BitExact);
        assert!(!k.honored);
        // and split_p is now honored, with its (deliberately non-BitExact) effect
        let sp = knob("split_p").unwrap();
        assert!(sp.honored, "the arg_max override honors split_p");
        assert_eq!(sp.effect, ValueEffect::ReassociatesFp);
        assert!(Schedule::parse_kv("split_p=4").is_ok());
    }

    #[test]
    fn tile_is_honored_and_parses() {
        let s = Schedule::parse_kv("tile=1024").unwrap();
        assert_eq!(s.tile, Some(1024));
        assert_eq!(s.arm_key(), "tile=1024");
        assert_eq!(s.tile_or(64).unwrap(), 1024);
        assert_eq!(Schedule::DEFAULT.tile_or(64).unwrap(), 64);
        assert_eq!(s.value_effect(), ValueEffect::BitExact);
    }

    #[test]
    fn value_effect_takes_the_worst_set_knob() {
        let s = Schedule { k_unroll: Some(16), split_p: Some(4), ..Schedule::DEFAULT };
        assert_eq!(s.value_effect(), ValueEffect::ReassociatesFp);
    }

    #[test]
    fn legal_arms_are_default_plus_one_knob_each_and_all_valid() {
        let arms = Schedule::DEFAULT.legal_arms();
        // default first — the no-change arm is always present
        assert_eq!(arms[0], Schedule::DEFAULT);
        assert!(arms.iter().all(|a| a.validate().is_ok()));
        assert!(arms.iter().any(|a| a.k_unroll == Some(16)));
        assert!(arms.iter().any(|a| a.tile == Some(1024)));
        // one-knob neighborhood: never two knobs set at once
        assert!(arms.iter().all(|a| a.set_values().len() <= 1));
        // only honored knobs appear
        assert!(arms.iter().all(|a| a.unhonored_knobs().is_empty()));
    }

    #[test]
    fn a_pinned_knob_is_not_re_expanded_but_others_still_are() {
        let pinned = Schedule { k_unroll: Some(8), ..Schedule::DEFAULT };
        let expanded = pinned.legal_arms();
        assert!(expanded.iter().all(|a| a.k_unroll == Some(8)));
        assert_eq!(expanded[0], pinned);
        // tile was left unset, so it is still proposed
        assert!(expanded.iter().any(|a| a.tile.is_some()));
        assert!(expanded.iter().all(|a| a.set_values().len() <= 2));
    }

    #[test]
    fn arm_key_is_field_ordered_and_stable() {
        let s = Schedule { tile: Some(256), k_unroll: Some(4), ..Schedule::DEFAULT };
        assert_eq!(s.arm_key(), "k_unroll=4,tile=256");
        assert_eq!(Schedule::parse_kv("tile=256,k_unroll=4").unwrap().arm_key(), s.arm_key());
    }

    #[test]
    fn parse_kv_accepts_spaces_and_trailing_comma() {
        let s = Schedule::parse_kv(" k_unroll = 32 , ").unwrap();
        assert_eq!(s.k_unroll, Some(32));
        assert_eq!(s.arm_key(), "k_unroll=32");
    }

    #[test]
    fn the_pruning_mask_takes_out_only_the_targeted_resource() {
        let hot_ub = HwSignature { ub: 0.9, ..Default::default() };
        let tile = Schedule { tile: Some(512), ..Schedule::DEFAULT };
        let unroll = Schedule { k_unroll: Some(16), ..Schedule::DEFAULT };
        assert!(tile.mask_blocked(&hot_ub, crate::hardware::SATURATION), "tile targets UB");
        assert!(!unroll.mask_blocked(&hot_ub, crate::hardware::SATURATION), "k_unroll targets the vector pipe");
        assert!(!Schedule::DEFAULT.mask_blocked(&hot_ub, crate::hardware::SATURATION), "default asks for nothing");
    }

    #[test]
    fn a_multi_knob_arm_survives_when_one_of_its_resources_is_free() {
        let hot_ub = HwSignature { ub: 0.9, ..Default::default() };
        // tile is pinned by UB, but buffer_num targets latency, which is free.
        let both = Schedule { tile: Some(512), buffer_num: Some(2), ..Schedule::DEFAULT };
        assert!(!both.mask_blocked(&hot_ub, crate::hardware::SATURATION));
        // ...and with latency pinned too, nothing is left to gain.
        let all_hot = HwSignature { ub: 0.9, latency: 0.9, ..Default::default() };
        assert!(both.mask_blocked(&all_hot, crate::hardware::SATURATION));
    }

    #[test]
    fn every_honored_knob_declares_a_target_so_the_mask_has_something_to_use() {
        for k in KNOBS.iter().filter(|k| k.honored) {
            assert!(k.target.is_some(), "honored knob {} has no pruning target", k.name);
        }
    }

    #[test]
    fn registry_covers_every_field_so_nothing_can_hide() {
        // Every settable field must appear in KNOBS, and every honored knob
        // must have an arm set (otherwise it is unreachable by the bandit).
        let probe = Schedule { k_unroll: Some(1), tile: Some(64), split_p: Some(1),
                               row_block: Some(1), buffer_num: Some(1),
                               split_chunk: Some(64), desc_count: Some(1) };
        assert_eq!(probe.set_values().len(), KNOBS.len());
        for k in KNOBS.iter().filter(|k| k.honored) {
            assert!(k.arms.is_some(), "honored knob {} has no arms", k.name);
        }
    }
}
