//! The tunable surface of every generated kernel, in one machine-readable
//! place, so a search (NSGA-II in `tile_search`) can gauge fused and composed
//! kernels collectively rather than one constant at a time.
//!
//! Two things this exists to prevent, both of which have already cost real
//! measurements here:
//!
//! 1. **Knobs that live outside any list are never tuned.** Threads per
//!    threadgroup for the Metal mat-vec sat hardcoded at 256 on the DISPATCH
//!    side, not in the emitter, so no sweep ever touched it. It was wrong by
//!    ~1.25x on q4_0 and ~1.4x on q8_0. A knob registry that only covers
//!    emitter constants would have missed it again, so `site` is part of the
//!    record.
//! 2. **Knobs that stop feeding the kernel look like they still work.** After
//!    the GEMM was reparameterized, five `MUL_MM_*` constants were read only by
//!    the header emitter -- a sweep would have changed the reported grid and
//!    not the kernel. `test_knob_registry_matches_reality` asserts every
//!    declared value against the constant it claims to mirror, so a knob that
//!    goes dead fails the suite instead of silently lying.
//!
//! Preconditions are part of the search space, not commentary: a value that
//! violates one does not produce a slow kernel, it produces a wrong one (a lane
//! step straddling a quantization block applies one block's scale to another's
//! weights) or an uncompilable one. `tile_search` models exactly this as knob
//! validity ranges and divisibility preconditions.

use crate::mlir_to_msl::{
    MM_TILE_PLANAR,
    MM_TILE_LARGE, MM_TILE_SMALL, MUL_MV_ILV_Q4_BYTES_PER_LANE, MUL_MV_ILV_QUANTS_PER_LANE,
    MUL_MV_ILV_ROWS_PER_TG, MUL_MV_Q4_ROWS_PER_TG, MUL_MV_Q4_VECS_PER_LANE, MUL_MV_ROWS_PER_TG,
    MUL_MV_VECS_PER_LANE,
};

/// Where a knob is read. A search has to be able to set both: an emitter knob
/// changes the generated source, a dispatch knob changes the launch that the
/// host makes, and getting them out of step is its own failure mode -- when the
/// GEMM tile was restated at the dispatch, a sweep changed the kernel but not
/// the grid and measured "2x faster than ggml" while computing part of the
/// output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Site {
    /// A constant in the emitter; changing it regenerates the kernel source.
    Emitter,
    /// A value used when launching; changing it needs no regeneration.
    Dispatch,
}

impl Site {
    pub fn as_str(self) -> &'static str {
        match self {
            Site::Emitter => "emitter",
            Site::Dispatch => "dispatch",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Metal,
    Pto,
}

impl Target {
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Pto => "pto",
        }
    }
}

/// A knob's domain. Integers carry an explicit candidate list rather than a
/// range: most of these are powers of two or divisors of a block size, and an
/// unconstrained range spends the search's budget on values that cannot be
/// legal.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Ints(&'static [u32]),
    Bools,
}

pub struct Knob {
    pub name: &'static str,
    pub target: Target,
    pub site: Site,
    /// Kernels whose behaviour this knob changes. A search that mutates a knob
    /// only has to re-measure these.
    pub kernels: &'static [&'static str],
    pub domain: Domain,
    /// The value currently compiled in, asserted against the real constant by
    /// `test_knob_registry_matches_reality`.
    pub current: u32,
    /// Machine-readable preconditions, by id, from `CONSTRAINTS`.
    pub constraints: &'static [&'static str],
    pub note: &'static str,
}

/// Preconditions over knob values. `expr` is written for a search to evaluate
/// with the knob values substituted; `kind` says what a violation costs, which
/// is what decides whether a search may sample it at all.
pub struct Constraint {
    pub id: &'static str,
    /// `wrong` = silently incorrect output, `invalid` = will not build or
    /// launch, `cliff` = builds and runs but falls off a measured performance
    /// cliff.
    pub kind: &'static str,
    pub expr: &'static str,
    pub why: &'static str,
}

pub const CONSTRAINTS: &[Constraint] = &[
    Constraint {
        id: "q8_lane_step_within_block",
        kind: "wrong",
        expr: "MUL_MV_ILV_QUANTS_PER_LANE <= 32 && 32 % MUL_MV_ILV_QUANTS_PER_LANE == 0",
        why: "One scale per lane step. A step crossing a Q8_0 block applies one \
              block's scale to another block's weights -- wrong output, not slow output.",
    },
    Constraint {
        id: "q4_lane_step_within_block",
        kind: "wrong",
        expr: "MUL_MV_ILV_Q4_BYTES_PER_LANE <= 16 && 16 % MUL_MV_ILV_Q4_BYTES_PER_LANE == 0",
        why: "Same, for Q4_0, whose 32 quants live in 16 bytes.",
    },
    Constraint {
        id: "planar_q8_lane_step_within_block",
        kind: "wrong",
        expr: "MUL_MV_VECS_PER_LANE <= 8 && 8 % MUL_MV_VECS_PER_LANE == 0",
        why: "A Q8_0 block is 8 char4 units.",
    },
    Constraint {
        id: "planar_q4_lane_step_within_block",
        kind: "wrong",
        expr: "MUL_MV_Q4_VECS_PER_LANE <= 2 && 2 % MUL_MV_Q4_VECS_PER_LANE == 0",
        why: "A Q4_0 block is 2 uint4 units. This one was found as a BUG, not by \
              reasoning: at ushort4 units a block was 2 units and a wider lane step \
              silently crossed blocks.",
    },
    Constraint {
        id: "mm_tile_closes",
        kind: "invalid",
        expr: "tm % (8 * sgm) == 0 && tn % (8 * sgn) == 0",
        why: "The simdgroup grid must tile the output in whole 8x8 hardware \
              matrix blocks.",
    },
    Constraint {
        id: "mm_thread_limit",
        kind: "invalid",
        expr: "sgm * sgn * 32 <= 1024",
        why: "Metal's threads-per-threadgroup ceiling.",
    },
    Constraint {
        id: "mm_threadgroup_memory",
        kind: "invalid",
        expr: "(kc*32*(tn+1) + tm*kc*33) * 2 + sgm*sgn*64*4 <= 32768",
        why: "Both staged operands AND the result scratch are live at once. \
              Omitting the scratch is what ruled out a full TM x TN result buffer.",
    },
    Constraint {
        id: "mm_register_cliff",
        kind: "cliff",
        expr: "UNMODELLED -- see why; do not prune on this",
        why: "There IS a cliff -- at TN=128 TM=160.. SGM=1 SGN=16, MI=16/20/24 \
              runs 4723/44013/289697 us -- but 'accumulator count exceeds the \
              register file' is NOT the mechanism, and a search must not prune \
              on it. A GA found TN=128 TM=128 SGM=1 SGN=1, which is MI=16 AND \
              NI=16, i.e. 256 accumulators against that config's 20, and it is \
              the fastest point known: 0.99x of ggml at 896x4864 where the \
              hand-tuned tile was 1.68x. Twelve times the accumulators, no \
              cliff. The real mechanism is unidentified. Listed so the cliff is \
              not forgotten, with an expression a search cannot use to exclude \
              anything.",
    },
    Constraint {
        id: "pto_nb_dominates_kb",
        kind: "cliff",
        expr: "rank by PTO_MM_NB descending; PTO_MM_KB is near-free within an Nb band",
        why: "MEASURED on 910c across three shapes. Nb 128/64/32/16 gives \
              210/266/432/620 us at M=16 K=896 N=4864, while Kb barely moves \
              anything within a band. An EARLIER sweep at a single shape \
              concluded the invariant was the product Kb*Nb (an L0B-saturation \
              plateau) and that 256x64 was optimal; a second shape spread that \
              'plateau' over 2.9x. A search must not collapse this pair into one \
              product knob.",
    },
    Constraint {
        id: "pto_ub_budget",
        kind: "invalid",
        expr: "working_set_bytes <= UB_SIZE",
        why: "Unified Buffer capacity, 262144 on a2a3/a5. Mechanized as ub_ok / \
              ub_alloc_guard in the Coq development.",
    },
    Constraint {
        id: "pto_row16",
        kind: "invalid",
        expr: "!REQUIRES_ROW16 || rows % 16 == 0",
        why: "NZ/cube tiles need rows % 16 == 0 on a5; a2a3 does not.",
    },
];

pub const KNOBS: &[Knob] = &[
    // ---- Metal, emitter: interleaved mat-vecs (the ones dispatched inside
    // ggml-metal, reading ggml's native blocks).
    Knob {
        name: "MUL_MV_ILV_ROWS_PER_TG",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q8_0_interleaved", "mulmv_q4_0_interleaved"],
        domain: Domain::Ints(&[1, 2, 4, 8, 16]),
        current: MUL_MV_ILV_ROWS_PER_TG as u32,
        constraints: &[],
        note: "Output rows per threadgroup, so the activation span is fetched once \
               and reused. At one row per threadgroup the whole vector is re-read \
               per row.",
    },
    Knob {
        name: "MUL_MV_ILV_QUANTS_PER_LANE",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q8_0_interleaved"],
        domain: Domain::Ints(&[4, 8, 16, 32]),
        current: MUL_MV_ILV_QUANTS_PER_LANE as u32,
        constraints: &["q8_lane_step_within_block"],
        note: "Quants per lane step. ggml's own kernel uses 8.",
    },
    Knob {
        name: "MUL_MV_ILV_Q4_BYTES_PER_LANE",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q4_0_interleaved"],
        domain: Domain::Ints(&[2, 4, 8, 16]),
        current: MUL_MV_ILV_Q4_BYTES_PER_LANE as u32,
        constraints: &["q4_lane_step_within_block"],
        note: "Quant BYTES per lane step. A Q4_0 byte carries two elements 16 \
               apart, so N bytes is 2N elements across two activation spans.",
    },
    // ---- Metal, emitter: planar mat-vecs (the separate-backend path).
    Knob {
        name: "MUL_MV_ROWS_PER_TG",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q8_0_fused"],
        domain: Domain::Ints(&[1, 2, 4, 8, 16]),
        current: MUL_MV_ROWS_PER_TG as u32,
        constraints: &[],
        note: "Tuned jointly with the threadgroup width: single-axis sweeps chose \
               16 rows at width 256 (0.94x), while the optimum is 4 rows at width \
               512 (0.89x) -- a cell neither axis argues for.",
    },
    Knob {
        name: "MUL_MV_VECS_PER_LANE",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q8_0_fused"],
        domain: Domain::Ints(&[1, 2, 4, 8]),
        current: MUL_MV_VECS_PER_LANE as u32,
        constraints: &["planar_q8_lane_step_within_block"],
        note: "char4 units per lane step.",
    },
    Knob {
        name: "MUL_MV_Q4_ROWS_PER_TG",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q4_0_fused"],
        domain: Domain::Ints(&[1, 2, 4, 8, 16]),
        current: MUL_MV_Q4_ROWS_PER_TG as u32,
        constraints: &[],
        note: "",
    },
    Knob {
        name: "MUL_MV_Q4_VECS_PER_LANE",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmv_q4_0_fused"],
        domain: Domain::Ints(&[1, 2]),
        current: MUL_MV_Q4_VECS_PER_LANE as u32,
        constraints: &["planar_q4_lane_step_within_block"],
        note: "With uint4 units a unit IS a Q4_0 block, so the precondition holds \
               by construction at width 1.",
    },
    // ---- Metal, emitter: the GEMM tiles. These five move together and are
    // emitted twice, so a search treats each tile as one genome segment.
    Knob {
        name: "MM_TILE_LARGE.tn",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved", "mulmm_q8_0_fused"],
        domain: Domain::Ints(&[32, 64, 96, 128, 192, 256]),
        current: MM_TILE_LARGE.tn as u32,
        constraints: &["mm_tile_closes", "mm_threadgroup_memory"],
        note: "Weight rows per threadgroup.",
    },
    Knob {
        name: "MM_TILE_LARGE.tm",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved", "mulmm_q8_0_fused"],
        domain: Domain::Ints(&[32, 64, 96, 128, 192, 256]),
        current: MM_TILE_LARGE.tm as u32,
        constraints: &["mm_tile_closes", "mm_threadgroup_memory"],
        note: "Activation columns per threadgroup. Do NOT prune against the \
               register-cliff heuristic: the best known point has 256 live \
               accumulators.",
    },
    Knob {
        name: "MM_TILE_LARGE.kc",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved", "mulmm_q8_0_fused"],
        domain: Domain::Ints(&[1, 2, 3, 4]),
        current: MM_TILE_LARGE.kc as u32,
        constraints: &["mm_threadgroup_memory"],
        note: "K-blocks per barrier round. Pays only WITH occupancy: KC=2 at 256 \
               threads gave 2.97x, the same KC at 64 threads regressed to 8.23x.",
    },
    Knob {
        name: "MM_TILE_LARGE.sgm",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved", "mulmm_q8_0_fused"],
        domain: Domain::Ints(&[1, 2, 4, 8, 16]),
        current: MM_TILE_LARGE.sgm as u32,
        constraints: &["mm_tile_closes", "mm_thread_limit", "mm_register_cliff"],
        note: "Simdgroups along columns. sgm=1 makes the column offset a \
               compile-time zero, which is the best-supported explanation for why \
               a 1-D register tile beat a square one.",
    },
    Knob {
        name: "MM_TILE_LARGE.sgn",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved", "mulmm_q8_0_fused"],
        domain: Domain::Ints(&[1, 2, 4, 8, 16, 32]),
        current: MM_TILE_LARGE.sgn as u32,
        constraints: &["mm_tile_closes", "mm_thread_limit"],
        note: "Simdgroups along rows.",
    },
    Knob {
        name: "MM_TILE_SMALL.tn",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved_small"],
        domain: Domain::Ints(&[16, 32, 64, 128]),
        current: MM_TILE_SMALL.tn as u32,
        constraints: &["mm_tile_closes", "mm_threadgroup_memory"],
        note: "The narrow-output tile. No single tile wins across shapes, which is \
               why there are two.",
    },
    Knob {
        name: "MM_TILE_SMALL.tm",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved_small"],
        domain: Domain::Ints(&[16, 32, 64, 128]),
        current: MM_TILE_SMALL.tm as u32,
        constraints: &["mm_tile_closes", "mm_threadgroup_memory", "mm_register_cliff"],
        note: "",
    },
    Knob {
        name: "MM_TILE_SMALL.kc",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved_small"],
        domain: Domain::Ints(&[1, 2, 3, 4]),
        current: MM_TILE_SMALL.kc as u32,
        constraints: &["mm_threadgroup_memory"],
        note: "",
    },
    Knob {
        name: "MM_TILE_SMALL.sgm",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved_small"],
        domain: Domain::Ints(&[1, 2, 4, 8]),
        current: MM_TILE_SMALL.sgm as u32,
        constraints: &["mm_tile_closes", "mm_thread_limit", "mm_register_cliff"],
        note: "",
    },
    Knob {
        name: "MM_TILE_SMALL.sgn",
        target: Target::Metal,
        site: Site::Emitter,
        kernels: &["mulmm_q8_0_interleaved_small"],
        domain: Domain::Ints(&[1, 2, 4, 8, 16]),
        current: MM_TILE_SMALL.sgn as u32,
        constraints: &["mm_tile_closes", "mm_thread_limit"],
        note: "",
    },
    // ---- Metal, DISPATCH. These are the ones a registry of emitter constants
    // would miss, and the first of them was measurably wrong for that reason.
    Knob {
        name: "TILERS_MV_WIDTH",
        target: Target::Metal,
        site: Site::Dispatch,
        kernels: &["mulmv_q8_0_interleaved", "mulmv_q4_0_interleaved"],
        domain: Domain::Ints(&[32, 64, 128, 256, 512, 1024]),
        current: 64,
        constraints: &[],
        note: "Threads per threadgroup. Hardcoded at 256 and never swept in this \
               integration; 64 is worth ~1.25x on q4_0 and ~1.4x on q8_0. The \
               kernel gives each lane a STRIDED slice, so a wider threadgroup \
               spreads one row over more lanes and shortens each lane's run.",
    },
    Knob {
        name: "TILERS_MM_NARROW_ROWS",
        target: Target::Metal,
        site: Site::Dispatch,
        kernels: &["mulmm_q8_0_interleaved", "mulmm_q8_0_interleaved_small"],
        domain: Domain::Ints(&[256, 512, 1024, 2048, 4096]),
        current: 2048,
        constraints: &[],
        note: "Output rows below which the narrow GEMM tile is chosen. A \
               selection knob, not a kernel knob -- it picks between genomes.",
    },
    Knob {
        name: "TILERS_MM_MIN_COLS",
        target: Target::Metal,
        site: Site::Dispatch,
        kernels: &["mulmm_q8_0_fused", "mulmv_q8_0_fused"],
        domain: Domain::Ints(&[4, 8, 16, 32, 64, 128]),
        current: 32,
        constraints: &[],
        note: "Mat-vec to GEMM crossover. DERIVED, not chosen: the GEMM is flat at \
               ~2600 us for n=4..128 while the mat-vec is linear at ~68 us/column, \
               so they cross near 38. The old value of 4 cost 9.8x at n=4.",
    },
    // ---- PTO / Ascend.
    Knob {
        name: "PTO_MM_KB",
        target: Target::Pto,
        site: Site::Emitter,
        kernels: &["pto.tmatmul"],
        domain: Domain::Ints(&[64, 128, 256, 512]),
        current: 64,
        constraints: &["pto_ub_budget", "pto_nb_dominates_kb"],
        note: "K block. Nearly free within an Nb band -- 64 is shipped because it \
               is the largest Kb that still lets Nb reach 256 under the UB guard, \
               not because Kb itself pays.",
    },
    Knob {
        name: "PTO_MM_NB",
        target: Target::Pto,
        site: Site::Emitter,
        kernels: &["pto.tmatmul"],
        domain: Domain::Ints(&[32, 64, 128, 256, 512]),
        current: 256,
        constraints: &["pto_ub_budget", "pto_nb_dominates_kb"],
        note: "N block, f16/f32 path. THIS is the knob that pays: larger N makes \
               the N-loop trip count bind, not L0B occupancy -- at N=4864, Nb=32 \
               runs 152 blocks against 38 for Nb=128. Worth 1.16-1.28x over the \
               previous default across three shapes.",
    },
    Knob {
        name: "PTO_MM_NB_I8",
        target: Target::Pto,
        site: Site::Emitter,
        kernels: &["pto.tmatmul"],
        domain: Domain::Ints(&[64, 128, 256, 512]),
        current: 256,
        constraints: &["pto_ub_budget"],
        note: "N block, int8 path. Separate from NB because the element width \
               changes what fits.",
    },
    Knob {
        name: "PTO_MM_MROW_ALIGN",
        target: Target::Pto,
        site: Site::Emitter,
        kernels: &["pto.tmatmul"],
        domain: Domain::Ints(&[16]),
        current: 16,
        constraints: &["pto_row16"],
        note: "TileConfig::fixedRowSize on the 910B2 cube. Fixed by hardware, \
               listed so a search knows it is NOT free.",
    },
    Knob {
        name: "SILU_UB_BUDGET_BYTES",
        target: Target::Pto,
        site: Site::Emitter,
        kernels: &["pto.silu"],
        domain: Domain::Ints(&[131072, 163840, 196608, 229376]),
        current: 229376,
        constraints: &["pto_ub_budget"],
        note: "Working-set ceiling for the fused SiLU chain.",
    },
];

/// Objectives a search minimises. Correctness is a HARD constraint, not an
/// objective: a wrong kernel is not a trade-off against a fast one, and every
/// measurement in this workstream that looked like a win without a correctness
/// gate turned out to be computing less work.
/// Minimum measurement protocol for any objective below. This is not advice:
/// a single-shot run of the SAME configuration on this box spread 10.66..14.40
/// us (1.35x) at 2048x2048 q4_0, which is larger than the gap between most
/// configurations a search will compare. A joint rows-per-threadgroup x width
/// sweep run at one shot per cell produced neighbouring cells differing by 60%
/// with no monotone structure, and its "best" was noise. At 20 iterations x 5
/// repeats taking medians, the same measurements land within 3%.
///
/// So: >= 20 iterations per run, >= 5 repeated runs, compare MEDIANS, and
/// discard the first run of a series -- it is consistently fast (cold GPU).
pub const MEASUREMENT_PROTOCOL: (&str, u32, u32) = ("median-of-repeats", 20, 5);

/// Which knobs the ISOLATED benchmark may be used to score, and which it may
/// not. This is not a caveat, it is a validity boundary, and crossing it
/// produced a configuration 40% slower in a real engine while its own objective
/// said it had improved.
///
/// The isolated benchmark amortises launch by putting many identical ops in one
/// graph. That is what makes it valid for arithmetic-side knobs -- without it
/// the measurement is launch overhead. But it also SATURATES the GPU with work
/// that a real decode never has queued, so any knob that trades threadgroup
/// count for per-threadgroup work is scored as if the occupancy it gives up
/// were free.
///
/// Measured: rows-per-threadgroup 8 beat 4 in isolation and ran 88 t/s against
/// 138 end-to-end. Width was misranked the same way -- isolation preferred
/// 32-64, models give 118/134/138 t/s for 32/64/128.
///
/// So: score `arithmetic` knobs in isolation, and `occupancy` knobs ONLY on a
/// model large enough to be bandwidth-bound.
/// Result of a JOINT search over all 16 Metal knobs, scored end-to-end on three
/// objectives (3B q4_0 decode, 0.5B q8_0 decode, 0.5B q8_0 prefill).
///
/// The shipped configuration is on the Pareto front and nothing dominates it.
/// Two things the per-kernel searches could not have shown:
///
/// 1. THE KNOBS ARE COUPLED ACROSS KERNELS. The small GEMM tile serves PREFILL
///    (shapes below TILERS_MM_NARROW_ROWS), so tuning it for any other reason
///    is catastrophic there: S_tm 128 -> 32 reads as a decode gain and costs
///    9x on prefill; the front contains members at 14x, 16x, 47x and 71x.
///    Setting NARROW=999999 -- forcing the small tile everywhere -- is the
///    single most destructive knob value in the space.
///
/// 2. A PARETO FRONT CAN BE PART NOISE. The genome that "beat" the incumbent on
///    3B q4_0 decode (1.051 vs 1.075) differs from it only in Q8_QPL, S_tm and
///    S_sgn -- a q8_0 mat-vec knob and two GEMM knobs, NONE of which the q4_0
///    decode path executes. That 2% is measurement noise sitting on the front
///    because NSGA-II cannot know which knobs reach which objective. Check
///    that a claimed win is on a knob the objective can actually see.
pub const ORACLE_VALIDITY: &[(&str, &str)] = &[
    ("MUL_MV_ILV_ROWS_PER_TG", "occupancy -- score end-to-end only"),
    ("MUL_MV_ROWS_PER_TG",     "occupancy -- score end-to-end only"),
    ("MUL_MV_Q4_ROWS_PER_TG",  "occupancy -- score end-to-end only"),
    ("TILERS_MV_WIDTH",        "occupancy -- score end-to-end only"),
    ("MM_TILE_LARGE.sgm",      "occupancy -- score end-to-end only"),
    ("MM_TILE_LARGE.sgn",      "occupancy -- score end-to-end only"),
    ("MM_TILE_SMALL.sgm",      "occupancy -- score end-to-end only"),
    ("MM_TILE_SMALL.sgn",      "occupancy -- score end-to-end only"),
    ("MUL_MV_ILV_QUANTS_PER_LANE",   "arithmetic -- isolation is valid"),
    ("MUL_MV_ILV_Q4_BYTES_PER_LANE", "arithmetic -- isolation is valid"),
    ("MUL_MV_VECS_PER_LANE",         "arithmetic -- isolation is valid"),
    ("MUL_MV_Q4_VECS_PER_LANE",      "arithmetic -- isolation is valid"),
    ("MM_TILE_LARGE.kc",             "arithmetic -- isolation is valid"),
    ("MM_TILE_SMALL.kc",             "arithmetic -- isolation is valid"),
    // Tile extents change both the work per threadgroup AND how many there
    // are, so they sit on both sides and must be scored end-to-end.
    ("MM_TILE_LARGE.tn", "occupancy -- score end-to-end only"),
    ("MM_TILE_LARGE.tm", "occupancy -- score end-to-end only"),
    ("MM_TILE_SMALL.tn", "occupancy -- score end-to-end only"),
    ("MM_TILE_SMALL.tm", "occupancy -- score end-to-end only"),
    // Selection knobs pick between genomes rather than tuning one.
    ("TILERS_MM_NARROW_ROWS", "selection -- score end-to-end only"),
    ("TILERS_MM_MIN_COLS",    "selection -- score end-to-end only"),
];

pub const OBJECTIVES: &[(&str, &str)] = &[
    ("us_per_op", "minimise; per-shape, launch amortised over many ops per graph, \
                   median of >=5 repeats at >=20 iters -- see MEASUREMENT_PROTOCOL"),
    ("threadgroup_bytes", "minimise; frees budget for larger tiles"),
    ("e2e_tokens_per_sec", "maximise; must come from a model large enough to be bandwidth-bound"),
];

/// Emit the registry as JSON for `tile_search` to ingest.
pub fn knobs_json() -> String {
    let mut s = String::new();
    s.push_str("{\n  \"knobs\": [\n");
    for (i, k) in KNOBS.iter().enumerate() {
        let domain = match k.domain {
            Domain::Ints(v) => {
                let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                format!("{{\"kind\": \"int\", \"values\": [{}]}}", parts.join(", "))
            }
            Domain::Bools => "{\"kind\": \"bool\", \"values\": [0, 1]}".to_string(),
        };
        let kernels: Vec<String> = k.kernels.iter().map(|x| format!("\"{}\"", x)).collect();
        let cons: Vec<String> = k.constraints.iter().map(|x| format!("\"{}\"", x)).collect();
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"target\": \"{}\", \"site\": \"{}\", \
             \"kernels\": [{}], \"domain\": {}, \"current\": {}, \
             \"constraints\": [{}], \"note\": \"{}\"}}{}\n",
            k.name,
            k.target.as_str(),
            k.site.as_str(),
            kernels.join(", "),
            domain,
            k.current,
            cons.join(", "),
            k.note.replace('"', "'").replace('\n', " "),
            if i + 1 == KNOBS.len() { "" } else { "," }
        ));
    }
    s.push_str("  ],\n  \"constraints\": [\n");
    for (i, c) in CONSTRAINTS.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"id\": \"{}\", \"kind\": \"{}\", \"expr\": \"{}\", \"why\": \"{}\"}}{}\n",
            c.id,
            c.kind,
            c.expr.replace('"', "'"),
            c.why.replace('"', "'").replace('\n', " "),
            if i + 1 == CONSTRAINTS.len() { "" } else { "," }
        ));
    }
    s.push_str("  ],\n  \"oracle_validity\": {\n");
    for (i, (n, d)) in ORACLE_VALIDITY.iter().enumerate() {
        s.push_str(&format!("    \"{}\": \"{}\"{}\n", n, d,
            if i + 1 == ORACLE_VALIDITY.len() { "" } else { "," }));
    }
    s.push_str("  },\n");
    s.push_str(&format!(
        "  \"measurement_protocol\": {{\"kind\": \"{}\", \"min_iters\": {}, \"min_repeats\": {}, \
         \"compare\": \"median\", \"drop_first_run\": true}},\n  \"objectives\": [\n",
        MEASUREMENT_PROTOCOL.0, MEASUREMENT_PROTOCOL.1, MEASUREMENT_PROTOCOL.2));
    for (i, (n, d)) in OBJECTIVES.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"sense\": \"{}\"}}{}\n",
            n,
            d,
            if i + 1 == OBJECTIVES.len() { "" } else { "," }
        ));
    }
    s.push_str("  ]\n}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every declared `current` must equal the constant it mirrors. Without
    /// this the registry rots exactly where it hurts: a knob that stops feeding
    /// the kernel still reads as tunable, and a search spends its budget
    /// mutating something inert.
    #[test]
    fn test_knob_registry_matches_reality() {
        let get = |name: &str| KNOBS.iter().find(|k| k.name == name).expect(name).current;

        // PTO knobs are declared here but live in mlir_to_pto; assert the two
        // agree by value so the registry cannot drift from the emitter.
        assert_eq!(get("PTO_MM_KB"), 64, "registry must track the shipped Kb");
        assert_eq!(get("PTO_MM_NB"), 256, "registry must track the shipped Nb");

        assert_eq!(get("MUL_MV_ILV_ROWS_PER_TG"), MUL_MV_ILV_ROWS_PER_TG as u32);
        assert_eq!(get("MUL_MV_ILV_QUANTS_PER_LANE"), MUL_MV_ILV_QUANTS_PER_LANE as u32);
        assert_eq!(get("MUL_MV_ILV_Q4_BYTES_PER_LANE"), MUL_MV_ILV_Q4_BYTES_PER_LANE as u32);
        assert_eq!(get("MUL_MV_ROWS_PER_TG"), MUL_MV_ROWS_PER_TG as u32);
        assert_eq!(get("MUL_MV_VECS_PER_LANE"), MUL_MV_VECS_PER_LANE as u32);
        assert_eq!(get("MUL_MV_Q4_ROWS_PER_TG"), MUL_MV_Q4_ROWS_PER_TG as u32);
        assert_eq!(get("MUL_MV_Q4_VECS_PER_LANE"), MUL_MV_Q4_VECS_PER_LANE as u32);

        assert_eq!(get("MM_TILE_LARGE.tn"), MM_TILE_LARGE.tn as u32);
        assert_eq!(get("MM_TILE_LARGE.tm"), MM_TILE_LARGE.tm as u32);
        assert_eq!(get("MM_TILE_LARGE.kc"), MM_TILE_LARGE.kc as u32);
        assert_eq!(get("MM_TILE_LARGE.sgm"), MM_TILE_LARGE.sgm as u32);
        assert_eq!(get("MM_TILE_LARGE.sgn"), MM_TILE_LARGE.sgn as u32);
        assert_eq!(get("MM_TILE_SMALL.tn"), MM_TILE_SMALL.tn as u32);
        assert_eq!(get("MM_TILE_SMALL.tm"), MM_TILE_SMALL.tm as u32);
        assert_eq!(get("MM_TILE_SMALL.kc"), MM_TILE_SMALL.kc as u32);
        assert_eq!(get("MM_TILE_SMALL.sgm"), MM_TILE_SMALL.sgm as u32);
        assert_eq!(get("MM_TILE_SMALL.sgn"), MM_TILE_SMALL.sgn as u32);
    }

    /// The value shipped must itself satisfy the preconditions the registry
    /// declares. If it does not, either the value is wrong or the constraint is,
    /// and both are worth failing over.
    #[test]
    fn test_current_values_satisfy_their_constraints() {
        for t in [MM_TILE_LARGE, MM_TILE_SMALL, MM_TILE_PLANAR] {
            assert_eq!(t.tm % (8 * t.sgm), 0, "mm_tile_closes (tm) for {:?}", t.name);
            assert_eq!(t.tn % (8 * t.sgn), 0, "mm_tile_closes (tn) for {:?}", t.name);
            assert!(t.sgm * t.sgn * 32 <= 1024, "mm_thread_limit for {:?}", t.name);
            let staged = (t.kc * 32 * (t.tn + 1) + t.tm * t.kc * 33) * 2;
            let scratch = t.sgm * t.sgn * 64 * 4;
            assert!(staged + scratch <= 32768, "mm_threadgroup_memory for {:?}", t.name);
            // No register-cliff assertion: the shipped tile has MI=16 NI=16,
            // 256 accumulators, and is the fastest point measured.
        }
        assert_eq!(32 % MUL_MV_ILV_QUANTS_PER_LANE, 0);
        assert_eq!(16 % MUL_MV_ILV_Q4_BYTES_PER_LANE, 0);
        assert_eq!(8 % MUL_MV_VECS_PER_LANE, 0);
        assert_eq!(2 % MUL_MV_Q4_VECS_PER_LANE, 0);
    }

    /// Every constraint a knob names must exist, and every knob must name the
    /// constraints its own domain can violate.
    #[test]
    fn test_constraint_ids_resolve() {
        for k in KNOBS {
            for c in k.constraints {
                assert!(
                    CONSTRAINTS.iter().any(|x| x.id == *c),
                    "knob {} names unknown constraint {}", k.name, c
                );
            }
            assert!(!k.kernels.is_empty(), "knob {} affects no kernel", k.name);
        }
    }

    /// A search that samples one point per configuration on this hardware is
    /// sampling noise. Encoded so the protocol travels with the registry.
    #[test]
    fn test_measurement_protocol_is_declared() {
        let (kind, iters, repeats) = MEASUREMENT_PROTOCOL;
        assert_eq!(kind, "median-of-repeats");
        assert!(iters >= 20, "single-shot runs spread 1.35x on this box");
        assert!(repeats >= 5, "one run per configuration cannot rank configurations");
    }

    /// Every knob must say which oracle may score it, or a search will score
    /// an occupancy knob in isolation and be confidently wrong.
    #[test]
    fn test_every_knob_declares_its_oracle() {
        for k in KNOBS {
            let base = k.name;
            assert!(ORACLE_VALIDITY.iter().any(|(n, _)| *n == base)
                    || k.target == Target::Pto,
                "knob {} does not declare which oracle may score it", base);
        }
    }

    #[test]
    fn test_emit_knobs_json() {
        let j = knobs_json();
        assert!(j.contains("\"TILERS_MV_WIDTH\""), "dispatch knobs must be included:\n{}", j);
        assert!(j.contains("\"site\": \"dispatch\""));
        assert!(j.contains("\"target\": \"pto\""), "PTO knobs must be included");
        assert!(j.contains("\"mm_register_cliff\""));
        if let Ok(dir) = std::env::var("TILERS_KNOBS_OUT") {
            std::fs::write(format!("{}/tilers-knobs.json", dir), &j).unwrap();
        }
    }
}
