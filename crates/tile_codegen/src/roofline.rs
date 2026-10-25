//! Roofline accounting: is this kernel bandwidth-bound, and what transfer
//! granularity does that require?
//!
//! Written from a measurement rather than a principle. `add_rms_norm_dynamic_quant`
//! scored 69.05/100 against a leader's 80.1, and the whole of that difference is
//! the performance term (`compile 20 + function 30 + performance 50`): the kernel
//! is correct and 2.5x slower than the vendor's fused op. Diffing the two per
//! case on 910b separates the regimes cleanly:
//!
//! | regime | ours | vendor |
//! |---|---|---|
//! | large (about 470 MB, memory-bound) | **354 GB/s** median | **672 GB/s** median |
//! | small (under 50 MB) | 91 GB/s | 92 GB/s |
//!
//! So it is not a tiling or a barrier problem in general -- on small shapes we
//! already match the vendor. It is that we reach **53 %** of the vendor's achieved
//! bandwidth once the operator is memory-bound, and how bad it gets depends on the
//! row extent: 469.8 MB at `[8192,8192]` takes us 1473 us (319 GB/s) against the
//! vendor's 577 us (814 GB/s), while `[524288,128]` -- the same bytes, narrow rows
//! -- takes us 4462 us (105 GB/s) to the vendor's 299 GB/s.
//!
//! The vendor ships **four shape-specialised variants** of this one fused op
//! (`AddRmsNormDynamicQuant_fp16_high_performance_0..3`, `blockDim: -1`, one
//! workspace, `core_type: AIV`), so shape specialisation is what it is doing
//! instead. This module is the Rust-side form of that finding: it decides which
//! regime a shape is in, and states the transfer granularity a strategy must
//! reach to be admissible there -- a guard, not a knob, because a strategy that
//! misses it cannot be rescued by tuning afterwards.
//!
//! The CANN guidance already cited in the generator ("a single transfer wants to
//! be >= 16 KB", cann-learning-hub 08_performance_optimization) is the mechanism:
//! below that, the transfer is latency- and drain-bound rather than
//! bandwidth-bound, which is why the narrow-row case is the worst one -- D=128 at
//! 2 bytes is a 256-byte row.

use super::layout::GuardedTransform;

/// Bytes in a gigabyte, for the bandwidth arithmetic (decimal, matching the
/// GB/s figures the profiler reports).
pub const BYTES_PER_GB: f64 = 1e9;

/// A single transfer below this cannot reach the achievable bandwidth: it is
/// paid for in latency and pipeline drains instead. CANN's own guidance.
pub const MIN_TRANSFER_BYTES: u64 = 16 * 1024;

/// Fraction of the roofline anchor a strategy must reach before it counts as
/// bandwidth-bound. 0.75 sits above our 53 % and below the vendor's 100 %.
pub const BOUND_FRACTION: f64 = 0.75;

/// The practical roofline: what the vendor's fused op demonstrably sustains on
/// this substrate, not the HBM datasheet figure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roofline {
    pub anchor_gbps: f64,
}

impl Roofline {
    /// Median achieved by the vendor op on the large arnq cases on 910b.
    pub const ASCEND_910B_MEASURED: Roofline = Roofline {
        anchor_gbps: 672.0,
    };

    pub fn bound_gbps(&self) -> f64 {
        self.anchor_gbps * BOUND_FRACTION
    }
}

/// Memory traffic of one invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Traffic {
    pub read_bytes: u64,
    pub write_bytes: u64,
}

impl Traffic {
    pub fn new(read_bytes: u64, write_bytes: u64) -> Traffic {
        Traffic {
            read_bytes,
            write_bytes,
        }
    }

    /// Add + RMS-norm + per-token int8 quant: read x1, x2; write xOut in the
    /// source width and y at one byte per element. Gamma is negligible and the
    /// scale is one float per row.
    pub fn add_rms_norm_dynamic_quant(rows: u64, extent: u64, elem_bytes: u64) -> Traffic {
        let n = rows * extent;
        Traffic {
            read_bytes: 2 * n * elem_bytes,
            write_bytes: n * elem_bytes + n,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.read_bytes + self.write_bytes
    }

    pub fn gb(&self) -> f64 {
        self.total_bytes() as f64 / BYTES_PER_GB
    }
}

/**
 * Elements stageable per batch, per the launcher's batching formula
 * (`per = pitchE_*15 + 104`, `R = 143360/per`, so `R * pitchE_ ~= 9557`).
 *
 * **This is NOT a UB-safe cap and must not gate staging on its own.** It is only
 * valid underneath the launcher's `D <= tile` gate, and relaxing that gate to
 * this formula faulted the device with 507035 (vector core exception) on D in
 * (4096, 9557]. Measured 2026-09-26; the fault also invalidated the apparent
 * 1.85x win at D=8192, which came from over-allocating UB and got away with it
 * at that one shape.
 */
pub const BATCH_FORMULA_CAP_2B: u64 = 9557;

/// The cap that actually bounds UB for this buffer set, from the launcher's own
/// `cap = 176128 / (3*es + 3 + 16)`: the largest STAGE this kernel can hold, and
/// therefore the largest row that may be staged whole.
pub const TILE_CAP_2B: u64 = 7045;

/// The two caps must never be confused for one another, so the relationship is
/// checked at compile time.
const _: () = assert!(
    BATCH_FORMULA_CAP_2B > TILE_CAP_2B,
    "the batch formula's cap is looser than the UB-safe one; planning against it is what faulted the device"
);

/// Whether a whole row may be staged as one transfer.
pub fn whole_row_admissible(extent: u64, tile_cap: u64) -> bool {
    extent <= tile_cap
}

/// The per-element UB footprint a staging decision may spend if it wants
/// `target_elems` staged inside `ub_bytes`.
///
/// This is the quantity that decides whether the granularity target is reachable
/// at all: a 16 KB transfer of 2-byte elements needs 8192 elements staged, which
/// inside the 176128-byte UB allows at most 21.5 bytes per element. This kernel
/// spends `3*es + 3 + 16 = 25`, which is why D=8192 cannot reach 16 KB however
/// the tile and rows are chosen -- the footprint has to come down first.
pub fn max_footprint_bytes(ub_bytes: u64, target_elems: u64) -> f64 {
    if target_elems == 0 {
        return 0.0;
    }
    ub_bytes as f64 / target_elems as f64
}

/// What the UB can actually support at a given extent, once the real cap is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feasibility {
    /// Stageable with the whole-row batching the kernel already has.
    WholeRow(StagingPlan),
    /// Only reachable by staging a CHUNK of each row, which the kernel does not
    /// do yet: its batching path assumes the staged pitch is the whole row.
    /// `bytes_per_transfer` says how far that gets -- check it against the
    /// target, because it may still be short.
    NeedsChunked(StagingPlan),
    /// The UB cannot hold two rows of even the smallest useful chunk.
    Impossible,
}

/// Plan staging under the UB-safe cap.
///
/// There is deliberately **no way to pass a cap in**. An earlier version took the
/// cap as an argument, and passing the batching formula's 9557 instead of the
/// tile formula's 7045 produced a whole-row plan at D=8192 with a 16 KB transfer
/// -- which measured 1.85x faster and faulted the device with 507035 on other D
/// in the same band. A caller cannot be trusted to choose the safe cap, so the
/// choice is not exposed.
pub fn plan_staging(extent: u64, elem_bytes: u64, target_bytes: u64) -> Feasibility {
    let cap = tile_cap(elem_bytes);
    if whole_row_admissible(extent, cap) {
        // Under the gate the launcher's own row count is the one in force, and it
        // is what must be reproduced: `per = pitchE_*15 + 104`, `R = 140KB/per`.
        // Using `cap / extent` here instead is a cruder estimate and disagrees
        // with the kernel (D=1024: 6 against the launcher's 9).
        let pitch = extent.div_ceil(32) * 32;
        // The launcher's constant is `2*sizeof(TSrc) + 1 + 2 + 4 + 4` = 2*es + 11.
        let per = pitch * (2 * elem_bytes + 11) + 104;
        let mut rows = if per > 0 { (140 * 1024) / per } else { 0 };
        rows = rows.clamp(1, 128);
        return Feasibility::WholeRow(StagingPlan::transfer(extent, extent, elem_bytes, rows));
    }
    // Chunked: the largest whole-block chunk that still allows two rows.
    let mut chunk = 64u64;
    while chunk * 2 <= extent && chunk * 2 * 2 <= cap {
        chunk *= 2;
    }
    let rows = cap / chunk;
    if rows < 2 {
        return Feasibility::Impossible;
    }
    let _ = target_bytes;
    Feasibility::NeedsChunked(StagingPlan::transfer(extent, chunk, elem_bytes, rows))
}

/// The joint `(tile, rows_per_chunk)` the launcher cannot express.
///
/// The launcher maximises `tile` first and only then asks how many **whole rows**
/// fit, so for `D > tile` it gets fewer than two (at D=14336 and 16384 the integer
/// division gives R = 0) and silently falls to one row-chunk per transfer -- 8 KB,
/// under the guidance, 315 GB/s measured against 500. Picking both together
/// instead reaches the 16 KB transfer from the same budget:
///
/// | D | launcher | this plan | transfer | measured |
/// |---|---|---|---|---|
/// | 1024 | tile 1024, R 9 | tile 1024, R 9 | 18 KB | 469-473 GB/s |
/// | 2048 | tile 2048, R 4 | tile 2048, R 4 | 16 KB | 469-473 |
/// | 4096 | tile 4096, R 2 | tile 4096, R 2 | 16 KB | 530-536 |
/// | 8192 | tile 4096, R **1** | tile 4096, R **2** | 16 KB | should be ~500 |
/// | 16384 | tile 4096, R **0** | tile 4096, R **2** | 16 KB | should be ~500 |
///
/// It agrees with the launcher wherever the launcher is already batching, which is
/// the check that it is not just a different rule:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingPlan {
    pub tile: u64,
    pub rows_per_chunk: u64,
    pub bytes_per_transfer: u64,
}

impl StagingPlan {
    pub fn staging(&self, extent: u64, elem_bytes: u64) -> Staging {
        Staging::new(extent, self.tile, elem_bytes, self.rows_per_chunk)
    }

    pub fn transfer(extent: u64, tile: u64, elem_bytes: u64, rows_per_chunk: u64) -> StagingPlan {
        StagingPlan {
            tile,
            rows_per_chunk,
            bytes_per_transfer: rows_per_chunk * extent.min(tile).max(1) * elem_bytes,
        }
    }

    pub fn meets_granularity(&self) -> bool {
        self.bytes_per_transfer >= MIN_TRANSFER_BYTES
    }
}

/// The UB-safe cap for a dtype, computed exactly as the launcher does:
/// `176128 / (3*es + 3 + 16)`. Everything that plans a stage must size against
/// this and not against the batching formula's 140 KB (see
/// [`BATCH_FORMULA_CAP_2B`]).
pub fn tile_cap(elem_bytes: u64) -> u64 {
    if elem_bytes == 0 {
        return 0;
    }
    176_128 / (3 * elem_bytes + 3 + 16)
}

/// How a strategy actually moves that traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferShape {
    pub bytes_per_transfer: u64,
    pub transfers: u64,
}

/// A staged strategy stated the way the kernel stages it: **one chunk of a row
/// at a time**, `rows_per_chunk` rows at a time.
///
/// This is the correction the first version of this module needed. The
/// granularity that matters is the CHUNK, not the row: at `D=8192` with
/// `tile=4096` a row is one 16 KB row only if a whole row is staged, and it is
/// not -- the launcher's `tile` is 4096, so the transfer is an 8 KB chunk. That
/// is exactly the observed split:
///
/// | path | transfer | measured |
/// |---|---|---|
/// | plain, `D > tile` | 1 row-chunk = 8 KB | 319, 311, 303, 321 GB/s |
/// | batched, `D <= tile` | R rows x chunk >= 16 KB | 536, 530, 473, 469 GB/s |
///
/// The same 8 KB chunk reaches ~500 GB/s when R rows are staged with it and
/// ~315 GB/s when a single row-chunk is transferred on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staging {
    pub chunk_elems: u64,
    pub rows_per_chunk: u64,
    pub elem_bytes: u64,
}

impl Staging {
    /// The launcher takes the chunk to be `min(extent, tile)`.
    pub fn new(extent: u64, tile: u64, elem_bytes: u64, rows_per_chunk: u64) -> Staging {
        Staging {
            chunk_elems: extent.min(tile).max(1),
            rows_per_chunk: rows_per_chunk.max(1),
            elem_bytes,
        }
    }

    pub fn bytes_per_transfer(&self) -> u64 {
        self.rows_per_chunk * self.chunk_elems * self.elem_bytes
    }

    pub fn transfers(&self, rows: u64) -> u64 {
        let chunks_per_row = rows.max(1);
        let _ = chunks_per_row;
        (rows.div_ceil(self.rows_per_chunk)).max(1)
    }

    /// Scalar reads out of UB, one per row per batch. Each is a pipeline drain,
    /// and on the narrow-row shapes this is what is left once the transfer is
    /// large enough: `[524288,128]` batches 70 rows per transfer and pays 70
    /// scalar reads per batch, which is why it sits at 105 GB/s while a wide
    /// chunk with the same byte count reaches ~500.
    pub fn scalar_drains(&self, rows: u64) -> u64 {
        rows
    }

    pub fn meets_granularity(&self) -> bool {
        self.bytes_per_transfer() >= MIN_TRANSFER_BYTES
    }
}

impl TransferShape {
    pub fn new(bytes_per_transfer: u64, transfers: u64) -> TransferShape {
        TransferShape {
            bytes_per_transfer,
            transfers,
        }
    }

    /// For a kernel that stages `rows_per_batch` rows of `extent` elements at
    /// `elem_bytes` each, one transfer per batch.
    pub fn batched(rows: u64, extent: u64, elem_bytes: u64, rows_per_batch: u64) -> TransferShape {
        let batches = (rows + rows_per_batch - 1) / rows_per_batch.max(1);
        TransferShape {
            bytes_per_transfer: rows_per_batch.max(1) * extent * elem_bytes,
            transfers: batches.max(1),
        }
    }

    /// For a kernel that stages one tile per transfer.
    pub fn per_tile(rows: u64, extent: u64, elem_bytes: u64) -> TransferShape {
        TransferShape {
            bytes_per_transfer: extent * elem_bytes,
            transfers: rows.max(1),
        }
    }

    pub fn meets_granularity(&self) -> bool {
        self.bytes_per_transfer >= MIN_TRANSFER_BYTES
    }
}

/// Achieved bandwidth over a measured duration.
pub fn achieved_gbps(bytes: u64, micros: f64) -> f64 {
    if micros <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / (micros / 1e6) / BYTES_PER_GB
}

/// Which regime a shape is in. This is the Rust-side stand-in for the vendor's
/// four compiled variants: the regime decides what has to be done, and the point
/// of naming it is that the same schedule cannot serve all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Few enough bytes that latency, not bandwidth, dominates. Our kernel
    /// already matches the vendor here, so nothing should be changed for it.
    Latency,
    /// Bandwidth-bound with rows wide enough that a row staging is a large
    /// transfer.
    BandwidthWide,
    /// Bandwidth-bound with rows too narrow to make one transfer large -- the
    /// worst case, `[524288,128]`. Needs many rows per transfer.
    BandwidthNarrow,
}

/// Row extent, in bytes, below which one row cannot make a worthwhile transfer.
pub const NARROW_ROW_BYTES: u64 = MIN_TRANSFER_BYTES / 2;
/// Total traffic below which the shape is latency-dominated on this substrate.
pub const LATENCY_TOTAL_BYTES: u64 = 50 * 1024 * 1024;

pub fn classify(traffic: Traffic, extent: u64, elem_bytes: u64) -> Regime {
    if traffic.total_bytes() < LATENCY_TOTAL_BYTES {
        return Regime::Latency;
    }
    if extent * elem_bytes < NARROW_ROW_BYTES {
        Regime::BandwidthNarrow
    } else {
        Regime::BandwidthWide
    }
}

/// The rows-per-transfer a strategy needs to be admissible in this regime.
///
/// Returns `None` in the latency regime: nothing needs to change there, and
/// saying so is more useful than returning a number that invites a change.
pub fn rows_per_transfer_needed(regime: Regime, extent: u64, elem_bytes: u64) -> Option<u64> {
    match regime {
        Regime::Latency => None,
        _ => {
            let row_bytes = (extent * elem_bytes).max(1);
            let need = MIN_TRANSFER_BYTES.div_ceil(row_bytes);
            Some(need.max(1))
        }
    }
}

/// The guarded-transform entries this module owns, in the same ledger as
/// [`crate::layout::TRANSFORMS`]: what may fire, on what, for what, and the trap.
pub const ROOFLINE_TRANSFORMS: &[GuardedTransform] = &[
    GuardedTransform {
        name: "widen_the_transfer",
        requires: "TransferShape::meets_granularity (bytes_per_transfer >= 16 KB)",
        reward: "the memory-bound regime: 354 GB/s -> the vendor's 672 GB/s median",
        trap: "tuning the tile without the granularity: on the small shapes the \
               kernel already matches the vendor, so a change there is pure risk",
    },
    GuardedTransform {
        name: "shape_specialised_variant",
        requires: "Regime is not Latency (the shape actually is bandwidth-bound)",
        reward: "the vendor's four compiled variants for one fused op; one \
                 schedule cannot serve wide and narrow rows",
        trap: "one kernel for all shapes: [8192,8192] and [524288,128] move the \
               same 470 MB and differ 3x in achieved bandwidth",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured pair this module was written from: 469.8 MB, ours 1473 us
    /// against the vendor's 577 us.
    #[test]
    fn our_bandwidth_in_the_memory_bound_regime_is_about_half_the_vendors() {
        let tr = Traffic::add_rms_norm_dynamic_quant(8192, 8192, 2);
        assert_eq!(tr.total_bytes(), 469_762_048);
        let ours = achieved_gbps(tr.total_bytes(), 1473.2);
        let vendor = achieved_gbps(tr.total_bytes(), 577.3);
        assert!((ours - 318.9).abs() < 2.0, "ours {ours}");
        assert!((vendor - 813.7).abs() < 2.0, "vendor {vendor}");
        assert!(ours < 0.5 * vendor);
        assert!(ours < Roofline::ASCEND_910B_MEASURED.bound_gbps());
        assert!(vendor >= Roofline::ASCEND_910B_MEASURED.bound_gbps());
    }

    /// And the regime where we already match: the model must NOT claim a gap
    /// there, or it would send the search after shapes that have nothing to give.
    #[test]
    fn the_small_regime_matches_the_vendor_and_must_not_be_flagged() {
        // [512,512] fp16: 1.8 MB, ours 11.3 us, vendor 10.0 us.
        let tr = Traffic::add_rms_norm_dynamic_quant(512, 512, 2);
        assert_eq!(classify(tr, 512, 2), Regime::Latency);
        assert_eq!(rows_per_transfer_needed(Regime::Latency, 512, 2), None);
        let ours = achieved_gbps(tr.total_bytes(), 11.3);
        let vendor = achieved_gbps(tr.total_bytes(), 10.0);
        assert!(vendor / ours < 1.2, "{ours} vs {vendor}");
    }

    /// The worst case is narrow rows, not wide ones: same bytes, 3x apart.
    #[test]
    fn narrow_rows_are_the_worst_regime_not_wide_ones() {
        let tr = Traffic::add_rms_norm_dynamic_quant(524_288, 128, 2);
        assert_eq!(tr.total_bytes(), 469_762_048);
        assert_eq!(classify(tr, 128, 2), Regime::BandwidthNarrow);
        let ours = achieved_gbps(tr.total_bytes(), 4461.9);
        assert!(ours < 110.0, "ours {ours}");
        // one row is 256 bytes, far below the guidance, so many rows per transfer
        let need = rows_per_transfer_needed(Regime::BandwidthNarrow, 128, 2).unwrap();
        assert_eq!(need, 64);
        assert_eq!(need * 128 * 2, 16_384);

        // the same bytes with wide rows lands in the other bandwidth regime
        assert_eq!(classify(tr, 8192, 2), Regime::BandwidthWide);
        let wide_need = rows_per_transfer_needed(Regime::BandwidthWide, 8192, 2).unwrap();
        assert_eq!(wide_need, 1, "a wide row is already one 16 KB transfer");
        assert!(wide_need < need);
    }

    /// The granularity guard, on both sides of the line.
    #[test]
    fn the_granularity_guard_agrees_with_the_measured_gap() {
        let one_row = TransferShape::per_tile(8192, 128, 2);
        assert_eq!(one_row.bytes_per_transfer, 256);
        assert!(!one_row.meets_granularity(), "D=128 is the failing case");

        let batched = TransferShape::batched(8192, 128, 2, 64);
        assert_eq!(batched.bytes_per_transfer, 16_384);
        assert!(batched.meets_granularity());

        // a wide row already satisfies it with one row per transfer
        assert!(TransferShape::per_tile(8192, 8192, 2).meets_granularity());
    }

    #[test]
    fn the_plain_path_stages_8kb_and_that_is_the_measured_slow_regime() {
        // D=8192 > tile=4096: one row-chunk per transfer.
        let plain = Staging::new(8192, 4096, 2, 1);
        assert_eq!(plain.chunk_elems, 4096);
        assert_eq!(plain.bytes_per_transfer(), 8192);
        assert!(
            !plain.meets_granularity(),
            "8 KB is under the 16 KB guidance -- this is the codegen change"
        );

        // Staging two rows per chunk clears it, and costs no extra traffic.
        let two = Staging::new(8192, 4096, 2, 2);
        assert_eq!(two.bytes_per_transfer(), 16_384);
        assert!(two.meets_granularity());

        // The measured medians either side of that line, same byte count.
        let tr = Traffic::add_rms_norm_dynamic_quant(8192, 8192, 2);
        let plain_gb = achieved_gbps(tr.total_bytes(), 1473.2);
        let batched_gb = achieved_gbps(tr.total_bytes(), 875.4);
        assert!((plain_gb - 318.9).abs() < 2.0, "{plain_gb}");
        assert!((batched_gb - 536.7).abs() < 2.0, "{batched_gb}");
        assert!(batched_gb > 1.6 * plain_gb);
    }

    #[test]
    fn the_batched_regime_already_clears_the_guard_for_wide_chunks() {
        // D=4096 == tile: R rows of a 4096-element chunk.
        assert!(Staging::new(4096, 4096, 2, 4).meets_granularity());
        assert_eq!(Staging::new(4096, 4096, 2, 4).bytes_per_transfer(), 32_768);
        // and a chunk as small as 2048 clears it at 4 rows
        assert!(Staging::new(2048, 4096, 2, 4).meets_granularity());
    }

    #[test]
    fn narrow_rows_clear_the_guard_and_are_still_slow_so_granularity_is_not_the_whole_story() {
        // [524288,128]: 64 rows x 256 B = 16 KB, guard satisfied, and yet 105 GB/s.
        let narrow = Staging::new(128, 4096, 2, 64);
        assert_eq!(narrow.bytes_per_transfer(), 16_384);
        assert!(narrow.meets_granularity());
        let tr = Traffic::add_rms_norm_dynamic_quant(524_288, 128, 2);
        let gb = achieved_gbps(tr.total_bytes(), 4461.9);
        assert!(gb < 110.0, "{gb}");
        // The remaining cause is the per-row scalar read out of UB, one drain each.
        assert_eq!(narrow.scalar_drains(524_288), 524_288);
    }

    #[test]
    fn the_plan_agrees_with_the_launcher_where_the_launcher_already_batches() {
        // D <= tile: the launcher's whole-row batching is already at granularity.
        for (d, tile, rows) in [(1024u64, 1024u64, 9u64), (2048, 2048, 4), (4096, 4096, 2)] {
            match plan_staging(d, 2, MIN_TRANSFER_BYTES) {
                Feasibility::WholeRow(p) => {
                    assert_eq!(p.tile, tile, "D={d}");
                    assert_eq!(p.rows_per_chunk, rows, "D={d}");
                    assert!(p.meets_granularity());
                }
                other => panic!("D={d} should be a whole-row plan, got {other:?}"),
            }
        }
        // The two caps are different quantities, which is the trap: planning
        // against the batch formula is what faulted the device. (Asserted at
        // compile time above, where the compiler can see both are constants.)
        assert_eq!(tile_cap(2), TILE_CAP_2B);
        assert_eq!(tile_cap(2), 7045);
    }

    /// The correction the device forced. Relaxing `D <= tile` to the batch
    /// formula looked like a 1.85x win at D=8192 (1159 us -> 626 us, 405 ->
    /// 751 GB/s) and FAULTED with 507035 on other D in (4096, 9557]. The gate is
    /// UB safety, and the batch formula is only valid underneath it.
    #[test]
    fn a_whole_row_above_the_tile_cap_is_unsafe_and_d8192_cannot_reach_16kb() {
        assert!(whole_row_admissible(4096, TILE_CAP_2B));
        assert!(
            !whole_row_admissible(8192, TILE_CAP_2B),
            "the tile cap is the row that fits a stage: 7045 elements"
        );
        // The transfer we want, against what the UB can stage:
        assert_eq!(8192 * 2, 16_384);
        const { assert!(TILE_CAP_2B * 2 < 16_384, "7045 elements is only 14090 bytes") };
        assert_eq!(
            plan_staging(8192, 2, MIN_TRANSFER_BYTES),
            Feasibility::NeedsChunked(StagingPlan {
                tile: 2048,
                rows_per_chunk: 3,
                bytes_per_transfer: 12_288
            }),
            "chunked staging reaches only 12 KB -- better than the 8 KB plain path \
             and still short of the 16 KB that makes a transfer bandwidth-bound"
        );
        // which is why the target is not reachable by staging alone:
        const { assert!(12_288 < MIN_TRANSFER_BYTES) };
        assert!(matches!(
            plan_staging(4096, 2, MIN_TRANSFER_BYTES),
            Feasibility::WholeRow(_)
        ));
    }

    /// What would actually have to change: the footprint per staged element, not
    /// the tile.
    #[test]
    fn reaching_16kb_at_d8192_needs_a_smaller_footprint_not_a_bigger_tile() {
        let need = max_footprint_bytes(176_128, 8192);
        assert!((need - 21.5).abs() < 0.05, "{need}");
        let ours = 3.0 * 2.0 + 3.0 + 16.0; // 3*es + 3 + 16 = 25
        assert_eq!(ours, 25.0);
        assert!(
            ours > need,
            "the kernel spends {ours} B/element against {need} available"
        );
        // and the row is over the cap by more than rounding:
        const { assert!(8192u64 > TILE_CAP_2B) };
    }

    #[test]
    fn a_dtype_whose_cap_is_too_small_is_impossible_not_a_bad_tile() {
        // 1 KB elements: cap = 176128 / 3091 = 56, which cannot hold two
        // 64-element chunks, so there is no sound staging at all.
        assert_eq!(tile_cap(1024), 56);
        assert_eq!(plan_staging(8192, 1024, MIN_TRANSFER_BYTES), Feasibility::Impossible);
        // and a zero-width element does not divide by zero
        assert_eq!(tile_cap(0), 0);
    }

    #[test]
    fn the_ledger_names_regimes_not_knobs() {
        assert_eq!(ROOFLINE_TRANSFORMS.len(), 2);
        assert!(ROOFLINE_TRANSFORMS[0].requires.contains("meets_granularity"));
        assert!(ROOFLINE_TRANSFORMS[1].trap.contains("same 470 MB"));
    }

    #[test]
    fn zero_duration_is_not_a_division_by_zero() {
        assert_eq!(achieved_gbps(1000, 0.0), 0.0);
        assert_eq!(achieved_gbps(1000, -1.0), 0.0);
    }
}
