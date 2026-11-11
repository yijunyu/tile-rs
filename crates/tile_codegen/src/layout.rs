//! Layout algebra: when a bulk (one contiguous) transfer is legal, and when a
//! strided transfer's gap is expressible in whole 32-byte blocks.
//!
//! This module exists because a *proxy* predicate here cost two scored cases on
//! `add_rms_norm_dynamic_quant` (`job_f6ff1c397d0a`, 18/20), and because the
//! same trap is available to every kernel that stages several rows in UB.
//!
//! The batching path stages `rows` rows in UB. A row holds `extent` valid
//! elements but the UB row stride is `pitch` elements, because a downstream
//! narrow output wants a wider stride. When the pitch IS the row, the rows are
//! contiguous in GM and one transfer replaces `rows` of them. The kernel tested
//!
//! ```text
//!     gap == 0   where   gap = pitch_bytes - align_up(extent_bytes, 32)
//! ```
//!
//! and read that as "the pitch is the row". It is not the same predicate.
//! `gap == 0` says the 32-byte *padding* reaches the pitch, which is also true
//! when the row falls short of the padding -- i.e. whenever
//! `extent_bytes % 32 != 0`. In that case the packed transfer lays row `i` at
//! `i * extent_bytes` while every reader indexes `i * pitch_bytes`, so rows
//! `1..rows-1` shift by the padding and the row reductions average in a
//! neighbour's tail. Silent, shape-dependent, and correct at `rows == 1`, which
//! is what made it look like a dtype bug for a whole session.
//!
//! Measured on 910b (rows=64, f16, `extent` 1023 and 4093): `scaleOut` wrong by
//! 7-29 %; bit-clean at rows <= 48, where a block holds one row and there is
//! nothing to displace. The shape predicate is `rows >= 2` and, for a 2-byte
//! source, `extent % 32` in 17..31. Of that job's 20 cases exactly two satisfy
//! it, and they were the only two failures.
//!
//! Both traps are now unrepresentable through [`RowLayout::plan`]:
//!
//! * [`PlanReason::PitchIsRow`] is the *only* reason that yields
//!   [`TransferPlan::Packed`], and it is the invariant itself, not a proxy.
//! * a gap that is not a whole number of 32-byte blocks is an [`LayoutError`]
//!   rather than a truncated integer division: `gap_bytes / 32` is `0` for any
//!   `0 < gap_bytes < 32`, so the old code silently turned a nonzero gap into
//!   none.
//!
//! [`RowLayout::gap_zero_proxy`] keeps the buggy predicate as an oracle so a
//! test can show where it disagrees, and [`proxy_disagrees`] is the shape set
//! to hand a regression matrix.

/// The transfer granularity both the DMA engine and `DataCopyPad` pad to.
pub const BLOCK_BYTES: usize = 32;

/// The UB row pitch rounds up to this many **elements**, not to a number of
/// blocks: the downstream int8 output wants a stride that is aligned for every
/// source width, so the pitch is `align_up(extent, 32)` regardless of
/// `elem_bytes`. Getting this wrong makes the algebra predict the wrong shapes
/// (an early version of this module rounded to a block instead, which mis-called
/// extent 2053).
pub const PITCH_ELEMS: usize = 32;

const fn align_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

/// A row-strided 2-D layout in UB, in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    pub rows: usize,
    /// Valid elements per row -- the `D` of the kernel.
    pub extent: usize,
    /// Elements between consecutive row starts.
    pub pitch: usize,
    pub elem_bytes: usize,
}

/// What the emitter should generate for a batch of rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferPlan {
    /// One contiguous transfer; only legal when the pitch is the row.
    Packed { bytes: usize },
    /// `blocks` transfers of `block_bytes`, skipping `gap_blocks` whole blocks
    /// of UB between them (`DataCopyPad` has already padded each block).
    Strided {
        blocks: u32,
        block_bytes: usize,
        gap_blocks: u32,
    },
}

/// Why a plan is legal. Carried explicitly so an emitter cannot pick `Packed`
/// from a predicate that merely correlates with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanReason {
    /// `pitch_bytes == extent_bytes`: rows are contiguous in GM and the reader's
    /// stride matches, so one transfer reproduces the layout exactly.
    PitchIsRow,
    /// The 32-byte padding is itself the pitch, so no extra gap is needed.
    PaddingIsPitch,
    /// The pitch is wider than the padded row by this many whole blocks.
    WholeBlockGap(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    NoRows,
    PitchNarrowerThanRow {
        pitch_bytes: usize,
        row_bytes: usize,
    },
    PitchBelowPaddedRow {
        pitch_bytes: usize,
        padded_bytes: usize,
    },
    /// A nonzero gap that cannot be expressed as whole 32-byte blocks. The old
    /// code divided instead of checking, which is how a gap became zero.
    GapNotBlockAligned {
        gap_bytes: usize,
    },
}

impl RowLayout {
    /// Elements of `elem_bytes` in one 32-byte block.
    pub const fn elems_per_block(elem_bytes: usize) -> usize {
        BLOCK_BYTES / elem_bytes
    }

    /// The natural layout: the pitch the kernel actually uses, which is the
    /// extent rounded up to [`PITCH_ELEMS`] elements rather than to a block.
    pub fn natural(rows: usize, extent: usize, elem_bytes: usize) -> RowLayout {
        RowLayout {
            rows,
            extent,
            pitch: align_up(extent, PITCH_ELEMS),
            elem_bytes,
        }
    }

    pub fn with_pitch(rows: usize, extent: usize, pitch: usize, elem_bytes: usize) -> RowLayout {
        RowLayout {
            rows,
            extent,
            pitch,
            elem_bytes,
        }
    }

    pub fn extent_bytes(&self) -> usize {
        self.extent * self.elem_bytes
    }

    pub fn pitch_bytes(&self) -> usize {
        self.pitch * self.elem_bytes
    }

    /// What `DataCopyPad` actually occupies per row: the row padded to a block.
    pub fn padded_extent_bytes(&self) -> usize {
        align_up(self.extent_bytes(), BLOCK_BYTES)
    }

    /// UB bytes between the padded end of one row and the start of the next.
    pub fn gap_bytes(&self) -> usize {
        self.pitch_bytes().saturating_sub(self.padded_extent_bytes())
    }

    /// The invariant a packed transfer needs: the reader's stride is the row.
    pub fn packed_reproduces_layout(&self) -> bool {
        self.pitch_bytes() == self.extent_bytes()
    }

    /// The predicate the kernel actually used. Kept as an oracle for tests and
    /// for the regression matrix -- **not** for codegen.
    pub fn gap_zero_proxy(&self) -> bool {
        self.gap_bytes() == 0
    }

    /// The single entry point the codegen should use.
    pub fn plan(&self) -> Result<(TransferPlan, PlanReason), LayoutError> {
        if self.rows == 0 {
            return Err(LayoutError::NoRows);
        }
        let row_bytes = self.extent_bytes();
        let pitch_bytes = self.pitch_bytes();
        if pitch_bytes < row_bytes {
            return Err(LayoutError::PitchNarrowerThanRow {
                pitch_bytes,
                row_bytes,
            });
        }
        if self.packed_reproduces_layout() {
            return Ok((
                TransferPlan::Packed {
                    bytes: self.rows * row_bytes,
                },
                PlanReason::PitchIsRow,
            ));
        }
        let padded = self.padded_extent_bytes();
        if pitch_bytes < padded {
            return Err(LayoutError::PitchBelowPaddedRow {
                pitch_bytes,
                padded_bytes: padded,
            });
        }
        let gap = pitch_bytes - padded;
        if !gap.is_multiple_of(BLOCK_BYTES) {
            return Err(LayoutError::GapNotBlockAligned { gap_bytes: gap });
        }
        let gap_blocks = (gap / BLOCK_BYTES) as u32;
        let reason = if gap_blocks == 0 {
            PlanReason::PaddingIsPitch
        } else {
            PlanReason::WholeBlockGap(gap_blocks)
        };
        Ok((
            TransferPlan::Strided {
                blocks: self.rows as u32,
                block_bytes: row_bytes,
                gap_blocks,
            },
            reason,
        ))
    }
}

/// Shapes where the proxy disagrees with the invariant -- the exact set the old
/// code was wrong on, and the set to put in a regression matrix.
pub fn proxy_disagrees(extent: usize, elem_bytes: usize) -> bool {
    let l = RowLayout::natural(2, extent, elem_bytes);
    l.gap_zero_proxy() && !l.packed_reproduces_layout()
}

/// One row of the safe-transformation ledger: a rewrite, the predicate it may
/// fire on, what it buys, and the trap if the predicate is faked.
///
/// The bandit arms are the values; these are the *guards* on them, which is why
/// they live here rather than in a kernel comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardedTransform {
    pub name: &'static str,
    /// The invariant that must hold. Checked, never inferred from a proxy.
    pub requires: &'static str,
    pub reward: &'static str,
    pub trap: &'static str,
}

pub const TRANSFORMS: &[GuardedTransform] = &[
    GuardedTransform {
        name: "batch_rows_one_transfer",
        requires: "RowLayout::packed_reproduces_layout (pitch_bytes == extent_bytes)",
        reward: "rows short transfers -> 1; 20 KB instead of 512 B at rows=40, D=128, f32",
        trap: "gap_zero_proxy: pitch == align_up(extent_bytes, 32) is also true when \
               extent_bytes % 32 != 0, which packs rows at extent_bytes into a \
               padded layout and displaces rows 1..R-1",
    },
    GuardedTransform {
        name: "strided_batch_with_gap",
        requires: "gap_bytes % 32 == 0 (PlanReason::PaddingIsPitch or WholeBlockGap)",
        reward: "the only correct batching transfer when the pitch is not the row",
        trap: "integer-dividing the gap: gap_bytes / 32 == 0 for 0 < gap_bytes < 32, \
               silently removing a nonzero gap",
    },
    GuardedTransform {
        name: "rows_per_block",
        requires: "rows > cores, or R stays 1 and the batching path is never taken",
        reward: "one block per row costs the per-row drain, see NOTES 149",
        trap: "a layout bug here hides at rows <= cores: one row per block means \
               nothing to displace, so it reads as a dtype or a value bug",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The two extents `job_f6ff1c397d0a` failed on, and the neighbours it did
    /// not: 1023 and 4093 were wrong, 2053 and 2048 were clean.
    #[test]
    fn proxy_disagrees_on_exactly_the_measured_extents() {
        // 2-byte source, the sweep's own list.
        for d in [1023usize, 4093, 4095, 4159, 127, 191, 255, 2047, 4031] {
            assert!(
                proxy_disagrees(d, 2),
                "extent {d} was measurably wrong and must be predicted"
            );
        }
        for d in [1024usize, 1088, 2048, 2053, 2112, 4096, 8192] {
            assert!(
                !proxy_disagrees(d, 2),
                "extent {d} was measurably clean and must not be predicted"
            );
        }
    }

    /// The closed form: for a 2-byte source the disagreement is `extent % 32`
    /// in 17..31, which is wider than the `extent % 64` in {61,62,63} first read
    /// off the failing cases -- that was an over-fit to the extents sampled.
    #[test]
    fn disagreement_is_extent_mod_32_in_17_to_31_for_two_byte_sources() {
        let mut seen: Vec<usize> = Vec::new();
        for d in 1..4200usize {
            let want = (17..32).contains(&(d % 32));
            assert_eq!(proxy_disagrees(d, 2), want, "extent {d}");
            if want {
                seen.push(d % 32);
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, (17..32).collect::<Vec<_>>());
    }

    /// The fix itself: 1023 now takes the strided path at zero extra gap,
    /// because the padding is the pitch. `Packed` is never selected for it.
    #[test]
    fn packed_is_never_chosen_when_the_pitch_is_not_the_row() {
        for d in 1..4200usize {
            let l = RowLayout::natural(64, d, 2);
            let (plan, reason) = l.plan().expect("natural layouts always plan");
            if matches!(plan, TransferPlan::Packed { .. }) {
                assert!(
                    l.packed_reproduces_layout(),
                    "Packed at extent {d} without the invariant"
                );
                assert_eq!(reason, PlanReason::PitchIsRow);
            } else {
                assert!(!l.packed_reproduces_layout() || reason != PlanReason::PitchIsRow);
            }
        }
    }

    #[test]
    fn the_measured_failure_now_plans_as_strided_with_no_extra_gap() {
        // This is the whole fix in one assertion.
        let l = RowLayout::natural(64, 1023, 2);
        assert!(l.gap_zero_proxy(), "the old predicate was true here");
        assert!(!l.packed_reproduces_layout(), "and the invariant was false");
        assert_eq!(
            l.plan(),
            Ok((
                TransferPlan::Strided {
                    blocks: 64,
                    block_bytes: 2046,
                    gap_blocks: 0
                },
                PlanReason::PaddingIsPitch
            ))
        );

        // 4093 is the harness's case 12: pitch 4096, row 8186 bytes, padded 8192.
        let l = RowLayout::natural(64, 4093, 2);
        assert!(l.gap_zero_proxy() && !l.packed_reproduces_layout());
        assert_eq!(
            l.plan(),
            Ok((
                TransferPlan::Strided {
                    blocks: 64,
                    block_bytes: 8186,
                    gap_blocks: 0
                },
                PlanReason::PaddingIsPitch
            ))
        );
    }

    /// A genuinely wider pitch skips whole blocks -- 2053 pads 27 elements, so
    /// the pitch 2080 is 32 bytes clear of the padded row.
    #[test]
    fn a_wider_pitch_becomes_a_whole_block_gap() {
        let l = RowLayout::natural(4, 2053, 2);
        assert_eq!(l.pitch_bytes(), 4160);
        assert_eq!(l.padded_extent_bytes(), 4128);
        assert_eq!(l.gap_bytes(), 32);
        assert_eq!(
            l.plan(),
            Ok((
                TransferPlan::Strided {
                    blocks: 4,
                    block_bytes: 4106,
                    gap_blocks: 1
                },
                PlanReason::WholeBlockGap(1)
            ))
        );
    }

    /// The trap the second guard closes: a sub-block gap divided by 32 is zero.
    #[test]
    fn a_sub_block_gap_is_an_error_not_a_truncation() {
        // pitch 1030 elements = 2060 bytes, row 1023 = 2046, padded 2048.
        let l = RowLayout::with_pitch(2, 1023, 1030, 2);
        assert_eq!(l.gap_bytes(), 12);
        assert_eq!(
            l.plan(),
            Err(LayoutError::GapNotBlockAligned { gap_bytes: 12 })
        );
        // and the naive division would have said "no gap":
        assert_eq!(l.gap_bytes() / BLOCK_BYTES, 0);
    }

    #[test]
    fn pitch_narrower_than_the_row_is_rejected() {
        let l = RowLayout::with_pitch(2, 1023, 1000, 2);
        assert!(matches!(
            l.plan(),
            Err(LayoutError::PitchNarrowerThanRow { .. })
        ));
        assert_eq!(RowLayout::natural(0, 1023, 2).plan(), Err(LayoutError::NoRows));
    }

    /// The pitch is aligned in ELEMENTS, for every width -- not to a block.
    #[test]
    fn the_pitch_aligns_to_elements_not_to_blocks() {
        assert_eq!(RowLayout::natural(2, 1023, 2).pitch, 1024);
        assert_eq!(RowLayout::natural(2, 2053, 2).pitch, 2080);
        assert_eq!(RowLayout::natural(2, 2049, 4).pitch, 2080);
        assert_eq!(RowLayout::elems_per_block(4), 8);
        assert_eq!(RowLayout::elems_per_block(2), 16);
        // The trap is NOT 2-byte-specific -- f32 disagrees too (extent 25 is
        // one) at different extents. The harness's 20 arnq cases are all
        // f16/bf16, so only the 2-byte band has measured evidence.
        assert!(proxy_disagrees(25, 4));
        // What must hold for every width is the invariant, not a closed form:
        for bytes in [2usize, 4] {
            for d in 1..4200usize {
                let l = RowLayout::natural(2, d, bytes);
                if matches!(l.plan(), Ok((TransferPlan::Packed { .. }, _))) {
                    assert!(l.packed_reproduces_layout(), "{bytes}-byte extent {d}");
                }
            }
        }
    }

    #[test]
    fn the_ledger_names_the_invariant_not_a_proxy() {
        assert_eq!(TRANSFORMS.len(), 3);
        for t in TRANSFORMS {
            assert!(!t.requires.is_empty() && !t.trap.is_empty() && !t.reward.is_empty());
        }
        assert!(TRANSFORMS[0].requires.contains("pitch_bytes == extent_bytes"));
        assert!(TRANSFORMS[0].trap.contains("gap_zero_proxy"));
    }
}
