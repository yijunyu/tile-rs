//! Hardware-aware pruning — KernelBand's masked UCB, at the codegen seam.
//!
//! The paper's Assumption 1 bounds the gain from applying a strategy by what
//! the hardware still has left to give: there is no point retiling a kernel
//! whose unified buffer is already saturated. It makes that a mask
//!
//! ```text
//! M_{i,s} = 1[ h(centroid)[Target(s)] < theta_sat ],   theta_sat = 0.75
//! ```
//!
//! over the pairing of kernel cluster `i` with strategy `s`, and then runs UCB
//! only over the unmasked actions.
//!
//! Two deviations, both because the substrate is Ascend and not an H100:
//!
//! 1. **The signature is our counters, not NCU's.** The paper reads DRAM / L2 /
//!    SM peak-throughput percentages from Nsight Compute. The nearest thing we
//!    have is the `PipeUtilization_*.csv` that `bench/profile_op.sh` already
//!    writes: `aiv_vec_ratio`, `aiv_mte2_ratio`, an HBM share, and — because
//!    Ascend's tiling catastrophe is UB overrun and not "low occupancy" — the
//!    fraction of the ADDRESSABLE UB ceiling in use. That last one is the one
//!    the benchmark data actually punishes: `bench/lint_ub_ceiling.py` records
//!    the measured top as 164864 bytes with 155648 safe, and over-allocating is
//!    an aicore fault rather than a slow kernel.
//! 2. **The mapping is a hypothesis, marked as one.** Which resource a knob
//!    targets is declared per knob in [`crate::schedule::KNOBS`] and every entry
//!    is to be CALIBRATED against measurements, not assumed from the paper's
//!    NVIDIA numbers (Tiling 14.4 % success there, Fusion 75 %). A wrong Target
//!    does not corrupt anything -- it only prunes the wrong arm -- so this is
//!    safe to be wrong about, which is why it is a declared table and not
//!    buried in the selection code.

/// A scarce resource a strategy can be trying to recover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resource {
    /// HBM / MTE3 bandwidth.
    Dram,
    /// The MTE2 (GM -> UB) pipe.
    Mte2,
    /// The vector pipe's issue slots.
    Vector,
    /// Unified-buffer capacity — Ascend's own cliff.
    Ub,
    /// Latency hiding rather than a throughput ceiling; the stall share.
    Latency,
}

/// Measured headroom of one kernel, as fractions in `[0, 1]`.
///
/// All-zero is the "unmeasured" signature and it prunes NOTHING, so a caller
/// without counters gets exactly the unmasked behaviour instead of a silently
/// different search.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HwSignature {
    pub dram: f64,
    pub mte2: f64,
    pub vector: f64,
    pub ub: f64,
    pub latency: f64,
}

/// Saturation threshold, the paper's `theta_sat`.
pub const SATURATION: f64 = 0.75;

impl HwSignature {
    pub fn get(&self, r: Resource) -> f64 {
        match r {
            Resource::Dram => self.dram,
            Resource::Mte2 => self.mte2,
            Resource::Vector => self.vector,
            Resource::Ub => self.ub,
            Resource::Latency => self.latency,
        }
    }

    /// True when this resource has no headroom left for its strategies.
    pub fn saturated(&self, r: Resource, theta: f64) -> bool {
        self.get(r) >= theta
    }

    /// True when nothing at all is saturated — the no-pruning case.
    pub fn has_headroom(&self) -> bool {
        [Resource::Dram, Resource::Mte2, Resource::Vector, Resource::Ub, Resource::Latency]
            .iter()
            .any(|r| !self.saturated(*r, SATURATION))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_signature_prunes_nothing() {
        let s = HwSignature::default();
        for r in [Resource::Dram, Resource::Mte2, Resource::Vector, Resource::Ub, Resource::Latency] {
            assert!(!s.saturated(r, SATURATION), "{r:?} must not be called saturated at 0");
        }
        assert!(s.has_headroom());
    }

    #[test]
    fn saturation_is_inclusive_at_the_threshold() {
        let s = HwSignature { dram: 0.75, ..Default::default() };
        assert!(s.saturated(Resource::Dram, SATURATION), "exactly at theta counts as saturated");
        let s = HwSignature { dram: 0.7499, ..Default::default() };
        assert!(!s.saturated(Resource::Dram, SATURATION));
    }

    #[test]
    fn a_kernel_with_every_resource_pinned_has_no_headroom() {
        let s = HwSignature { dram: 0.99, mte2: 0.99, vector: 0.99, ub: 0.99, latency: 0.99 };
        assert!(!s.has_headroom());
    }
}
