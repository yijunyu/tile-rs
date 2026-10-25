//! Trace-driven clustering — KernelBand's `phi(k)`, and the cluster-shared bandit.
//!
//! The paper's Assumption 2 is that expected reward is Lipschitz-continuous in
//! execution behaviour: kernels with similar bottlenecks respond similarly to
//! the same strategy. That is what lets one expensive measurement inform
//! another kernel — and it is the reason the state is keyed by CLUSTER rather
//! than by kernel. Keyed per kernel, every measurement on the first kernel of a
//! family is thrown away when the second one arrives.
//!
//! `phi` is five numbers. The paper uses [normalised time, registers/thread,
//! shared memory per block, block dimension, occupancy]; ours are the counters
//! that actually exist on this substrate (`bench/profile_op.sh` ->
//! `PipeUtilization_*.csv`):
//!
//! ```text
//! phi(k) = [time, aiv_vec_ratio, aiv_mte2_ratio, ub_frac, dram_frac]
//! ```
//!
//! All five are expected in `[0, 1]`; `time` is the caller's call (normalise it
//! against the baseline, not against the other candidates — a per-batch maximum
//! makes the clusters drift as the frontier changes, which defeats the point of
//! reusing history).
//!
//! Determinism is a requirement, not a nicety. k-means is seeded by
//! farthest-point initialisation, never by random restarts: a clustering that
//! differed between two identical runs would make the bandit's history
//! unreproducible, and unreproducible history cannot be audited when a
//! submission goes wrong. The cost is a slightly worse partition than multi-restart
//! k-means would find, on five well-separated dimensions.

/// The paper's `phi(k)`: five normalised behaviour counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BehaviourVector {
    /// Execution time, normalised by the caller (0..1).
    pub time: f64,
    /// Vector-pipe utilisation.
    pub vec: f64,
    /// MTE2 (GM -> UB) utilisation.
    pub mte2: f64,
    /// Fraction of the ADDRESSABLE unified-buffer ceiling in use. This is the
    /// cliff this substrate actually punishes (see `bench/lint_ub_ceiling.py`),
    /// so it is a better third dimension here than occupancy would be.
    pub ub: f64,
    /// HBM share.
    pub dram: f64,
}

impl BehaviourVector {
    pub fn components(&self) -> [f64; 5] {
        [self.time, self.vec, self.mte2, self.ub, self.dram]
    }

    pub fn distance(&self, other: &Self) -> f64 {
        self.components()
            .iter()
            .zip(other.components().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    pub fn mean_of(points: &[BehaviourVector]) -> BehaviourVector {
        if points.is_empty() {
            return BehaviourVector::default();
        }
        let n = points.len() as f64;
        let mut acc = [0.0f64; 5];
        for p in points {
            for (a, c) in acc.iter_mut().zip(p.components().iter()) {
                *a += c;
            }
        }
        for a in acc.iter_mut() {
            *a /= n;
        }
        BehaviourVector { time: acc[0], vec: acc[1], mte2: acc[2], ub: acc[3], dram: acc[4] }
    }
}

/// Cluster centroids. `assign` is what the tracker uses; `k` is the number of
/// groups the paper settles on (`K = 3`).
#[derive(Clone, Debug, PartialEq)]
pub struct Centroids {
    pub points: Vec<BehaviourVector>,
}

impl Centroids {
    pub fn k(&self) -> usize {
        self.points.len()
    }

    /// Index of the nearest centroid. Ties break to the lower index, so the
    /// assignment is total and deterministic.
    pub fn assign(&self, p: &BehaviourVector) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, c) in self.points.iter().enumerate() {
            let d = c.distance(p);
            if d < best_d {
                best = i;
                best_d = d;
            }
        }
        best
    }
}

/// Deterministic k-means: farthest-point seeding, then Lloyd iterations.
///
/// `k >= points.len()` gives every point its own centroid rather than an error —
/// a frontier of two kernels has two behaviours, and refusing to cluster it
/// would leave the tracker in its fallback keying for no benefit.
pub fn kmeans(points: &[BehaviourVector], k: usize, iters: usize) -> Centroids {
    if points.is_empty() || k == 0 {
        return Centroids { points: Vec::new() };
    }
    if k >= points.len() {
        return Centroids { points: points.to_vec() };
    }
    // Farthest-point init: start from the first point (stable), then repeatedly
    // take the point furthest from everything chosen so far. No rng to seed.
    let mut centroids: Vec<BehaviourVector> = vec![points[0]];
    while centroids.len() < k {
        let mut best_idx = 0usize;
        let mut best_d = f64::NEG_INFINITY;
        for (i, p) in points.iter().enumerate() {
            let d = centroids.iter().map(|c| c.distance(p)).fold(f64::INFINITY, f64::min);
            if d > best_d {
                best_d = d;
                best_idx = i;
            }
        }
        centroids.push(points[best_idx]);
    }
    // Lloyd.
    for _ in 0..iters {
        let mut groups: Vec<Vec<BehaviourVector>> = vec![Vec::new(); centroids.len()];
        for p in points {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (i, c) in centroids.iter().enumerate() {
                let d = c.distance(p);
                if d < best_d {
                    best = i;
                    best_d = d;
                }
            }
            groups[best].push(*p);
        }
        let mut moved = false;
        for (i, g) in groups.iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            let m = BehaviourVector::mean_of(g);
            if m != centroids[i] {
                moved = true;
            }
            centroids[i] = m;
        }
        if !moved {
            break; // converged: identical centroids would loop forever
        }
    }
    Centroids { points: centroids }
}

/// Kernels seen so far, their behaviour, and the current partition.
///
/// Re-clustering is periodic (`tau` pulls, the paper's `tau = 10`) because
/// behaviour is a measurement and the frontier moves; between reclusterings the
/// assignment is frozen so a kernel cannot change cluster mid-comparison, which
/// would silently split one kernel's history across two states.
#[derive(Clone, Debug)]
pub struct ClusterTracker {
    tau: usize,
    k: usize,
    kernels: Vec<(String, BehaviourVector)>,
    centroids: Option<Centroids>,
    since_recluster: usize,
}

impl ClusterTracker {
    pub fn new(tau: usize, k: usize) -> Self {
        Self { tau: tau.max(1), k: k.max(1), kernels: Vec::new(), centroids: None, since_recluster: 0 }
    }

    /// Record (or update) a kernel's behaviour vector.
    pub fn observe(&mut self, kernel: &str, phi: BehaviourVector) {
        match self.kernels.iter_mut().find(|(n, _)| n == kernel) {
            Some(slot) => slot.1 = phi,
            None => self.kernels.push((kernel.to_string(), phi)),
        }
    }

    pub fn note_pull(&mut self) {
        self.since_recluster += 1;
    }

    pub fn kernels(&self) -> &[(String, BehaviourVector)] {
        &self.kernels
    }

    /// Re-cluster if `tau` pulls have passed and there is something to cluster.
    /// Returns whether the partition changed.
    pub fn recluster_if_due(&mut self) -> bool {
        if self.since_recluster < self.tau || self.kernels.is_empty() {
            return false;
        }
        self.since_recluster = 0;
        let points: Vec<BehaviourVector> = self.kernels.iter().map(|(_, p)| *p).collect();
        let next = kmeans(&points, self.k, 20);
        let changed = self.centroids.as_ref() != Some(&next);
        self.centroids = Some(next);
        changed
    }

    pub fn centroids(&self) -> Option<&Centroids> {
        self.centroids.as_ref()
    }

    /// Cluster of `kernel`, or `None` before the first clustering.
    pub fn cluster_of(&self, kernel: &str) -> Option<usize> {
        let c = self.centroids.as_ref()?;
        let (_, phi) = self.kernels.iter().find(|(n, _)| n == kernel)?;
        Some(c.assign(phi))
    }

    /// The key a `BanditState` should be stored under.
    ///
    /// Before the first clustering this is per-kernel (`{arch}__{kernel}`), so
    /// no history is ever lost waiting for a partition. Afterwards it is per
    /// cluster (`{arch}__cluster{i}`), which is the whole point: a strategy that
    /// paid off on one member of a behaviour family is already credited on its
    /// siblings. The `arch` prefix is preserved in both forms — statistics must
    /// never cross chips, whatever the clustering says.
    pub fn state_key(&self, arch: &str, kernel: &str) -> String {
        match self.cluster_of(kernel) {
            Some(i) => format!("{arch}__cluster{i}"),
            None => format!("{arch}__{kernel}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(time: f64, vec: f64, mte2: f64) -> BehaviourVector {
        BehaviourVector { time, vec, mte2, ub: 0.3, dram: 0.2 }
    }

    #[test]
    fn two_obvious_groups_are_separated() {
        let pts = vec![v(0.1, 0.9, 0.1), v(0.12, 0.88, 0.12), v(0.11, 0.91, 0.11),
                       v(0.9, 0.1, 0.85), v(0.88, 0.12, 0.87), v(0.91, 0.11, 0.86)];
        let c = kmeans(&pts, 2, 20);
        assert_eq!(c.k(), 2);
        let a = c.assign(&pts[0]);
        let b = c.assign(&pts[5]);
        assert_ne!(a, b, "the two groups landed in one cluster");
        // every member of a group is assigned with its group
        for p in &pts[0..3] {
            assert_eq!(c.assign(p), a);
        }
        for p in &pts[3..6] {
            assert_eq!(c.assign(p), b);
        }
    }

    #[test]
    fn clustering_is_deterministic_across_runs() {
        let pts = vec![v(0.1, 0.9, 0.1), v(0.4, 0.5, 0.4), v(0.9, 0.1, 0.85), v(0.45, 0.52, 0.42)];
        let a = kmeans(&pts, 3, 20);
        let b = kmeans(&pts, 3, 20);
        assert_eq!(a, b, "same input must give the same partition, bit for bit");
    }

    #[test]
    fn more_clusters_than_kernels_gives_each_its_own() {
        let pts = vec![v(0.1, 0.9, 0.1), v(0.9, 0.1, 0.85)];
        let c = kmeans(&pts, 3, 20);
        assert_eq!(c.k(), 2);
        assert_ne!(c.assign(&pts[0]), c.assign(&pts[1]));
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let empty: Vec<BehaviourVector> = Vec::new();
        assert_eq!(kmeans(&empty, 3, 5).k(), 0);
        assert_eq!(kmeans(&[v(0.1, 0.1, 0.1)], 0, 5).k(), 0);
    }

    #[test]
    fn the_tracker_reclusters_on_tau_not_every_pull() {
        let mut t = ClusterTracker::new(10, 2);
        t.observe("a", v(0.1, 0.9, 0.1));
        t.observe("b", v(0.9, 0.1, 0.85));
        for _ in 0..9 {
            t.note_pull();
            assert!(!t.recluster_if_due(), "must not recluster before tau");
        }
        assert!(t.centroids().is_none(), "nothing to compare against before tau");
        t.note_pull();
        // The FIRST partition is a change by definition (none -> some); what
        // must not change is the partition once the behaviour is unchanged.
        assert!(t.recluster_if_due(), "the first partition is a change from none");
        assert!(t.centroids().is_some());
        for _ in 0..10 {
            t.note_pull();
        }
        assert!(!t.recluster_if_due(), "unchanged behaviour must re-derive the same partition");
    }

    #[test]
    fn the_state_key_is_per_kernel_until_there_is_a_partition() {
        let mut t = ClusterTracker::new(10, 2);
        t.observe("arg_max", v(0.1, 0.9, 0.1));
        assert_eq!(t.state_key("dav-c220", "arg_max"), "dav-c220__arg_max");
        for _ in 0..10 {
            t.note_pull();
        }
        t.recluster_if_due();
        assert_eq!(t.state_key("dav-c220", "arg_max"), "dav-c220__cluster0");
        // arch is still in the key, whatever the cluster says
        assert!(t.state_key("dav-c310", "arg_max").starts_with("dav-c310__"));
    }

    #[test]
    fn behaviourally_similar_kernels_share_one_state_but_a_distant_one_does_not() {
        let mut t = ClusterTracker::new(1, 2);
        t.observe("a", v(0.10, 0.90, 0.10));
        t.observe("b", v(0.11, 0.89, 0.11));   // sibling of a
        t.observe("z", v(0.90, 0.10, 0.85));   // a different bottleneck entirely
        t.note_pull();
        t.recluster_if_due();
        assert_eq!(t.state_key("dav-c220", "a"), t.state_key("dav-c220", "b"));
        assert_ne!(t.state_key("dav-c220", "a"), t.state_key("dav-c220", "z"));
    }

    #[test]
    fn an_unseen_kernel_maps_to_the_nearest_existing_cluster() {
        let mut t = ClusterTracker::new(1, 2);
        t.observe("a", v(0.10, 0.90, 0.10));
        t.observe("z", v(0.90, 0.10, 0.85));
        t.note_pull();
        t.recluster_if_due();
        // a newcomer with no history: nearest centroid decides
        t.observe("new", v(0.12, 0.88, 0.12));
        assert_eq!(t.state_key("dav-c220", "new"), t.state_key("dav-c220", "a"));
    }
}
