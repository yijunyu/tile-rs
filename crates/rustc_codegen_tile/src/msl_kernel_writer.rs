//! `KernelWriter` — emit hazardous MSL constructs and charge their obligations
//! in the same call.
//!
//! See `docs/MSL_HARDENING_PLAN.md` (D4, D5, D6). The problem this solves: our
//! kernels depend on launch-geometry facts the host cannot see and the compiler
//! cannot check, and today those facts live only in comments — or nowhere. Two
//! live examples, both currently correct and both silent if violated:
//!
//!   * `laguna_head_rms_norm_rope_neox` reduces with a power-of-two tree
//!     (`for (step = nth>>1; step; step >>= 1)`). Dispatch a non-power-of-two
//!     threadgroup and it computes a PARTIAL SUM — no crash, no UB, no failing
//!     gate, just a wrong RMS that reads as a quantization artifact. It is
//!     correct today only because the host happens to pass `HEAD_DIM = 128`.
//!   * `laguna_attention_decode_gqa_f16` returns without writing when
//!     `head_dim != 128`, so its output buffer is consumed stale.
//!
//! The mechanism (D6): a hazardous construct is reachable ONLY through a
//! `KernelWriter` method, and that method both writes the MSL and records the
//! obligation the MSL now carries. One call site produces the code and the claim
//! about the code, so they cannot drift the way a comment does. This is the same
//! discipline as wrapping an intrinsic in a checked function rather than
//! `format!`-ing it: recover, at the emission boundary, the invariants that
//! emitting text throws away.
//!
//! Deliberately NOT a full wrapper over MSL (D7). Trivia — braces, comments,
//! scalar arithmetic — keeps going through plain `writeln!` to the inner buffer.
//! Gating everything would be a 46k-line rewrite whose bulk carries no
//! invariant. Only constructs that charge an obligation are gated.

use std::fmt::Write as _;

/// A predicate the host launch must satisfy for the emitted kernel to be
/// correct. Data, never code (D5): a table can be printed, diffed against the
/// previous emit, and unit-tested with no GPU and no model loaded — which is
/// exactly what stops it rotting the way the comments did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Obligation {
    /// Threads-per-threadgroup must be a power of two. Charged by tree
    /// reductions, whose `step >>= 1` walk only covers the array when the
    /// element count is a power of two.
    ThreadsPowerOfTwo { because: &'static str },
    /// Threads-per-threadgroup must not exceed `max` — typically the capacity of
    /// a threadgroup array the reduction indexes with `tid`.
    ThreadsAtMost { max: u32, because: &'static str },
    /// Threadgroup memory the kernel statically allocates, in bytes.
    ThreadgroupBytes { bytes: u32 },
    /// A buffer is addressed through an index READ FROM ANOTHER BUFFER, so it
    /// must hold at least `product(dims)` elements and the index must be
    /// bounds-checked before it becomes an address.
    ///
    /// This is the class that produces silent garbage rather than a fault: on
    /// Metal an out-of-range address is not trapped, so a bad routing id or a
    /// wrong stride reads neighbouring memory and the model simply returns
    /// wrong numbers. Upstream MLX's `gather_qmm` with `lhsIndices` returns NaN
    /// on NVFP4 for what is very likely this reason, and it is still unfixed —
    /// the discoverer routed around it with a custom kernel rather than
    /// repairing it.
    ///
    /// `dims` names the runtime dimensions whose product bounds the buffer, so
    /// the host can evaluate the bound against the model it actually loaded.
    BufferMinElems {
        buffer: &'static str,
        dims: &'static [&'static str],
        because: &'static str,
    },
    /// A lookup table staged into threadgroup memory by giving every lane an
    /// equal, contiguous run of entries, where the run length is a literal.
    /// The run length times the lanes a threadgroup has must equal the table,
    /// so the kernel only stages it correctly at exactly `n` simdgroups. Fewer
    /// leaves part of the table unwritten and the kernel reads whatever was
    /// there; more runs past the end of it. Neither faults on Metal.
    ///
    /// The simdgroup count is a function constant, so the host chooses it at
    /// pipeline creation and the kernel cannot check it.
    SimdgroupsExactly {
        n: u32,
        table: &'static str,
        because: &'static str,
    },
    /// A runtime dimension that indexes a FIXED-SIZE array must not exceed that
    /// array's capacity.
    ///
    /// The array is sized at compile time; the loop bound comes in at dispatch.
    /// Nothing in the kernel can check it — a per-thread stack array has no
    /// length — so the predicate belongs to the host, and on Metal exceeding it
    /// corrupts neighbouring memory rather than faulting. That makes this the
    /// quiet-wrong-answer class rather than the crash class.
    DimAtMost {
        dim: &'static str,
        max: u32,
        array: &'static str,
        because: &'static str,
    },
    /// A floating-point expression chain whose ROUNDING BOUNDARIES are pinned:
    /// FP contraction is disabled and the chain's operations appear in the
    /// reference order, so the emitted arithmetic is bit-identical to the
    /// sequence the contract names.
    ///
    /// This is the class that survives every gate except the one that matters.
    /// A re-associated or FMA-fused chain is *more* accurate, not less, so it
    /// passes tolerance-based tests and roundtrip checks — and then flips a
    /// near-tie argmax and fails an exact-token gate somewhere unrelated, weeks
    /// later. Bit-exactness is not accuracy: a fused multiply-add keeps an
    /// intermediate at full width where the reference rounded, and the whole
    /// downstream comparison changes.
    ///
    /// `ops` is the reference operation sequence, checked as an ordered
    /// subsequence of the emitted text, so an edit that reorders or drops a step
    /// fails at CODEGEN time rather than in a token mismatch far from the cause.
    RoundingChainPinned {
        chain: &'static str,
        ops: &'static [&'static str],
        because: &'static str,
    },
    /// Threadgroups partition a writable range between them, and the mapping
    /// from threadgroup id to range is injective: every element is written by
    /// exactly one threadgroup. This is what licenses a plain store where a
    /// straddling tiling would need atomics or a reduction pass — the
    /// `partition_disjoint` discipline, mechanized in the Coq supplement.
    PartitionDisjoint {
        key: &'static str,
        over: &'static str,
        because: &'static str,
    },
    /// A MODEL dimension, divided by `divisor`, must not exceed the threadgroup
    /// width: `dim / divisor <= threads_per_threadgroup`. The kernel gives work to
    /// one thread per unit, so a narrower threadgroup simply leaves the tail
    /// undone — silently, because every thread that does exist behaves correctly.
    ///
    /// Distinct from `ThreadsAtMost`, which bounds geometry by a CONSTANT. This
    /// relates geometry to a runtime dim, so neither `check_launch` (thread count,
    /// no dims) nor `check_buffer_bounds` (dims, no thread count) can evaluate it
    /// alone -- see `check_dims_against_geometry`.
    DimAtMostThreads {
        dim: &'static str,
        divisor: u32,
        because: &'static str,
    },
    /// One MODEL dimension must not exceed another: `dim <= limit`. Distinct from
    /// `DimAtMost`, whose bound is a compile-time constant.
    DimAtMostDim {
        dim: &'static str,
        limit: &'static str,
        because: &'static str,
    },
    /// A MODEL dimension must be even. Cheap to state, and the kernel that needs it
    /// typically responds to an odd value by doing nothing at all.
    DimIsEven {
        dim: &'static str,
        because: &'static str,
    },
    /// A model dimension must be divisible by `k`. The block-quantized
    /// kernels derive their block count as `dim / k` with truncating
    /// division: a non-multiple leaves the remainder values unread, which
    /// is a silently-wrong dot product, not a fault.
    DimDivisibleBy {
        dim: &'static str,
        k: u32,
        because: &'static str,
    },
}

impl Obligation {
    /// One-line rendering for the generated contract table and for test output.
    pub fn describe(&self) -> String {
        match self {
            Obligation::ThreadsPowerOfTwo { because } => {
                format!("threads_per_threadgroup must be a power of two ({because})")
            }
            Obligation::ThreadsAtMost { max, because } => {
                format!("threads_per_threadgroup must be <= {max} ({because})")
            }
            Obligation::ThreadgroupBytes { bytes } => {
                format!("threadgroup memory: {bytes} bytes")
            }
            Obligation::SimdgroupsExactly { n, table, because } => format!(
                "simdgroups per threadgroup must be exactly {n}, because {table} is \
                 staged in equal fixed-length runs ({because})"
            ),
            Obligation::BufferMinElems {
                buffer,
                dims,
                because,
            } => format!(
                "buffer `{buffer}` must hold at least {} elements ({because})",
                dims.join(" * ")
            ),
            Obligation::DimAtMost {
                dim,
                max,
                array,
                because,
            } => format!("`{dim}` must be <= {max} (capacity of `{array}`; {because})"),
            Obligation::RoundingChainPinned {
                chain,
                ops,
                because,
            } => format!(
                "fp chain `{chain}` is rounding-pinned: contract(off) and the order {} ({because})",
                ops.join(" -> ")
            ),
            Obligation::PartitionDisjoint { key, over, because } => format!(
                "threadgroups partition {over} disjointly by {key}, so every element \
                 has exactly one writer ({because})"
            ),
            Obligation::DimAtMostThreads {
                dim,
                divisor,
                because,
            } => format!(
                "{dim}/{divisor} must not exceed threads_per_threadgroup ({because})"
            ),
            Obligation::DimAtMostDim {
                dim,
                limit,
                because,
            } => format!("{dim} must not exceed {limit} ({because})"),
            Obligation::DimIsEven { dim, because } => {
                format!("{dim} must be even ({because})")
            }
            Obligation::DimDivisibleBy { dim, k, because } => {
                format!("{dim} must be divisible by {k} ({because})")
            }
        }
    }
}

thread_local! {
    /// Obligations charged during the current [`record`] scope.
    ///
    /// A side channel exists because the 13 Laguna emitters are still
    /// `fn(out: &mut String)` — D6's signature migration is incremental, and a
    /// big-bang change to all of them would have to land before ANY contract
    /// could be generated. This lets the table come up now and the signatures
    /// migrate one kernel at a time underneath it. It is not the end state:
    /// once an emitter takes `&mut KernelWriter` it returns its ledger directly
    /// via `finish()` and stops relying on this.
    static RECORDING: std::cell::RefCell<Option<Vec<Obligation>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `emit` and capture both the MSL it wrote and the obligations it charged.
///
/// Nesting is not supported and is a programming error rather than a silent
/// merge — a nested scope would attribute an inner kernel's obligations to the
/// outer one, which is precisely the misattribution this whole mechanism exists
/// to prevent.
pub fn record<F: FnOnce(&mut String)>(emit: F) -> (String, Vec<Obligation>) {
    RECORDING.with(|r| {
        assert!(
            r.borrow().is_none(),
            "record() is not reentrant: a nested scope would attribute the inner \
             kernel's obligations to the outer one"
        );
        *r.borrow_mut() = Some(Vec::new());
    });
    let mut out = String::new();
    emit(&mut out);
    let obligations = RECORDING.with(|r| r.borrow_mut().take().unwrap_or_default());
    (out, obligations)
}

/// Wraps the emission buffer and the obligation ledger. Hazardous constructs go
/// through the methods; everything else goes through [`KernelWriter::raw`].
pub struct KernelWriter<'a> {
    out: &'a mut String,
    obligations: Vec<Obligation>,
}

impl<'a> KernelWriter<'a> {
    pub fn new(out: &'a mut String) -> Self {
        Self {
            out,
            obligations: Vec::new(),
        }
    }

    /// Ungated passthrough for trivia (D7). Named `raw` rather than `writeln` so
    /// that reaching for it in a review reads as a deliberate choice not to
    /// charge an obligation.
    pub fn raw(&mut self, line: &str) {
        self.out.push_str(line);
        self.out.push('\n');
    }

    /// Ungated passthrough for a blank line.
    pub fn blank(&mut self) {
        self.out.push('\n');
    }

    /// Obligations charged so far, in charge order.
    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }

    /// Release the borrow on the output buffer and hand back the ledger. This is
    /// the shape a migrated emitter wants: write the kernel, then return its
    /// contract to the caller assembling the generated contract table.
    pub fn finish(self) -> Vec<Obligation> {
        self.obligations
    }

    fn charge(&mut self, o: Obligation) {
        if !self.obligations.contains(&o) {
            self.obligations.push(o.clone());
        }
        // Mirror into the enclosing `record` scope, if any, so a contract can be
        // collected from emitters that have not yet migrated their signature.
        RECORDING.with(|r| {
            if let Some(v) = r.borrow_mut().as_mut() {
                if !v.contains(&o) {
                    v.push(o);
                }
            }
        });
    }

    /// Declare a `threadgroup` scratch array, charging its byte cost.
    ///
    /// `elems` is the element count; `elem_bytes` the size of one element. Both
    /// are needed as numbers (not just the emitted text) so the charge is real.
    pub fn threadgroup_array(
        &mut self,
        ty: &str,
        name: &str,
        elems: u32,
        elem_bytes: u32,
        comment: Option<&str>,
    ) {
        let mut line = format!("    threadgroup {ty} {name}[{elems}];");
        if let Some(c) = comment {
            let _ = write!(line, "  // {c}");
        }
        self.raw(&line);
        self.charge(Obligation::ThreadgroupBytes {
            bytes: elems * elem_bytes,
        });
    }

    /// Record that this kernel addresses `buffer` through an index read from
    /// another buffer, and VERIFY that the emitted source already guards that
    /// index before using it as an address.
    ///
    /// This is an adoption bridge for hand-written emitters (D6's signature
    /// migration is incremental): it charges the obligation without changing a
    /// byte of output, so existing correct kernels gain contract coverage
    /// immediately. But it is not a bare assertion — `guard_snippet` must
    /// actually appear in what has been emitted so far, so a future edit that
    /// deletes the bounds check fails at CODEGEN time rather than shipping a
    /// kernel whose contract claims a guard it no longer has.
    ///
    /// Panics if the guard is absent. That is deliberate: a contract that can
    /// be wrong is worse than no contract, because the host then trusts it.
    pub fn charge_bounds_checked_index(
        &mut self,
        buffer: &'static str,
        dims: &'static [&'static str],
        guard_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(guard_snippet),
            "kernel addresses `{buffer}` through an index but the expected \
             bounds guard is not in the emitted source.\n  expected to find: \
             {guard_snippet:?}\nAn unguarded index-derived address on Metal is \
             silent corruption, not a fault. Either restore the guard or stop \
             charging this obligation -- do not let the contract claim a check \
             the kernel does not perform."
        );
        self.charge(Obligation::BufferMinElems {
            buffer,
            dims,
            because,
        });
    }

    /// Charge threadgroup memory an already-written declaration allocates, and
    /// VERIFY that declaration is in the emitted source.
    ///
    /// Adoption bridge for emitters that still write their own `threadgroup`
    /// line; [`Self::threadgroup_array`] is the migrated form that emits and
    /// charges together. Emitted bytes are unchanged.
    pub fn charge_threadgroup_bytes(&mut self, bytes: u32, decl_snippet: &str) {
        assert!(
            self.out.contains(decl_snippet),
            "charging {bytes} threadgroup bytes, but the declaration is not in \
             the emitted source.\n  expected to find: {decl_snippet:?}\n\
             The contract would then claim an allocation the kernel does not \
             make, and the host would size its launch against a fiction."
        );
        self.charge(Obligation::ThreadgroupBytes { bytes });
    }

    /// Record that a runtime dimension indexes a fixed-size array, and VERIFY
    /// the declaration is in the emitted source.
    ///
    /// Adoption bridge like [`Self::charge_bounds_checked_index`]: emitted bytes
    /// are unchanged. `decl_snippet` must appear in what has been emitted, so
    /// resizing or deleting the array without revisiting the bound fails at
    /// CODEGEN rather than shipping a contract that names a capacity the kernel
    /// no longer has.
    ///
    /// Unlike the bounds-checked-index case there is no in-kernel guard to
    /// verify, and that is the point: a per-thread array carries no length, so
    /// the check CANNOT live in the kernel. It belongs to the host, which is
    /// exactly why it needs to be written down.
    pub fn charge_dim_bound(
        &mut self,
        dim: &'static str,
        max: u32,
        array: &'static str,
        decl_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(decl_snippet),
            "charging `{dim} <= {max}` for array `{array}`, but its declaration \
             is not in the emitted source.\n  expected to find: {decl_snippet:?}\n\
             The contract would then name a capacity the kernel does not have. \
             Either restore the declaration or stop charging this obligation."
        );
        self.charge(Obligation::DimAtMost {
            dim,
            max,
            array,
            because,
        });
    }

    /// The pragma that disables FP contraction. Named once so the emitter and
    /// the verifier below cannot drift apart.
    pub const FP_CONTRACT_OFF: &'static str = "#pragma clang fp contract(off)";

    /// Emit the contraction-off pragma. Call before a chain whose rounding
    /// boundaries must match a reference sequence bit for bit.
    pub fn pin_rounding(&mut self) {
        self.raw(&format!("    {}", Self::FP_CONTRACT_OFF));
    }

    /// Record that this kernel computes `chain`, and VERIFY that the emitted
    /// source both disables FP contraction and spells `ops` in order.
    ///
    /// Adoption bridge, same shape as [`Self::charge_bounds_checked_index`]: it
    /// changes no output, so an existing hand-written emitter gains coverage the
    /// moment it charges. But it is not a bare assertion — both the pragma and
    /// the operation ORDER are checked against what was actually emitted.
    ///
    /// The order check is an ordered-subsequence match, not a substring match:
    /// `ops` must occur left to right with anything permitted between them. That
    /// tolerates the naming and whitespace an emitter chooses while still failing
    /// if a step is dropped or two steps swap — which is exactly the edit that
    /// changes the last bit while looking harmless in review.
    ///
    /// Panics if either check fails. A contract that can be wrong is worse than
    /// no contract, because the host then trusts it.
    pub fn charge_rounding_chain(
        &mut self,
        chain: &'static str,
        ops: &'static [&'static str],
        because: &'static str,
    ) {
        assert!(
            self.out.contains(Self::FP_CONTRACT_OFF),
            "fp chain `{chain}` is charged as rounding-pinned but the emitted \
             source does not disable FP contraction.\n  expected to find: {:?}\n\
             Without it the compiler may fuse a multiply-add, keeping an \
             intermediate at full width where the reference rounded. That is \
             MORE accurate and still wrong: it flips near-tie comparisons and \
             fails exact-output gates far from here. Either emit the pragma \
             (see `pin_rounding`) or stop charging this obligation.",
            Self::FP_CONTRACT_OFF
        );

        let mut cursor = 0usize;
        for (i, op) in ops.iter().enumerate() {
            match self.out[cursor..].find(op) {
                Some(at) => cursor += at + op.len(),
                None => panic!(
                    "fp chain `{chain}`: step {i} ({op:?}) does not appear after the \
                     preceding steps in the emitted source.\n  reference order: {}\n\
                     Either the step was dropped, or two steps were reordered. Both \
                     change the rounding boundaries, so the emitted chain is no \
                     longer the one this contract names.",
                    ops.join(" -> ")
                ),
            }
        }

        self.charge(Obligation::RoundingChainPinned {
            chain,
            ops,
            because,
        });
    }

    /// Record that threadgroups partition a writable range disjointly, and VERIFY
    /// both halves of the claim against the emitted source: that the index is
    /// derived from the threadgroup id, and that the store is a PLAIN store.
    ///
    /// Adoption bridge — emitted bytes are unchanged, so an existing hand-written
    /// emitter gains the contract without moving a byte (D8).
    ///
    /// Both checks matter, and the second is the one that decays. Disjointness is
    /// what licenses storing without atomics; if a later edit makes two
    /// threadgroups share an output element, the plain store silently keeps
    /// whichever landed last instead of racing visibly. Requiring the store text
    /// here means such an edit has to come past this contract.
    ///
    /// This is the vocabulary the (B) composition pilot needs: expert-aligned
    /// partitioning is the one mechanism the mlx.fast trunk evidence says
    /// transfers, and it is quantization-independent.
    pub fn charge_partition_disjoint(
        &mut self,
        key: &'static str,
        over: &'static str,
        index_snippet: &str,
        store_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(index_snippet),
            "partition `{key}` is charged as disjoint, but the emitted source does \
             not derive its index the way the contract claims.\n  expected to find: \
             {index_snippet:?}\nDisjointness follows from the index being a function \
             of the threadgroup id; if that derivation changed, the claim does not hold."
        );
        assert!(
            self.out.contains(store_snippet),
            "partition `{key}` is charged as disjoint, but the emitted store does not \
             match.\n  expected to find: {store_snippet:?}\nA disjoint partition is \
             what permits a PLAIN store here. If the store moved or became a \
             read-modify-write, two threadgroups sharing an element would silently \
             keep the last writer rather than fail."
        );
        self.charge(Obligation::PartitionDisjoint { key, over, because });
    }

    /// Record that `dim / divisor` units of work are handed one-per-thread, and
    /// VERIFY the emitted source contains the guard that drops the surplus threads.
    ///
    /// Adoption bridge: emitted bytes are unchanged.
    ///
    /// The guard is what makes this a SILENT failure rather than a loud one. Every
    /// thread that exists does its own unit correctly, so a threadgroup narrower
    /// than `dim / divisor` produces a partially-processed result with no fault and
    /// no diagnostic. Requiring the guard text here means removing or rewriting it
    /// has to come past this contract.
    pub fn charge_dim_at_most_threads(
        &mut self,
        dim: &'static str,
        divisor: u32,
        guard_snippet: &str,
        because: &'static str,
    ) {
        assert!(divisor > 0, "divisor must be non-zero");
        assert!(
            self.out.contains(guard_snippet),
            "charging `{dim}/{divisor} <= threads` but the emitted source does not \
             contain the guard that drops surplus threads.\n  expected to find: \
             {guard_snippet:?}\nWithout that guard the shape of the obligation is \
             different, so the contract would describe a kernel this is not."
        );
        self.charge(Obligation::DimAtMostThreads {
            dim,
            divisor,
            because,
        });
    }

    /// Record `dim <= limit` between two MODEL dimensions, verifying the guard.
    ///
    /// Adoption bridge; emitted bytes unchanged. The guard is required in the
    /// source because it is what turns a violation into a SILENT no-op: the kernel
    /// returns having written nothing, so the caller consumes a stale buffer rather
    /// than seeing a fault.
    pub fn charge_dim_at_most_dim(
        &mut self,
        dim: &'static str,
        limit: &'static str,
        guard_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(guard_snippet),
            "charging `{dim} <= {limit}` but the emitted source does not contain \
             the guard.\n  expected to find: {guard_snippet:?}\nThe contract would \
             then describe a check the kernel does not perform."
        );
        self.charge(Obligation::DimAtMostDim {
            dim,
            limit,
            because,
        });
    }

    /// Record that `dim` must be even, verifying the guard. Same silent-no-op
    /// reasoning as [`Self::charge_dim_at_most_dim`].
    pub fn charge_dim_is_even(
        &mut self,
        dim: &'static str,
        guard_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(guard_snippet),
            "charging `{dim} is even` but the emitted source does not contain the \
             parity guard.\n  expected to find: {guard_snippet:?}"
        );
        self.charge(Obligation::DimIsEven { dim, because });
    }

    /// Record that `dim` must be divisible by `k`, verifying that the emitted
    /// source contains the derivation the obligation is about (typically the
    /// truncating block count, e.g. `ne00 / QK_MXFP4`). There is no in-kernel
    /// guard to verify and none is possible: the kernel cannot detect the
    /// dropped remainder — every thread that runs behaves correctly over the
    /// blocks that exist, and the tail values are simply never read. The check
    /// belongs to the host, which is why it must be written down.
    /// Charge [`Obligation::SimdgroupsExactly`] for a table staged in equal
    /// fixed-length runs, one run per lane.
    ///
    /// The simdgroup count is DERIVED here rather than passed in: a table of
    /// `table_entries` covered by runs of `run_len` over 32-lane simdgroups
    /// needs `table_entries / (run_len * 32)` of them, and anything that does
    /// not divide exactly is a staging that covers no whole number of
    /// simdgroups and so cannot be stated as this obligation at all. Stating
    /// the number instead of deriving it is how a contract comes to disagree
    /// with the kernel it describes.
    ///
    /// `evidence_snippet` must be the run-length declaration itself, so the
    /// charge fails if the staging it is about is edited away.
    pub fn charge_simdgroups_exactly(
        &mut self,
        table: &'static str,
        table_entries: u32,
        run_len: u32,
        evidence_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(evidence_snippet),
            "charging a simdgroup count for {table} but the emitted source does \
             not contain the fixed-run staging this obligation is about.\n  \
             expected to find: {evidence_snippet:?}\nEither restore it or stop \
             charging — the contract must not claim a dependency the kernel no \
             longer has."
        );
        let lanes = run_len * 32;
        assert!(
            run_len > 0 && lanes > 0 && table_entries % lanes == 0,
            "{table}: a run of {run_len} per lane covers {lanes} entries per \
             simdgroup, which does not divide the {table_entries} the table \
             holds — this staging spans no whole number of simdgroups, so there \
             is no count to charge"
        );
        let n = table_entries / lanes;
        self.charge(Obligation::SimdgroupsExactly { n, table, because });
    }

    pub fn charge_dim_divisible_by(
        &mut self,
        dim: &'static str,
        k: u32,
        evidence_snippet: &str,
        because: &'static str,
    ) {
        assert!(
            self.out.contains(evidence_snippet),
            "charging `{dim} divisible by {k}` but the emitted source does not \
             contain the truncating derivation this obligation is about.\n  \
             expected to find: {evidence_snippet:?}\nEither restore it or stop \
             charging — the contract must not claim a dependency the kernel no \
             longer has."
        );
        self.charge(Obligation::DimDivisibleBy { dim, k, because });
    }

    /// Emit the power-of-two threadgroup tree reduction over `scratch[tid]`,
    /// charging the two obligations it creates.
    ///
    /// The emitted form is exactly the shape already shipping — the barrier sits
    /// INSIDE the loop and the loop bound is threadgroup-uniform, so no barrier
    /// is reached divergently. `capacity` is the element count of `scratch`,
    /// which bounds the legal threadgroup size because the pre-loop seed writes
    /// `scratch[tid]` for every thread.
    ///
    /// `combine` is the reduction body written in terms of `scratch[tid]` and
    /// `scratch[tid + step]` (e.g. `"scratch[tid] += scratch[tid + step]"`), so
    /// sum/max/min share this path and its obligations.
    pub fn tree_reduce(&mut self, threads_expr: &str, scratch: &str, capacity: u32, combine: &str) {
        self.raw(&format!(
            "    for (uint step = {threads_expr} >> 1u; step != 0u; step >>= 1u) {{"
        ));
        self.raw(&format!("        if (tid < step) {combine};"));
        self.raw("        threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.raw("    }");
        self.charge(Obligation::ThreadsPowerOfTwo {
            because: "power-of-two tree reduction: a non-power-of-two thread count \
                      leaves elements unfolded and yields a partial result",
        });
        self.charge(Obligation::ThreadsAtMost {
            max: capacity,
            because: "every thread seeds scratch[tid] before the reduction",
        });
        let _ = scratch; // named for call-site readability; bound is `capacity`
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_reduce_charges_both_obligations_and_emits_the_shipping_shape() {
        let mut buf = String::new();
        let obs = {
            let mut k = KernelWriter::new(&mut buf);
            k.tree_reduce("nth", "scratch", 1024, "scratch[tid] += scratch[tid + step]");
            k.finish()
        };

        assert!(buf.contains("for (uint step = nth >> 1u; step != 0u; step >>= 1u) {"));
        assert!(buf.contains("if (tid < step) scratch[tid] += scratch[tid + step];"));
        assert!(buf.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));

        assert!(
            obs.iter()
                .any(|o| matches!(o, Obligation::ThreadsPowerOfTwo { .. })),
            "a tree reduction must charge the power-of-two obligation; that is the \
             whole point of routing it through KernelWriter"
        );
        assert!(
            obs.iter()
                .any(|o| matches!(o, Obligation::ThreadsAtMost { max: 1024, .. })),
            "the scratch capacity must bound the legal threadgroup size"
        );
    }

    #[test]
    fn threadgroup_array_charges_its_bytes() {
        let mut buf = String::new();
        let obs = {
            let mut k = KernelWriter::new(&mut buf);
            k.threadgroup_array("float", "scratch", 1024, 4, None);
            k.finish()
        };
        assert!(buf.contains("threadgroup float scratch[1024];"));
        assert_eq!(obs, vec![Obligation::ThreadgroupBytes { bytes: 4096 }]);
    }

    #[test]
    fn obligations_are_deduplicated_not_accumulated() {
        // A kernel with two reductions must not charge the same predicate twice:
        // the contract is a set of requirements, not a log of emissions.
        let mut buf = String::new();
        let obs = {
            let mut k = KernelWriter::new(&mut buf);
            k.tree_reduce("nth", "scratch", 1024, "scratch[tid] += scratch[tid + step]");
            k.tree_reduce(
                "nth",
                "scratch",
                1024,
                "scratch[tid] = max(scratch[tid], scratch[tid + step])",
            );
            k.finish()
        };
        assert_eq!(obs.len(), 2, "expected exactly the two distinct predicates");
    }

    /// The SwiGLU epilogue chain, in the order the Laguna reference spells it.
    /// Used as the worked example because it is the chain whose bit-exactness
    /// argument motivated this obligation.
    const SWIGLU: &[&str] = &[
        "exp(abs(gate))",
        "1 + exp_abs",
        "1 / denominator",
        "gate * sigmoid",
        "silu * up",
    ];

    fn emit_swiglu(k: &mut KernelWriter, ops: &[&str]) {
        k.pin_rounding();
        for op in ops {
            k.raw(&format!("    /* step */ {op};"));
        }
    }

    #[test]
    fn rounding_chain_charges_when_pragma_and_order_are_present() {
        let mut buf = String::new();
        let obs = {
            let mut k = KernelWriter::new(&mut buf);
            emit_swiglu(&mut k, SWIGLU);
            k.charge_rounding_chain("swiglu", SWIGLU, "bit-exact vs the staged epilogue");
            k.finish()
        };
        assert!(buf.contains(KernelWriter::FP_CONTRACT_OFF));
        assert!(
            obs.iter()
                .any(|o| matches!(o, Obligation::RoundingChainPinned { chain: "swiglu", .. })),
            "a pinned chain must appear in the ledger"
        );
    }

    #[test]
    #[should_panic(expected = "does not disable FP contraction")]
    fn rounding_chain_rejects_a_missing_pragma() {
        // The chain is spelled correctly but contraction is left on: the
        // compiler is free to fuse, so the contract would be claiming a
        // guarantee the kernel does not provide.
        let mut buf = String::new();
        let mut k = KernelWriter::new(&mut buf);
        for op in SWIGLU {
            k.raw(&format!("    /* step */ {op};"));
        }
        k.charge_rounding_chain("swiglu", SWIGLU, "bit-exact");
    }

    #[test]
    // Fails at step 4, not step 3: the reordered `gate * sigmoid` still matches
    // (later than intended), and it is the following `silu * up` that can no
    // longer be found AFTER it. The swap is still caught -- an ordered
    // subsequence match detects it at the second of the two swapped steps.
    #[should_panic(expected = "step 4")]
    fn rounding_chain_rejects_reordered_steps() {
        // THE mutation test. Swapping the sigmoid and silu steps is the edit
        // that looks harmless in review, keeps compiling, stays numerically
        // close -- and changes the last bit. If this does not panic, the whole
        // obligation is decorative.
        let reordered: &[&str] = &[
            "exp(abs(gate))",
            "1 + exp_abs",
            "1 / denominator",
            "silu * up",
            "gate * sigmoid",
        ];
        let mut buf = String::new();
        let mut k = KernelWriter::new(&mut buf);
        emit_swiglu(&mut k, reordered);
        k.charge_rounding_chain("swiglu", SWIGLU, "bit-exact");
    }

    #[test]
    #[should_panic(expected = "step 2")]
    fn rounding_chain_rejects_a_dropped_step() {
        let dropped: &[&str] = &[
            "exp(abs(gate))",
            "1 + exp_abs",
            "gate * sigmoid",
            "silu * up",
        ];
        let mut buf = String::new();
        let mut k = KernelWriter::new(&mut buf);
        emit_swiglu(&mut k, dropped);
        k.charge_rounding_chain("swiglu", SWIGLU, "bit-exact");
    }

    #[test]
    fn raw_charges_nothing() {
        let mut buf = String::new();
        let obs = {
            let mut k = KernelWriter::new(&mut buf);
            k.raw("    // trivia");
            k.finish()
        };
        assert!(obs.is_empty());
        assert_eq!(buf, "    // trivia\n");
    }
}
