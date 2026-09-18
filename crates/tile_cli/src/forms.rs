//! The form taxonomy: what a kernel representation IS, and how a file is mapped to one.
//!
//! Four extensions in this domain are overloaded — `.py` by nki/aie/tpu, `.mlir` by
//! mlir/linalg/pto/rvv, `.c` by gaudi/hexagon, `.cpp` by ttmetal/cpp — so an extension
//! is a hint, not a decision procedure. Resolution is: explicit `-f`, then content
//! magic, then extension, then an error that NAMES the candidates.
//!
//! `.rs` is reserved for tile-rs unconditionally and is never sniffed for anything else.
//!
//! The table is DATA. Adding a 17th backend is a row here plus a routes edge, not a new
//! code path — mirroring how `TargetRegistry::register` makes adding a codegen target a
//! one-line change.

use std::fmt;

/// How much a result produced along an edge should be trusted.
///
/// A mangled heading is visible; a numerically wrong kernel is invisible until it
/// corrupts a run. So this is printed on every conversion, not hidden in `--verbose`.
/// Where a `Validated` claim's evidence lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    /// A run recorded in this repo, at the named place.
    Here(&'static str),
    /// The claim comes from a project that has the hardware. It may be true; this repo
    /// carries nothing that shows it, and says so rather than implying it does.
    Inherited,
    /// An inherited claim that a run HERE contradicts, with the place that records what
    /// happened. Not the same as Unvalidated: unvalidated means nobody looked, refuted
    /// means somebody did and it did not work, and a caller picking a backend has to be
    /// able to tell those apart.
    Refuted(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fidelity {
    /// Mechanically checked emit path (the generality matrix + emit-purity contract).
    Exact,
    /// Additionally checked against a CPU reference on that hardware, and WHERE that
    /// check is recorded.
    ///
    /// #012. This was `Validated(hw)` alone, printed on the last line of every conversion
    /// -- by the design doc's own account "worth more than anything else the tool prints".
    /// Eight backends carried it and this repo holds evidence for three; the rest came
    /// from projects that do have the hardware, so the claims may well be true and
    /// downgrading them here would replace a possibly-true claim with a definitely-wrong
    /// one.
    ///
    /// So the claim is not changed -- it is ATTRIBUTED. `Evidence::Here` names the file
    /// that records the run; `Evidence::Inherited` says plainly that no run is recorded in
    /// this repo. The tool's own D4 rule is that an unmeasured quantity refuses rather
    /// than printing a plausible number, and a fidelity class is exactly a measurement
    /// claim.
    Validated(&'static str, Evidence),
    /// Emits correctly; never run on that target.
    Unvalidated,
    /// Produced by a lift. Needs review before use.
    Synthesised,
}

impl fmt::Display for Fidelity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fidelity::Exact => write!(f, "exact"),
            Fidelity::Validated(hw, Evidence::Here(where_)) => {
                write!(f, "exact, validated on {hw} ({where_})")
            }
            Fidelity::Validated(hw, Evidence::Inherited) => {
                write!(
                    f,
                    "exact, validated on {hw} — inherited, no run recorded here"
                )
            }
            Fidelity::Validated(hw, Evidence::Refuted(where_)) => {
                write!(f, "emits, but REFUTED on {hw} here — {where_}")
            }
            Fidelity::Unvalidated => write!(f, "exact, unvalidated on hardware"),
            Fidelity::Synthesised => write!(f, "synthesised"),
        }
    }
}

/// What a form IS FOR — that is, who consumes it.
///
/// This is the answer to the question the `lowering` backlog cluster kept re-asking in
/// four different shapes. `level` says how far down a representation sits; it does not
/// say what `tile` owes that representation, so "we cannot read AscendC" and "`.pico.s`
/// stops short of instruction words" both looked like missing features. Under this rule
/// they are different categories, and only one of them is a gap.
///
/// A form is one of:
///
/// * [`Role::Mlir`] — for further compiler passes to process. tile-rs owns it end to
///   end: it reads it, rewrites it, and writes it back. Optimization lives here. Only
///   `mlir` and `linalg` qualify: `rvv` and `pto` are MLIR *syntax* but no tile-rs pass
///   consumes them, so by the consumer test they are egress. That reclassification is
///   the rule earning its keep — `level` had them with the IR forms, which made a
///   deliberate hand-off look like two missing readers.
/// * [`Role::Source`] — for a DOWNSTREAM TOOLCHAIN to consume. tile-rs owes correct text
///   and the name of the toolchain that takes it. It does **not** owe a reader: reading
///   target source back is a frontend, which is a separate product, not a missing
///   milestone. This is what makes the 14-writers/3-readers asymmetry a design rather
///   than a debt.
/// * [`Role::Binary`] — loaded and executed directly by the harness, with no compile
///   step. No form is one today, which is a fact worth stating rather than an omission:
///   it is why `-r` compiles Metal source at runtime, and why turning an op program into
///   instruction words and a container would be a NEW form rather than more of an
///   existing one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Mlir,
    Source,
    Binary,
    /// Neither consumed, processed, nor executed — a dump for a human.
    ///
    /// Kept deliberately visible rather than folded into one of the three: by the rule
    /// above this is not a form at all, and marking it is more honest than picking the
    /// nearest label and hiding the exception.
    Diagnostic,
}

impl Role {
    /// Who takes this form next.
    pub fn consumer(self) -> &'static str {
        match self {
            Role::Mlir => "tile-rs compiler passes",
            Role::Source => "a downstream toolchain",
            Role::Binary => "the run harness, directly",
            Role::Diagnostic => "a human reader",
        }
    }

    /// Is tile-rs expected to be able to READ this form?
    ///
    /// Only its own IR. A `Source` form that cannot be read is complete, not partial.
    pub fn owes_a_reader(self) -> bool {
        matches!(self, Role::Mlir)
    }

    /// Can a run harness execute an artifact of this form without a compile step?
    pub fn directly_executable(self) -> bool {
        matches!(self, Role::Binary)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Role::Mlir => "mlir",
            Role::Source => "source",
            Role::Binary => "binary",
            Role::Diagnostic => "diagnostic",
        })
    }
}

/// A kernel representation.
///
/// `level` is what makes lower/lift/optimize a *derived* fact rather than three code
/// paths: 3 = tile-rs source, 2 = IR, 1 = target source. `tile -> msl` descends, so it
/// is a lowering; the reverse ascends, so it is a lift; equal levels mean optimization.
#[derive(Clone, Copy, Debug)]
pub struct Form {
    /// Stable string id. Serialized into MCP responses, attempt records and route
    /// reports, so it is frozen and covered by a compatibility test.
    pub id: &'static str,
    pub display: &'static str,
    /// What this form is for; see [`Role`]. `level` says how far down it sits, `kind`
    /// says who consumes it — and only the second determines what `tile` owes it.
    pub role: Role,
    pub level: u8,
    pub exts: &'static [&'static str],
    /// Discriminating content markers. Empty means "the fallback for its extension".
    pub magic: &'static [&'static str],
    /// Can tile-rs READ this form? tile-rs has 1.5 readers and 16 writers; this is the
    /// field that keeps that fact honest instead of implied.
    pub readable: bool,
    pub writable: bool,
    pub fidelity: Fidelity,
    /// Accelerator family this form runs on, for the detected-platform default.
    pub family: &'static str,
    /// The form this one REFINES, when it is a strictly narrower case of another.
    ///
    /// `rvv` is linalg with a RISC-V triple stamped on it — per the README it "has no
    /// translator of its own; LLVM already owns RVV codegen, so it wraps the linalg
    /// egress". So an rvv module also matches linalg's magic, and without this field
    /// the more general form wins by table order and rvv becomes unreachable.
    pub refines: Option<&'static str>,
}

impl Form {
    /// The extension used when deriving an output name.
    pub fn primary_ext(&self) -> &'static str {
        self.exts.first().copied().unwrap_or("out")
    }
    pub fn can_optimize_in_place(&self) -> bool {
        // Same-form optimization needs a READER. `msl -> msl` would mean writing a
        // Metal frontend: a second product, not a milestone.
        self.readable
    }

    /// Is this form's read-side ABSENCE a gap, or is it complete as designed?
    ///
    /// The distinction the backlog was missing. `cpp` (AscendC) is not readable and owes
    /// no reader, so #004 is a request for a frontend rather than a defect. `mlir` is
    /// readable and must stay so, because the passes have nowhere else to run.
    pub fn reader_is_owed(&self) -> bool {
        self.role.owes_a_reader()
    }
}

/// The table. Ordering matters only for deterministic error messages.
pub const FORMS: &[Form] = &[
    Form {
        id: "tile",
        role: Role::Source,
        display: "tile-rs DSL",
        level: 3,
        exts: &["rs"],
        magic: &[],
        readable: true,
        writable: true,
        fidelity: Fidelity::Exact,
        family: "*",
        refines: None,
    },
    // ── IR family (level 2) ──────────────────────────────────────────────────────
    Form {
        id: "mlir",
        role: Role::Mlir,
        display: "MLIR (generic)",
        level: 2,
        exts: &["mlir"],
        magic: &[],
        readable: true,
        writable: true,
        fidelity: Fidelity::Exact,
        family: "*",
        refines: None,
    },
    Form {
        id: "linalg",
        role: Role::Mlir,
        display: "MLIR linalg dialect",
        level: 2,
        exts: &["mlir"],
        magic: &["linalg."],
        readable: true,
        writable: true,
        fidelity: Fidelity::Validated("CPU", Evidence::Here("docs/cli/INTEGRATION.md")),
        family: "cpu",
        refines: None,
    },
    Form {
        id: "rvv",
        role: Role::Source,
        display: "linalg + RISC-V triple",
        level: 2,
        exts: &["mlir"],
        magic: &["riscv64"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "riscv",
        // An rvv module IS a linalg module plus a triple, so it carries linalg's marker
        // too. Without this it resolves to linalg and rvv becomes unreachable.
        refines: Some("linalg"),
    },
    Form {
        id: "pto",
        role: Role::Source,
        display: "PTO-MLIR",
        level: 2,
        exts: &["mlir"],
        magic: &["pto."],
        readable: false,
        writable: true,
        // Run from this tree on a 910b: the emitted PTO compiles through
        // ptoas, builds for dav-c220-vec, launches, and is numerically exact
        // for elementwise and for rms_norm at every shape that fills the
        // block. It needs a prologue ptoas does not emit -- `set_mask_norm()`
        // plus a full vector mask -- without which every VEC op faults while
        // MTE is unaffected. Partial tail blocks are still wrong.
        fidelity: Fidelity::Validated(
            "Ascend 910B",
            Evidence::Here("needs a mask prologue ptoas omits; cannbench-tilers NOTES 240"),
        ),
        family: "ascend",
        refines: None,
    },
    // ── Target source (level 1) ──────────────────────────────────────────────────
    Form {
        id: "msl",
        role: Role::Source,
        display: "Metal Shading Language",
        level: 1,
        exts: &["metal"],
        magic: &["kernel void"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated(
            "Apple GPU",
            Evidence::Here("docs/cli/INTEGRATION.md, 3 machines"),
        ),
        family: "apple-gpu",
        refines: None,
    },
    Form {
        id: "gpu",
        role: Role::Source,
        display: "CUDA C",
        level: 1,
        exts: &["cu"],
        magic: &["__global__"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated("NVIDIA", Evidence::Inherited),
        family: "nvidia",
        refines: None,
    },
    Form {
        id: "musa",
        role: Role::Source,
        display: "MUSA",
        level: 1,
        exts: &["mu"],
        magic: &["musa_runtime.h"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "moore",
        refines: None,
    },
    Form {
        id: "spirv",
        role: Role::Source,
        display: "GLSL -> SPIR-V",
        level: 1,
        exts: &["comp"],
        magic: &["layout(set = 0"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated(
            "Vulkan/MoltenVK",
            Evidence::Here("docs/cli/INTEGRATION.md, crates/tile_cli/assets/harness/vulkan.c"),
        ),
        family: "vulkan",
        refines: None,
    },
    Form {
        id: "nki",
        role: Role::Source,
        display: "AWS Trainium NKI",
        level: 1,
        exts: &["py"],
        magic: &["@nki.jit"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated("Trainium trn1", Evidence::Inherited),
        family: "trainium",
        refines: None,
    },
    Form {
        id: "aie",
        role: Role::Source,
        display: "AMD Ryzen AI IRON",
        level: 1,
        exts: &["py"],
        magic: &["from aie.iron"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "amd-npu",
        refines: None,
    },
    Form {
        id: "tpu",
        role: Role::Source,
        display: "JAX / Pallas",
        level: 1,
        exts: &["py"],
        magic: &["pallas"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated("TPU v5e", Evidence::Inherited),
        family: "tpu",
        refines: None,
    },
    Form {
        id: "bang",
        role: Role::Source,
        display: "Cambricon BANG-C",
        level: 1,
        exts: &["mlu"],
        magic: &["__mlu_entry__"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "cambricon",
        refines: None,
    },
    Form {
        id: "gaudi",
        role: Role::Source,
        display: "Intel Gaudi TPC-C",
        level: 1,
        exts: &["c"],
        magic: &["tpc-clang"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated("Gaudi", Evidence::Inherited),
        family: "gaudi",
        refines: None,
    },
    Form {
        id: "hexagon",
        role: Role::Source,
        display: "Qualcomm HVX / QNN",
        level: 1,
        exts: &["c"],
        magic: &["hvx_"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "hexagon",
        refines: None,
    },
    Form {
        id: "ttmetal",
        role: Role::Source,
        display: "Tenstorrent Tensix",
        level: 1,
        exts: &["cpp"],
        magic: &["void MAIN"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "tenstorrent",
        refines: None,
    },
    Form {
        id: "cpp",
        role: Role::Source,
        display: "AscendC C++",
        level: 1,
        // `.h` and `.hpp` are here because AscendC kernels are routinely written as
        // headers -- cannbench's tile_mm_core.h is one -- and backlog #004 hit exactly
        // that: a `.h` was not identified at all. They are also in WEAK_EXTS, without
        // which this row would claim every C header on the machine.
        exts: &["cce", "cpp", "h", "hpp"],
        // `using namespace AscendC;` is the idiom, and it has no `::`.
        magic: &["AscendC::", "namespace AscendC", "__aicore__"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Validated("Ascend 910B", Evidence::Inherited),
        family: "ascend",
        refines: None,
    },
    Form {
        id: "pico",
        role: Role::Source,
        display: "Huawei PICO / SVP_NNN intrinsic program",
        level: 1,
        // The target language is the intrinsic set itself: 91 named opcodes, emitted as
        // an intrinsic program plus a JSON manifest. Alone among the targets it ends in
        // no vendor compiler -- "linking" means rebuilding the .om container.
        exts: &["pico.s"],
        magic: &["pico", "SVP_NNN"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "pico",
        refines: None,
    },
    Form {
        id: "csl",
        role: Role::Source,
        display: "Cerebras CSL",
        level: 1,
        exts: &["csl"],
        magic: &["comptime"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Unvalidated,
        family: "cerebras",
        refines: None,
    },
    Form {
        id: "debug",
        role: Role::Diagnostic,
        display: "tile-rs debug banner",
        level: 2,
        exts: &["mlir.txt"],
        magic: &["tile-rs debug target"],
        readable: false,
        writable: true,
        fidelity: Fidelity::Exact,
        family: "*",
        refines: None,
    },
];

/// The value `TILERS_CODEGEN_PATH` wants for this form.
///
/// **Not the same string as the form id**, and finding that out cost a debugging session:
/// `TILERS_CODEGEN_PATH=msl` gets you "Could not determine ASCEND_HOME_PATH", because an
/// unrecognised value falls through to the Ascend default rather than being rejected. The
/// mapping is written down here so nobody has to discover it twice.
pub fn codegen_path(id: &str) -> Option<&'static str> {
    Some(match id {
        "msl" => "metal",
        "gpu" => "cuda",
        "spirv" => "vulkan",
        // The rest name themselves. Returned from the table rather than echoing the
        // caller's `&str`, so the result really is 'static.
        "cpp" => "cpp",
        "pto" => "pto",
        "nki" => "nki",
        "aie" => "aie",
        "bang" => "bang",
        "gaudi" => "gaudi",
        "musa" => "musa",
        "linalg" => "linalg",
        "rvv" => "rvv",
        "tpu" => "tpu",
        "csl" => "csl",
        "pico" => "pico",
        "hexagon" => "hexagon",
        "ttmetal" => "ttmetal",
        _ => return None,
    })
}

/// Extensions too common in the wild for a match on the extension alone to be evidence.
///
/// Normally a single claimant resolves by extension: only `msl` claims `.metal`, so a
/// `.metal` file is Metal. `.h` is different — it is every C project's header, and
/// letting the one row that claims it win by default would make every header on the
/// machine an AscendC kernel. Here the content has to say so.
const WEAK_EXTS: &[&str] = &["h", "hpp"];

pub fn by_id(id: &str) -> Option<&'static Form> {
    FORMS.iter().find(|f| f.id == id)
}

/// Every form that claims this extension.
pub fn by_ext(ext: &str) -> Vec<&'static Form> {
    FORMS.iter().filter(|f| f.exts.contains(&ext)).collect()
}

/// How a form was decided. Reported to the user, because a heuristic that does not
/// say it was a heuristic is a bug generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    /// `.rs` — reserved, never sniffed.
    Reserved,
    /// `-f/--from` on the command line.
    Forced,
    /// Content magic picked it out of the extension's candidates.
    Magic,
    /// The extension claimed exactly one form.
    Extension,
}

impl fmt::Display for How {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            How::Reserved => write!(f, ".rs reserved"),
            How::Forced => write!(f, "forced by -f"),
            How::Magic => write!(f, "sniffed: extension ∩ content"),
            How::Extension => write!(f, "extension"),
        }
    }
}

#[derive(Debug)]
pub enum FormError {
    /// The extension is shared and no magic matched. Names the candidates.
    Ambiguous {
        ext: String,
        candidates: Vec<&'static str>,
    },
    /// Nothing claims this extension and no magic matched.
    Unknown { ext: String },
    /// `-f <id>` named a form that does not exist.
    NoSuchForm { id: String },
}

impl fmt::Display for FormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormError::Ambiguous { ext, candidates } => write!(
                f,
                ".{ext} is ambiguous ({}); pass -f <form> to say which",
                candidates.join(", ")
            ),
            FormError::Unknown { ext } => write!(
                f,
                "cannot tell what kind of kernel .{ext} is, and no content marker matched; \
                 pass -f <form> (see `tile --list-forms`)"
            ),
            FormError::NoSuchForm { id } => {
                write!(f, "no such form \"{id}\" (see `tile --list-forms`)")
            }
        }
    }
}

/// The resolution, plus any disagreement worth warning about.
#[derive(Debug)]
pub struct Resolved {
    pub form: &'static Form,
    pub how: How,
    /// Set when `-f` was given but the content says otherwise. Not an error — the user
    /// may be converting a file someone mislabelled — but never silent.
    pub contradicted_by: Option<&'static Form>,
}

/// Extension of a path, lowercased. Handles the one two-part extension in the table.
pub fn ext_of(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    // Two-part extensions, longest first.
    for two in [".mlir.txt", ".pico.s"] {
        if name.ends_with(two) {
            return two[1..].to_string();
        }
    }
    match name.rsplit_once('.') {
        Some((_, e)) if !e.is_empty() => e.to_string(),
        _ => String::new(),
    }
}

/// Which form does the content look like, considering only these candidates?
///
/// When several match, a form that REFINES another beats the one it refines — an rvv
/// module carries linalg's marker as well as its own, and the narrower answer is the
/// true one.
fn magic_pick(content: &str, candidates: &[&'static Form]) -> Option<&'static Form> {
    let matches: Vec<&'static Form> = candidates
        .iter()
        .filter(|f| !f.magic.is_empty())
        .filter(|f| f.magic.iter().any(|m| content.contains(m)))
        .copied()
        .collect();
    // Drop any form that another match refines.
    let refined_away: Vec<&str> = matches.iter().filter_map(|f| f.refines).collect();
    matches
        .iter()
        .find(|f| !refined_away.contains(&f.id))
        .copied()
        .or_else(|| matches.first().copied())
}

/// Resolve the form of an input.
///
/// Order: explicit `-f`, then `.rs` reserved, then magic within the extension's
/// candidates, then the extension alone, then an error naming the candidates.
pub fn resolve_input(
    path: &str,
    content: &str,
    forced: Option<&str>,
) -> Result<Resolved, FormError> {
    if let Some(id) = forced {
        let form = by_id(id).ok_or_else(|| FormError::NoSuchForm { id: id.to_string() })?;
        // Warn if strong magic disagrees — but obey the user.
        let contradicted_by =
            magic_pick(content, &FORMS.iter().collect::<Vec<_>>()).filter(|f| f.id != form.id);
        return Ok(Resolved {
            form,
            how: How::Forced,
            contradicted_by,
        });
    }

    let ext = ext_of(path);

    // `.rs` is reserved for tile-rs unconditionally. No sniffing happens at all —
    // a tile-rs kernel is allowed to contain the string `__global__` in a comment.
    if ext == "rs" {
        return Ok(Resolved {
            form: by_id("tile").expect("tile form"),
            how: How::Reserved,
            contradicted_by: None,
        });
    }

    let candidates = by_ext(&ext);
    if let Some(f) = magic_pick(content, &candidates) {
        return Ok(Resolved {
            form: f,
            how: How::Magic,
            contradicted_by: None,
        });
    }

    // No magic matched. A single candidate with no magic of its own is the extension's
    // documented fallback (this is how a plain `.mlir` module resolves to `mlir` while
    // `.py` — whose three claimants all carry magic — correctly refuses).
    let fallbacks: Vec<_> = candidates.iter().filter(|f| f.magic.is_empty()).collect();
    if WEAK_EXTS.contains(&ext.as_str()) {
        // No magic matched and the extension is not evidence on its own.
        return Err(FormError::Unknown { ext });
    }
    match (candidates.len(), fallbacks.len()) {
        (0, _) => Err(FormError::Unknown { ext }),
        (1, _) => Ok(Resolved {
            form: candidates[0],
            how: How::Extension,
            contradicted_by: None,
        }),
        (_, 1) => Ok(Resolved {
            form: fallbacks[0],
            how: How::Extension,
            contradicted_by: None,
        }),
        _ => Err(FormError::Ambiguous {
            ext,
            candidates: candidates.iter().map(|f| f.id).collect(),
        }),
    }
}

/// Resolve the form of an output: explicit `-t`, else the extension of `-o`, else the
/// caller's default (the detected platform's native form).
pub fn resolve_output(
    out_path: Option<&str>,
    to: Option<&str>,
    default_form: &'static Form,
) -> Result<&'static Form, FormError> {
    resolve_output_from(out_path, to, default_form, None)
}

/// `resolve_output`, told what the INPUT was.
///
/// An output file has no content to sniff, so a shared extension is unresolvable by
/// construction — which is what `-t` is for. But two cases resolve without asking:
///
/// * the input's own form claims that extension, in which case `-o k2.mlir` on an mlir
///   input plainly means the same form (this is the optimization case, and refusing it
///   as "ambiguous" is pedantry);
/// * the extension has a single documented fallback — the form with no magic of its own
///   — which is the same rule that lets a plain `.mlir` module resolve on the way in.
pub fn resolve_output_from(
    out_path: Option<&str>,
    to: Option<&str>,
    default_form: &'static Form,
    input_form: Option<&'static Form>,
) -> Result<&'static Form, FormError> {
    if let Some(id) = to {
        return by_id(id).ok_or_else(|| FormError::NoSuchForm { id: id.to_string() });
    }
    let Some(path) = out_path else {
        return Ok(default_form);
    };
    if path == "-" {
        return Ok(default_form);
    }
    let ext = ext_of(path);
    if ext == "rs" {
        return Ok(by_id("tile").expect("tile form"));
    }
    let candidates = by_ext(&ext);
    if candidates.len() <= 1 {
        return match candidates.first() {
            None => Err(FormError::Unknown { ext }),
            Some(f) => Ok(f),
        };
    }
    // The input's own form wins: `-o k2.mlir` on an mlir input means mlir.
    if let Some(inf) = input_form {
        if candidates.iter().any(|c| c.id == inf.id) {
            return Ok(inf);
        }
    }
    // Otherwise the extension's documented fallback, if it has exactly one.
    let fallbacks: Vec<_> = candidates.iter().filter(|f| f.magic.is_empty()).collect();
    if fallbacks.len() == 1 {
        return Ok(fallbacks[0]);
    }
    Err(FormError::Ambiguous {
        ext,
        candidates: candidates.iter().map(|f| f.id).collect(),
    })
}

/// lower / lift / optimize, derived from the level difference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Lower,
    Lift,
    Optimize,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Lower => write!(f, "lower"),
            Kind::Lift => write!(f, "lift"),
            Kind::Optimize => write!(f, "optimize"),
        }
    }
}

#[cfg(test)]
mod role_tests {
    use super::*;

    #[test]
    fn every_form_has_a_role_consistent_with_its_level() {
        // The IR forms are the ones the passes process; everything at level 1 is target
        // source for somebody else's toolchain. A row that violates this is either
        // mis-levelled or mis-roled, and both are worth failing over.
        for f in FORMS {
            match f.role {
                Role::Mlir => assert_eq!(f.level, 2, "{} is Mlir but not level 2", f.id),
                // Level 2 is allowed here and means MLIR-SYNTAX EGRESS: the file is
                // MLIR, but no tile-rs pass consumes it, so the consumer is external.
                // `rvv` is the clearest case -- the README says it "has no translator of
                // its own; LLVM already owns RVV codegen, so it wraps the linalg
                // egress". Classifying it by syntax put it with the IR forms and made it
                // look like a form we had failed to write a reader for.
                Role::Source => assert!(
                    (1..=3).contains(&f.level),
                    "{} is Source but level {}",
                    f.id,
                    f.level
                ),
                Role::Binary => assert_eq!(f.level, 0, "{} is Binary but not level 0", f.id),
                Role::Diagnostic => {}
            }
        }
    }

    #[test]
    fn only_the_ir_forms_are_owed_a_reader() {
        // This is the whole point of the taxonomy: tile-rs writes 14 target-source forms
        // and reads none of them, and that is COMPLETE rather than partial. Before this
        // rule the same fact read as a 14-way hole and produced backlog issues.
        for f in FORMS {
            if f.reader_is_owed() {
                assert!(
                    f.readable,
                    "{} is an IR form that the passes cannot read",
                    f.id
                );
            }
        }
        // Exactly two. `tile` is also readable -- we own a frontend for our own DSL --
        // but that is a capability we chose, not one the taxonomy requires, and the
        // difference is why "3 readers, 14 writers" is a shape rather than a shortfall.
        let owed: Vec<&str> = FORMS
            .iter()
            .filter(|f| f.reader_is_owed())
            .map(|f| f.id)
            .collect();
        assert_eq!(owed, vec!["mlir", "linalg"]);
    }

    #[test]
    fn no_form_is_directly_executable_yet_and_that_is_why_the_harness_compiles() {
        // Stated rather than left implicit. `-r` compiles Metal source at runtime
        // precisely because nothing tile-rs writes can just be loaded and run. When a
        // binary form is added, the harness gains a path that skips the compiler -- and
        // this test is the one that should change.
        assert!(
            !FORMS.iter().any(|f| f.role.directly_executable()),
            "a Binary form exists now; the run harness should no longer compile it"
        );
    }

    #[test]
    fn a_source_form_names_who_consumes_it() {
        let msl = by_id("msl").unwrap();
        assert_eq!(msl.role, Role::Source);
        assert_eq!(msl.role.consumer(), "a downstream toolchain");
        assert!(!msl.reader_is_owed());
    }

    #[test]
    fn the_debug_dump_is_flagged_as_not_really_a_form() {
        // It is neither consumed by a toolchain, processed by a pass, nor executed. By
        // the rule it is not a form; marking it beats picking the nearest label.
        assert_eq!(by_id("debug").unwrap().role, Role::Diagnostic);
    }
}

#[cfg(test)]
mod codegen_path_tests {
    use super::*;

    #[test]
    fn the_codegen_path_is_not_always_the_form_id() {
        // The three that differ. An unrecognised TILERS_CODEGEN_PATH does not error --
        // it falls through to the Ascend default and fails with a message about
        // ASCEND_HOME_PATH, which points at entirely the wrong thing.
        assert_eq!(codegen_path("msl"), Some("metal"));
        assert_eq!(codegen_path("gpu"), Some("cuda"));
        assert_eq!(codegen_path("spirv"), Some("vulkan"));
    }

    #[test]
    fn every_writable_target_form_has_a_codegen_path() {
        for f in FORMS {
            if f.level == 1 || f.id == "pto" || f.id == "linalg" || f.id == "rvv" {
                assert!(
                    codegen_path(f.id).is_some(),
                    "{} can be emitted but has no TILERS_CODEGEN_PATH",
                    f.id
                );
            }
        }
    }

    #[test]
    fn a_form_that_is_not_a_codegen_target_has_no_path() {
        assert_eq!(codegen_path("tile"), None);
        assert_eq!(codegen_path("mlir"), None);
        assert_eq!(codegen_path("debug"), None);
    }
}

#[cfg(test)]
mod output_tests {
    use super::*;

    #[test]
    fn an_ascendc_header_is_identified_by_its_content() {
        // backlog #004: `tile tile_mm_core.h -i` said "cannot tell what kind of kernel
        // .h is" about a file whose first lines say `using namespace AscendC`.
        let src = "#include \"kernel_operator.h\"\nusing namespace AscendC;\n";
        let r = resolve_input("tile_mm_core.h", src, None).expect("resolve");
        assert_eq!(r.form.id, "cpp");
        assert_eq!(r.how, How::Magic);
    }

    #[test]
    fn a_header_with_no_ascend_marker_is_still_refused() {
        // The extension alone must not claim it: `.h` is every C project's header.
        let e = resolve_input("vector.h", "struct v { float x; };\n", None).unwrap_err();
        assert!(e.to_string().contains("cannot tell"), "{e}");
    }

    #[test]
    fn an_ambiguous_output_extension_takes_the_inputs_form() {
        // `tile k.mlir -o k2.mlir` obviously means mlir. Refusing it as ambiguous
        // because `.mlir` is claimed by four forms is pedantry, and it makes the whole
        // same-form optimization case unreachable.
        let default = by_id("linalg").unwrap();
        let input = by_id("mlir").unwrap();
        let f = resolve_output_from(Some("k2.mlir"), None, default, Some(input)).unwrap();
        assert_eq!(f.id, "mlir");
    }

    #[test]
    fn an_ambiguous_output_falls_back_when_the_input_does_not_claim_it() {
        // A `.rs` input writing `.mlir` has no claim on the extension, so the
        // extension's documented fallback decides — the same rule the sniffer uses.
        let default = by_id("linalg").unwrap();
        let input = by_id("tile").unwrap();
        let f = resolve_output_from(Some("out.mlir"), None, default, Some(input)).unwrap();
        assert_eq!(f.id, "mlir");
    }

    #[test]
    fn an_extension_with_no_fallback_still_demands_dash_t() {
        // `.py` is claimed by three forms and none of them is a fallback, so there is
        // genuinely nothing to infer.
        let default = by_id("linalg").unwrap();
        let input = by_id("mlir").unwrap();
        let e = resolve_output_from(Some("out.py"), None, default, Some(input)).unwrap_err();
        assert!(e.to_string().contains("nki"), "{e}");
    }

    #[test]
    fn dash_t_still_beats_every_inference() {
        let default = by_id("linalg").unwrap();
        let input = by_id("mlir").unwrap();
        let f = resolve_output_from(Some("k2.mlir"), Some("pto"), default, Some(input)).unwrap();
        assert_eq!(f.id, "pto");
    }
}

pub fn kind_of(from: &Form, to: &Form) -> Kind {
    match to.level.cmp(&from.level) {
        std::cmp::Ordering::Less => Kind::Lower,
        std::cmp::Ordering::Greater => Kind::Lift,
        std::cmp::Ordering::Equal => Kind::Optimize,
    }
}

#[cfg(test)]
mod evidence_tests {
    use super::*;

    /// #012. A `Validated` claim now says where its evidence is. A pointer at a file that
    /// does not exist would be worse than the unattributed claim it replaced: it looks
    /// checkable and is not. So every path named is opened.
    #[test]
    fn every_evidence_pointer_names_a_file_that_exists() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("the repo root is two levels above this crate")
            .to_path_buf();
        let mut checked = 0;
        for f in FORMS {
            let Fidelity::Validated(hw, Evidence::Here(w)) = f.fidelity else {
                continue;
            };
            // The pointer may name a file plus a note ("..., 3 machines"); the path is the
            // part before the first comma, and each comma-separated entry that looks like
            // a path is checked.
            for part in w.split(',') {
                let part = part.trim();
                if !part.contains('/') {
                    continue;
                }
                assert!(
                    root.join(part).exists(),
                    "{}: claims validation on {hw} recorded in `{part}`, which does not \
                     exist. An evidence pointer that cannot be followed is worse than no \
                     pointer -- it looks checkable.",
                    f.id
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 3,
            "at least the three attributed backends were checked"
        );
    }

    /// The two kinds must stay distinguishable in what the tool PRINTS, not only in the
    /// type. The whole issue was nine claims rendered identically.
    #[test]
    fn an_inherited_claim_never_renders_like_a_recorded_one() {
        let here = Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md"));
        let up = Fidelity::Validated("TPU v5e", Evidence::Inherited);
        let (a, b) = (here.to_string(), up.to_string());
        assert!(a.contains("docs/cli/INTEGRATION.md"), "{a}");
        assert!(
            b.contains("inherited") && b.contains("no run recorded here"),
            "an unattributed claim must say so on the line it is printed on: {b}"
        );
        assert!(
            !b.contains('('),
            "no file is named for a claim that has none: {b}"
        );
    }
}
