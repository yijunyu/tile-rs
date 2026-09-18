//! `tile <kernel> -i` — what this kernel IS, before anything changes it.
//!
//! This is not a line counter. `tile_codegen`'s DEFAULT features (std-only, no LLVM,
//! builds anywhere) already carry the analysis: tile planning, type promotion, the
//! RAW/WAR/WAW hazard model, and the NPU resource bounds. The CLI is the first
//! user-facing surface for the last two, which arrived in commits `be53759`
//! ("NPU resource bounds as codegen invariants") and `63d1147` ("give tile-rs the
//! hazard model it lacked").
//!
//! The rule inherited from `f10513c` and enforced here: **an unmeasured architecture
//! refuses**. A wrong lowering is silent — an unfired NaN test, a half-written
//! destination — so a bound that was never measured on this chip reports "unmeasured",
//! never a plausible number borrowed from a chip that was.

use crate::forms::{Form, How};
use std::fmt::Write as _;
use tile_codegen::hazard::{barrier_points, hazards, unsynced, BufId, Edge, VecOp};
use tile_codegen::{plan_tiles, HardwareParams, ScalarType, TilePlan};

/// One kernel found in the input.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub op_count: usize,
    pub dtypes: Vec<ScalarType>,
    pub extents: Vec<usize>,
}

impl Kernel {
    /// Elements the kernel walks, from the largest declared extent.
    pub fn numel(&self) -> usize {
        self.extents.iter().copied().max().unwrap_or(0)
    }
    /// Widest dtype, which is what the tile plan must be sized against.
    pub fn widest(&self) -> ScalarType {
        self.dtypes
            .iter()
            .copied()
            .max_by_key(|d| d.bytes())
            .unwrap_or(ScalarType::F32)
    }
}

/// The hazard picture: what must be synchronised, and what was left unsynchronised.
#[derive(Debug, Default)]
pub struct Hazards {
    pub edges: Vec<Edge>,
    pub barriers: Vec<usize>,
    pub unsynced: Vec<Edge>,
}

/// Resource bounds against the selected target — or the refusal.
#[derive(Debug)]
pub enum Bounds {
    /// Every check answered, with its verdict.
    Checked {
        arch: &'static str,
        results: Vec<(&'static str, Result<(), String>)>,
    },
    /// The architecture has never been measured, so no check may answer.
    Unmeasured { arch: &'static str, why: String },
    /// The source declares no extent, so the checks have nothing to check.
    ///
    /// #009: `bounds (n/a): UB ok; repeat ok; DMA stride ok;` was printed for a kernel
    /// whose tile plan reported "not computable (no declared extent)" one line above. The
    /// checks ran against a DEFAULT tile size and every one of them answered `ok` --
    /// "which is the worst of the three possible answers: a planner reading this concludes
    /// the tiling is fine".
    ///
    /// There are three answers, not two, and this is the third.
    NoInput { arch: &'static str, why: String },
}

#[derive(Debug)]
pub struct Profile {
    pub path: String,
    pub form: &'static Form,
    pub how: How,
    pub kernels: Vec<Kernel>,
    pub plan: Option<TilePlan>,
    pub cores: usize,
    pub tile: usize,
    pub hazards: Hazards,
    pub bounds: Bounds,
}

/// Hardware parameters for a target family.
///
/// `ascend-950` is the deliberate refusal case: arch35 is a different instruction set
/// from arch32, so a 910B capacity would approve precisely the tilings that fail there.
/// `ascend-950pr` and `ascend-950dt` share that ISA target (`dav-3510`) but are
/// different silicon and different eval images — pick the SKU when it is known.
pub fn params_for(family: &str) -> HardwareParams {
    match family {
        "ascend" => HardwareParams::ascend_910b(),
        "ascend-950" => HardwareParams::ascend_950_unmeasured(),
        "ascend-950pr" => HardwareParams::ascend_950pr_unmeasured(),
        "ascend-950dt" => HardwareParams::ascend_950dt_unmeasured(),
        // A CUDA or Metal backend has no unified buffer, no 8-bit repeat field and no
        // 16-bit stride: measured, every limit zero, every check correctly answers Ok.
        _ => HardwareParams::default(),
    }
}

/// Physical cores to plan against. A conservative default per family; `--cores`
/// overrides. Wrong here means a wrong `blocks`, not a wrong answer.
///
/// `0` = unknown — `plan_tiles` refuses a zero core count rather than inventing a
/// grid. The 950 SKUs return `0`: `GetCoreNumAiv()` has never been printed from a
/// graded job on either, and a borrowed 48 would be a 910B number presented as a
/// 950 fact (same failure mode D4 forbids for capacities).
pub fn cores_for(family: &str) -> usize {
    match family {
        "ascend" => 48,
        "ascend-950" | "ascend-950pr" | "ascend-950dt" => 0,
        "apple-gpu" => 32,
        "nvidia" => 80,
        _ => 8,
    }
}

// ── Extraction ────────────────────────────────────────────────────────────────────
// M1 reads structure out of the text directly. The real MLIR parser (`mlir_parse`)
// arrives with the emitters under the `emitters` feature in M2; until then this must
// work with no LLVM and no parser, which is also what makes it testable everywhere.

/// Kernel entry points, per form family.
pub fn kernel_names(form_id: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |s: String| {
        if !s.is_empty() && !out.contains(&s) {
            out.push(s)
        }
    };
    for line in text.lines() {
        // Stop at a comment. The corpus kernel's own comment says it is "safe for it to
        // mention __global__ and kernel void", and this scan duly found a kernel called
        // `in` -- from "kernel void in a comment". Fourth appearance of reading code out
        // of prose; the file written to trap the sniffer caught the profiler instead.
        let t = line.split("//").next().unwrap_or("").trim();
        // MLIR family. Both spellings: `func.func @` for the tensor/linalg dialects, and
        // `llvm.func @` for the LLVM-dialect modules the emitters actually consume.
        for marker in ["func.func @", "llvm.func @"] {
            if let Some(rest) = t.split(marker).nth(1) {
                push(ident(rest));
            }
        }
        // tile-rs source
        if form_id == "tile" {
            if let Some(rest) = t.strip_prefix("pub fn ").or_else(|| t.strip_prefix("fn ")) {
                push(ident(rest));
            }
        }
        // Target sources: each backend's entry idiom.
        for marker in [
            "kernel void ",
            "__global__ void ",
            "__mlu_entry__ void ",
            "void MAIN",
        ] {
            if let Some(rest) = t.split(marker).nth(1) {
                push(ident(rest));
            }
        }
        if let Some(rest) = t.strip_prefix("def ") {
            push(ident(rest));
        }
    }
    out
}

fn ident(s: &str) -> String {
    s.trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Scalar types named anywhere in the text, in a stable order.
pub fn dtypes_in(text: &str) -> Vec<ScalarType> {
    // Longest spellings first so `float16` is not shadowed by a `float` prefix scan.
    const NAMES: &[(&str, ScalarType)] = &[
        ("bfloat16", ScalarType::BF16),
        ("float16", ScalarType::F16),
        ("float32", ScalarType::F32),
        ("float64", ScalarType::F64),
        ("bf16", ScalarType::BF16),
        ("f16", ScalarType::F16),
        ("f32", ScalarType::F32),
        ("f64", ScalarType::F64),
        ("half", ScalarType::F16),
        ("int64", ScalarType::I64),
        ("int32", ScalarType::I32),
        ("i64", ScalarType::I64),
        ("i32", ScalarType::I32),
        ("i16", ScalarType::I16),
        ("i8", ScalarType::I8),
    ];
    let mut found: Vec<ScalarType> = Vec::new();
    for (name, ty) in NAMES {
        if text.contains(name) && !found.contains(ty) {
            found.push(*ty);
        }
    }
    found.sort();
    found
}

/// Declared extents.
///
/// Two spellings, because the two dialects say it differently: `tensor<1024xf32>` and
/// `memref<256x256xf32>` carry the shape in the type, while an LLVM-dialect module
/// carries it in the `llvm.mlir.constant` values the `__tile_*` intrinsics are handed.
/// A naive scan of every `<...>` group reads the `1` out of `!llvm.ptr<1>` and calls it
/// an extent, which is how a 1024-element kernel came to report `numel 1`.
pub fn extents_in(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut push = |n: usize| {
        if n > 1 && !out.contains(&n) {
            out.push(n);
        }
    };

    // Shaped types: tensor / memref / vector only. `ptr<1>` is an address space.
    for kw in ["tensor<", "memref<", "vector<"] {
        let mut rest = text;
        while let Some(i) = rest.find(kw) {
            rest = &rest[i + kw.len()..];
            let end = rest.find('>').unwrap_or(rest.len());
            for part in rest[..end].split('x') {
                if let Ok(n) = part.trim().parse::<usize>() {
                    push(n);
                }
            }
        }
    }

    // tile-rs source: the extents ride in the const generics of a view type,
    // `GmView<1, 1024, f32>`. Without this a kernel whose shape is right there in its
    // signature reports "not computable".
    for kw in ["GmView<", "GmViewMut<"] {
        let mut rest = text;
        while let Some(i) = rest.find(kw) {
            rest = &rest[i + kw.len()..];
            let end = rest.find('>').unwrap_or(rest.len());
            for part in rest[..end].split(',') {
                if let Ok(n) = part.trim().parse::<usize>() {
                    push(n);
                }
            }
        }
    }

    // LLVM dialect: the extents ride in the constants.
    let kw = "llvm.mlir.constant(";
    let mut rest = text;
    while let Some(i) = rest.find(kw) {
        rest = &rest[i + kw.len()..];
        let end = rest.find(')').unwrap_or(rest.len());
        let val = rest[..end].split(':').next().unwrap_or("").trim();
        if let Ok(n) = val.parse::<usize>() {
            push(n);
        }
    }

    out.sort_unstable();
    out
}

/// Build the vector-op sequence the hazard model consumes, from SSA assignments
/// of the shape `%dst = dialect.op %a, %b`.
pub fn vec_ops(text: &str) -> Vec<VecOp> {
    let mut names: Vec<String> = Vec::new();
    let id_of = |n: &str, names: &mut Vec<String>| -> BufId {
        if let Some(i) = names.iter().position(|x| x == n) {
            BufId(i)
        } else {
            names.push(n.to_string());
            BufId(names.len() - 1)
        }
    };
    let mut ops = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        let Some(eq) = t.find(" = ") else { continue };
        if !t.starts_with('%') {
            continue;
        }
        let dst_name = t[..eq].trim().to_string();
        let rhs = &t[eq + 3..];
        let opname: String = rhs.chars().take_while(|c| !c.is_whitespace()).collect();
        if opname.is_empty() {
            continue;
        }
        let srcs: Vec<BufId> = rhs
            .split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')')
            .filter(|s| s.starts_with('%'))
            .map(|s| id_of(s, &mut names))
            .collect();
        let dst = id_of(&dst_name, &mut names);
        ops.push(VecOp::new(opname, dst, &srcs));
    }
    ops
}

/// Profile one input.
pub fn profile(
    path: &str,
    text: &str,
    form: &'static Form,
    how: How,
    family: &str,
    tile: usize,
) -> Profile {
    let names = kernel_names(form.id, text);
    let dtypes = dtypes_in(text);
    let extents = extents_in(text);
    let op_count = text
        .lines()
        .filter(|l| l.trim().starts_with('%') && l.contains(" = "))
        .count();

    let kernels: Vec<Kernel> = if names.is_empty() {
        vec![Kernel {
            name: "<unnamed>".into(),
            op_count,
            dtypes: dtypes.clone(),
            extents: extents.clone(),
        }]
    } else {
        names
            .into_iter()
            .map(|name| Kernel {
                name,
                op_count,
                dtypes: dtypes.clone(),
                extents: extents.clone(),
            })
            .collect()
    };

    let cores = cores_for(family);
    let numel = kernels.iter().map(|k| k.numel()).max().unwrap_or(0);
    let plan = if numel > 0 {
        plan_tiles(numel, tile, cores)
    } else {
        None
    };

    let ops = vec_ops(text);
    let edges = hazards(&ops);
    let barriers = barrier_points(&ops);
    let un = unsynced(&ops, &barriers);
    let hz = Hazards {
        edges,
        barriers,
        unsynced: un,
    };

    let hp = params_for(family);
    let widest = kernels
        .first()
        .map(|k| k.widest())
        .unwrap_or(ScalarType::F32);
    // No declared extent means no tile to check. Answering `ok` here is answering a
    // question that was never asked -- see `Bounds::NoInput`.
    let bounds = if plan.is_none() && numel == 0 {
        Bounds::NoInput {
            arch: if hp.npu_arch.is_empty() {
                "n/a"
            } else {
                hp.npu_arch
            },
            why: "the source declares no extent, so there is no tile size to check a \
                  capacity against"
                .to_string(),
        }
    } else if !hp.measured {
        Bounds::Unmeasured {
            arch: hp.npu_arch,
            why: format!(
                "no capacity has been MEASURED on {}; another chip's numbers would \
                 approve tilings it cannot hold",
                if hp.chip.is_empty() {
                    hp.npu_arch.to_string()
                } else {
                    format!("{} / {}", hp.npu_arch, hp.chip)
                }
            ),
        }
    } else {
        let dtype_bytes = widest.bytes();
        Bounds::Checked {
            arch: if hp.npu_arch.is_empty() {
                "n/a"
            } else {
                hp.npu_arch
            },
            results: vec![
                ("UB", hp.check_ub(tile * dtype_bytes)),
                ("repeat", hp.check_repeat(tile, dtype_bytes)),
                ("DMA stride", hp.check_dma_stride(tile * dtype_bytes, "row")),
            ],
        }
    };

    Profile {
        path: path.to_string(),
        form,
        how,
        kernels,
        plan,
        cores,
        tile,
        hazards: hz,
        bounds,
    }
}

impl Profile {
    /// The human rendering. Every line here is asserted by a scenario in
    /// `features/04_profile_and_optimize.feature`.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "{}", self.path);
        let _ = writeln!(s, "  form: {} ({})", self.form.id, self.how);
        for k in &self.kernels {
            let _ = writeln!(s, "  kernel: {}  [{} ops]", k.name, k.op_count);
        }
        let names: Vec<&str> = self
            .kernels
            .first()
            .map(|k| k.dtypes.iter().map(|d| d.name()).collect())
            .unwrap_or_default();
        if !names.is_empty() {
            let widest = self.kernels[0].widest();
            let _ = writeln!(
                s,
                "  dtypes: {} -> promotes to {} ({} bytes)",
                names.join(", "),
                widest.name(),
                widest.bytes()
            );
            if !widest.has_vector_compare() {
                let _ = writeln!(
                    s,
                    "  note: {} has no vector compare path; a predicate on it drops to scalar",
                    widest.name()
                );
            }
        }
        if let Some(k) = self.kernels.first() {
            if !k.extents.is_empty() {
                let _ = writeln!(s, "  extents: {:?}  numel {}", k.extents, k.numel());
            }
        }
        match &self.plan {
            Some(p) => {
                let _ = writeln!(
                    s,
                    "  tile plan: {} x {} cores -> {} full + {} tail, {} blocks, {} rounds/block",
                    self.tile, self.cores, p.full, p.tail, p.blocks, p.rounds_per_block
                );
            }
            None => {
                if self.cores == 0 && self.kernels.iter().any(|k| k.numel() > 0) {
                    // cores_for returned unknown (950 PR/DT: GetCoreNumAiv never
                    // printed). Saying "no declared extent" here would be a lie;
                    // inventing 48 would be a 910B fact.
                    let _ = writeln!(
                        s,
                        "  tile plan: UNKNOWN — core count has not been measured on this \
                         architecture (never fall back to 40/48)"
                    );
                } else {
                    let _ = writeln!(s, "  tile plan: not computable (no declared extent)");
                }
            }
        }
        let _ = writeln!(
            s,
            "  hazards: {} edges, {} barrier points, {} unsynchronised",
            self.hazards.edges.len(),
            self.hazards.barriers.len(),
            self.hazards.unsynced.len()
        );
        for e in &self.hazards.unsynced {
            // An unsynchronised edge is a defect, not a note.
            let _ = writeln!(s, "  DEFECT: unsynchronised {:?}", e);
        }
        match &self.bounds {
            Bounds::Unmeasured { arch, why } => {
                let _ = writeln!(s, "  bounds: UNMEASURED on {arch}");
                let _ = writeln!(s, "    {why}");
                let _ = writeln!(
                    s,
                    "    unmeasured is not the same as unlimited; no bound is reported"
                );
            }
            Bounds::NoInput { arch, why } => {
                let _ = writeln!(s, "  bounds ({arch}): UNKNOWN — nothing to check");
                let _ = writeln!(s, "    {why}");
                let _ = writeln!(
                    s,
                    "    unknown is not ok: a check with no input has not passed, and a \
                     tiling\n    that was never examined must not read as one that was"
                );
            }
            Bounds::Checked { arch, results } => {
                let _ = write!(s, "  bounds ({arch}):");
                for (name, r) in results {
                    match r {
                        Ok(()) => {
                            let _ = write!(s, " {name} ok;");
                        }
                        Err(e) => {
                            let _ = write!(s, " {name} FAIL ({e});");
                        }
                    }
                }
                let _ = writeln!(s);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {

    /// #009. `bounds (n/a): UB ok; repeat ok; DMA stride ok;` was printed for a kernel
    /// whose tile plan said "not computable (no declared extent)" one line above: the
    /// checks ran against a DEFAULT tile size and all three answered ok.
    ///
    /// The issue's own words: "which is the worst of the three possible answers -- a
    /// planner reading this concludes the tiling is fine". There ARE three answers, and a
    /// check with no input has not passed.
    #[test]
    fn a_check_with_no_input_says_unknown_rather_than_ok() {
        // No extents anywhere: nothing declares how much data this moves.
        let f = forms::by_id("cpp").or_else(|| forms::by_id("msl")).unwrap();
        let p = profile(
            "k.cpp",
            "extern \"C\" void transpose_kernel(void* x, void* y) {}\n",
            f,
            How::Extension,
            "apple-gpu",
            8192,
        );
        assert!(
            matches!(p.bounds, Bounds::NoInput { .. }),
            "no declared extent must give NoInput, got {:?}",
            p.bounds
        );
        let out = p.render();
        assert!(out.contains("UNKNOWN"), "{out}");
        assert!(
            !out.contains("UB ok"),
            "a capacity check with no tile size must not report ok:\n{out}"
        );
        // And the reason is spelled out, because "unknown" beside "ok" elsewhere in the
        // same report is exactly what made the old line misleading.
        assert!(out.contains("unknown is not ok"), "{out}");
    }

    /// The other side of it: a source that DOES declare an extent must still be checked.
    /// A change that made everything answer "unknown" would satisfy the test above and destroy
    /// the feature.
    #[test]
    fn a_declared_extent_is_still_checked() {
        let mlir = "\
module {
  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %a, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
";
        let f = forms::by_id("msl").unwrap();
        let p = profile("k.mlir", mlir, f, How::Extension, "apple-gpu", 8192);
        assert!(
            matches!(p.bounds, Bounds::Checked { .. }),
            "a declared extent must still be checked, got {:?}",
            p.bounds
        );
    }
    use super::*;
    use crate::forms;

    const MLIR: &str = "\
module {
  func.func @softmax(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = arith.maximumf %arg0, %arg0 : tensor<1024xf32>
    %1 = arith.subf %arg0, %0 : tensor<1024xf32>
    %2 = math.exp %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
}
";

    #[test]
    fn a_comment_mentioning_an_entry_idiom_does_not_become_a_kernel() {
        let src = "// safe to mention __global__ and kernel void in a comment\n                   pub fn softmax_1d() {}\n";
        assert_eq!(kernel_names("tile", src), vec!["softmax_1d".to_string()]);
    }

    #[test]
    fn kernel_name_comes_from_func_func() {
        assert_eq!(kernel_names("mlir", MLIR), vec!["softmax".to_string()]);
    }

    #[test]
    fn a_tile_rs_kernel_declares_its_shape_in_the_view_type() {
        // `GmView<1, 1024, f32>` says the shape in the signature. A profiler that cannot
        // read it reports "not computable" about a kernel that stated its own extent.
        let src = "pub fn k(input: GmView<1, 1024, f32>, output: GmViewMut<1, 1024, f32>) {}";
        assert_eq!(extents_in(src), vec![1024]);
    }

    #[test]
    fn dtypes_and_extents_are_extracted() {
        assert_eq!(dtypes_in(MLIR), vec![ScalarType::F32]);
        assert_eq!(extents_in(MLIR), vec![1024]);
    }

    #[test]
    fn ssa_assignments_become_vector_ops_for_the_hazard_model() {
        let ops = vec_ops(MLIR);
        assert_eq!(ops.len(), 3, "three SSA assignments");
        // %1 reads %0, so there is a real dependence to find.
        let edges = hazards(&ops);
        assert!(!edges.is_empty(), "expected at least one hazard edge");
    }

    #[test]
    fn a_metal_target_has_no_ascend_limits_and_every_check_passes() {
        let f = forms::by_id("msl").unwrap();
        let p = profile("k.mlir", MLIR, f, How::Extension, "apple-gpu", 8192);
        match p.bounds {
            Bounds::Checked { results, .. } => {
                assert!(results.iter().all(|(_, r)| r.is_ok()), "{results:?}");
            }
            Bounds::NoInput { .. } => {
                panic!("this source declares an extent, so the checks have an input")
            }
            Bounds::Unmeasured { .. } => panic!("metal must not be unmeasured"),
        }
    }

    #[test]
    fn an_unmeasured_arch_refuses_instead_of_borrowing_910b_numbers() {
        let f = forms::by_id("cpp").unwrap();
        let p = profile("k.mlir", MLIR, f, How::Extension, "ascend-950", 8192);
        match p.bounds {
            Bounds::NoInput { .. } => {
                panic!("this source declares an extent, so the checks have an input")
            }
            Bounds::Unmeasured { arch, .. } => assert_eq!(arch, "DAV_3510"),
            Bounds::Checked { .. } => {
                panic!("arch35 has no measured capacity; it must refuse, not answer")
            }
        }
        let r = p.render();
        assert!(r.contains("UNMEASURED"), "{r}");
        assert!(r.contains("not the same as unlimited"), "{r}");
    }

    #[test]
    fn the_950_skus_refuse_and_do_not_invent_a_core_count() {
        // PR and DT share DAV_3510; neither may borrow 910B's 48 cores.
        for family in ["ascend-950", "ascend-950pr", "ascend-950dt"] {
            assert_eq!(cores_for(family), 0, "{family}");
            let f = forms::by_id("cpp").unwrap();
            let p = profile("k.mlir", MLIR, f, How::Extension, family, 8192);
            assert!(matches!(p.bounds, Bounds::Unmeasured { .. }), "{family}");
            let r = p.render();
            assert!(r.contains("UNKNOWN — core count"), "{family}: {r}");
            assert!(!r.contains("not computable"), "{family}: {r}");
        }
        // The DT SKU names its silicon in the refusal, so PR-measured numbers
        // cannot be mistaken for a DT measurement.
        let dt = params_for("ascend-950dt");
        assert_eq!(dt.chip, "Ascend950DT_9582");
        assert!(dt
            .check_vendor_op("Cat")
            .unwrap_err()
            .contains("no OP JSON"));
        let pr = params_for("ascend-950pr");
        assert_eq!(pr.chip, "Ascend950PR_9589");
        assert!(
            pr.check_vendor_op("Cat").is_ok(),
            "PR has the dyn-shape set"
        );
        // 910B still plans against a measured core count.
        assert_eq!(cores_for("ascend"), 48);
    }

    #[test]
    fn a_measured_ascend_answers_its_bounds() {
        let f = forms::by_id("cpp").unwrap();
        let p = profile("k.mlir", MLIR, f, How::Extension, "ascend", 8192);
        assert!(matches!(
            p.bounds,
            Bounds::Checked {
                arch: "DAV_2201",
                ..
            }
        ));
    }

    #[test]
    fn the_tile_plan_caps_blocks_at_the_core_count() {
        let f = forms::by_id("mlir").unwrap();
        let p = profile("k.mlir", MLIR, f, How::Extension, "ascend", 128);
        let plan = p.plan.expect("plan");
        assert_eq!(plan.full, 8);
        assert!(plan.blocks <= p.cores);
    }

    #[test]
    fn render_names_the_form_and_how_it_was_decided() {
        let f = forms::by_id("mlir").unwrap();
        let p = profile("k.mlir", MLIR, f, How::Magic, "none", 8192);
        let r = p.render();
        assert!(r.contains("form: mlir"), "{r}");
        assert!(r.contains("sniffed"), "{r}");
        assert!(r.contains("tile plan:"), "{r}");
        assert!(r.contains("hazards:"), "{r}");
    }
}
