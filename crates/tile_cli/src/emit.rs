//! Emit: MLIR text in, target source out.
//!
//! Every emitter here is a **pure function of its input** — the same contract
//! `tile_spec`'s `backend_emit_purity.feature` holds them to, and the reason `-O0`
//! through `-O2` can promise byte-identical output forever. No LLVM, no toolchain, no
//! network: this is string in, string out, in process.
//!
//! Without the `emitters` feature the dispatch still exists and still names every form —
//! it simply reports that this build cannot take the route. That distinction is the
//! whole point of exit code 4: the capability exists, just not here.

use crate::forms::Form;

/// Why an emit could not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// The emitter ran and refused the input.
    Rejected(String),
    /// This build has no emitter for the form, though tile-rs does.
    NotCompiledIn {
        form: &'static str,
        feature: &'static str,
    },
    /// No emitter exists anywhere for this form.
    NoEmitter { form: &'static str },
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Rejected(e) => write!(f, "{e}"),
            EmitError::NotCompiledIn { form, feature } => write!(
                f,
                "this build has no {form} emitter; rebuild with --features {feature}"
            ),
            EmitError::NoEmitter { form } => {
                write!(f, "tile-rs has no emitter for {form}")
            }
        }
    }
}

/// Is there an emitter for this form in THIS build?
pub fn have_emitter(form: &str) -> bool {
    cfg!(feature = "emitters") && emitter_exists(form)
}

/// Does tile-rs have an emitter for this form at all, in any build?
pub fn emitter_exists(form: &str) -> bool {
    if form == "pico" {
        // The 16th target lives in a sibling checkout, so it exists only where that
        // checkout was compiled in.
        return cfg!(feature = "pico");
    }
    matches!(
        form,
        "gpu"
            | "musa"
            | "spirv"
            | "msl"
            | "nki"
            | "aie"
            | "bang"
            | "gaudi"
            | "tpu"
            | "csl"
            | "hexagon"
            | "ttmetal"
            | "linalg"
            | "rvv"
            | "pto"
    )
}

/// Emit `mlir` as `form`'s target source.
pub fn emit(form: &Form, mlir: &str) -> Result<String, EmitError> {
    if mlir.trim().is_empty() {
        // The emitters' own contract, surfaced early so every form fails the same way.
        return Err(EmitError::Rejected("empty MLIR module".into()));
    }
    if !emitter_exists(form.id) {
        return Err(EmitError::NoEmitter { form: form.id });
    }
    #[cfg(feature = "pico")]
    if form.id == "pico" {
        return crate::mlir_to_pico::convert_mlir_to_pico(mlir).map_err(EmitError::Rejected);
    }
    #[cfg(not(feature = "emitters"))]
    {
        Err(EmitError::NotCompiledIn {
            form: form.id,
            feature: "emitters",
        })
    }
    #[cfg(feature = "emitters")]
    {
        dispatch(form.id, mlir).map_err(EmitError::Rejected)
    }
}

/// Emit for a FORM, by the same path `emit` takes.
///
/// `dispatch` is keyed on the form id and does not know about `pico`, which `emit`
/// handles ahead of it because its emitter lives in a sibling checkout. Anything that
/// reasons about emitter behaviour must go through here instead, or it silently sees
/// "unknown target 'pico'" and concludes the backend does nothing.
///
/// That was not hypothetical: `ignored_intrinsics` called `dispatch` directly, so the
/// silent-copy detector had never once examined pico -- its baseline always errored and
/// it returned "nothing ignored" for every module. And the op-coverage report, written
/// on top of it, put a column of dashes under a backend that lowers eighteen of the
/// eighteen ops it was being asked about.
#[cfg(feature = "emitters")]
pub fn emit_for(form: &Form, mlir: &str) -> Result<String, String> {
    #[cfg(feature = "pico")]
    if form.id == "pico" {
        return crate::mlir_to_pico::convert_mlir_to_pico(mlir);
    }
    dispatch(form.id, mlir)
}

#[cfg(not(feature = "emitters"))]
pub fn emit_for(form: &Form, _mlir: &str) -> Result<String, String> {
    Err(format!(
        "this build has no {} emitter; rebuild with --features emitters",
        form.id
    ))
}

#[cfg(feature = "emitters")]
fn dispatch(form: &str, mlir: &str) -> Result<String, String> {
    match form {
        "gpu" => crate::mlir_to_gpu::convert_mlir_to_gpu(mlir),
        "musa" => crate::mlir_to_musa::convert_mlir_to_musa(mlir),
        "spirv" => crate::mlir_to_spirv::convert_mlir_to_spirv(mlir),
        "msl" => crate::mlir_to_msl::convert_mlir_to_msl(mlir),
        "nki" => crate::mlir_to_nki::convert_mlir_to_nki(mlir),
        "aie" => crate::mlir_to_aie::convert_mlir_to_aie(mlir),
        "bang" => crate::mlir_to_bang::convert_mlir_to_bang(mlir),
        "gaudi" => crate::mlir_to_gaudi::convert_mlir_to_gaudi(mlir),
        "tpu" => crate::mlir_to_tpu::convert_mlir_to_tpu(mlir),
        "csl" => crate::mlir_to_csl::convert_mlir_to_csl(mlir),
        "hexagon" => crate::mlir_to_hexagon::convert_mlir_to_hexagon(mlir),
        "ttmetal" => crate::mlir_to_ttmetal::convert_mlir_to_ttmetal(mlir),
        "linalg" => crate::mlir_to_linalg::convert_mlir_to_linalg(mlir),
        "rvv" => crate::mlir_to_rvv::convert_mlir_to_rvv(mlir),
        "pto" => crate::mlir_to_pto::convert_mlir_to_pto(mlir),
        other => Err(format!("unknown target '{other}'")),
    }
}

/// An intrinsic the emitter produced no output for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ignored {
    pub intrinsic: String,
    /// How many times it appears in the module.
    pub occurrences: usize,
}

/// Which of this module's tile intrinsics did the emitter actually use?
///
/// ## Why this exists
///
/// Lowering is a **partial** function that is written as a total one. An emitter's
/// dispatch ends in a default arm that produces output rather than refusing, so an
/// intrinsic with no arm yields a kernel that compiles, passes the purity check, and
/// computes something else entirely. Backlog #005 caught `__tile_matmul_f32` emitting a
/// plain copy: three buffers in, `p1[gid] = p0[gid]` out, exit 0, fidelity reported as
/// "exact, validated on Apple GPU".
///
/// `tile_spec`'s generality matrix cannot catch this, because it only ever feeds
/// canonical snippets — every arm it exercises is one that exists.
///
/// ## The check
///
/// Emit twice: once as written, once with the intrinsic's **name** replaced by one no
/// emitter can possibly handle. The module's shape, operands and types are untouched, so
/// the only thing that changed is which arm can fire. **If the output is byte-identical,
/// the emitter did not use that intrinsic** — whatever it produced, it produced without
/// reference to the operation the kernel is made of.
///
/// This does not need the emitters to change, which matters: they live in a tree this
/// crate `#[path]`-includes and does not own.
#[cfg(feature = "emitters")]
/// Do two emitter outputs say the same thing, once the intrinsic's name is normalised?
///
/// Pulled out so it can be tested without an emitter that misbehaves on demand. The four
/// that did -- gpu, nki, tpu and bang -- now refuse outright, which is the better outcome
/// and also means the regression they demonstrated can no longer be reproduced through
/// them. The logic still has to be guarded for the next emitter that quotes what it drops.
pub fn outputs_agree_modulo_name(a: &str, b: &str, name: &str, probe: &str) -> bool {
    a.replace(name, probe) == *b
}

/// Tile-level ops `tile_std` declares with the `(dst, src, rows, cols)` signature.
///
/// Derived from `crates/tile_std/src/tile.rs`, not invented here, and
/// `every_probe_op_is_actually_declared` parses that file and fails if the two drift. A
/// coverage report built from a stale list would be the same class of untruth as the
/// silent copies it exists to expose.
///
/// Semantically these are not all elementwise -- `transpose`, `reduce_max` and `argmax`
/// change the shape of what they produce. That does not matter to this probe: the question
/// is whether the emitter has an arm for the intrinsic, not what the arm means.
/// How a probe kernel calls an op, because they do not all take the same arguments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProbeShape {
    /// `__tile_<op>_f32(dst, src, rows, cols)` — one input tile.
    Unary,
    /// `__tile_<op>_f32(dst, a, b, rows, cols)` — two input tiles, same extent.
    Binary,
    /// `__tile_<op>_f32(dst, a, b, m, k, n)` — three dimensions.
    Matmul,
}

/// The ops the coverage probe asks every backend about, and how to call each.
///
/// Chosen by a rule, not by taste: every `tile_std` intrinsic whose declaration matches one
/// of the three shapes above. Anything else — an attention kernel's dozen strides, a
/// quantized matvec's block layout — needs a template of its own and is honestly absent
/// rather than silently counted, which is what the note under the table says.
///
/// The shapes are also the boundary of what this report is FOR, and that is a decision
/// rather than an accident. `tile_std`'s most common signature is not one of these three:
/// 26 intrinsics take `(src, dst, ne0, nb_src, nb_dst)`, the DS4 `_scalar` family ported
/// from one backend's model work. `_f32_scalar` appears 61 times in `mlir_to_msl` and zero
/// times in every other emitter. Probing them would add twenty-six rows reading `yes` under
/// msl and `-` under all ten others: true, and it would bury the portability signal this
/// table exists to carry under ops that were never meant to be portable. They are out of
/// scope, said out loud, rather than left as an oversight for someone to "complete".
///
/// It began as 18 ops of one shape under the name PROBE_UNARY_OPS, which was three short of
/// even that shape and contained `argmax`, `transpose` and `softmax`. Widening it to the
/// rule found kv_cache_update missing from eight backends; adding the other two shapes puts
/// max, min, matvec and MATMUL — the headline op — into a report that had never asked about
/// any of them.
pub const PROBE_OPS: &[(&str, ProbeShape)] = &[
    ("abs", ProbeShape::Unary),
    ("absmax", ProbeShape::Unary),
    ("argmax", ProbeShape::Unary),
    ("cast_f16", ProbeShape::Unary),
    ("causal_mask", ProbeShape::Unary),
    ("exp", ProbeShape::Unary),
    ("kv_cache_update", ProbeShape::Unary),
    ("kv_cache_update_prefill", ProbeShape::Unary),
    ("log", ProbeShape::Unary),
    ("neg", ProbeShape::Unary),
    ("reduce_max", ProbeShape::Unary),
    ("reduce_sum", ProbeShape::Unary),
    ("relu", ProbeShape::Unary),
    ("rsqrt", ProbeShape::Unary),
    ("sigmoid", ProbeShape::Unary),
    ("silu", ProbeShape::Unary),
    ("softmax", ProbeShape::Unary),
    ("softplus", ProbeShape::Unary),
    ("sqrt", ProbeShape::Unary),
    ("tanh", ProbeShape::Unary),
    ("transpose", ProbeShape::Unary),
    ("max", ProbeShape::Binary),
    ("min", ProbeShape::Binary),
    ("matvec", ProbeShape::Binary),
    ("matmul", ProbeShape::Matmul),
    ("matmul_transposed", ProbeShape::Matmul),
    ("argmin", ProbeShape::Unary),
    ("cast_bf16", ProbeShape::Unary),
    ("cast_i8", ProbeShape::Unary),
    ("cvt_f16", ProbeShape::Unary),
    ("init_sort_buf", ProbeShape::Unary),
    ("mxfp4_value", ProbeShape::Unary),
];
/// A minimal kernel whose only compute op is `__tile_<op>_f32`.
///
/// 64x64 rather than 1024x1024, and the difference is not cosmetic. A 1024x1024 f32 tile
/// is 4 MB, and `pto` enforces the Ascend Unified Buffer bound of 256 KB -- so every probe
/// came back refused and the report scored that backend 0 of 18. It was behaving
/// correctly and my measurement was wrong, which is the same shape as everything else
/// this file exists to catch: a plausible-looking number meaning something other than
/// what it says. 64x64 is 16 KB and fits every target's budget.
pub fn probe_module(op: &str, shape: ProbeShape) -> String {
    let head = "module {\n        llvm.func @probe(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>) attributes {hacc.entry} {\n        ^bb0:\n        %n = llvm.mlir.constant(64 : i32) : i32\n        %x = llvm.call @__tile_load_f32(%a, %n, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n";
    // The second input is loaded from the SAME buffer on purpose. The probe asks whether
    // an emitter has an arm, not whether it computes correctly -- that is what `-r` and
    // the three-way comparison are for -- so a distinct buffer would buy nothing and cost
    // a third pointer argument in every kernel signature.
    let call = match shape {
        ProbeShape::Unary => format!(
            "%y = llvm.call @__tile_{op}_f32(%x, %x, %n, %n) : (i32, i32, i32, i32) -> i32\n"
        ),
        ProbeShape::Binary => format!(
            "%y = llvm.call @__tile_{op}_f32(%x, %x, %x, %n, %n) :              (i32, i32, i32, i32, i32) -> i32\n"
        ),
        ProbeShape::Matmul => format!(
            "%y = llvm.call @__tile_{op}_f32(%x, %x, %x, %n, %n, %n) :              (i32, i32, i32, i32, i32, i32) -> i32\n"
        ),
    };
    format!(
        "{head}         {call}                  llvm.call @__tile_store_f32(%b, %y, %n, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n         llvm.return\n}}\n}}\n"
    )
}

/// Does `form`'s emitter lower `op`?
///
/// Measured by asking it, not by reading a table somebody maintains by hand. Only
/// meaningful because a backend that cannot lower an op now REFUSES: while a gap emitted
/// a plausible copy, every cell of this matrix would have said yes.
pub fn lowers_op(form: &Form, op: &str, shape: ProbeShape) -> bool {
    let mlir = probe_module(op, shape);
    match emit_for(form, &mlir) {
        Err(_) => false,
        // Emitted, but did it use the op? An emitter that produced output without
        // touching the intrinsic has not lowered it, whatever the exit code says.
        Ok(_) => ignored_intrinsics(form, &mlir).is_empty(),
    }
}

/// The op-support matrix, as text.
pub fn coverage_report() -> String {
    let forms: Vec<&Form> = crate::forms::FORMS
        .iter()
        .filter(|f| f.writable && emitter_exists(f.id) && f.id != "debug")
        .collect();

    // Width from the longest name actually in the set, not a constant that happened to fit
    // when the set was shorter. Adding kv_cache_update_prefill to the probe ran the name
    // straight into the first column: "kv_cache_updateyes". The +2 is the gutter.
    let name_w = PROBE_OPS
        .iter()
        .map(|(o, _)| o.len())
        .chain(std::iter::once("total".len()))
        .max()
        .unwrap_or(14)
        + 2;

    let mut s = String::new();
    s.push_str(&format!("{:<name_w$}", "op"));
    for f in &forms {
        s.push_str(&format!("{:<9}", f.id));
    }
    s.push('\n');
    s.push_str(&"-".repeat(name_w + 9 * forms.len()));
    s.push('\n');

    let mut totals = vec![0usize; forms.len()];
    for (op, shape) in PROBE_OPS {
        s.push_str(&format!("{op:<name_w$}"));
        for (i, f) in forms.iter().enumerate() {
            let ok = lowers_op(f, op, *shape);
            if ok {
                totals[i] += 1;
            }
            s.push_str(&format!("{:<9}", if ok { "yes" } else { "-" }));
        }
        s.push('\n');
    }
    s.push_str(&"-".repeat(name_w + 9 * forms.len()));
    s.push('\n');
    s.push_str(&format!("{:<name_w$}", "total"));
    for t in &totals {
        s.push_str(&format!("{:<9}", format!("{t}/{}", PROBE_OPS.len())));
    }
    s.push('\n');
    s.push_str(
        "\nMeasured by lowering a probe kernel for each op, not read from a table. A dash \
         means\nthe backend refuses that op or emits without using it -- it is a gap that \
         ANNOUNCES\nitself, which is the property the emitters are held to. It does not \
         mean the op is\nimpossible there, only that nobody has written the arm.\n",
    );
    // What the denominator is, said out loud. `18/18` reads as "everything", and it is
    // not: the row set is this probe's list, while tile_std declares intrinsics by the
    // couple of hundred. A backend at full marks here has covered the ops that were
    // asked about -- which is a real and useful claim, and a smaller one than the score
    // looks. The number is taken from the list rather than written down beside it,
    // because a count restated in prose is the thing that drifts.
    s.push_str(&format!(
        "\nThe denominator is this probe's {} ops, NOT the whole intrinsic surface, which \
         is much\nlarger. {}/{} means every op ASKED ABOUT lowers, not that the backend \
         lowers everything.\nThe set is the portable tile-level ops; single-backend kernel \
         families (the DS4\n`_scalar` group and friends) are deliberately outside it, not \
         overlooked.\n",
        PROBE_OPS.len(),
        PROBE_OPS.len(),
        PROBE_OPS.len()
    ));
    s
}

#[cfg(feature = "emitters")]
pub fn ignored_intrinsics(form: &Form, mlir: &str) -> Vec<Ignored> {
    let baseline = match emit_for(form, mlir) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let module = crate::mlir::parse(mlir);
    let mut names: Vec<String> = Vec::new();
    for o in module.ops() {
        if let Some(n) = o.intrinsic() {
            // Loads and stores are plumbing; every emitter handles them, and a kernel
            // that is only a load and a store legitimately IS a copy.
            if n.contains("load") || n.contains("store") {
                continue;
            }
            if !names.iter().any(|x| x == n) {
                names.push(n.to_string());
            }
        }
    }

    let mut out = Vec::new();
    for name in names {
        let occurrences = mlir.matches(&name).count();
        // A name no arm can match, of the same shape so nothing else shifts.
        const PROBE: &str = "__tile_unhandled_probe";
        let masked = mlir.replace(&name, PROBE);
        // Only the Ok case is interesting. An ERROR under masking means the emitter DID
        // depend on the name, which is the healthy answer and needs no branch -- it used to
        // be an explicit `Err(_) => {}` arm carrying this comment, which is the only thing
        // that arm was for.
        if let Ok(s) = emit_for(form, &masked) {
            // Compare with the NAME ITSELF normalised out of both outputs.
            //
            // A straight equality test has a false negative, and a real emitter hits
            // it: `mlir_to_gpu` writes `// TODO: unhandled intrinsic: <name>` and then
            // emits nothing for the op. Renaming therefore changes the output -- by
            // exactly the one word inside a comment -- and the detector concluded the
            // emitter had depended on the name. It had not; it had only quoted it
            // while ignoring it, which is the case this check exists to find.
            //
            // Normalising both sides means "identical apart from where the name is
            // echoed", which is the question actually being asked.
            if outputs_agree_modulo_name(&baseline, &s, &name, PROBE) {
                out.push(Ignored {
                    intrinsic: name,
                    occurrences,
                });
            }
        }
    }
    out
}

#[cfg(not(feature = "emitters"))]
pub fn ignored_intrinsics(_form: &Form, _mlir: &str) -> Vec<Ignored> {
    Vec::new()
}

impl std::fmt::Display for Ignored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} appears {} time(s) and changed nothing in the output — the emitter has no \
             arm for it and fell through",
            self.intrinsic, self.occurrences
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms;

    /// A kernel whose only compute op is an intrinsic NOTHING implements and nothing will.
    ///
    /// Deliberately synthetic. This was `__tile_relu_f32` until msl grew an arm for it,
    /// at which point both tests using it started failing -- for the third time in this
    /// file, a fix removed the witness that demonstrated the bug. A real op is a moving
    /// target: anyone may implement it tomorrow, and the test silently stops testing.
    const UNHANDLED_KERNEL: &str = "module {\n\
      llvm.func @k(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>) attributes {hacc.entry} {\n\
      ^bb0:\n\
      %n = llvm.mlir.constant(1024 : i32) : i32\n\
      %x = llvm.call @__tile_load_f32(%a, %n, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
      %y = llvm.call @__tile_frobnicate_f32(%x, %n, %n) : (i32, i32, i32) -> i32\n\
      llvm.call @__tile_store_f32(%b, %y, %n, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
      llvm.return\n}\n}\n";

    const MATMUL_F32: &str = "\
module {
  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %m = llvm.mlir.constant(64 : i32) : i32
    %k = llvm.mlir.constant(128 : i32) : i32
    %n = llvm.mlir.constant(64 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
";

    #[cfg(feature = "emitters")]
    #[test]
    fn the_msl_emitter_now_refuses_rather_than_needing_this_detector() {
        // Backlog #005, and the reason this test changed shape.
        //
        // It used to assert that `ignored_intrinsics` DETECTED `__tile_matmul_f32`
        // emitting a plain copy: three buffers in, `p1[gid] = p0[gid]` out, exit 0. The
        // detector worked by renaming the intrinsic and observing that the output did not
        // change -- proof the emitter never looked at it.
        //
        // The msl emitter now has a refusing default arm, so that module does not lower
        // at all. The detector's whole technique needs a BASELINE to compare against, and
        // there is no longer one to get. That is a strictly better outcome than detecting
        // the copy after the fact, so this asserts the refusal instead.
        // Anchored to an intrinsic that will never exist, and that is the point.
        //
        // This asserted a refusal of __tile_matmul_f32, then of __tile_matvec_f32, and
        // both were implemented within the hour -- msl is 25 of 25 on the probe now, so
        // there is no real op left to refuse. A guard on the REFUSING DEFAULT ARM should
        // not be a hostage to coverage progress: every time the gap it used got filled,
        // the guard broke and the temptation was to delete it rather than move it. A name
        // nothing declares cannot be implemented out from under it.
        let unknown = MATMUL_F32
            .replace("__tile_matmul_f32", "__tile_frobnicate_f32")
            .replace(
                "(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32)",
                "(%a, %a, %b, %m, %n) : (i32, i32, i32, i32, i32)",
            );
        let msl = forms::by_id("msl").unwrap();
        let err = dispatch(msl.id, &unknown).expect_err("must not lower to a copy");
        assert!(err.contains("__tile_frobnicate_f32"), "{err}");
        assert!(
            err.to_lowercase().contains("copy"),
            "does not say what it avoided: {err}"
        );
        // And with no baseline, the detector reports nothing -- correctly. A caller only
        // reaches it after a conversion SUCCEEDED, so this state is unreachable in the
        // tool; asserted here so the two mechanisms cannot silently disagree.
        assert!(ignored_intrinsics(msl, &unknown).is_empty());
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn an_emitter_that_only_quotes_the_intrinsic_does_not_escape_the_detector() {
        // Tested against the comparison itself rather than a live emitter.
        //
        // The four that demonstrated this -- gpu, nki, tpu, bang -- now refuse outright,
        // so the bug can no longer be reproduced through them. That is the better
        // outcome and it removes the witness, which is exactly when a regression creeps
        // back in.
        //
        // The shape that escaped: output identical except where the emitter echoed the
        // name it was dropping.
        let base = "// TODO: unhandled intrinsic: __tile_frobnicate_f32\n    p1[i] = 0;\n";
        let renamed = "// TODO: unhandled intrinsic: __tile_unhandled_probe\n    p1[i] = 0;\n";
        assert!(
            outputs_agree_modulo_name(
                base,
                renamed,
                "__tile_frobnicate_f32",
                "__tile_unhandled_probe"
            ),
            "an emitter that only quotes the name would escape again"
        );
        // And a real difference must still read as a real difference, or the detector
        // would flag every working lowering.
        let different =
            "// TODO: unhandled intrinsic: __tile_unhandled_probe\n    p1[i] = frob(x);\n";
        assert!(!outputs_agree_modulo_name(
            base,
            different,
            "__tile_frobnicate_f32",
            "__tile_unhandled_probe"
        ));
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn the_emitters_that_quoted_what_they_dropped_now_refuse() {
        // The false negative this closes, and it was a real emitter rather than a
        // hypothetical: `mlir_to_gpu` writes
        //
        //     // TODO: unhandled intrinsic: __tile_relu_f32
        //     p1[goff] = %y;
        //
        // and emits nothing for the op -- it even leaks the MLIR SSA name into CUDA, so
        // the result cannot compile. But renaming the intrinsic CHANGED the output, by
        // exactly the one word inside that comment, so a straight equality test concluded
        // the emitter had depended on the name. It had only quoted it while ignoring it,
        // which is precisely the case this check exists to find.
        //
        // Six emitters escaped that way. Normalising the name out of both sides before
        // comparing brought all six back.
        // gpu wrote `p1[goff] = %y;` -- an MLIR SSA name in CUDA -- and nki, tpu and
        // bang did the same into Python and C. All four refuse now, and refusing means
        // no file is written at all, so there is nothing to leak.
        for id in ["gpu", "nki", "tpu", "bang"] {
            let f = forms::by_id(id).unwrap();
            if !emitter_exists(id) {
                continue;
            }
            let err = dispatch(id, UNHANDLED_KERNEL)
                .expect_err(&format!("{id} must refuse an intrinsic it cannot lower"));
            assert!(err.contains("__tile_frobnicate_f32"), "{id}: {err}");
            let _ = f;
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn no_emitted_target_source_contains_an_mlir_ssa_name() {
        // `mlir_to_gpu` wrote `p1[goff] = %y;` into its CUDA: the MLIR SSA name of an op
        // it had dropped, in a language with no such syntax. It could not compile, and it
        // was written with exit 0 and reported as [exact, validated on NVIDIA].
        //
        // The narrow fix is that the emitter refuses. This is the general property: an
        // SSA name from the INPUT must never survive into target source. Checked against
        // the names actually present rather than by pattern-matching `%`, which is a
        // modulo operator in most of these languages and would false-positive forever.
        let module = crate::mlir::parse(UNHANDLED_KERNEL);
        let mut ssa: Vec<String> = Vec::new();
        for o in module.ops() {
            if let Some(r) = &o.result {
                if r.len() > 1 {
                    ssa.push(r.clone());
                }
            }
        }
        assert!(!ssa.is_empty(), "the fixture has no SSA results to leak");
        for f in forms::FORMS {
            if !f.writable || !emitter_exists(f.id) {
                continue;
            }
            let Ok(out) = dispatch(f.id, UNHANDLED_KERNEL) else {
                continue; // refused: nothing was written, so nothing leaked
            };
            // Comments may quote the source -- pico's listing documents each op as
            // `; [1] relu %y <- %x`, which is a record of what it lowered, not a leak.
            // Code may not: `%y` in CUDA, Python or C is a parse error.
            for line in out.lines() {
                let t = line.trim_start();
                if t.starts_with("//") || t.starts_with('#') || t.starts_with(';') {
                    continue;
                }
                for name in &ssa {
                    assert!(
                        !line.contains(name.as_str()),
                        "{} leaked the MLIR SSA name {name} into code: {line}",
                        f.id
                    );
                }
            }
        }
    }

    /// A minimal kernel whose only compute op is the named tile-level unary intrinsic.
    fn unary_module(op: &str) -> String {
        format!(
            "module {{\n\
             llvm.func @k(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n\
             ^bb0:\n\
             %n = llvm.mlir.constant(1024 : i32) : i32\n\
             %x = llvm.call @__tile_load_f32(%a, %n, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             %y = llvm.call @__tile_{op}_f32(%x, %x, %n, %n) : (i32, i32, i32, i32) -> i32\n\
             llvm.call @__tile_store_f32(%b, %y, %n, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             llvm.return\n}}\n}}\n"
        )
    }

    /// Tile-level unary ops `tile_std` declares. Each is `(dst, src, rows, cols)`.
    const DECLARED_UNARY: &[&str] = &[
        "relu", "tanh", "abs", "sqrt", "softplus", "sigmoid", "exp", "log", "neg", "silu",
    ];

    #[test]
    fn every_probe_op_is_actually_declared_by_tile_std() {
        // The coverage report is only worth reading if its rows are real ops. A stale
        // list would report gaps that do not exist, or miss ones that do -- the same
        // class of untruth as the silent copies this file exists to expose.
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tile_std/src/tile.rs"),
        )
        .expect("tile_std source");
        for (op, shape) in PROBE_OPS {
            // Checked against the shape the probe will actually CALL it with. When this
            // only knew the unary signature it could not have caught the entry
            // "matvec_f32", which the template turns into __tile_matvec_f32_f32 -- a name
            // nothing declares, so every backend "failed" to lower it and the row read
            // 0 of 11 for an op that linalg and rvv both implement.
            let decl = match shape {
                ProbeShape::Unary => {
                    format!("fn __tile_{op}_f32(dst: u32, src: u32, rows: u32, cols: u32)")
                }
                ProbeShape::Binary => {
                    format!("fn __tile_{op}_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32)")
                }
                ProbeShape::Matmul => {
                    format!("fn __tile_{op}_f32(dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32)")
                }
            };
            assert!(
                src.contains(&decl),
                "PROBE_OPS lists {op} as {shape:?}, which tile_std does not declare with \
                 that signature. The probe would call __tile_{op}_f32 and get a name \
                 nothing defines, scoring every backend a false gap."
            );
        }
    }

    /// Every op of a probeable shape is probed, not just the ones somebody remembered.
    ///
    /// The list was 18 of the 21 unary-shaped intrinsics, missing cast_f16 and both
    /// kv_cache_update variants for no recorded reason -- and kv_cache_update turned out to
    /// be a gap in eight backends that the report had never asked about. A set defined by a
    /// rule has to be checked against the rule, or it drifts back to a list.
    #[test]
    fn every_probeable_intrinsic_is_in_the_probe_set() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tile_std/src/tile.rs"),
        )
        .expect("tile_std source");
        let shapes = [
            (
                ProbeShape::Unary,
                "dst: u32, src: u32, rows: u32, cols: u32",
            ),
            (
                ProbeShape::Binary,
                "dst: u32, a: u32, b: u32, rows: u32, cols: u32",
            ),
            (
                ProbeShape::Matmul,
                "dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32",
            ),
        ];
        for (shape, args) in shapes {
            let needle = format!("({args})");
            for line in src.lines() {
                let Some(rest) = line.trim().strip_prefix("pub fn __tile_") else {
                    continue;
                };
                if !rest.contains(&needle) {
                    continue;
                }
                let Some(name) = rest.split('(').next() else {
                    continue;
                };
                let Some(op) = name.strip_suffix("_f32") else {
                    continue;
                };
                assert!(
                    PROBE_OPS.iter().any(|(o, s)| o == &op && *s == shape),
                    "tile_std declares __tile_{op}_f32 with the {shape:?} shape and the \
                     probe never asks about it, so no backend's gap in it is visible"
                );
            }
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn the_coverage_probe_reaches_pico_like_every_other_backend() {
        // pico is emitted ahead of `dispatch`, which is keyed on the form id and does not
        // know it. Anything calling `dispatch` directly sees "unknown target 'pico'" and
        // concludes the backend does nothing -- which is what the first version of the
        // report said, and what `ignored_intrinsics` had been quietly assuming since it
        // was written.
        let pico = forms::by_id("pico").unwrap();
        if !emitter_exists("pico") {
            return;
        }
        let lowered = PROBE_OPS
            .iter()
            .filter(|(op, shape)| lowers_op(pico, op, *shape))
            .count();
        assert!(
            lowered > 0,
            "the probe cannot reach pico: it reports {lowered} of {} ops",
            PROBE_OPS.len()
        );
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn the_probe_fits_every_backends_memory_budget() {
        // A 1024x1024 f32 tile is 4 MB and pto enforces the Ascend Unified Buffer bound
        // of 256 KB, so every probe came back refused and the report scored it 0 of 18.
        // The backend was right and the measurement was wrong.
        let m = probe_module("exp", ProbeShape::Unary);
        assert!(m.contains("constant(64 : i32)"), "the probe tile grew: {m}");
        let pto = forms::by_id("pto").unwrap();
        if emitter_exists("pto") {
            assert!(
                lowers_op(pto, "exp", ProbeShape::Unary),
                "pto refuses the probe again -- check the tile size against its UB budget"
            );
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn every_declared_unary_op_is_lowered_or_refused_on_every_backend() {
        // The generalisation of the single-fixture test below, and the reason it matters:
        // `tile_std` DECLARES more than any backend implements. Of these ten, Metal has
        // six, CUDA four, PICO eight. Every gap used to emit a silent copy, which is why
        // nobody had counted them.
        //
        // The property is not "every backend implements every op" -- that is false and
        // will stay false. It is that a gap must ANNOUNCE itself: lower it, or refuse.
        // Emitting a plausible kernel for an op you do not implement is the one outcome
        // that is never acceptable.
        for op in DECLARED_UNARY {
            let mlir = unary_module(op);
            for f in forms::FORMS {
                if !f.writable || !emitter_exists(f.id) || f.id == "debug" {
                    continue;
                }
                let Ok(out) = dispatch(f.id, &mlir) else {
                    continue; // refused: the honest answer for an op it lacks
                };
                // It lowered. The only question left is whether it USED the intrinsic,
                // and the detector is what answers that -- not a search for the op's
                // name in the output.
                //
                // My first version looked for the name and got this wrong immediately:
                // `mlir_to_linalg` lowers relu as `arith.maximumf %in, %zero`, which is
                // exactly right and contains "relu" nowhere. A correct lowering carries
                // the op's SEMANTICS, not its spelling, so the name test failed a
                // backend that was doing its job.
                let ignored = ignored_intrinsics(f, &mlir);
                assert!(
                    ignored.is_empty(),
                    "{} emitted {} bytes for __tile_{op}_f32 without using it: {ignored:?}",
                    f.id,
                    out.len()
                );
            }
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn no_emitter_can_silently_ignore_an_intrinsic_any_more() {
        // The property this milestone is actually about. For every writable form, an
        // intrinsic no emitter handles must produce one of:
        //
        //   * a refusal at emit time (msl has the arm), or
        //   * a detector hit, or
        //   * a real lowering (pico folds relu to vvmax and says so in its manifest).
        //
        // What must not happen is output with nothing said, which is what six of these
        // did before the normalisation fix above.
        for f in forms::FORMS {
            if !f.writable || !emitter_exists(f.id) || f.id == "debug" {
                continue;
            }
            match dispatch(f.id, UNHANDLED_KERNEL) {
                // Refused outright: the strongest answer.
                Err(_) => {}
                Ok(out) => {
                    let flagged = !ignored_intrinsics(f, UNHANDLED_KERNEL).is_empty();
                    // Not flagged is only acceptable if the emitter really lowered it.
                    // `vvmax` is relu on PICO; a form that neither refuses, nor is
                    // flagged, nor shows the op, is the silent case.
                    let handled = out.contains("vvmax") || out.contains("relu");
                    assert!(
                        flagged || handled,
                        "{} emitted {} bytes for an unhandled intrinsic and nothing said so",
                        f.id,
                        out.len()
                    );
                }
            }
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn the_detector_still_guards_the_emitters_that_have_no_refusing_arm() {
        // Only `msl` grew the arm. Fifteen other emitters can still fall through, and
        // this detector is what stands between them and a silent wrong kernel -- so it
        // must keep working, and must keep NOT firing on a lowering that is real.
        //
        // aie is checked rather than msl precisely because it has no arm yet: a healthy
        // answer here is the detector doing its job on an emitter that needs it.
        let aie = forms::by_id("aie").unwrap();
        let f16 = MATMUL_F32.replace("f32", "f16");
        if dispatch(aie.id, &f16).is_ok() {
            // A real lowering must not be flagged; that is the half that makes the
            // detector usable rather than noise.
            assert!(
                ignored_intrinsics(aie, &f16).is_empty(),
                "a working lowering was flagged as ignored"
            );
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn an_intrinsic_the_emitter_handles_is_not_flagged() {
        // The same module in f16 lowers to a real matmul, so renaming the intrinsic must
        // change the output. A check that flagged this too would be useless.
        let msl = forms::by_id("msl").unwrap();
        let f16 = MATMUL_F32.replace("f32", "f16");
        assert!(
            ignored_intrinsics(msl, &f16).is_empty(),
            "a working lowering was flagged as ignored"
        );
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn a_kernel_that_is_only_a_load_and_a_store_is_not_flagged() {
        // Plumbing is not an operation. A kernel that legitimately IS a copy must not be
        // reported as one that fell through to a copy.
        let msl = forms::by_id("msl").unwrap();
        let src = "\
module {
  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(64 : i32) : i32
    %t = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
";
        assert!(ignored_intrinsics(msl, src).is_empty());
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn the_canonical_softmax_is_fully_covered() {
        let msl = forms::by_id("msl").unwrap();
        let src = include_str!("../testdata/forms/softmax.mlir");
        assert!(
            ignored_intrinsics(msl, src).is_empty(),
            "the corpus kernel is not fully lowered"
        );
    }

    #[test]
    fn an_empty_module_is_rejected_by_every_form_identically() {
        for id in ["msl", "gpu", "linalg"] {
            let f = forms::by_id(id).unwrap();
            assert_eq!(
                emit(f, "   \n  "),
                Err(EmitError::Rejected("empty MLIR module".into())),
                "{id} accepted an empty module"
            );
        }
    }

    #[test]
    fn a_form_with_no_emitter_anywhere_says_so_distinctly() {
        let f = forms::by_id("tile").unwrap();
        assert_eq!(
            emit(f, "module {}"),
            Err(EmitError::NoEmitter { form: "tile" }),
            "tile-rs is not an emit target"
        );
    }

    #[test]
    fn every_lowering_edge_either_has_an_emitter_or_names_the_feature_gating_it() {
        // The invariant is NOT "every target has an open emitter" — `cpp` (AscendC) is a
        // closed target whose edge is real and whose emitter is not in this tree. What
        // must never happen is an edge with neither: that is a route the planner offers
        // and nothing can take, and the user would meet it as a crash rather than a
        // refusal.
        for e in crate::routes::edges() {
            if e.kind() != forms::Kind::Lower || e.to == "mlir" || e.to == "debug" {
                continue;
            }
            let open = emitter_exists(e.to);
            let gated = matches!(e.need, crate::routes::Need::Feature(_));
            assert!(
                open || gated,
                "edge lowers to {} with no open emitter and no feature gate",
                e.to
            );
        }
    }

    #[test]
    fn the_closed_targets_are_exactly_the_ones_without_an_open_emitter() {
        // Freezing this stops a target quietly changing sides. `pto`'s emitter IS open
        // (msl and gpu import mlir_to_pto). `cpp` is closed; `pico` lives in a sibling
        // repository and comes in behind its own feature.
        let closed: Vec<&str> = forms::FORMS
            .iter()
            .filter(|f| f.writable && f.id != "tile" && f.id != "mlir" && f.id != "debug")
            .filter(|f| !emitter_exists(f.id))
            .map(|f| f.id)
            .collect();
        // `pico` is closed only when its sibling checkout was not compiled in, so the
        // expectation follows the build rather than being asserted flat.
        let expected: Vec<&str> = if cfg!(feature = "pico") {
            vec!["cpp"]
        } else {
            vec!["cpp", "pico"]
        };
        assert_eq!(closed, expected, "the closed-target set moved");
    }
}
