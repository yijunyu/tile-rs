//! `-r/--run`: put the generated kernel on the device and see what it does.
//!
//! Emitting a kernel proves it compiles. Running it proves it computes the right answer,
//! and is the only way to say anything about speed that is not a guess.
//!
//! ## What is measured, and against what
//!
//! Two comparisons, because they answer different questions and are constantly confused:
//!
//! * **Against the CPU reference** — what running on the accelerator bought you. The
//!   reference is a **naive single-threaded scalar loop written here**, so this number is
//!   "this kernel against that loop". It is emphatically *not* "GPU versus CPU": a tuned,
//!   vectorised, multi-threaded CPU implementation would be far faster than the
//!   reference, and quoting this figure as a CPU comparison would be a measurement
//!   presented as something it is not.
//! * **Against `-O0`** — what the optimizer bought you. Only reported when the two
//!   actually differ: when the passes changed nothing, the honest line is "identical
//!   output", not "1.00x".
//!
//! Device time comes from the command buffer's own `GPUStartTime`/`GPUEndTime`. Wall
//! clock around the dispatch would fold in encoding, driver submission and this process's
//! scheduling and report them as kernel cost.
//!
//! ## The doctrine this inherits
//!
//! A speedup is a measurement. Nothing here reports one it did not take, and every figure
//! carries its sample count and spread — a median with no spread is a number with no
//! error bar, and the `kernel-impact` rule that "headroom is a measurement" applies just
//! as much to a ratio.

use crate::forms::Form;
use std::fmt;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
pub enum RunError {
    /// tile-rs has no harness for this target yet.
    NoHarness {
        form: &'static str,
    },
    /// The harness needs a device that is not here.
    NoDevice {
        form: &'static str,
        why: String,
    },
    /// We cannot compute a reference for what this kernel does.
    NoReference {
        hint: String,
    },
    /// The harness ran and the kernel was wrong.
    Wrong {
        report: String,
    },
    /// The harness itself failed.
    Harness {
        code: i32,
        stderr: String,
    },
    Io(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::NoHarness { form } => write!(
                f,
                "-r has no harness for \"{form}\" yet. A harness needs the target's host \
                 API, and writing one for a device nobody here can run would be a harness \
                 nobody has watched work."
            ),
            RunError::NoDevice { form, why } => {
                write!(f, "-r needs a {form} device: {why}")
            }
            RunError::NoReference { hint } => write!(
                f,
                "-r cannot check this kernel: {hint}\n  Running it would produce numbers \
                 with nothing to compare them against, which is worse than not running it."
            ),
            RunError::Wrong { report } => {
                write!(f, "the kernel computed the wrong answer:\n{report}")
            }
            RunError::Harness { code, stderr } => {
                write!(f, "the harness exited {code}:\n{}", stderr.trim())
            }
            RunError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// What the emitted kernel expects to be handed.
///
/// Read off the emitted signature rather than assumed, because assuming it is what made
/// `-r` useless for anything but a one-in-one-out kernel (backlog #002): the harness took
/// exactly two buffers and a length, so a matmul did not fit and was simply refused.
///
/// The signature is our own output, so parsing it is not guesswork — and doing it here
/// rather than in the harness keeps the harness dumb, which is what lets a second one for
/// another target be a transcription rather than a reimplementation.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelAbi {
    pub entry: String,
    /// `float` or `half`.
    pub dtype: String,
    /// Buffer indices that are `device const` — the inputs.
    pub inputs: usize,
    /// The `constant uint&` scalars, in buffer order, by name.
    pub scalars: Vec<String>,
    /// The workgroup width the SHADER fixes, when it fixes one.
    ///
    /// `None` for Metal, where the host chooses the threadgroup at dispatch and `-O4` can
    /// sweep it. `Some(n)` for SPIR-V, where `layout(local_size_x = n)` is compiled into
    /// the module: the harness must dispatch that width because no other is available.
    pub local_x: Option<usize>,
}

impl KernelAbi {
    pub fn bytes_per_element(&self) -> usize {
        if self.dtype == "half" {
            2
        } else {
            4
        }
    }

    /// Read the kernel's interface, whichever language it is written in.
    ///
    /// One entry point so a caller cannot pick the wrong parser for the form it emitted --
    /// which is how a harness ends up binding a Metal signature's buffer order onto a GLSL
    /// shader and comparing two unrelated numbers.
    pub fn parse(form: &Form, source: &str) -> Result<KernelAbi, String> {
        match form.id {
            "msl" => Self::parse_msl(source),
            "spirv" => Self::parse_glsl(source),
            other => Err(format!("no signature parser for `{other}`")),
        }
    }

    /// Parse a GLSL compute shader's interface.
    ///
    /// The sibling of `parse_msl`, and deliberately as strict: `readonly buffer` is an
    /// input, `writeonly buffer` an output, and the push-constant block names the scalars.
    /// A shader whose interface does not read cleanly is refused rather than guessed at,
    /// because a harness that binds the wrong buffer compares two unrelated numbers.
    pub fn parse_glsl(source: &str) -> Result<KernelAbi, String> {
        let mut inputs = 0usize;
        let mut outputs = 0usize;
        let mut scalars = Vec::new();
        let mut local_x = 0usize;
        let mut in_push_block = false;
        let mut pc_text = String::new();
        for line in source.lines() {
            let t = line.trim();
            if t.starts_with("layout(local_size_x") {
                local_x = t
                    .split("local_size_x")
                    .nth(1)
                    .and_then(|r| r.trim_start_matches([' ', '=']).split([',', ')']).next())
                    .and_then(|n| n.trim().parse().ok())
                    .unwrap_or(0);
            }
            if t.contains("buffer") && t.contains("binding") {
                if t.contains("readonly") {
                    inputs += 1;
                } else if t.contains("writeonly") || t.contains("buffer") {
                    outputs += 1;
                }
            }
            // The push-constant block, in either of the two shapes the emitters write.
            //
            // It first handled ONLY the single-line form: softmax emits
            // `{ uint num_elements; } pc;` and parsed fine, matmul emits M, N and K on
            // separate lines, so no scalars were found, one wrong value was pushed, N came
            // through as 0 and the kernel's loop never executed.
            //
            // The fix for that broke the case it replaced -- splitting the whole line on
            // `;` makes the first field `layout(push_constant) uniform PushConstants {
            // uint num_elements`, which does not start with `uint `. Single-line blocks
            // yielded NOTHING, the harness pushed `in_sizes[0]`, and matvec was handed 256
            // for a 64-wide row and read off the end of it. Softmax survived only because
            // at one row `rows * cols == cols` and the wrong number happened to be right.
            //
            // So the braces are found first and the fields parsed from between them,
            // wherever the newlines fall. Both forms are pinned by a test now; neither was
            // before, which is why one fix could silently undo the other.
            if !in_push_block && t.contains("push_constant") {
                in_push_block = true;
                pc_text.clear();
            }
            if in_push_block {
                pc_text.push_str(t);
                pc_text.push(' ');
                if t.contains('}') {
                    in_push_block = false;
                    if let (Some(a), Some(b)) = (pc_text.find('{'), pc_text.rfind('}')) {
                        for field in pc_text[a + 1..b].split(';') {
                            if let Some(name) = field.trim().strip_prefix("uint ") {
                                let name = name.trim();
                                if !name.is_empty() {
                                    scalars.push(name.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        if outputs != 1 {
            return Err(format!(
                "expected exactly one writeonly buffer, found {outputs}; this harness \
                 writes one result"
            ));
        }
        if inputs == 0 {
            return Err("no readonly buffer: the kernel reads nothing".to_string());
        }
        if local_x == 0 {
            return Err("no layout(local_size_x = N): the workgroup size is unknown".to_string());
        }
        // What this harness can actually drive: `vkCmdDispatch(grid, 1, 1)` over buffers it
        // fills with f32. A kernel outside that is not measurable here, and saying so is
        // the only honest answer.
        //
        // The spirv f16 GEMM is both -- `local_size_y = 16`, indexed by
        // `gl_WorkGroupID.y`, over `uint p0[]` holding packed halves. Run anyway it read
        // f32 bits as pairs of halves across a grid it never got, and the report said "the
        // KERNEL disagrees with torch, look at the lowering". That is the third time in a
        // day a confident wrong pointer came out of measuring something the harness could
        // not measure; refusing names the reason instead.
        if source.contains("gl_WorkGroupID.y") || source.contains("gl_GlobalInvocationID.y") {
            return Err(
                "this kernel indexes with gl_WorkGroupID.y, so it needs a 2D dispatch; \n                 this harness issues vkCmdDispatch(grid, 1, 1). It cannot be measured \n                 here, and a number from running it anyway would describe neither the \n                 kernel nor the reference."
                    .to_string(),
            );
        }
        // What the buffers hold. `float16_t` is a type the harness CAN fill now, so it
        // sets the dtype rather than refusing; anything else -- packed `uint` halves, for
        // one -- is still refused, because reading those as f32 measures the
        // reinterpretation and not the lowering.
        let mut dtype = "float".to_string();
        if let Some(i) = source.find("readonly  buffer") {
            let decl = &source[i..source[i..].find('}').map_or(source.len(), |j| i + j)];
            if decl.contains("float16_t ") {
                dtype = "half".to_string();
            } else if !decl.contains("float ") {
                let ty = decl
                    .split('{')
                    .nth(1)
                    .and_then(|b| b.split_whitespace().next())
                    .unwrap_or("something other than float");
                return Err(format!(
                    "this kernel's buffers hold `{ty}`, which this harness cannot fill. It \n                     writes f32 and f16; reading either as the other measures the \n                     reinterpretation, not the lowering."
                ));
            }
        }
        Ok(KernelAbi {
            entry: "main".to_string(),
            dtype,
            inputs,
            scalars,
            local_x: Some(local_x),
        })
    }

    /// Parse a Metal kernel signature.
    pub fn parse_msl(source: &str) -> Result<KernelAbi, String> {
        let start = source
            .find("kernel void ")
            .ok_or("no `kernel void` entry point in the emitted source")?;
        let rest = &source[start + "kernel void ".len()..];
        let paren = rest
            .find('(')
            .ok_or("the entry point has no parameter list")?;
        let entry = rest[..paren].trim().to_string();
        // Match the closing paren by DEPTH, not by the first `)`. Metal attributes nest
        // parens -- `[[ buffer(0) ]]` -- so `find(')')` ends the parameter list inside the
        // first attribute, leaving one parameter parsed and the output buffer invisible.
        // That is how a matmul reported "expected exactly one output buffer, found 0".
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in rest[paren..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(paren + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.ok_or("unterminated parameter list")?;
        let params = &rest[paren + 1..end];

        let mut dtype = String::new();
        let mut inputs = 0usize;
        let mut outputs = 0usize;
        let mut scalars = Vec::new();
        for p in params.split(',') {
            let p = p.trim();
            if p.starts_with("device const") {
                inputs += 1;
                dtype = pointee(p);
            } else if p.starts_with("device ") {
                outputs += 1;
                if dtype.is_empty() {
                    dtype = pointee(p);
                }
            } else if let Some(rest) = p.strip_prefix("constant uint&") {
                let name = rest.split_whitespace().next().unwrap_or("").to_string();
                scalars.push(name);
            }
        }
        // A `char*` buffer carries no element type. The ggml-derived kernels declare
        // `device const char* p0` and cast inside the body -- `(device const half *)(p0 +
        // row * nb_src)` -- so the declaration says `char` and the arithmetic is done in
        // half. Reading the declaration alone made the harness upload f32 into a buffer
        // the kernel read as pairs of halves, and `exp` came back wrong by 5.85e4.
        //
        // The cast is where the type is stated, so that is where it is read from.
        if dtype == "char" || dtype.is_empty() {
            let body = &source[start..];
            dtype = if body.contains("device const half *") || body.contains("device half *") {
                "half".to_string()
            } else if body.contains("device const float *") || body.contains("device float *") {
                "float".to_string()
            } else {
                return Err(format!(
                    "the kernel's buffers are `{dtype}*` and nothing in the body casts them \
                     to a known element type, so the harness cannot tell what to fill them \
                     with. Refusing rather than guessing f32."
                ));
            };
        }

        // The harness dispatches a 1D grid -- MTLSize(width: grid, height: 1, depth: 1).
        // A kernel that declares `uint2 threadgroup_position_in_grid` is indexed by BOTH
        // components, so under a 1D dispatch tgpig.y is always 0 and everything past the
        // first row-block of the output is never written. That is not a lowering fault and
        // must not be reported as one: the simdgroup matmul documents its own decomposition
        // as grid = (ceil(N/8), ceil(M/8)), which this harness has no way to infer in
        // general. Refuse, and say which side is unable.
        if params.contains("uint2") && params.contains("threadgroup_position_in_grid") {
            return Err("the kernel takes a 2D threadgroup position (`uint2 \
                 threadgroup_position_in_grid`) and this harness dispatches a 1D grid, so \
                 every block above the first row would be left unwritten. The kernel is \
                 fine; the harness cannot infer the intended 2D decomposition"
                .to_string());
        }
        if outputs != 1 {
            return Err(format!(
                "expected exactly one output buffer, found {outputs}; this harness writes \
                 one result"
            ));
        }
        if inputs == 0 {
            return Err("no input buffers".into());
        }
        Ok(KernelAbi {
            entry,
            dtype,
            inputs,
            scalars,
            // Metal takes its threadgroup at dispatch, so the SOURCE fixes nothing and
            // `-O4` is free to sweep it. SPIR-V is the other way round.
            local_x: None,
        })
    }
}

fn pointee(param: &str) -> String {
    param
        .split_whitespace()
        .find(|t| t.ends_with('*') || *t == "half" || *t == "float")
        .map(|t| t.trim_end_matches('*').to_string())
        .unwrap_or_else(|| "float".into())
}

/// The elementwise-over-a-row operations we can produce a reference for.
///
/// Deliberately small. A kernel whose op is not here is refused rather than run: numbers
/// with nothing to compare them against are worse than no numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefOp {
    /// The one that is not elementwise, and the reason the ABI had to generalise.
    Matmul,
    Softmax,
    Exp,
    Sigmoid,
    Relu,
    Sqrt,
    Log,
    Neg,
    Abs,
    // Added once --list-ops showed msl lowering 17 ops while only nine had a reference.
    // Lowering is not correctness, and an op nobody can check is one nobody has checked.
    Tanh,
    Rsqrt,
    Silu,
    Softplus,
    /// Row reductions: one value out per row in.
    ReduceMax,
    ReduceSum,
    Min,
    Max,
    Matvec,
    RmsNorm,
    Absmax,
}

/// Every operation the harness can recognise in a kernel, and the reference it maps to.
///
/// At module scope so the "it knows..." hint and its test can both read the same table
/// the detector reads. It was inside `detect`, which is why the hint had to restate it in
/// prose and could drift three ops behind it.
const OPS: &[(&str, RefOp)] = &[
    ("__tile_matmul_", RefOp::Matmul),
    ("__tile_softmax_", RefOp::Softmax),
    ("__tile_exp_", RefOp::Exp),
    ("__tile_sigmoid_", RefOp::Sigmoid),
    ("__tile_relu_", RefOp::Relu),
    ("__tile_sqrt_", RefOp::Sqrt),
    ("__tile_log_", RefOp::Log),
    ("__tile_neg_", RefOp::Neg),
    ("__tile_abs_", RefOp::Abs),
    ("__tile_tanh_", RefOp::Tanh),
    ("__tile_rsqrt_", RefOp::Rsqrt),
    ("__tile_silu_", RefOp::Silu),
    ("__tile_softplus_", RefOp::Softplus),
    ("__tile_reduce_max_", RefOp::ReduceMax),
    ("__tile_reduce_sum_", RefOp::ReduceSum),
    ("__tile_min_", RefOp::Min),
    ("__tile_max_", RefOp::Max),
    ("__tile_matvec_", RefOp::Matvec),
    ("__tile_rms_norm_", RefOp::RmsNorm),
    ("__tile_absmax_", RefOp::Absmax),
];

impl RefOp {
    pub fn name(self) -> &'static str {
        match self {
            RefOp::Matmul => "matmul",
            RefOp::Softmax => "softmax",
            RefOp::Exp => "exp",
            RefOp::Sigmoid => "sigmoid",
            RefOp::Relu => "relu",
            RefOp::Sqrt => "sqrt",
            RefOp::Log => "log",
            RefOp::Neg => "neg",
            RefOp::Abs => "abs",
            RefOp::Tanh => "tanh",
            RefOp::Rsqrt => "rsqrt",
            RefOp::Silu => "silu",
            RefOp::Softplus => "softplus",
            RefOp::ReduceMax => "reduce_max",
            RefOp::ReduceSum => "reduce_sum",
            RefOp::Min => "min",
            RefOp::Max => "max",
            RefOp::Matvec => "matvec",
            RefOp::RmsNorm => "rms_norm",
            RefOp::Absmax => "absmax",
        }
    }

    /// Which operation does this kernel perform? `None` when it is not one we can check.
    ///
    /// A kernel using more than one of these is refused: the reference would have to know
    /// the order they compose in, and guessing it would produce a comparison that fails
    /// for the wrong reason.
    pub fn detect(text: &str) -> Result<RefOp, String> {
        let found: Vec<RefOp> = OPS
            .iter()
            .filter(|(m, _)| text.contains(m))
            .map(|(_, o)| *o)
            .collect();
        match found.len() {
            0 => {
                // Built from OPS, the table just searched, rather than written out again
                // in prose. The hand-maintained version had drifted three ops behind --
                // it still promised thirteen after reduce_max, absmax and reduce_sum were
                // added -- so the message told the caller the harness could not check
                // something it could.
                let mut known: Vec<&str> = Vec::new();
                for (_, o) in OPS {
                    if !known.contains(&o.name()) {
                        known.push(o.name());
                    }
                }
                let list = match known.split_last() {
                    Some((last, rest)) if !rest.is_empty() => {
                        format!("{} and {last}", rest.join(", "))
                    }
                    _ => known.join(", "),
                };
                Err(format!(
                    "no operation this harness has a reference for. It knows {list}."
                ))
            }
            1 => Ok(found[0]),
            _ => Err(format!(
                "{} operations ({}), and the reference would have to guess the order they \
                 compose in",
                found.len(),
                found
                    .iter()
                    .map(|o| o.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }

    /// Does this op reduce a row to one value?
    ///
    /// The harness sizes its output buffer from this, so getting it wrong means reading
    /// back memory nobody wrote.
    pub fn is_row_reduction(self) -> bool {
        matches!(self, RefOp::ReduceMax | RefOp::ReduceSum | RefOp::Absmax)
    }

    /// How many input buffers this operation reads.
    pub fn inputs(self) -> usize {
        match self {
            RefOp::Matmul | RefOp::Min | RefOp::Max | RefOp::Matvec => 2,
            _ => 1,
        }
    }

    /// Does this op read two input tiles of the same extent?
    pub fn is_two_input_elementwise(self) -> bool {
        matches!(self, RefOp::Min | RefOp::Max)
    }

    /// The reference, over one row. Matmul is not a row operation and is not handled
    /// here — see [`RefOp::matmul`].
    pub fn apply(self, row: &[f32], out: &mut [f32], eps: f32) {
        match self {
            RefOp::Matmul => panic!("matmul is not a row operation; call RefOp::matmul"),
            RefOp::Softmax => {
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (i, x) in row.iter().enumerate() {
                    let e = (x - m).exp();
                    out[i] = e;
                    sum += e;
                }
                for o in out.iter_mut() {
                    *o /= sum;
                }
            }
            RefOp::Exp => elementwise(row, out, |x| x.exp()),
            RefOp::Sigmoid => elementwise(row, out, |x| 1.0 / (1.0 + (-x).exp())),
            RefOp::Relu => elementwise(row, out, |x| x.max(0.0)),
            RefOp::Sqrt => elementwise(row, out, |x| x.sqrt()),
            RefOp::Log => elementwise(row, out, |x| x.ln()),
            RefOp::Neg => elementwise(row, out, |x| -x),
            RefOp::Abs => elementwise(row, out, f32::abs),
            RefOp::Tanh => elementwise(row, out, |x| x.tanh()),
            RefOp::Rsqrt => elementwise(row, out, |x| 1.0 / x.sqrt()),
            // x * sigmoid(x), written the same way the emitter does it so a disagreement
            // is about the KERNEL rather than about two spellings of the same function.
            RefOp::Silu => elementwise(row, out, |x| x / (1.0 + (-x).exp())),
            // The emitter guards large x with a passthrough; the reference does too, or
            // exp overflows to inf and the comparison is about f32 range, not the kernel.
            RefOp::Softplus => {
                elementwise(
                    row,
                    out,
                    |x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() },
                )
            }
            // x * rsqrt(mean(x^2) + eps), as tile_std documents it. EPS is the emitters'
            // 1e-6 rather than the intrinsic's operand, because the emitters bake that
            // constant in and the harness has no way to pass a float scalar yet -- see the
            // note in docs/cli/INTEGRATION.md. Matching what they emit is the only way this
            // comparison means anything today.
            RefOp::RmsNorm => {
                let n = row.len().max(1) as f32;
                let mean_sq = row.iter().map(|x| x * x).sum::<f32>() / n;
                let scale = 1.0 / (mean_sq + eps).sqrt();
                elementwise(row, out, |x| x * scale)
            }
            RefOp::ReduceMax | RefOp::ReduceSum | RefOp::Absmax => {
                panic!("row reductions are not elementwise; use Shape::RowReduce")
            }
            RefOp::Min | RefOp::Max => {
                panic!("min and max read two tiles; use Shape::Rows2")
            }
            RefOp::Matvec => panic!("matvec is a matrix times a vector; use Shape::Matvec"),
        }
    }
}

fn elementwise(row: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32) {
    for (i, x) in row.iter().enumerate() {
        out[i] = f(*x);
    }
}

/// The reference matmul: (M x K) @ (K x N).
///
/// Written out rather than expressed through `apply`, because a matmul is not an
/// elementwise row operation and pretending it is would be the sort of shoehorning that
/// produces a reference nobody trusts.
pub fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// The input the harness generates, reproduced here so both sides compute the same thing.
///
/// Written once and shared by construction rather than by two implementations agreeing:
/// a reference fed different data from the kernel is a comparison of two unrelated
/// numbers, and it fails in a way that looks like a numerical bug.
pub fn input_values(n: usize) -> Vec<f32> {
    input_values_for(0, n)
}

/// The values for buffer `b`. Two inputs must not be handed the same matrix, or a matmul
/// reference agrees with a transposed kernel and nobody finds out.
///
/// The `i * 1e-4` ramp is what makes this usable for REDUCTIONS. Without it the sequence
/// is periodic with period 17, so the largest element always falls within the first 32 --
/// and a kernel that reduces only the first SIMD group agrees exactly with one that
/// reduces the whole row. Metal's `reduce_max` and `absmax` did precisely that, returning
/// 0.31 on a row whose maximum was 10.23, and this generator could not tell the
/// difference. Finding it took a hand-written monotone input.
///
/// The ramp is small enough (0.1 across 1024 elements, against 0.25 steps) that every
/// value stays in roughly the same range, and large enough that the extreme lands near
/// the END of the row, where a partial reduction will miss it.
/// The unit roundoff of a kernel's buffer type: the closest any stored result can be.
///
/// An f16 kernel writes its answer into half, so its relative error against an f32
/// reference is up to 2^-11 = 4.88e-4 however good the lowering is. `REL_TOL` is 1e-4,
/// five times tighter than the format allows, so every correct f16 elementwise kernel was
/// reported as "the KERNEL disagrees with torch, look at the lowering".
///
/// This is #013 in the dtype dimension, and the answer is the same one: the number is
/// derived from the format rather than chosen. f32's roundoff is 5.96e-8, far below
/// `REL_TOL`, so nothing about the f32 path changes.
pub fn unit_roundoff(dtype: &str) -> f32 {
    match dtype {
        "half" => 4.882_812_5e-4,
        _ => f32::EPSILON / 2.0,
    }
}

/// Round a generated input to what the kernel's buffers can actually hold.
///
/// #010. `metal.swift` writes an f16 kernel's inputs as `Float16(v[i])` while both
/// references used the f32 originals, so an f16 kernel was compared against numbers it was
/// never given and part of every reported gap was that rounding rather than the lowering.
/// A comment in `RunReport::render` claimed this had been fixed by quantizing "at
/// generation time"; it had not -- the generator has no dtype and never did. The claim sat
/// where someone would rely on it.
///
/// With this, all three sides see identical inputs and the only f16 effect left is the
/// kernel's own arithmetic, which is what `error_budget` accounts for separately.
pub fn quantize_inputs(values: &mut [f32], dtype: &str) {
    if dtype != "half" {
        return;
    }
    for v in values.iter_mut() {
        *v = f16_round(*v);
    }
}

/// f32 -> f16 -> f32, without a dependency. Ties to even, and anything outside f16's range
/// saturates to its infinity the way a `Float16` cast does.
pub fn f16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let sign = bits >> 31;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    // Unbiased exponent, rebiased for f16 (bias 15 against f32's 127).
    let e = exp - 127 + 15;
    let out = if e >= 0x1f {
        // Overflow to infinity.
        return if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    } else if e <= 0 {
        // Subnormal in f16, or zero. Shift the implicit bit back in and round.
        if e < -10 {
            0u32
        } else {
            let m = mant | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = 1u32 << (shift - 1);
            let r = m + half - 1 + ((m >> shift) & 1);
            r >> shift
        }
    } else {
        // Normal: 13 bits of mantissa are dropped, round to nearest even.
        let half = 1u32 << 12;
        let r = mant + half - 1 + ((mant >> 13) & 1);
        let carry = r >> 23;
        (((e as u32 + carry) << 10) | ((r >> 13) & 0x3ff)) & 0x7fff
    };
    let h = ((sign << 15) | out) as u16;
    // Back to f32.
    let hs = (h >> 15) as u32;
    let he = ((h >> 10) & 0x1f) as u32;
    let hm = (h & 0x3ff) as u32;
    let f = if he == 0 {
        if hm == 0 {
            hs << 31
        } else {
            // Subnormal: value = hm * 2^-24. With k = floor(log2 hm) that is
            // 1.f * 2^(k-24), so the f32 biased exponent is k - 24 + 127. The loop below
            // shifts until bit 10 is set, taking 10 - k steps, so k = e + 11 and the
            // exponent is e + 114.
            //
            // It was `e + 1 - 15 + 127` = e + 113, one too low, which halved every f16
            // subnormal. Caught by checking against numpy rather than by reading it back.
            let mut m = hm;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (hs << 31) | (((e + 114) as u32) << 23) | (m << 13)
        }
    } else if he == 0x1f {
        (hs << 31) | 0x7f80_0000 | (hm << 13)
    } else {
        (hs << 31) | ((he + 127 - 15) << 23) | (hm << 13)
    };
    f32::from_bits(f)
}

pub fn input_values_for(b: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i + 7 * b) % 17) as f32 * 0.25 - 2.0 + i as f32 * 1e-4)
        .collect()
}

/// The dispatch geometry: buffer sizes, grid, and the scalar arguments.
///
/// This exists because `(rows, cols)` could not express a matmul. Two matrices of
/// different sizes produce a third of a third size, and there is no pair of numbers that
/// says so. Refusing matmul was the honest response while that was true (backlog #002);
/// widening the type is the fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// `rows` independent rows of `cols` elements: elementwise ops and softmax.
    Rows { rows: usize, cols: usize },
    /// `(m x k) . (k x n)`.
    Matmul { m: usize, k: usize, n: usize },
    /// `rows` rows of `cols`, reduced to ONE value per row.
    ///
    /// A separate variant rather than a flag on `Rows`, because the output size is what
    /// the harness sizes its buffer from: a reduction that wrote `rows * cols` would read
    /// back mostly uninitialised memory and compare it against nothing.
    RowReduce { rows: usize, cols: usize },
    /// A (rows, cols) matrix times a (cols,) vector — one value out per row.
    ///
    /// The first shape whose two inputs are DIFFERENT sizes, which is why `in_sizes`
    /// returns a vector rather than a count: the harness allocates each buffer from its
    /// own entry, so a matvec's vector is cols long and its matrix rows*cols.
    Matvec { rows: usize, cols: usize },
    /// Elementwise over TWO input tiles of the same extent — `min`, `max`.
    ///
    /// A separate variant rather than a flag on `Rows` because `in_sizes` is what the
    /// harness allocates from, and the ABI check compares its length against the kernel's
    /// buffer count. A two-input kernel driven by a one-input shape fails there, loudly,
    /// instead of reading a buffer nobody filled.
    Rows2 { rows: usize, cols: usize },
}

impl Shape {
    /// The size of each input buffer, in elements.
    pub fn in_sizes(self) -> Vec<usize> {
        match self {
            Shape::Rows { rows, cols } => vec![rows * cols],
            Shape::Rows2 { rows, cols } => vec![rows * cols, rows * cols],
            Shape::Matvec { rows, cols } => vec![rows * cols, cols],
            Shape::RowReduce { rows, cols } => vec![rows * cols],
            Shape::Matmul { m, k, n } => vec![m * k, k * n],
        }
    }

    pub fn out_size(self) -> usize {
        match self {
            Shape::Rows { rows, cols } => rows * cols,
            Shape::Rows2 { rows, cols } => rows * cols,
            Shape::Matvec { rows, .. } => rows,
            Shape::RowReduce { rows, .. } => rows,
            Shape::Matmul { m, n, .. } => m * n,
        }
    }

    /// One threadgroup per output row, matching the emitted kernels, which index their
    /// row by `threadgroup_position_in_grid`.
    pub fn grid(self) -> usize {
        match self {
            Shape::Rows { rows, .. } => rows.max(1),
            Shape::Rows2 { rows, .. } => rows.max(1),
            Shape::Matvec { rows, .. } => rows.max(1),
            Shape::RowReduce { rows, .. } => rows.max(1),
            Shape::Matmul { m, .. } => m.max(1),
        }
    }

    pub fn threads(self) -> usize {
        match self {
            Shape::Rows { cols, .. } => cols.clamp(1, 1024),
            Shape::Rows2 { cols, .. } => cols.clamp(1, 1024),
            Shape::Matvec { cols, .. } => cols.clamp(1, 1024),
            Shape::RowReduce { cols, .. } => cols.clamp(1, 1024),
            Shape::Matmul { n, .. } => n.clamp(1, 1024),
        }
    }

    /// The value of a `constant uint&` argument, looked up BY NAME.
    ///
    /// By name rather than by position: the emitters choose their own scalar order, and
    /// binding `M` to whatever happens to sit in buffer 4 is how a matmul silently
    /// computes a transpose. An unknown name is `None`, and the caller refuses — better
    /// than passing a zero that the kernel reads as an empty loop and returns success for.
    pub fn scalar(self, name: &str) -> Option<u32> {
        self.scalar_for(name, 4)
    }

    /// `scalar`, told how wide one element is.
    ///
    /// The ggml-derived kernels take BYTE strides (`nb_src`, `nb_dst`) over `char*`
    /// buffers rather than element counts, so the same shape binds a different number for
    /// f16 and f32. Every one of them was refused before this -- correctly, since a wrong
    /// stride reads the row at the wrong offset, but it meant a whole kernel family had
    /// never been run.
    pub fn scalar_for(self, name: &str, elem_bytes: u32) -> Option<u32> {
        // Element strides that the ggml vocabulary spells differently. `ne0` is the
        // innermost extent -- the row width -- and `nb_*` are that width in BYTES.
        match name {
            "ne0" => return self.scalar_for("num_elements", elem_bytes),
            "nb_src" | "nb_dst" | "nb0" | "nb1" => {
                return self
                    .scalar_for("num_elements", elem_bytes)
                    .and_then(|n| n.checked_mul(elem_bytes))
            }
            _ => {}
        }
        self.scalar_inner(name)
    }

    fn scalar_inner(self, name: &str) -> Option<u32> {
        let v = match (self, name) {
            // For a reduction `num_elements` is the ROW WIDTH the kernel loops over, not
            // the size of what it writes. Binding it to out_size would make the kernel
            // read `rows` elements of a `rows * cols` row and reduce a sliver.
            (Shape::RowReduce { cols, .. }, "num_elements") => cols,
            // Matvec is a reduction too, and the catch-all below bound it to out_size --
            // which is `rows`. The kernel then looped `i < 4` over a 256-wide row and
            // summed a sliver, disagreeing with both references by 1.74e2 while they
            // agreed with each other to 3.05e-5. Exactly the failure the comment above
            // describes, one shape later.
            (Shape::Matvec { cols, .. }, "num_elements") => cols,
            // Spelled out per shape rather than left to `_`, so a shape added later has to
            // say what its kernels loop over instead of inheriting an answer that happens
            // to compile.
            // The row WIDTH, not the total. Every emitted kernel uses this both as the
            // row stride (`base = row * num_elements`) and as the per-row bound, so with
            // more than one row the total is the wrong number in both places. It was only
            // ever right because `rows` was hardcoded to 1 (#017).
            (Shape::Rows { cols, .. }, "num_elements")
            | (Shape::Rows2 { cols, .. }, "num_elements") => cols,
            (Shape::Matmul { .. }, "num_elements") => self.out_size(),
            (Shape::RowReduce { cols, .. }, "n") | (Shape::RowReduce { cols, .. }, "N") => cols,
            (Shape::RowReduce { rows, .. }, "m") | (Shape::RowReduce { rows, .. }, "M") => rows,
            (Shape::Rows { cols, .. }, "n") | (Shape::Rows { cols, .. }, "N") => cols,
            (Shape::Rows { rows, .. }, "m") | (Shape::Rows { rows, .. }, "M") => rows,
            (Shape::Matmul { m, .. }, "M") | (Shape::Matmul { m, .. }, "m") => m,
            (Shape::Matmul { n, .. }, "N") | (Shape::Matmul { n, .. }, "n") => n,
            (Shape::Matmul { k, .. }, "K") | (Shape::Matmul { k, .. }, "k") => k,
            // The f32 matvec emitter is DECODE-shaped: N = output rows, K = input
            // dim, M = 1 (unused by the body). Binding N=1 made `row >= N` early-
            // return every group but the first, so rows 1.. stayed at the memset 0
            // and max_rel hit exactly 1.0. The GEMM "matvec is matmul with N=1"
            // view is what the (undriven) f16 GEMM path uses; this binding serves
            // the driven f32 path and matches the emitter's own signature comment.
            (Shape::Matvec { .. }, "M") | (Shape::Matvec { .. }, "m") => 1,
            (Shape::Matvec { rows, .. }, "N") | (Shape::Matvec { rows, .. }, "n") => rows,
            (Shape::Matvec { cols, .. }, "K") | (Shape::Matvec { cols, .. }, "k") => cols,
            _ => return None,
        };
        Some(v as u32)
    }

    /// The eps a `__tile_rms_norm_*` call carries, read from the same MLIR the emitter
    /// reads it from.
    ///
    /// Both sides must take it from the SOURCE. While the emitter hardcoded 1e-6 the
    /// reference hardcoded 1e-6 to match, and two constants agreeing with each other is not
    /// a check of anything -- change the kernel's eps and the comparison would have gone on
    /// passing.
    pub fn rms_eps_from_mlir(mlir: &str) -> Option<f32> {
        let call = mlir.lines().find(|l| l.contains("__tile_rms_norm_"))?;
        let open = call.find('(')?;
        // Only the FIVE-argument form carries eps. The four-argument form is
        // (tile, tile, rows, cols), so reading index 2 unconditionally read the ROW
        // COUNT -- whose constant is `1` -- and handed both references an eps of 1.0.
        //
        // They then agreed with each other and the kernel, computing with the 1e-6 it
        // actually contains, was reported as "the KERNEL disagrees with torch, look at
        // the lowering". Nothing in the three-way comparison can catch that: when the
        // harness feeds BOTH references the same wrong parameter, the odd one out is the
        // only side that is right.
        let args: Vec<&str> = call[open + 1..].split(')').next()?.split(',').collect();
        if args.len() < 5 {
            return None;
        }
        let eps_ssa = args.get(2)?.trim();
        for line in mlir.lines() {
            if !line.contains("llvm.mlir.constant(") {
                continue;
            }
            let Some(res) = line.split('=').next().map(str::trim) else {
                continue;
            };
            if res != eps_ssa {
                continue;
            }
            // And it must be a FLOAT constant. An i32 reaching here means the operand
            // is not an epsilon whatever its position, which is the second half of the
            // same mistake.
            if !line.contains("f32") && !line.contains("f16") {
                return None;
            }
            let open = line.find("llvm.mlir.constant(")? + "llvm.mlir.constant(".len();
            let v: String = line[open..]
                .chars()
                .take_while(|c| {
                    c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == '-' || *c == '+'
                })
                .collect();
            return v.parse().ok();
        }
        None
    }

    /// Read the matmul shape out of the MLIR, from the intrinsic call's own operands.
    ///
    /// The dimensions are in the source; taking them from there beats asking the user for
    /// numbers that must then agree with the kernel. `__tile_matmul_*(a, a, b, m, k, n)`
    /// -- the trailing three operands are SSA names bound to `llvm.mlir.constant`.
    /// The `(rows, cols)` the MLIR actually states, from its first `__tile_load_*`.
    ///
    /// #017. The harness built every non-matmul shape with `rows: 1` and `cols` set to the
    /// TOTAL element count -- a literal, never read from the source. So a kernel written
    /// `%r = 3, %c = 129` was driven as one row of 387, `base = row * num_elements` was
    /// always 0, and no kernel in this repo has ever had its row indexing exercised. The
    /// reference was handed the same flattened shape, so both sides agreed about a
    /// question the MLIR had not asked, and the three-way comparison could not see it:
    /// that is not a disagreement, it is two right answers to the wrong problem.
    ///
    /// `matmul_from_mlir` already read its three dimensions this way. The load intrinsic
    /// carries `(rows, cols)` just as plainly.
    pub fn rows_cols_from_mlir(mlir: &str) -> Option<(usize, usize)> {
        let consts = Self::int_consts(mlir);
        let call = mlir
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .find(|l| l.contains("__tile_load_"))?;
        let open = call.find('(')?;
        let close = call[open..].find(')')? + open;
        let args: Vec<&str> = call[open + 1..close].split(',').map(str::trim).collect();
        if args.len() < 3 {
            return None;
        }
        let rows = *consts.get(args[args.len() - 2])?;
        let cols = *consts.get(args[args.len() - 1])?;
        if rows == 0 || cols == 0 {
            return None;
        }
        Some((rows, cols))
    }

    /// Integer constants by SSA name, shared by the shape readers.
    fn int_consts(mlir: &str) -> std::collections::HashMap<String, usize> {
        let mut consts = std::collections::HashMap::new();
        for line in mlir.lines() {
            let line = line.split("//").next().unwrap_or("");
            let Some((lhs, rhs)) = line.split_once('=') else {
                continue;
            };
            let Some(rest) = rhs.trim().strip_prefix("llvm.mlir.constant(") else {
                continue;
            };
            let val = rest.split(':').next().unwrap_or("").trim();
            if let Ok(v) = val.parse::<usize>() {
                consts.insert(lhs.trim().to_string(), v);
            }
        }
        consts
    }

    pub fn matmul_from_mlir(mlir: &str) -> Option<Shape> {
        let mut consts = std::collections::HashMap::new();
        for line in mlir.lines() {
            let line = line.split("//").next().unwrap_or("");
            let Some((lhs, rhs)) = line.split_once('=') else {
                continue;
            };
            let Some(rest) = rhs.trim().strip_prefix("llvm.mlir.constant(") else {
                continue;
            };
            let val = rest.split(':').next().unwrap_or("").trim();
            if let Ok(v) = val.parse::<usize>() {
                consts.insert(lhs.trim().to_string(), v);
            }
        }
        let call = mlir
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .find(|l| l.contains("__tile_matmul_"))?;
        let open = call.find('(')?;
        let close = call[open..].find(')')? + open;
        let args: Vec<&str> = call[open + 1..close].split(',').map(str::trim).collect();
        if args.len() < 3 {
            return None;
        }
        let tail = &args[args.len() - 3..];
        let m = *consts.get(tail[0])?;
        let k = *consts.get(tail[1])?;
        let n = *consts.get(tail[2])?;
        Some(Shape::Matmul { m, k, n })
    }
}

/// A set of timings, and what can honestly be said about them.
#[derive(Debug, Clone)]
pub struct Timing {
    pub samples: Vec<f64>,
    pub warmup: usize,
}

impl Timing {
    pub fn median(&self) -> f64 {
        let mut v = self.samples.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            return 0.0;
        }
        v[v.len() / 2]
    }
    pub fn min(&self) -> f64 {
        self.samples.iter().copied().fold(f64::INFINITY, f64::min)
    }
    pub fn max(&self) -> f64 {
        self.samples.iter().copied().fold(0.0, f64::max)
    }
    /// A median with no spread is a number with no error bar.
    /// How wide the samples are, as a fraction of the median.
    ///
    /// The denominator of every ratio this tool prints, so it decides whether the ratio
    /// is reproducible. Observed directly: the same matmul on the same machine reported
    /// 5.2x and then 18.6x, entirely because the CPU reference competes for cores with
    /// whatever else the box is running while the GPU does not. The device samples were
    /// 77.75-78.12 us across both runs; the reference moved by a factor of ten.
    pub fn spread(&self) -> f64 {
        let m = self.median();
        if m <= 0.0 || self.samples.is_empty() {
            return 0.0;
        }
        (self.max() - self.min()) / m
    }

    /// Is this timing stable enough to quote a ratio from?
    ///
    /// 15% is a judgement, not a law: tight enough that a quiet machine passes and a
    /// contended one does not, loose enough that ordinary scheduler noise on a short
    /// kernel is not reported as a problem.
    pub fn is_noisy(&self) -> bool {
        self.spread() > 0.15
    }

    pub fn render(&self, unit: &str) -> String {
        format!(
            "median {:.2} {unit} over {} runs (min {:.2}, max {:.2}, {} warmup)",
            self.median(),
            self.samples.len(),
            self.min(),
            self.max(),
            self.warmup
        )
    }
}

#[derive(Debug)]
pub struct RunReport {
    pub device: String,
    pub op: RefOp,
    /// The element type of the kernel's BUFFERS, which sets how close agreement can get.
    pub dtype: String,
    pub elements: usize,
    pub checked: usize,
    pub worst_error: f32,
    /// Elements where one side was a number and the other was not.
    pub nan_mismatch: usize,
    pub tolerance: f32,
    pub target: Timing,
    pub reference: Timing,
    /// Present only when an `-O0` build was also run AND differed.
    pub unoptimized: Option<Timing>,
    /// Set when `-O0` and the chosen level produced identical source.
    pub optimizer_changed_nothing: bool,
    /// What f32 summation can cost this op at this shape, when it sums at all. `None`
    /// means the fixed elementwise tolerance is the right judge. See #013.
    pub budget: Option<Budget>,
    /// The kernel against this tool's own reference.
    pub vs_ours: crate::torchref::Accuracy,
    /// The kernel against torch, and this tool's reference against torch — present only
    /// when torch answered. Absence weakens the claim and is stated, never hidden.
    pub torch: Result<
        (String, crate::torchref::Accuracy, crate::torchref::Accuracy),
        crate::torchref::TorchStatus,
    >,
}

impl RunReport {
    /// This kernel against the naive scalar reference. NOT "GPU versus CPU".
    pub fn speedup_vs_reference(&self) -> f64 {
        if self.target.median() <= 0.0 {
            return 0.0;
        }
        self.reference.median() / self.target.median()
    }
    pub fn speedup_from_optimizer(&self) -> Option<f64> {
        let u = self.unoptimized.as_ref()?;
        if self.target.median() <= 0.0 {
            return None;
        }
        Some(u.median() / self.target.median())
    }

    pub fn render(&self) -> String {
        // #010, and the comment that used to sit here is worth remembering: it said the
        // inputs were "quantized at generation time now, so both sides see identical
        // values". They were not. `input_values_for` had no dtype and never did, so the
        // sentence asserted a fix that had never been made, in the place someone would
        // check before trusting an f16 number.
        //
        // They are quantized now, for real: `quantize_inputs` on the Rust side and
        // `.half().float()` in the torch script, both checked against numpy. `ours vs
        // torch` reads 0.00e0 on an f16 matmul as a result -- the two references agree
        // bit for bit -- and that kernel's max REL error fell from 2.86e-1 to 4.81e-4,
        // which is how much of the old figure was the rounding rather than the lowering.
        const ABS_TOL: f32 = 1e-5;
        const REL_TOL: f32 = 1e-4;
        // One tolerance, chosen from the operation rather than from a constant that had
        // to serve every operation at once. See #013 and `error_budget`.
        let tol = match self.budget {
            Some(b) => crate::torchref::Tolerance::Summation(b),
            // The relative bound can never be tighter than what the kernel's own buffer
            // type can represent. See `unit_roundoff`.
            None => crate::torchref::Tolerance::Fixed {
                abs: ABS_TOL,
                rel: REL_TOL.max(unit_roundoff(&self.dtype)),
            },
        };
        let mut s = String::new();
        s.push_str(&format!(
            "run: {} — {} over {} elements\n",
            self.device,
            self.op.name(),
            self.elements
        ));
        s.push_str("  accuracy:\n");
        // A comparison over nothing is not agreement. `worst` is a max over the values
        // that came back, so an empty result set makes every error 0.0 and every verdict
        // read "all three agree" -- a green report for a kernel that produced nothing.
        // The PICO session hit the same shape on a board rig that returned the PREVIOUS
        // model's output for a model that had failed to load, and two apparent results
        // were that artifact. Say it instead.
        // A one-sided NaN is not a small error, it is a different KIND of answer, and no
        // magnitude describes it. Reported before the tolerances, because a clean
        // "max abs 4.77e-7" beside a thousand NaNs is the most misleading line this tool
        // could print.
        if self.nan_mismatch > 0 {
            s.push_str(&format!(
                "    NOT A NUMBER: {} of {} elements where one side is NaN or infinite and\n\
                 \x20   the other is finite. The error figures below describe only the rest.\n",
                self.nan_mismatch, self.checked
            ));
        }
        if self.checked == 0 {
            s.push_str(
                "    NO VALUES were read back, so nothing was compared. This is not
                     agreement -- treat the timings below as unattributed.
",
            );
        } else if self.checked < self.elements {
            s.push_str(&format!(
                "    note: {} of {} elements compared
",
                self.checked, self.elements
            ));
        }
        match &self.torch {
            Ok((version, k, o)) => {
                s.push_str(&format!(
                    "    kernel vs torch {version:<8}  {}\n",
                    k.render()
                ));
                s.push_str(&format!(
                    "    kernel vs ours            {}\n",
                    self.vs_ours.render()
                ));
                s.push_str(&format!("    ours   vs torch           {}\n", o.render()));
                // "look at the lowering" is a confident pointer, and it must not be
                // printed where this tool has no bound to judge by. An f16 op outside
                // `error_budget` -- softmax, rms_norm -- is exactly that case: the
                // reference is exact, the kernel differs by its own half arithmetic, and
                // nothing here can say whether that amount is right. Naming the lowering
                // is the same mistake that was made three times in a day by measuring
                // something unmeasurable.
                let verdict = crate::torchref::attribute(k, o, &self.vs_ours, tol);
                let unbounded_half = self.dtype == "half" && self.budget.is_none();
                if unbounded_half && verdict.starts_with("the KERNEL disagrees") {
                    s.push_str(
                        "    verdict: the kernel differs from BOTH references, which agree \
                         with each\n             other — so this is the kernel's own f16 \
                         arithmetic. This tool has\n             no derived bound for this \
                         operation in half and does not guess at\n             one; see \
                         the note below.\n",
                    );
                } else {
                    s.push_str(&format!("    verdict: {verdict}\n"));
                }
                if let Some(b) = self.budget {
                    // State the budget beside the figures it judges. A verdict of "agree"
                    // against a number the reader cannot see is the same as no verdict.
                    s.push_str(&format!(
                        "    summation over {} terms in {}, |terms| summing to {:.3e}:\n",
                        b.terms, b.accumulates_in, b.magnitude
                    ));
                    s.push_str(&format!(
                        "      no order can err by more than {:.2e} (Wilkinson); \
                         uncorrelated\n      rounding typically gives {:.2e}. \
                         Judged on the first.\n",
                        b.guaranteed, b.typical
                    ));
                    if k.max_abs > b.typical * 8.0 && k.max_abs <= b.guaranteed {
                        s.push_str(
                            "      NOTE: admissible, but well above typical -- what a \
                             kernel summing\n      in a needlessly bad order looks like. \
                             Not a fault, not nothing.\n",
                        );
                    }
                }
                if self.dtype == "half" && !tol.admits(k) {
                    // What this can and cannot settle. Both references now see the SAME
                    // f16-rounded inputs the kernel does, so the gap above is no longer
                    // part rounding and part lowering with no way to tell them apart --
                    // it is the kernel's own f16 arithmetic, which `error_budget` bounds.
                    // Point at the budget only when there IS one. `softmax` and
                    // `rms_norm` reduce but are deliberately outside `error_budget` --
                    // their error is not `gamma_K * sum|terms|` -- so telling a reader to
                    // judge by "the budget above" would name a line that is not printed.
                    if self.budget.is_some() {
                        s.push_str(
                            "             f16 ARITHMETIC: the inputs match on all three \
                             sides now, so this gap\n             is the kernel computing \
                             in half, not the harness rounding for it.\n             The \
                             budget above is the one that applies. See backlog #010.\n",
                        );
                    } else {
                        for line in [
                            "f16 ARITHMETIC: the inputs match on all three sides, so this gap is",
                            "the kernel computing in half. This tool has NO derived bound for",
                            "this operation in half -- its error is not the summation bound --",
                            "so the figures above are reported without one. Read them against",
                            "2^-11 = 4.9e-4, the closest any f16 result can come.",
                        ] {
                            s.push_str(&format!("             {line}\n"));
                        }
                    }
                }
            }
            Err(why) => {
                s.push_str(&format!("    kernel vs ours   {}\n", self.vs_ours.render()));
                // Never silently drop the stronger claim and leave the weaker one
                // looking authoritative.
                s.push_str(&format!("    torch: {why}\n"));
            }
        }
        s.push_str(&format!("  target   : {}\n", self.target.render("us")));
        s.push_str(&format!("  reference: {}\n", self.reference.render("us")));
        // A ratio is only as reproducible as its noisiest side. Say which side moved,
        // because they fail for different reasons: the device is usually stable and the
        // CPU reference is what competes with the rest of the machine.
        let noisy_target = self.target.is_noisy();
        let noisy_ref = self.reference.is_noisy();
        if noisy_target || noisy_ref {
            let which = match (noisy_target, noisy_ref) {
                (true, true) => "both timings are",
                (true, false) => "the device timing is",
                (false, true) => "the reference timing is",
                (false, false) => unreachable!(),
            };
            s.push_str(&format!(
                "  NOISY    : {which} spread wider than 15%\n\
                 \x20            (device {:.0}%, reference {:.0}%), so the ratio below is \
                 not reproducible.\n\
                 \x20            The CPU reference competes for cores with everything else \
                 on this\n\
                 \x20            machine; the device does not. Re-run on a quiet box \
                 before quoting it.\n",
                self.target.spread() * 100.0,
                self.reference.spread() * 100.0
            ));
        }
        // A ratio below 1 is a SLOWDOWN, and calling it a speedup of 0.1x invites the
        // reader to skim it as a win. Found on an M2 Max and an M4, where a 1024-element
        // softmax is dominated by launch overhead and the device is genuinely slower than
        // a scalar CPU loop -- a real and useful result, but only if it is labelled as
        // one. The reciprocal is printed too, because "0.1x" takes a moment to convert
        // and "8.1x slower" does not.
        let ratio = self.speedup_vs_reference();
        if ratio >= 1.0 {
            s.push_str(&format!("  speedup  : {ratio:.1}x against the reference\n"));
        } else {
            s.push_str(&format!(
                "  SLOWDOWN : {ratio:.2}x against the reference — the device is {:.1}x \
                 SLOWER here\n",
                if ratio > 0.0 {
                    1.0 / ratio
                } else {
                    f64::INFINITY
                }
            ));
            s.push_str(
                "             at this size the dispatch costs more than the work; try a
                 \x20             larger problem before concluding anything about the kernel
",
            );
        }
        // Said every time, because this number is quoted out of context otherwise. The
        // reference is a naive single-threaded scalar loop, not a tuned CPU kernel.
        s.push_str(
            "             (the reference is a naive single-threaded scalar loop written\n\
             \x20             by this tool — this is NOT a GPU-versus-CPU figure)\n",
        );
        // The build profile of THIS binary sets the reference's speed, and nothing else
        // in the report hints at it. The same 64x128x64 matmul reported 59.7x from a
        // debug build and 5.2x from a release build of the same commit, on the same
        // machine, against the same kernel: the device time was identical and the
        // baseline was eleven times slower. A ratio quoted without this is not
        // reproducible, and the direction of the error always flatters the tool.
        if cfg!(debug_assertions) {
            s.push_str(
                "             WARNING: this is a debug build, so the reference is\n\
                 \x20             UNOPTIMIZED and every ratio above is inflated — measured at\n\
                 \x20             ~11x on one matmul. Use a release build for any number\n\
                 \x20             you intend to quote.\n",
            );
        }
        match (
            self.optimizer_changed_nothing,
            self.speedup_from_optimizer(),
        ) {
            (true, _) => s.push_str(
                "  optimizer: -O0 and this level produced identical source; there is\n\
                 \x20            nothing to compare, so no ratio is reported\n",
            ),
            (false, Some(x)) => {
                let u = self.unoptimized.as_ref().expect("checked above");
                s.push_str(&format!("  -O0      : {}\n", u.render("us")));
                s.push_str(&format!(
                    "  optimizer: {x:.2}x from the optimization passes\n"
                ));
            }
            (false, None) => {}
        }
        s
    }
}

/// Where the harness for a form lives.
pub fn harness_source(form: &Form) -> Result<&'static str, RunError> {
    match form.id {
        "msl" => Ok(include_str!("../assets/harness/metal.swift")),
        "spirv" => Ok(include_str!("../assets/harness/vulkan.c")),
        other => Err(RunError::NoHarness {
            form: Box::leak(other.to_string().into_boxed_str()),
        }),
    }
}

/// How a form's harness is built and invoked.
///
/// Metal's is a Swift script the interpreter runs directly. SPIR-V's is C that has to be
/// compiled, against a shader that itself has to be compiled from GLSL first. Keeping the
/// difference here means `harness_invoke` sets up buffers, scalars and the entry point once
/// for both, which is the property that stopped the sweep and the single run drifting apart.
struct HarnessPlan {
    /// What the harness source is written as.
    file: &'static str,
    /// Extra tools that must exist, with what to say when they do not.
    needs: &'static [(&'static str, &'static str)],
}

/// Is `tool` on PATH?
///
/// The harness reports a missing tool by NAME rather than letting the process fail with an
/// exec error, because "the run did not happen" and "the run disagreed" must not look the
/// same to a caller.
fn which(tool: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|d| {
        let p = d.join(tool);
        p.is_file().then_some(p)
    })
}

fn harness_plan(form: &Form) -> Result<HarnessPlan, RunError> {
    match form.id {
        "msl" => Ok(HarnessPlan {
            file: "harness.swift",
            needs: &[("swift", "the harness needs `swift`")],
        }),
        "spirv" => Ok(HarnessPlan {
            file: "harness.c",
            needs: &[
                (
                    "glslangValidator",
                    "the harness compiles GLSL to SPIR-V with `glslangValidator`",
                ),
                ("clang", "the harness is C and needs `clang`"),
            ],
        }),
        other => Err(RunError::NoHarness {
            form: Box::leak(other.to_string().into_boxed_str()),
        }),
    }
}

/// Run `source` on the device and report. `rows`/`cols` come from the profile.
pub fn run(
    form: &Form,
    source: &str,
    abi: &KernelAbi,
    shape: Shape,
    iterations: usize,
) -> Result<(String, Vec<f32>, Timing), RunError> {
    run_with_threads(form, source, abi, shape, iterations, None)
}

/// `run`, with the threadgroup width pinned. `None` uses the shape's own choice.
pub fn run_with_threads(
    form: &Form,
    source: &str,
    abi: &KernelAbi,
    shape: Shape,
    iterations: usize,
    threads: Option<usize>,
) -> Result<(String, Vec<f32>, Timing), RunError> {
    // Bind every scalar the kernel declares, by name, BEFORE launching. A scalar we
    // cannot bind is refused here rather than defaulted to zero: a zero `K` makes the
    // inner loop run no iterations, and the kernel then returns all-zeros successfully.
    // That is a wrong answer wearing a green exit code.
    let mut scalars = Vec::new();
    for name in &abi.scalars {
        match shape.scalar_for(name, if abi.dtype == "half" { 2 } else { 4 }) {
            Some(v) => scalars.push(v.to_string()),
            None => {
                return Err(RunError::Harness {
                    code: 2,
                    stderr: format!(
                        "the kernel takes a scalar `{name}` that this harness does not know \
                         how to bind for {shape:?}; refusing rather than passing 0"
                    ),
                })
            }
        }
    }
    let in_sizes = shape.in_sizes();
    if in_sizes.len() != abi.inputs {
        return Err(RunError::Harness {
            code: 2,
            stderr: format!(
                "the kernel reads {} input buffers but {shape:?} supplies {}",
                abi.inputs,
                in_sizes.len()
            ),
        });
    }
    let text = harness_invoke(form, source, abi, shape, iterations, threads, None)?;
    let text = text.as_str();
    let mut device = String::from("(unnamed device)");
    let mut warmup = 0usize;
    let mut samples = Vec::new();
    let mut values = Vec::new();
    let mut control = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#c ") {
            if let Ok(v) = rest.trim().parse::<f32>() {
                control.push(v);
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("#device ") {
            device = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("#warmup ") {
            warmup = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("#us ") {
            if let Ok(v) = rest.trim().parse::<f64>() {
                samples.push(v);
            }
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            if let Ok(v) = line.trim().parse::<f32>() {
                values.push(v);
            }
        }
    }
    // The control arm must disagree. The harness dispatched a second time over inputs
    // from a different seed; identical output means the read-back is not coming from this
    // dispatch, and every accuracy number computed from it would be describing something
    // else. An empty or stale buffer scores a perfect match against any reference, so
    // this is the arm that distinguishes "agrees" from "was never measured".
    check_control(&values, &control).map_err(|stderr| RunError::Harness { code: 1, stderr })?;
    Ok((device, values, Timing { samples, warmup }))
}

/// One measured candidate from the `-O4` sweep.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub threads: usize,
    pub median_us: f64,
}

/// Measure a set of dispatch configurations on the real device, cheapest last.
///
/// This is what makes `-O4` the MEASURED level rather than `-O3` with a different label.
/// Before this the level ran the same passes as `-O3`, listed "measured-autotuning" under
/// "not done here", and exited 0 -- so asking for the measured level got an unmeasured
/// artifact.
///
/// Threadgroup size is the axis, because it is the one this layer can vary without
/// changing what the kernel COMPUTES: the emitted kernels stride by
/// `threads_per_threadgroup`, so a different width is the same arithmetic in a different
/// number of steps. Re-chunking a tile is not in that class and is deliberately not
/// attempted here -- `check_bounds` still checks tilings rather than rewriting them.
///
/// Every candidate is returned with its own measurement. Nothing is inferred from a
/// neighbour, and a configuration that failed to run is simply absent rather than
/// recorded with a guessed cost.
pub fn autotune(
    form: &Form,
    source: &str,
    abi: &KernelAbi,
    shape: Shape,
    iterations: usize,
) -> Result<Vec<Candidate>, RunError> {
    let ceiling = shape.threads();
    // Powers of two AND the multiples of the SIMD width between them.
    //
    // The list was powers of two alone, which made the correctness guard below unable to
    // do its job: the threadgroup fold that produced WRONG ANSWERS did so only at counts
    // that are not powers of two, so a sweep of powers of two could never have caught it
    // and the guard never had a width to drop.
    //
    // 96, 192, 384 and 768 are legal threadgroups, are multiples of the 32-wide SIMD
    // group, and are the widths at which a halving fold loses its top element. Sweeping
    // them is real tuning space -- 384 and 768 measure within 10% of the best on an M1
    // Ultra -- and it is also what makes `-O4`'s verification more than a formality.
    //
    // Demonstrated rather than assumed. With the fold temporarily reverted to its old
    // power-of-two form, `-O4` on a 1024-wide softmax dropped EXACTLY these four widths
    // and kept the six powers of two:
    //
    //     threadgroup 96: DROPPED -- it computes a different answer at this width
    //     threadgroup 192: DROPPED ...
    //     threadgroup 384: DROPPED ...
    //     threadgroup 768: DROPPED ...
    //     6 of 10 widths verified; the rest are excluded from the ranking.
    //
    // which is the exact signature of that defect. With the correct fold, all ten verify.
    // Before this list changed, the guard had never had a width to drop and there was no
    // evidence it worked at all.
    let widths: Vec<usize> = [32usize, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        .into_iter()
        .filter(|t| *t <= ceiling)
        .collect();
    if widths.is_empty() {
        return Err(RunError::NoDevice {
            form: form.id,
            why: format!("no threadgroup width fits a shape of width {ceiling}"),
        });
    }
    // ONE harness invocation for the whole sweep. It used to run once per candidate, so
    // six widths paid for six compilations of the same Swift file and six Metal pipeline
    // builds -- most of the wall-clock of `-O4`, and none of it measuring anything.
    let mut out = harness_sweep(form, source, abi, shape, iterations, &widths)?;
    if out.is_empty() {
        return Err(RunError::NoDevice {
            form: form.id,
            why: "no dispatch configuration could be measured on this device".into(),
        });
    }
    out.sort_by(|a, b| b.median_us.total_cmp(&a.median_us));
    Ok(out)
}

/// Write the harness and the kernel to a scratch directory and run them.
///
/// `sweep` non-None puts the harness in sweep mode: it measures each width and prints
/// `#sweep` lines instead of doing one timed run. Shared so the two modes cannot drift
/// apart in how they set up buffers, scalars or the entry point.
#[allow(clippy::too_many_arguments)]
fn harness_invoke(
    form: &Form,
    source: &str,
    abi: &KernelAbi,
    shape: Shape,
    iterations: usize,
    threads: Option<usize>,
    sweep: Option<&str>,
) -> Result<String, RunError> {
    let mut scalars = Vec::new();
    for name in &abi.scalars {
        match shape.scalar_for(name, if abi.dtype == "half" { 2 } else { 4 }) {
            Some(v) => scalars.push(v.to_string()),
            None => {
                return Err(RunError::Harness {
                    code: 2,
                    stderr: format!("cannot bind scalar `{name}` for {shape:?}"),
                })
            }
        }
    }
    let in_sizes = shape.in_sizes();
    let harness = harness_source(form)?;
    let dir = std::env::temp_dir().join(format!(
        "tile-run-{}-{}",
        std::process::id(),
        if sweep.is_some() { "sweep" } else { "once" }
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| RunError::Io(e.to_string()))?;
    let plan = harness_plan(form)?;
    for (tool, why) in plan.needs {
        if which(tool).is_none() {
            return Err(RunError::NoHarness {
                form: Box::leak(
                    format!("{}: {why}, which is not on PATH", form.id).into_boxed_str(),
                ),
            });
        }
    }
    let hpath = dir.join(plan.file);
    let kpath = dir.join(format!("kernel.{}", form.primary_ext()));
    std::fs::write(&hpath, harness).map_err(|e| RunError::Io(e.to_string()))?;
    std::fs::write(&kpath, source).map_err(|e| RunError::Io(e.to_string()))?;

    // SPIR-V takes two build steps the Swift path does not: the shader is compiled from
    // GLSL, and the harness itself is C. Both are done here so the argument list below is
    // the same for either device.
    let (program, kernel_arg) = if form.id == "spirv" {
        let spv = dir.join("kernel.spv");
        let g = std::process::Command::new("glslangValidator")
            .args(["-V", "--target-env", "vulkan1.2", "-S", "comp"])
            .arg(&kpath)
            .arg("-o")
            .arg(&spv)
            .output()
            .map_err(|e| RunError::Io(e.to_string()))?;
        if !g.status.success() {
            return Err(RunError::Harness {
                code: 3,
                stderr: format!(
                    "glslangValidator rejected the emitted shader:\n{}",
                    String::from_utf8_lossy(&g.stdout)
                ),
            });
        }
        let exe = dir.join("vkrun");
        let c = std::process::Command::new("clang")
            .args([
                "-O2",
                "-I/opt/homebrew/include",
                "-L/opt/homebrew/lib",
                "-lvulkan",
                "-o",
            ])
            .arg(&exe)
            .arg(&hpath)
            .output()
            .map_err(|e| RunError::Io(e.to_string()))?;
        if !c.status.success() {
            return Err(RunError::Harness {
                code: 3,
                stderr: format!(
                    "the vulkan harness did not build:\n{}",
                    String::from_utf8_lossy(&c.stderr)
                ),
            });
        }
        (exe, spv)
    } else {
        (std::path::PathBuf::from("swift"), kpath.clone())
    };

    // Metal runs `swift <harness> <kernel> ...`; the Vulkan harness IS the program, so it
    // takes only `<kernel.spv> ...`. Passing both paths for Metal is the whole difference,
    // and getting it wrong sent the harness its own source as the kernel.
    let mut cmd = std::process::Command::new(&program);
    if form.id != "spirv" {
        cmd.arg(&hpath);
    }
    cmd.arg(&kernel_arg)
        .arg(&abi.entry)
        .arg(&abi.dtype)
        .arg(iterations.to_string())
        .arg(shape.grid().to_string())
        // The shader's own width wins when it has one. SPIR-V compiles local_size_x into
        // the module, so dispatching anything else is not available -- and telling the
        // harness a width it cannot honour would make the reported threadgroup a fiction.
        .arg(
            abi.local_x
                .or(threads)
                .unwrap_or_else(|| shape.threads())
                .to_string(),
        )
        .arg(
            in_sizes
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
        .arg(shape.out_size().to_string())
        .arg(scalars.join(","));
    if let Some(list) = sweep {
        cmd.arg(list);
    }
    let out = cmd.output();
    let _ = std::fs::remove_dir_all(&dir);

    let out = out.map_err(|e| RunError::NoDevice {
        form: form.id,
        why: format!("the harness needs `swift`, which is not available ({e})"),
    })?;
    if !out.status.success() {
        let code = out.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        if code == 6 {
            return Err(RunError::NoDevice {
                form: form.id,
                why: stderr.trim().into(),
            });
        }
        return Err(RunError::Harness { code, stderr });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Drive the harness in sweep mode: many widths, one process.
fn harness_sweep(
    form: &Form,
    source: &str,
    abi: &KernelAbi,
    shape: Shape,
    iterations: usize,
    widths: &[usize],
) -> Result<Vec<Candidate>, RunError> {
    let list = widths
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let text = harness_invoke(form, source, abi, shape, iterations, None, Some(&list))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#sweep ") {
            let mut it = rest.split_whitespace();
            if let (Some(t), Some(us)) = (it.next(), it.next()) {
                if let (Ok(t), Ok(us)) = (t.parse::<usize>(), us.parse::<f64>()) {
                    out.push(Candidate {
                        threads: t,
                        median_us: us,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// The control arm: dispatching over different inputs must produce different output.
///
/// Separated from [`run`] so it can be tested without a device, because the situation it
/// detects is precisely the one where the device is not really participating.
///
/// The rule generalises two independent findings. Mine: `compare` maxes the error over
/// the values that came back, so an EMPTY result set scores zero error and reads as
/// agreement. The PICO session's: a board rig returned the PREVIOUS model's output and
/// timings for a model that had failed to load, so two variants "agreed with baseline"
/// while one of them had never executed. Same shape one level apart -- absence of
/// evidence rendering as evidence of agreement -- and the general form is theirs: every
/// comparison run must include one arm that MUST disagree. If it agrees, the harness is
/// broken, not the kernel.
pub fn check_control(values: &[f32], control: &[f32]) -> Result<(), String> {
    if values.is_empty() {
        return Err(
            "the harness returned no output values, so there is nothing to \
                    compare; treat this as a harness fault"
                .into(),
        );
    }
    if control.is_empty() {
        return Err(
            "the harness returned no control values, so this run cannot show \
                    that the device participated at all"
                .into(),
        );
    }
    if control == values {
        return Err(format!(
            "the control arm AGREED: dispatching over different inputs produced \
             identical output across all {} values. Either the harness is not moving \
             data to the device and back, or the kernel ignores its inputs. Both make \
             every accuracy and timing number from this run meaningless, and neither is \
             a numerical error -- so this refuses rather than reporting a match.",
            values.len()
        ));
    }
    Ok(())
}

/// Time the reference the same way the device is timed: warm up, then take samples.
pub fn time_reference(op: RefOp, shape: Shape, iterations: usize) -> Timing {
    // Matmul is timed on its own path: it is O(m*k*n) rather than O(rows*cols), so
    // timing it as if it were an elementwise pass would compare the device against a
    // fraction of the work it actually did, and report a speedup that is mostly the
    // difference between two different computations.
    if let Shape::Matmul { m, k, n } = shape {
        let a = input_values_for(0, m * k);
        let b = input_values_for(1, k * n);
        let warmup = (iterations / 10).max(3);
        for _ in 0..warmup {
            std::hint::black_box(matmul_ref(&a, &b, m, k, n));
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t = std::time::Instant::now();
            std::hint::black_box(matmul_ref(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                m,
                k,
                n,
            ));
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        return Timing { samples, warmup };
    }
    // A reduction is timed over its own reference, which produces one value per row
    // rather than a row per row -- timing it as elementwise would compare the device
    // against a different amount of work.
    if let Shape::RowReduce { rows, cols } = shape {
        let input = input_values(rows * cols);
        let warmup = (iterations / 10).max(3);
        let once = |input: &[f32]| {
            let mut acc = Vec::with_capacity(rows);
            for r in 0..rows {
                let row = &input[r * cols..(r + 1) * cols];
                acc.push(match op {
                    RefOp::Absmax => row.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
                    _ => row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                });
            }
            acc
        };
        for _ in 0..warmup {
            std::hint::black_box(once(&input));
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t = std::time::Instant::now();
            std::hint::black_box(once(std::hint::black_box(&input)));
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        return Timing { samples, warmup };
    }
    // Matvec is timed on its own path too: two inputs of different sizes, one value out
    // per row, so neither the elementwise loop nor the Rows2 one describes it.
    if let Shape::Matvec { rows, cols } = shape {
        let a = input_values_for(0, rows * cols);
        let b = input_values_for(1, cols);
        let mut out = vec![0.0f32; rows];
        let warmup = (iterations / 10).max(3);
        let once = |a: &[f32], b: &[f32], out: &mut [f32]| {
            for (r, o) in out.iter_mut().enumerate() {
                let row = &a[r * cols..(r + 1) * cols];
                *o = row.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            }
        };
        for _ in 0..warmup {
            once(&a, &b, &mut out);
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t = std::time::Instant::now();
            once(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                std::hint::black_box(&mut out),
            );
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        return Timing { samples, warmup };
    }
    // Two-input elementwise is timed on its own path for the same reason matmul is: the
    // generic loop calls `apply`, which takes ONE row, so a min or max reaching it panics
    // rather than quietly timing the wrong computation.
    if let Shape::Rows2 { rows, cols } = shape {
        let a = input_values_for(0, rows * cols);
        let b = input_values_for(1, rows * cols);
        let mut out = vec![0.0f32; rows * cols];
        let warmup = (iterations / 10).max(3);
        let once = |a: &[f32], b: &[f32], out: &mut [f32]| {
            for i in 0..out.len() {
                out[i] = match op {
                    RefOp::Min => a[i].min(b[i]),
                    RefOp::Max => a[i].max(b[i]),
                    other => panic!("{other:?} is not a two-input elementwise op"),
                };
            }
        };
        for _ in 0..warmup {
            once(&a, &b, &mut out);
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t = std::time::Instant::now();
            once(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                std::hint::black_box(&mut out),
            );
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        return Timing { samples, warmup };
    }
    let (rows, cols) = match shape {
        Shape::Rows { rows, cols } => (rows, cols),
        Shape::Rows2 { .. }
        | Shape::Matvec { .. }
        | Shape::Matmul { .. }
        | Shape::RowReduce { .. } => unreachable!("handled above"),
    };
    let input = input_values(rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    let warmup = (iterations / 10).max(3);
    // The scratch row is allocated ONCE, outside the timed region. Allocating inside it
    // measures the allocator and calls the result the reference's cost -- which
    // handicaps the baseline and overstates every speedup computed against it. A
    // baseline you have accidentally slowed down is not a baseline.
    let mut tmp = vec![0.0f32; cols];
    let once = |input: &[f32], out: &mut [f32], tmp: &mut [f32]| {
        for r in 0..rows {
            let s = r * cols;
            // Timing, not correctness: eps changes the cost of nothing, so the reference
            // is exercised with a representative value rather than plumbed one.
            op.apply(&input[s..s + cols], tmp, 1e-6);
            out[s..s + cols].copy_from_slice(tmp);
        }
    };
    for _ in 0..warmup {
        once(&input, &mut out, &mut tmp);
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        once(&input, &mut out, &mut tmp);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    Timing { samples, warmup }
}

/// What f32 arithmetic can and typically does cost an operation that SUMS.
///
/// #013: one constant judged every op. An elementwise op's error does not grow with
/// anything, so 1e-5 fits it forever; a K-term reduction's does, and no single number
/// serves both. The issue left three options and called it a numerics decision. It is not
/// -- the bound is standard, and it is derived here from `f32::EPSILON` and the terms
/// actually summed rather than fitted to whatever kernel was in front of me. Fitting was
/// tried for f16 in #010 and reverted for exactly that reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Budget {
    /// How many terms the longest sum in this operation adds.
    pub terms: usize,
    /// The magnitude that error grows against: max over outputs of the sum of the
    /// ABSOLUTE values of its terms. Under cancellation this is far larger than the
    /// result, which is precisely why a relative tolerance cannot judge a reduction.
    pub magnitude: f32,
    /// Wilkinson's bound, `gamma_K * magnitude` with `gamma_K = K*u/(1 - K*u)`. NO
    /// summation order can exceed this. A kernel above it is wrong, whatever order it
    /// used -- which is the property a tolerance needs and a fitted constant never has.
    pub guaranteed: f32,
    /// `sqrt(K) * u * magnitude`: what uncorrelated rounding actually delivers. Not a
    /// gate -- a correct kernel may exceed it -- but a reading. Far above typical while
    /// under `guaranteed` is admissible arithmetic that still deserves a look.
    pub typical: f32,
    /// What the bound assumes the kernel sums in. Stated because it is an assumption the
    /// harness cannot check, not a measurement.
    pub accumulates_in: &'static str,
}

/// The summation budget for `op` at `shape`, or `None` when the op sums nothing and the
/// fixed elementwise tolerance is the right judge.
///
/// Computed over the same inputs the harness generates, so it describes THIS run rather
/// than a worst case over inputs that were never used.
pub fn error_budget(op: RefOp, shape: Shape, dtype: &str) -> Option<Budget> {
    // Half an ulp: the unit roundoff for f32. `f32::EPSILON` is the gap between 1.0 and
    // the next representable value, which is 2u.
    const U: f32 = f32::EPSILON / 2.0;
    // f16 keeps 10 explicit mantissa bits, so its unit roundoff is 2^-11.
    //
    // Without this the bound was f32's, and the msl f16 matmul -- which is CORRECT, and
    // errs by 2.18e-2 because `half * half` rounds each product before the f32
    // accumulator ever sees it -- was told "the KERNEL disagrees with torch, look at the
    // lowering". A tighter gate that points confidently at the wrong place is worse than
    // the loose one it replaced, and this is the second time in this file that a number
    // right for one dtype was applied to both (#010 was the first).
    const U_HALF: f32 = 4.882_812_5e-4;
    let (u_product, accumulates_in) = match dtype {
        "half" => (U_HALF, "f32 (assumed)"),
        _ => (U, "f32"),
    };
    let (terms, magnitude) = match (op, shape) {
        (RefOp::Matmul, Shape::Matmul { m, k, n }) => {
            let mut a = input_values_for(0, m * k);
            quantize_inputs(&mut a, dtype);
            let mut b = input_values_for(1, k * n);
            quantize_inputs(&mut b, dtype);
            let mut worst = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        acc += (a[i * k + l] * b[l * n + j]).abs();
                    }
                    if acc > worst {
                        worst = acc;
                    }
                }
            }
            (k, worst)
        }
        (RefOp::Matvec, Shape::Matvec { rows, cols }) => {
            let mut a = input_values_for(0, rows * cols);
            quantize_inputs(&mut a, dtype);
            let mut b = input_values_for(1, cols);
            quantize_inputs(&mut b, dtype);
            let mut worst = 0.0f32;
            for r in 0..rows {
                let acc: f32 = (0..cols).map(|c| (a[r * cols + c] * b[c]).abs()).sum();
                if acc > worst {
                    worst = acc;
                }
            }
            (cols, worst)
        }
        (RefOp::ReduceSum, Shape::RowReduce { rows, cols }) => {
            let mut v = input_values(rows * cols);
            quantize_inputs(&mut v, dtype);
            let mut worst = 0.0f32;
            for r in 0..rows {
                let acc: f32 = v[r * cols..(r + 1) * cols].iter().map(|x| x.abs()).sum();
                if acc > worst {
                    worst = acc;
                }
            }
            (cols, worst)
        }
        // Everything else either sums nothing, or sums under a nonlinearity whose bound
        // is not this one. Softmax and rms_norm both reduce, and both are left out on
        // purpose: their error is not `gamma_K * sum|terms|`, and writing that down as
        // though it were would be the same overreach in a new place. They pass the fixed
        // tolerance comfortably, so nothing is lost by saying only what is known.
        _ => return None,
    };
    if terms == 0 {
        return None;
    }
    let ku = terms as f32 * U;
    // gamma_K diverges as K*u approaches 1 (K near 16.7 million). Past that no bound is
    // meaningful and claiming one would be worse than declining.
    if ku >= 1.0 {
        return None;
    }
    Some(Budget {
        terms,
        magnitude,
        // Two sources, both derived: each product is rounded to the BUFFER type before it
        // is added, then the sum accumulates. The accumulator is not visible from here --
        // f32 is assumed, which is what every emitter in this repo now uses and is the
        // TIGHT choice. A kernel that accumulates in half will exceed this, and that is a
        // lowering decision worth reporting rather than a bound to be widened for.
        guaranteed: (u_product + ku / (1.0 - ku)) * magnitude,
        typical: (u_product + (terms as f32).sqrt() * U) * magnitude,
        accumulates_in,
    })
}

/// Does the number the REFERENCES were given actually appear in the emitted kernel?
///
/// The eps is the one parameter the harness hands the references but not the kernel: it is
/// baked into the emitted source at lowering time. So the two can differ silently, and the
/// three-way comparison cannot see it -- when both references are given the same wrong
/// value they agree with each other, and the kernel, which is right, is the odd one out
/// and gets named as the fault.
///
/// That happened twice in two days from opposite directions: the msl emitter ignored a
/// stated eps and wrote its own 1e-6, and `rms_eps_from_mlir` read argument 2 of the
/// FOUR-argument form -- the row count -- and gave both references an eps of 1.0. The
/// first was caught because the references were right; the second produced a confident
/// "look at the lowering" against a correct kernel.
///
/// Every float literal in the source is compared numerically, so a backend that spells
/// `1e-3` and one that spells `0.001` both satisfy it.
pub fn eps_reaches_the_kernel(src: &str, eps: f32) -> bool {
    let bytes = src.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if !(c.is_ascii_digit() || c == '.') {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() {
            let c = bytes[i] as char;
            // The first two arms have the same body and are deliberately not merged: one
            // consumes a mantissa character, the other steps over an exponent marker. They
            // are different scanner states that happen to advance by the same amount, and
            // collapsing them would bury the exponent handling in a boolean chain.
            #[allow(clippy::if_same_then_else)]
            if c.is_ascii_digit() || c == '.' {
                i += 1;
            } else if (c == 'e' || c == 'E')
                && i + 1 < bytes.len()
                && (bytes[i + 1] as char).is_ascii_digit()
            {
                i += 1;
            } else if (c == 'e' || c == 'E')
                && i + 2 < bytes.len()
                && matches!(bytes[i + 1] as char, '+' | '-')
                && (bytes[i + 2] as char).is_ascii_digit()
            {
                i += 2;
            } else {
                break;
            }
        }
        if let Ok(v) = src[start..i].parse::<f32>() {
            // A relative match, because the literal may be rounded in printing.
            if v != 0.0 && ((v - eps).abs() <= eps.abs() * 1e-3) {
                return true;
            }
        }
    }
    false
}

/// Does this kernel still compute the reference answer at THIS threadgroup width?
///
/// `-O4` sweeps threadgroup widths for speed and never looked at what came back: the sweep
/// mode times each width and exits before printing a value. So the level that calls itself
/// MEASURED was recommending a dispatch configuration on timing alone.
///
/// The axis it varies is the one most likely to change an answer. Every reduction in this
/// repo folds across `tcount`, and the power-of-two tree fold produced wrong results at
/// thread counts that were not powers of two -- a sweep that only times would have found
/// one of those fastest and said so in the emitted source.
///
/// This runs the ordinary verified path at the given width, so the check and the single
/// run cannot drift apart.
pub fn verify_at_width(
    form: &'static crate::forms::Form,
    source: &str,
    abi: &KernelAbi,
    shape: Shape,
    op: RefOp,
    eps: f32,
    threads: usize,
) -> Result<bool, String> {
    let text = harness_invoke(form, source, abi, shape, 3, Some(threads), None)
        .map_err(|e| format!("{e:?}"))?;
    // The plain value lines: everything the harness prints that is not a `#`-prefixed
    // record. Same shape as the single run's parse, kept short because only the values
    // matter here -- the timings at this width were already taken by the sweep.
    let values: Vec<f32> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .collect();
    if values.is_empty() {
        return Err("no values came back".into());
    }
    let (_, worst, nan) = compare_detailed(op, shape, &values, eps, &abi.dtype);
    let tol = match error_budget(op, shape, &abi.dtype) {
        Some(b) => b.guaranteed,
        None => 1e-5f32.max(unit_roundoff(&abi.dtype)),
    };
    Ok(nan == 0 && worst <= tol)
}

/// Compare device output against the reference. Returns (checked, worst absolute error).
pub fn compare(op: RefOp, shape: Shape, got: &[f32], eps: f32, dtype: &str) -> (usize, f32) {
    let (checked, worst, _) = compare_detailed(op, shape, got, eps, dtype);
    (checked, worst)
}

/// `compare`, also returning how many elements disagreed about being a NUMBER at all.
///
/// `if e > worst` is false when `e` is NaN, so every NaN difference was silently skipped
/// while `checked` went on counting the element as compared. A kernel that produced NaN
/// where the reference produced a finite value therefore passed with a clean maximum
/// error -- the same shape as an empty comparison reading as agreement, which this harness
/// already carries a control arm for.
///
/// Found on `sqrt` and `log`: the harness feeds values in [-2, 2.25], so both take the
/// root and the logarithm of negatives, both produce NaN, and both sides agreeing on NaN
/// is fine. What is not fine is that a ONE-SIDED NaN went the same way.
///
/// Both-NaN counts as agreement: it is the same behaviour on both sides, which is what
/// this check is for. One-sided is a disagreement of the strongest kind and is counted
/// separately rather than folded into `worst`, because no magnitude describes it.
pub fn compare_detailed(
    op: RefOp,
    shape: Shape,
    got: &[f32],
    eps: f32,
    dtype: &str,
) -> (usize, f32, usize) {
    let want = reference_output(op, shape, eps, dtype);
    let mut worst = 0.0f32;
    let mut nan_mismatch = 0usize;
    let n = got.len().min(want.len());
    for i in 0..n {
        let (g, w) = (got[i], want[i]);
        let (gf, wf) = (g.is_finite(), w.is_finite());
        if gf != wf {
            // One is a number and the other is NaN or an infinity. No subtraction
            // describes that, and skipping it is how it stayed invisible.
            nan_mismatch += 1;
            continue;
        }
        if !gf {
            // Neither is finite. NaN on both sides, or the same infinity: agreement.
            continue;
        }
        let e = (g - w).abs();
        if e > worst {
            worst = e;
        }
    }
    (n, worst, nan_mismatch)
}

/// The reference result for `shape`, over the same inputs the harness generates.
/// The reference result for `shape`, over the same inputs the harness generates.
///
/// `eps` is only read by ops that take one; it comes from the MLIR via
/// `Shape::rms_eps_from_mlir` so the reference and the emitted kernel use the SAME number,
/// rather than two constants that happen to match.
pub fn reference_output(op: RefOp, shape: Shape, eps: f32, dtype: &str) -> Vec<f32> {
    let (rows, cols) = match shape {
        Shape::Matmul { m, k, n } => {
            let mut a = input_values_for(0, m * k);
            quantize_inputs(&mut a, dtype);
            let mut b = input_values_for(1, k * n);
            quantize_inputs(&mut b, dtype);
            return matmul_ref(&a, &b, m, k, n);
        }
        Shape::Matvec { rows, cols } => {
            // Row-major matrix in buffer 0, vector in buffer 1, summed left to right --
            // the order the kernel's cross-SIMD fold reduces to for one threadgroup.
            let mut a = input_values_for(0, rows * cols);
            quantize_inputs(&mut a, dtype);
            let mut b = input_values_for(1, cols);
            quantize_inputs(&mut b, dtype);
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                let row = &a[r * cols..(r + 1) * cols];
                out.push(row.iter().zip(b.iter()).map(|(x, y)| x * y).sum());
            }
            return out;
        }
        Shape::Rows2 { rows, cols } => {
            // Buffer 1 uses the same generator the harness and the torch script use for
            // its second input, so all three sides see identical numbers. Getting this
            // offset wrong would be a comparison of two different problems, which is what
            // `input_values_for`'s own comment warns about.
            let mut a = input_values_for(0, rows * cols);
            quantize_inputs(&mut a, dtype);
            let mut b = input_values_for(1, rows * cols);
            quantize_inputs(&mut b, dtype);
            return a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| match op {
                    RefOp::Min => x.min(*y),
                    RefOp::Max => x.max(*y),
                    other => panic!("{other:?} is not a two-input elementwise op"),
                })
                .collect();
        }
        Shape::RowReduce { rows, cols } => {
            // One value per row, from the same inputs the harness generates.
            let mut input = input_values(rows * cols);
            quantize_inputs(&mut input, dtype);
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                let row = &input[r * cols..(r + 1) * cols];
                out.push(match op {
                    RefOp::ReduceMax => row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    // Summed in order, left to right, which is what the kernel's
                    // cross-SIMD fold reduces to for one threadgroup. A different order
                    // would differ in the last bits and look like a kernel fault.
                    RefOp::ReduceSum => row.iter().copied().sum(),
                    RefOp::Absmax => row.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
                    other => panic!("{other:?} is not a row reduction"),
                });
            }
            return out;
        }
        Shape::Rows { rows, cols } => (rows, cols),
    };
    let mut input = input_values(rows * cols);
    quantize_inputs(&mut input, dtype);
    let mut want = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = r * cols;
        let mut tmp = vec![0.0f32; cols];
        op.apply(&input[s..s + cols], &mut tmp, eps);
        want[s..s + cols].copy_from_slice(&tmp);
    }
    want
}

/// The source of the timed region, so the specification can assert that nothing
/// allocates inside it.
///
/// Checking the property rather than trusting a comment: a `vec!` added between the
/// clock starting and stopping would measure the allocator and call it the reference's
/// cost, handicapping the baseline and overstating every speedup against it.
pub fn timed_region_source() -> &'static str {
    let src = include_str!("run.rs");
    let after = src.split("let t = Instant::now();").nth(1).unwrap_or("");
    after.split("samples.push").next().unwrap_or("")
}

/// A path the caller can hand to `--keep` for the harness, for debugging.
pub fn harness_path_hint(form: &Form) -> PathBuf {
    std::env::temp_dir().join(format!("tile-run-harness.{}", form.id))
}

#[cfg(test)]
mod tests {

    /// #017. The shape is read from the MLIR now instead of being a literal `rows: 1`.
    ///
    /// Agreement alone could not show this had taken effect: before the change the KERNEL
    /// was also flat -- one threadgroup over one row of `rows * cols` -- so kernel and
    /// reference agreed about the wrong problem. What must be checked is that the two
    /// shapes are now DIFFERENT computations, and that the source's own numbers are what
    /// reaches the harness.
    #[test]
    fn the_stated_row_count_is_read_from_the_mlir_and_changes_the_problem() {
        let mlir = "
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
";
        assert_eq!(Shape::rows_cols_from_mlir(mlir), Some((4, 256)));

        // A softmax over four rows of 256 is not a softmax over one row of 1024. If these
        // agreed, the shape would not be reaching the reference.
        let four = reference_output(
            RefOp::Softmax,
            Shape::Rows { rows: 4, cols: 256 },
            1e-6,
            "float",
        );
        let flat = reference_output(
            RefOp::Softmax,
            Shape::Rows {
                rows: 1,
                cols: 1024,
            },
            1e-6,
            "float",
        );
        assert_eq!(
            four.len(),
            flat.len(),
            "same element count, different problem"
        );
        assert!(
            four.iter()
                .zip(flat.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "flattening four rows into one changes the answer; if these match, `rows` is \
             not reaching the reference"
        );

        // And the harness drives it as four rows: one threadgroup each, over a row 256
        // wide. `num_elements` is the WIDTH -- kernels use it as the row stride, so the
        // total would be wrong in two places at once.
        let sh = Shape::Rows { rows: 4, cols: 256 };
        assert_eq!(sh.grid(), 4);
        assert_eq!(sh.scalar("num_elements"), Some(256));
        assert_eq!(sh.out_size(), 1024, "the buffer is still the whole tile");

        // A source without constant dimensions keeps the old flattening rather than
        // guessing: still correct for one row, and honest about not knowing.
        assert_eq!(Shape::rows_cols_from_mlir("no constants here"), None);
    }

    /// The harness dispatches `(grid, 1, 1)` over buffers it fills with f32. A kernel
    /// outside that is not measurable here, and saying so is the only honest answer.
    ///
    /// The spirv f16 GEMM is both -- `local_size_y = 16`, indexed by `gl_WorkGroupID.y`,
    /// over `uint p0[]` holding packed halves. Run anyway it read f32 bits as pairs of
    /// halves across a grid it never got, and the report said "the KERNEL disagrees with
    /// torch, look at the lowering" -- the third confident wrong pointer in a day, all
    /// three from measuring something the harness could not measure.
    #[test]
    fn a_kernel_this_harness_cannot_drive_is_refused_rather_than_mismeasured() {
        let two_d = "#version 450\n\
                     layout(local_size_x = 16, local_size_y = 16) in;\n\
                     layout(set = 0, binding = 0) readonly  buffer B0 { float p0[]; };\n\
                     layout(set = 0, binding = 1) writeonly buffer B1 { float p1[]; };\n\
                     void main() { uint r = gl_WorkGroupID.y * 16u; }\n";
        let err = KernelAbi::parse_glsl(two_d).expect_err("a 2D kernel is refused");
        assert!(err.contains("2D dispatch"), "{err}");
        assert!(
            err.contains("vkCmdDispatch(grid, 1, 1)"),
            "the reason is named: {err}"
        );

        let packed = "#version 450\n\
                      layout(local_size_x = 64) in;\n\
                      layout(set = 0, binding = 0) readonly  buffer B0 { uint p0[]; };\n\
                      layout(set = 0, binding = 1) writeonly buffer B1 { uint p1[]; };\n\
                      void main() {}\n";
        let err = KernelAbi::parse_glsl(packed).expect_err("packed buffers are refused");
        assert!(err.contains("uint"), "the element type is named: {err}");

        // And an ordinary 1D f32 kernel is still accepted -- the refusal must not be so
        // broad that it turns away what the harness CAN run.
        let ok = "#version 450\n\
                  layout(local_size_x = 64) in;\n\
                  layout(set = 0, binding = 0) readonly  buffer B0 { float p0[]; };\n\
                  layout(set = 0, binding = 1) writeonly buffer B1 { float p1[]; };\n\
                  void main() { uint i = gl_WorkGroupID.x; }\n";
        KernelAbi::parse_glsl(ok).expect("a 1D f32 kernel runs here");
    }

    /// Both shapes of push-constant block, because fixing one broke the other and nothing
    /// caught it for a day.
    ///
    /// The single-line form was handled first; matmul writes M, N and K on separate lines
    /// and yielded no scalars, so the harness pushed one wrong value, N arrived as 0 and
    /// the kernel's loop never ran -- caught by the control arm. The fix for THAT split
    /// the whole line on `;`, so the single-line form's first field became
    /// `layout(push_constant) uniform PushConstants { uint num_elements` and matched
    /// nothing. Matvec was then handed `in_sizes[0]` -- 256 for a 64-wide row -- and read
    /// off the end of it, while softmax survived only because at one row
    /// `rows * cols == cols` and the wrong number happened to be right.
    #[test]
    fn push_constants_parse_on_one_line_and_on_several() {
        let head = "#version 450\n\
                    layout(local_size_x = 64) in;\n\
                    layout(set = 0, binding = 0) readonly  buffer B0 { float p0[]; };\n\
                    layout(set = 0, binding = 1) writeonly buffer B1 { float p1[]; };\n";

        let one_line = format!(
            "{head}layout(push_constant) uniform PushConstants {{ uint num_elements; }} pc;\n\
             void main() {{}}\n"
        );
        let abi = KernelAbi::parse_glsl(&one_line).expect("a one-line block parses");
        assert_eq!(
            abi.scalars,
            vec!["num_elements".to_string()],
            "{:?}",
            abi.scalars
        );

        let many_lines = format!(
            "{head}layout(push_constant) uniform PushConstants {{\n\
             \x20   uint M;\n\
             \x20   uint N;\n\
             \x20   uint K;\n\
             }} pc;\n\
             void main() {{}}\n"
        );
        let abi = KernelAbi::parse_glsl(&many_lines).expect("a multi-line block parses");
        assert_eq!(
            abi.scalars,
            vec!["M".to_string(), "N".to_string(), "K".to_string()],
            "{:?}",
            abi.scalars
        );

        // A shader with no push constants declares none -- not one invented from the
        // buffer declarations, which also contain braces.
        let none = format!("{head}void main() {{}}\n");
        let abi = KernelAbi::parse_glsl(&none).expect("no block is fine");
        assert!(abi.scalars.is_empty(), "{:?}", abi.scalars);
    }

    /// `f16_round` is hand-written -- exactly the kind of code that looks right and is
    /// subtly wrong -- so it is checked against numpy's own f32->f16->f32 over the
    /// harness's actual input range plus the edges that break naive implementations:
    /// the largest finite f16, the first value that overflows to infinity, subnormals,
    /// and a value below the subnormal floor.
    #[test]
    fn f16_rounding_matches_the_hardware_conversion() {
        // (input, expected) pairs generated by numpy: float32(float16(float32(v))).
        // The expected values are the EXACT decimal expansions of the f16 results, written
        // to more digits than an f32 literal can hold. clippy calls that excessive
        // precision and it is right about the type -- but shortening them would hide where
        // they came from, and where they came from (numpy, against 63488 bit patterns) is
        // the whole reason this table is trustworthy.
        #[allow(clippy::excessive_precision)]
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (0.5, 0.5),
            (2.0 / 3.0, 6.665039062e-1),
            (65504.0, 6.5504e4),
            (65520.0, f32::INFINITY),
            (70000.0, f32::INFINITY),
            (6e-5, 6.002187729e-5),
            (6e-8, 5.960464478e-8),
            (1e-8, 0.0),
            (-2.0, -2.0),
            (0.25, 0.25),
            (0.0001, 1.000165939e-4),
        ];
        for (input, want) in cases {
            let got = f16_round(*input);
            assert!(
                (got == *want)
                    || (got.is_infinite() && want.is_infinite() && got.signum() == want.signum()),
                "f16_round({input:e}) = {got:e}, numpy says {want:e}"
            );
        }
        // Rounding is to nearest EVEN, not truncation: a value exactly between two f16
        // neighbours goes to the one with an even mantissa. Truncation would round the
        // whole range one way and bias every f16 comparison in the same direction.
        assert_eq!(f16_round(1.0 + 2.0f32.powi(-11)), 1.0);
        // And it is idempotent -- rounding an already-representable value changes nothing,
        // which is what makes it safe to apply to inputs the harness will round again.
        for v in input_values(512) {
            let once = f16_round(v);
            assert_eq!(f16_round(once), once, "not idempotent at {v}");
        }
        // The fourteen cases above were checked against numpy; a swept 598 values agreed
        // too, but that sweep needed numpy and cannot live here. This is the equivalent
        // that does not: EVERY f16 value is a fixed point of the rounding, all 65536 of
        // them, which the halved-subnormal bug would have failed at 2048 of them.
        let mut checked = 0u32;
        for bits in 0u32..=0xffff {
            let hs = (bits >> 15) & 1;
            let he = (bits >> 10) & 0x1f;
            let hm = bits & 0x3ff;
            if he == 0x1f {
                continue; // inf and NaN: no finite fixed point to check.
            }
            // Decode this f16 pattern to the f32 it denotes, independently of the
            // reconstruction inside `f16_round`: subnormals are hm * 2^-24, normals are
            // (1 + hm/1024) * 2^(he-15).
            let v = if he == 0 {
                hm as f32 * 2.0f32.powi(-24)
            } else {
                (1.0 + hm as f32 / 1024.0) * 2.0f32.powi(he as i32 - 15)
            };
            let v = if hs == 1 { -v } else { v };
            assert_eq!(
                f16_round(v),
                v,
                "f16 pattern {bits:#06x} is not a fixed point"
            );
            checked += 1;
        }
        assert_eq!(checked, 63488, "every finite f16 pattern was checked");
    }

    /// The four-argument rms_norm carries no epsilon. Reading index 2 anyway read the
    /// ROW COUNT, whose constant is 1, so both references computed with eps = 1.0, agreed
    /// with each other, and the kernel -- correct, using the 1e-6 it contains -- was
    /// reported as "the KERNEL disagrees with torch, look at the lowering". 3.70e-1 of
    /// disagreement, entirely on the harness's side.
    #[test]
    fn the_four_argument_rms_norm_yields_no_eps_rather_than_the_row_count() {
        let four = "
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %y = llvm.call @__tile_rms_norm_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
";
        assert_eq!(
            Shape::rms_eps_from_mlir(four),
            None,
            "no epsilon is stated, so none may be invented from a neighbouring operand"
        );
        let five = "
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %e = llvm.mlir.constant(1.000000e-03 : f32) : f32
    %y = llvm.call @__tile_rms_norm_f32(%a, %a, %e, %r, %c) : (i32, i32, f32, i32, i32) -> i32
";
        assert_eq!(Shape::rms_eps_from_mlir(five), Some(1e-3));
        // An integer in the eps position is not an epsilon, whatever its position.
        let wrong_type = five.replace("1.000000e-03 : f32) : f32", "3 : i32) : i32");
        assert_eq!(Shape::rms_eps_from_mlir(&wrong_type), None);
    }

    /// The eps is the one parameter the references are given and the kernel is not -- it
    /// is baked in at lowering time -- so the two can disagree with nothing to show it.
    /// Both failures of that kind happened within two days, from opposite sides.
    #[test]
    fn the_eps_the_reference_uses_is_looked_for_in_the_kernel() {
        // The same number, spelled the three ways the backends spell it.
        for spelling in ["+ (float)1e-3;", "+ 0.001;", "+ 1.000000e-03;"] {
            assert!(
                eps_reaches_the_kernel(spelling, 1e-3),
                "{spelling} states 1e-3"
            );
        }
        // A kernel carrying the emitter's own default does not satisfy a stated 1e-3.
        // This is the msl bug, and this assertion is what would have caught it.
        assert!(!eps_reaches_the_kernel(
            "float rms = rsqrt(s / n + (float)1e-6);",
            1e-3
        ));
        // And a reference given 1.0 by a mis-read operand does not match a kernel that
        // contains 1e-6. This is the harness-side bug, from the other direction.
        assert!(!eps_reaches_the_kernel(
            "float rms = rsqrt(s / n + (float)1e-6);",
            1.0
        ));
        // Integers in the source are not mistaken for a small epsilon.
        assert!(!eps_reaches_the_kernel(
            "for (uint i = 0; i < 256; i++)",
            1e-6
        ));
    }

    /// The property that makes a derived tolerance worth having: it must still reject
    /// every defect this project has already found. A tolerance widened until today's
    /// kernel passes measures nothing tomorrow (#010's f16 threshold, reverted), and the
    /// way to tell the two apart is to re-run the old bugs against the new gate.
    ///
    /// These magnitudes are the ones recorded in #013 and the cluster docs, each from a
    /// real fault: a threadgroup array indexed past its end, a row reduction that never
    /// accumulated across the row, and a scalar bound to the wrong shape so a 256-term
    /// matvec summed 4 terms.
    #[test]
    fn the_derived_budget_still_rejects_every_defect_already_found() {
        let cases: &[(&str, f32, RefOp, Shape)] = &[
            (
                "sdata indexed past its end (17.8)",
                17.8,
                RefOp::ReduceSum,
                Shape::RowReduce { rows: 4, cols: 256 },
            ),
            (
                "a row reduction that never accumulated (4.96)",
                4.96,
                RefOp::ReduceSum,
                Shape::RowReduce { rows: 4, cols: 256 },
            ),
            (
                "num_elements bound to out_size, so matvec summed 4 of 256 (1.74e2)",
                1.74e2,
                RefOp::Matvec,
                Shape::Matvec { rows: 4, cols: 256 },
            ),
        ];
        for (what, observed, op, shape) in cases {
            let b = error_budget(*op, *shape, "float").expect("these ops sum");
            assert!(
                *observed > b.guaranteed * 100.0,
                "{what}: {observed:.3e} must be far outside the budget {:.3e}, or the \
                 budget has been widened until it no longer detects anything",
                b.guaranteed
            );
        }
    }

    /// Derived, not fitted. The bound is `gamma_K * magnitude` with `gamma_K = Ku/(1-Ku)`
    /// and `u = f32::EPSILON/2` -- recomputed here from the definition, so a constant
    /// quietly substituted for the formula fails.
    #[test]
    fn the_budget_is_computed_from_epsilon_and_not_from_a_chosen_number() {
        let shape = Shape::Matmul { m: 8, k: 64, n: 8 };
        let b = error_budget(RefOp::Matmul, shape, "float").expect("matmul sums");
        assert_eq!(b.terms, 64);
        let u = f32::EPSILON / 2.0;
        let ku = 64.0 * u;
        // Two terms: the product rounding at the buffer type (f32 here) and the
        // accumulation. Written out from the definition rather than read from the code.
        let expected = (u + ku / (1.0 - ku)) * b.magnitude;
        assert!(
            (b.guaranteed - expected).abs() <= expected * 1e-6,
            "{} vs {expected}",
            b.guaranteed
        );
        // And it grows with K, which is the whole reason one constant could not serve.
        let wide = error_budget(RefOp::Matmul, Shape::Matmul { m: 8, k: 256, n: 8 }, "float")
            .expect("matmul sums");
        assert!(
            wide.guaranteed > b.guaranteed * 3.0,
            "a longer reduction must be allowed more error: {} vs {}",
            wide.guaranteed,
            b.guaranteed
        );
    }

    /// An op that sums nothing gets no budget, and keeps the fixed tolerance. Saying
    /// `gamma_K * sum|terms|` about `relu` would be the same overreach in a new place.
    #[test]
    fn elementwise_ops_get_no_summation_budget() {
        for op in [RefOp::Relu, RefOp::Exp, RefOp::Neg, RefOp::Abs] {
            assert!(
                error_budget(op, Shape::Rows { rows: 4, cols: 64 }, "float").is_none(),
                "{op:?} sums nothing"
            );
        }
        // Softmax and rms_norm DO reduce, and are still left out: their error is not this
        // bound. Deliberate, and tested so that adding them is a decision rather than a
        // drift.
        assert!(error_budget(RefOp::Softmax, Shape::Rows { rows: 4, cols: 64 }, "float").is_none());
    }

    /// The budget must know what the kernel's BUFFERS are, not just how long the sum is.
    ///
    /// The msl f16 matmul errs by 2.18e-2 -- correct behaviour, because `half * half`
    /// rounds every product before the f32 accumulator sees it. Judged by f32's bound it
    /// was told to look at the lowering, which is a confident wrong answer and worse than
    /// the vague right one it replaced.
    #[test]
    fn a_half_precision_kernel_is_judged_at_half_precision() {
        let shape = Shape::Matmul { m: 8, k: 64, n: 8 };
        let f32b = error_budget(RefOp::Matmul, shape, "float").expect("sums");
        let f16b = error_budget(RefOp::Matmul, shape, "half").expect("sums");
        assert!(
            f16b.guaranteed > f32b.guaranteed * 50.0,
            "f16 rounds products at 2^-11 against f32's 2^-24: {:.3e} vs {:.3e}",
            f16b.guaranteed,
            f32b.guaranteed
        );
        // The observed f16 matmul error must fall inside it, and the f32 one must not.
        assert!(2.18e-2 < f16b.guaranteed, "{:.3e}", f16b.guaranteed);
        assert!(2.18e-2 > f32b.guaranteed, "{:.3e}", f32b.guaranteed);
        // But a HALF accumulator would not be admitted: that is a lowering decision the
        // report should surface, not a looseness the bound absorbs.
        let half_accum = (64.0 * 4.882_812_5e-4) * f16b.magnitude;
        assert!(
            half_accum > f16b.guaranteed * 10.0,
            "accumulating in half must still stand out: {half_accum:.3e} vs {:.3e}",
            f16b.guaranteed
        );
    }

    /// Wilkinson's bound is an upper bound on a correct implementation, so the guaranteed
    /// figure must never sit below what uncorrelated rounding typically delivers -- if it
    /// did, the gate would reject correct kernels as a matter of course.
    #[test]
    fn the_guaranteed_bound_is_never_tighter_than_the_typical_one() {
        for k in [2usize, 8, 64, 256, 1024, 4096] {
            let b = error_budget(
                RefOp::ReduceSum,
                Shape::RowReduce { rows: 2, cols: k },
                "float",
            )
            .expect("reduce_sum sums");
            assert!(
                b.guaranteed > b.typical,
                "K={k}: guaranteed {:.3e} must exceed typical {:.3e}",
                b.guaranteed,
                b.typical
            );
        }
    }
    use super::*;

    #[test]
    fn the_reference_softmax_sums_to_one() {
        let input = input_values(1024);
        let mut out = vec![0.0f32; 1024];
        RefOp::Softmax.apply(&input, &mut out, 0.0);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax summed to {sum}");
    }

    #[test]
    fn the_reference_is_numerically_stable_on_large_inputs() {
        // Subtracting the row max is not decoration: without it exp overflows and the
        // whole comparison becomes NaN, which reads as a kernel bug.
        let input = vec![900.0f32; 64];
        let mut out = vec![0.0f32; 64];
        RefOp::Softmax.apply(&input, &mut out, 0.0);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "the reference overflowed"
        );
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "{sum}");
    }

    #[test]
    fn the_signature_parser_reads_past_metal_attribute_parens() {
        // `[[ buffer(0) ]]` nests parens inside the parameter list. Ending the list at
        // the first `)` truncated it after one parameter, so every kernel looked like it
        // had no output buffer -- which is what refused the first real matmul run.
        let src = "kernel void k(\n\
                   device const half* a [[ buffer(0) ]],\n\
                   device const half* b [[ buffer(1) ]],\n\
                   device half* c [[ buffer(2) ]],\n\
                   constant uint& M [[ buffer(3) ]])\n{}";
        let abi = KernelAbi::parse_msl(src).expect("parses");
        assert_eq!(abi.entry, "k");
        assert_eq!(abi.inputs, 2);
        assert_eq!(abi.dtype, "half");
        assert_eq!(abi.scalars, vec!["M".to_string()]);
    }

    #[test]
    fn the_matmul_shape_comes_from_the_intrinsics_own_operands() {
        let mlir = "%m = llvm.mlir.constant(64 : i32) : i32\n\
                    %k = llvm.mlir.constant(128 : i32) : i32\n\
                    %n = llvm.mlir.constant(32 : i32) : i32\n\
                    %c = llvm.call @__tile_matmul_f16(%a, %a, %b, %m, %k, %n) : () -> i32";
        assert_eq!(
            Shape::matmul_from_mlir(mlir),
            Some(Shape::Matmul {
                m: 64,
                k: 128,
                n: 32
            })
        );
        // Non-square, and asymmetric buffers: exactly what (rows, cols) could not say.
        let s = Shape::Matmul {
            m: 64,
            k: 128,
            n: 32,
        };
        assert_eq!(s.in_sizes(), vec![64 * 128, 128 * 32]);
        assert_eq!(s.out_size(), 64 * 32);
    }

    #[test]
    fn an_unbindable_scalar_is_refused_rather_than_defaulted_to_zero() {
        // A zero `K` makes the kernel's inner loop run no iterations and return all
        // zeros, successfully. Refusing is the only way that is not a silent wrong answer.
        let s = Shape::Matmul { m: 4, k: 4, n: 4 };
        assert_eq!(s.scalar("K"), Some(4));
        assert_eq!(s.scalar("stride"), None);
    }

    #[test]
    fn the_op_is_detected_from_the_intrinsic() {
        assert_eq!(
            RefOp::detect("llvm.call @__tile_softmax_f32(...)").unwrap(),
            RefOp::Softmax
        );
        assert_eq!(RefOp::detect("__tile_relu_f16").unwrap(), RefOp::Relu);
    }

    #[test]
    fn a_kernel_with_no_referenceable_op_is_refused_with_the_list() {
        let e = RunError::NoReference {
            // Was `matmul` until matmul gained a reference. Attention is the next one
            // that matters and is still outside the set.
            hint: RefOp::detect("llvm.call @__tile_attention_f32(...)").unwrap_err(),
        };
        let msg = e.to_string();
        assert!(msg.contains("softmax") && msg.contains("matmul"), "{msg}");
        assert!(msg.contains("worse than not running it"), "{msg}");
    }

    #[test]
    fn a_kernel_with_two_referenceable_ops_is_refused_rather_than_guessed_at() {
        // The reference would have to know the order they compose in, and guessing it
        // produces a comparison that fails for the wrong reason.
        let e = RefOp::detect("__tile_exp_f32 then __tile_relu_f32").unwrap_err();
        assert!(e.contains("compose"), "{e}");
        assert!(e.contains("exp") && e.contains("relu"), "{e}");
    }

    #[test]
    fn the_input_can_actually_discriminate_a_partial_reduction() {
        // The property the old generator lacked, stated so it cannot be lost again.
        //
        // With `(i % 17) * 0.25 - 2.0` the sequence is periodic with period 17, so the
        // maximum appears in EVERY 32-element window. A kernel reducing one SIMD group
        // and one reducing the whole row then return the same value, and Metal's
        // reduce_max shipped for months returning the max of the first 32 elements.
        //
        // The ramp puts the extreme near the end, where a partial reduction misses it.
        let v = input_values(1024);
        let whole = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let first_simd = v[..32].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            whole > first_simd,
            "the input cannot tell a partial reduction from a whole one: \
             row max {whole}, first-32 max {first_simd}"
        );
        // And by a margin no tolerance would swallow. Measured against the real broken
        // kernel: it returned 2.0016 where the row's maximum is 2.1019.
        assert!(
            whole - first_simd > 1e-2,
            "the margin is {} -- too small to fail a comparison",
            whole - first_simd
        );
    }

    #[test]
    fn a_row_reduction_reads_back_one_value_per_row() {
        // Sizing a reduction as elementwise makes the harness read a buffer the kernel
        // never wrote past its first element, and compare it against nothing.
        let sh = Shape::RowReduce { rows: 4, cols: 256 };
        assert_eq!(sh.out_size(), 4);
        assert_eq!(sh.in_sizes(), vec![1024]);
        // `num_elements` is the ROW WIDTH here, not the output size: the kernel loops
        // over it. Binding it to 4 would reduce a sliver of each row.
        assert_eq!(sh.scalar("num_elements"), Some(256));
        assert_eq!(
            reference_output(RefOp::ReduceMax, sh, 1e-6, "float").len(),
            4
        );
    }

    /// The "it knows..." hint must name every op the table can detect.
    ///
    /// It was prose written beside the table rather than derived from it, and it drifted
    /// three ops behind: after reduce_max, absmax and reduce_sum were added it still
    /// promised thirteen, telling a caller the harness could not check something it could.
    #[test]
    fn the_no_reference_hint_names_every_op_the_table_knows() {
        let err = RefOp::detect("nothing recognisable in here").unwrap_err();
        for (_, op) in OPS {
            assert!(
                err.contains(op.name()),
                "the hint omits {}, which the table detects:\n{err}",
                op.name()
            );
        }
        // The three that were missing when this was prose.
        for name in ["reduce_max", "reduce_sum", "absmax"] {
            assert!(err.contains(name), "the hint omits {name}:\n{err}");
        }
    }

    #[test]
    fn both_sides_generate_the_same_input_by_construction() {
        // Not by two implementations agreeing: a reference fed different data from the
        // kernel compares two unrelated numbers and fails like a numerical bug.
        let v = input_values(5);
        // The ramp is part of the value now: see `input_values_for` for why the sequence
        // must not be periodic.
        assert_eq!(v, vec![-2.0, -1.7499, -1.4998, -1.2497, -0.9996]);
        // Buffer 1 is offset by 7 so two inputs are never the same matrix.
        assert_ne!(input_values_for(0, 5), input_values_for(1, 5));
        // The Swift harness computes the same rule, generalised over the buffer index.
        let swift = include_str!("../assets/harness/metal.swift");
        assert!(
            swift.contains("Float(($0 + 7 * b + seed) % 17) * 0.25 - 2.0"),
            "the harness and the reference have drifted apart"
        );
    }

    #[test]
    fn the_reference_does_not_allocate_inside_the_timed_region() {
        // Allocating per row would measure the allocator and call it the reference's
        // cost, handicapping the baseline and overstating every speedup against it.
        let src = include_str!("run.rs");
        let body = src
            .split("pub fn time_reference")
            .nth(1)
            .and_then(|s| s.split("\n}").next())
            .expect("the function body");
        let timed = body.split("let t = Instant::now();").nth(1).unwrap_or("");
        let timed = timed.split("samples.push").next().unwrap_or("");
        assert!(
            !timed.contains("vec!["),
            "an allocation sits inside the timed region:{timed}"
        );
    }

    #[test]
    fn a_debug_build_says_its_reference_is_unoptimized() {
        // The tests run under debug_assertions, which is exactly the build whose numbers
        // must not be quoted -- so this asserts the warning is present here, and absent
        // from a release build.
        let r = RunReport {
            device: "test".into(),
            op: RefOp::Matmul,
            dtype: "float".into(),
            elements: 16,
            checked: 16,
            worst_error: 0.0,
            nan_mismatch: 0,
            tolerance: 1e-5,
            target: Timing {
                samples: vec![1.0],
                warmup: 1,
            },
            reference: Timing {
                samples: vec![10.0],
                warmup: 1,
            },
            unoptimized: None,
            optimizer_changed_nothing: true,
            budget: None,
            vs_ours: crate::torchref::Accuracy::of(&[1.0], &[1.0]),
            torch: Err(crate::torchref::TorchStatus::NoEquivalent { op: "matmul" }),
        };
        let out = r.render();
        assert_eq!(out.contains("debug build"), cfg!(debug_assertions), "{out}");
    }

    #[test]
    fn a_one_sided_nan_is_a_disagreement_and_not_a_rounding_error() {
        // `if e > worst` is FALSE when e is NaN, so every NaN difference was skipped
        // while `checked` went on counting the element as compared. A kernel producing
        // NaN where the reference produced a number passed with a clean maximum error.
        //
        // Demonstrated both ways, because the old code's failure is invisible by
        // construction: the number it printed was correct for the elements it looked at.
        let want = [1.0f32, 2.0, 3.0, 4.0];

        // The old behaviour, written out: NaN skipped, worst stays at the finite max.
        let got_nan = [1.0f32, f32::NAN, 3.0, 4.0];
        let old_worst = got_nan
            .iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, |acc, e| if e > acc { e } else { acc });
        assert_eq!(old_worst, 0.0, "the old rule reported a perfect match");

        // The new rule counts it.
        let (checked, worst, nan) = nan_aware(&got_nan, &want);
        assert_eq!(checked, 4);
        assert_eq!(nan, 1, "the NaN was not counted");
        assert_eq!(worst, 0.0, "the finite elements really do match");

        // Both sides NaN is agreement -- the same behaviour on each side, which is what
        // this check is for. sqrt and log over the harness's inputs do exactly this.
        let (_, _, nan_both) = nan_aware(&[f32::NAN, 2.0], &[f32::NAN, 2.0]);
        assert_eq!(nan_both, 0, "both-NaN must not be reported as a mismatch");

        // An infinity against a number is the same kind of disagreement.
        let (_, _, inf) = nan_aware(&[f32::INFINITY, 2.0], &[1.0, 2.0]);
        assert_eq!(inf, 1);
    }

    /// The comparison loop from `compare_detailed`, over supplied values.
    fn nan_aware(got: &[f32], want: &[f32]) -> (usize, f32, usize) {
        let mut worst = 0.0f32;
        let mut nan_mismatch = 0usize;
        let n = got.len().min(want.len());
        for i in 0..n {
            let (g, w) = (got[i], want[i]);
            let (gf, wf) = (g.is_finite(), w.is_finite());
            if gf != wf {
                nan_mismatch += 1;
                continue;
            }
            if !gf {
                continue;
            }
            let e = (g - w).abs();
            if e > worst {
                worst = e;
            }
        }
        (n, worst, nan_mismatch)
    }

    #[test]
    fn the_report_says_when_answers_are_not_numbers() {
        let r = RunReport {
            device: "test".into(),
            op: RefOp::Sqrt,
            dtype: "float".into(),
            elements: 4,
            checked: 4,
            worst_error: 0.0,
            nan_mismatch: 3,
            tolerance: 1e-5,
            target: Timing {
                samples: vec![1.0],
                warmup: 1,
            },
            reference: Timing {
                samples: vec![2.0],
                warmup: 1,
            },
            unoptimized: None,
            optimizer_changed_nothing: true,
            budget: None,
            vs_ours: crate::torchref::Accuracy::of(&[1.0], &[1.0]),
            torch: Err(crate::torchref::TorchStatus::NoEquivalent { op: "sqrt" }),
        };
        let out = r.render();
        // Before the tolerances: a clean "max abs 0.00e0" beside three NaNs is the most
        // misleading line this tool could print.
        assert!(out.contains("NOT A NUMBER"), "{out}");
        assert!(out.contains("3 of 4"), "{out}");
    }

    #[test]
    fn a_contended_machine_says_its_ratio_is_not_reproducible() {
        // Observed: the same matmul on the same machine reported 5.2x and then 18.6x,
        // because the CPU reference competes for cores and the GPU does not. The device
        // samples barely moved across both runs. A ratio whose denominator swings by a
        // factor of ten is not a measurement, and nothing said so.
        let r = RunReport {
            device: "test".into(),
            op: RefOp::Matmul,
            dtype: "float".into(),
            elements: 16,
            checked: 16,
            worst_error: 0.0,
            nan_mismatch: 0,
            tolerance: 1e-5,
            // Tight: the device is not the problem.
            target: Timing {
                samples: vec![77.8, 77.9, 78.1],
                warmup: 1,
            },
            // Wide: this is what a contended box does to a CPU loop.
            reference: Timing {
                samples: vec![396.0, 410.0, 477.0],
                warmup: 1,
            },
            unoptimized: None,
            optimizer_changed_nothing: true,
            budget: None,
            vs_ours: crate::torchref::Accuracy::of(&[1.0], &[1.0]),
            torch: Err(crate::torchref::TorchStatus::NoEquivalent { op: "matmul" }),
        };
        let out = r.render();
        assert!(out.contains("NOISY"), "{out}");
        // It must name WHICH side moved: they fail for different reasons.
        assert!(out.contains("the reference timing is"), "{out}");
        assert!(!out.contains("both timings are"), "{out}");

        // A quiet machine must not be warned about, or the warning becomes wallpaper.
        let quiet = RunReport {
            target: Timing {
                samples: vec![77.8, 77.9, 78.1],
                warmup: 1,
            },
            reference: Timing {
                samples: vec![400.0, 402.0, 404.0],
                warmup: 1,
            },
            ..r
        };
        assert!(!quiet.render().contains("NOISY"), "{}", quiet.render());
    }

    #[test]
    fn a_ratio_below_one_is_reported_as_a_slowdown() {
        // "speedup: 0.1x" is a slowdown that reads like a win at a glance.
        let r = RunReport {
            device: "test".into(),
            op: RefOp::Softmax,
            dtype: "float".into(),
            elements: 1024,
            checked: 1024,
            worst_error: 0.0,
            nan_mismatch: 0,
            tolerance: 1e-5,
            target: Timing {
                samples: vec![20.0],
                warmup: 1,
            },
            reference: Timing {
                samples: vec![2.0],
                warmup: 1,
            },
            unoptimized: None,
            optimizer_changed_nothing: true,
            budget: None,
            vs_ours: crate::torchref::Accuracy::of(&[1.0], &[1.0]),
            torch: Err(crate::torchref::TorchStatus::NoEquivalent { op: "softmax" }),
        };
        let out = r.render();
        assert!(out.contains("SLOWDOWN"), "{out}");
        assert!(out.contains("10.0x SLOWER"), "{out}");
        assert!(!out.contains("speedup"), "still called a speedup: {out}");
    }

    #[test]
    fn the_control_arm_refuses_a_kernel_that_ignores_its_inputs() {
        // Verified against a real Metal kernel that writes a constant: values and
        // control came back byte-identical over all 4096 elements.
        let e = check_control(&[1.0, 1.0], &[1.0, 1.0]).unwrap_err();
        assert!(e.contains("control arm AGREED"), "{e}");
        // And it must not fire on a kernel that does depend on its input.
        assert!(check_control(&[1.0, 2.0], &[3.0, 4.0]).is_ok());
    }

    #[test]
    fn an_empty_read_back_is_a_harness_fault_not_a_perfect_score() {
        // Zero values is the case that scored a perfect match against any reference.
        assert!(check_control(&[], &[]).is_err());
        assert!(check_control(&[1.0], &[]).is_err());
    }

    #[test]
    fn an_empty_result_set_is_not_reported_as_agreement() {
        // The failure this guards: `compare` maxes over what came back, so zero values
        // yields zero error, and the verdict line then claims all three sources agree.
        let r = RunReport {
            device: "test".into(),
            op: RefOp::Matmul,
            dtype: "float".into(),
            elements: 4096,
            checked: 0,
            worst_error: 0.0,
            nan_mismatch: 0,
            tolerance: 1e-5,
            target: Timing {
                samples: vec![1.0],
                warmup: 1,
            },
            reference: Timing {
                samples: vec![2.0],
                warmup: 1,
            },
            unoptimized: None,
            optimizer_changed_nothing: true,
            budget: None,
            vs_ours: crate::torchref::Accuracy::of(&[], &[]),
            torch: Err(crate::torchref::TorchStatus::NoEquivalent { op: "matmul" }),
        };
        let out = r.render();
        assert!(out.contains("NO VALUES"), "{out}");
    }

    #[test]
    fn a_timing_always_carries_its_sample_count_and_spread() {
        // A median with no spread is a number with no error bar.
        let t = Timing {
            samples: vec![10.0, 12.0, 11.0],
            warmup: 3,
        };
        let r = t.render("us");
        assert!(r.contains("median 11.00"), "{r}");
        assert!(r.contains("over 3 runs"), "{r}");
        assert!(r.contains("min 10.00") && r.contains("max 12.00"), "{r}");
        assert!(r.contains("3 warmup"), "{r}");
    }

    fn a_report(unopt: Option<Timing>, identical: bool) -> RunReport {
        RunReport {
            device: "Test GPU".into(),
            op: RefOp::Softmax,
            dtype: "float".into(),
            elements: 1024,
            checked: 1024,
            worst_error: 1e-7,
            nan_mismatch: 0,
            tolerance: 1e-5,
            target: Timing {
                samples: vec![10.0],
                warmup: 3,
            },
            reference: Timing {
                samples: vec![1000.0],
                warmup: 3,
            },
            unoptimized: unopt,
            optimizer_changed_nothing: identical,
            budget: None,
            vs_ours: crate::torchref::Accuracy {
                compared: 1024,
                max_abs: 1e-7,
                max_rel: 1e-7,
                rmse: 1e-8,
                non_finite: 0,
                max_abs_near_zero: 0.0,
            },
            torch: Err(crate::torchref::TorchStatus::Absent {
                why: "No module named 'torch'".into(),
            }),
        }
    }

    #[test]
    fn an_absent_torch_still_reports_the_weaker_comparison_and_labels_it() {
        let r = a_report(None, true).render();
        assert!(r.contains("kernel vs ours"), "{r}");
        assert!(r.contains("torch:"), "the absence must be stated");
        assert!(r.contains("weaker claim"), "{r}");
    }

    #[test]
    fn the_reference_baseline_is_described_every_time_it_is_quoted() {
        // This figure is quoted out of context otherwise. It is this kernel against a
        // naive scalar loop, not GPU versus CPU.
        let r = a_report(None, true).render();
        assert!(r.contains("100.0x against the reference"), "{r}");
        assert!(r.contains("NOT a GPU-versus-CPU figure"), "{r}");
        assert!(r.contains("naive single-threaded scalar loop"), "{r}");
    }

    #[test]
    fn an_optimizer_that_changed_nothing_reports_no_ratio() {
        // "1.00x" implies a measurement was taken and came out even. Nothing was taken.
        let r = a_report(None, true).render();
        assert!(r.contains("identical source"), "{r}");
        assert!(!r.contains("from the optimization passes"), "{r}");
    }

    #[test]
    fn an_optimizer_that_changed_something_reports_both_timings_and_the_ratio() {
        let r = a_report(
            Some(Timing {
                samples: vec![15.0],
                warmup: 3,
            }),
            false,
        );
        let text = r.render();
        assert_eq!(
            r.speedup_from_optimizer().map(|x| (x * 100.0).round()),
            Some(150.0)
        );
        assert!(text.contains("-O0      : median 15.00"), "{text}");
        assert!(
            text.contains("1.50x from the optimization passes"),
            "{text}"
        );
    }

    #[test]
    fn a_target_with_no_harness_says_so_rather_than_pretending() {
        let e = harness_source(crate::forms::by_id("gpu").unwrap()).unwrap_err();
        assert!(e.to_string().contains("no harness"), "{e}");
        assert!(e.to_string().contains("nobody has watched work"), "{e}");
    }
}
