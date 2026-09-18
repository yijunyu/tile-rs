//! Running the Metal backend on the GPU that is actually under this machine.
//!
//! The other three emulators are SHIMS. `emulate_gpu` redefines `__global__` and
//! `threadIdx` so CUDA becomes ordinary C; `emulate_nki` and `emulate_tpu` answer the
//! NKI and Pallas calls with numpy. Each approximates its target's semantics, and a
//! defect in the shim looks exactly like a defect in the emitter.
//!
//! Metal needs none of that. `xcrun metal` is the vendor's own compiler and this is an
//! Apple GPU, so the kernel is compiled by the real toolchain, dispatched to real
//! hardware, and the numbers read back are what the hardware produced. Nothing between
//! the emitter and the answer is mine.
//!
//! That buys two things the shims could not.
//!
//!   * THE REDUCTIONS. `emulate_gpu` skips `softmax`, `reduce_sum`, `reduce_max` and
//!     `absmax` because they lower to warp primitives and one thread cannot be a warp.
//!     Here a threadgroup is a threadgroup: the `threadgroup_barrier` folds run
//!     concurrently, and a lost barrier or a mis-strided fold produces a wrong number
//!     rather than being skipped.
//!   * `matvec`. Only four backends lower it -- linalg, rvv, spirv and msl -- and none
//!     of the three shimmable backends is among them, so `Shape::Matvec` had a
//!     reference and nothing at all driving it. It is driven here.
//!
//! THE THREADGROUP SIZE IS A VARIABLE, on purpose. The emitted fold carries a comment
//! claiming it is "correct for any tcount, not only powers of two". That is a claim
//! about code nobody had run. Every kernel is dispatched at 256, 96 and 33 threads --
//! a power of two, an even non-power, and an odd one -- and all three must agree with
//! the reference. 33 also exceeds no bound and divides nothing evenly, which is the
//! case a halving loop gets wrong.
//!
//! Needs `swiftc` and a Metal device; skips with a note otherwise.

use std::process::Command;
use tile_cli::run::{error_budget, reference_output, unit_roundoff, RefOp, Shape};
use tile_cli::torchref::{Accuracy, Tolerance};

/// Dispatches an emitted kernel and prints what it computed.
///
/// Buffers are classified by pipeline REFLECTION rather than by parsing the emitted
/// text: a read-only float buffer is an input, a writable one an output, a uint a
/// scalar matched BY NAME. If what the Metal compiler saw disagrees with what this
/// test described, the runner exits nonzero instead of dispatching something else --
/// which is the difference between a gate and a green light.
const RUNNER: &str = r#"
import Metal
import Foundation

func die(_ m: String) -> Never {
    FileHandle.standardError.write(("mslrun: " + m + "\n").data(using: .utf8)!)
    exit(1)
}

// The same generator as run::input_values_for.
func gen(_ b: Int, _ n: Int) -> [Float] {
    (0..<n).map { i in Float((i + 7 * b) % 17) * 0.25 - 2.0 + Float(i) * 1e-4 }
}

var path: String? = nil, groups = 1, tcount = 256, halfIO = false
var inSizes: [Int] = [], outSizes: [Int] = [], scalars: [String: UInt32] = [:]
var a = Array(CommandLine.arguments.dropFirst()), i = 0
while i < a.count {
    switch a[i] {
    case "--in":     i += 1; inSizes.append(Int(a[i])!)
    case "--out":    i += 1; outSizes.append(Int(a[i])!)
    case "--scalar": i += 1; let kv = a[i].split(separator: "="); scalars[String(kv[0])] = UInt32(kv[1])!
    case "--groups": i += 1; groups = Int(a[i])!
    case "--tcount": i += 1; tcount = Int(a[i])!
    case "--dtype":  i += 1; halfIO = (a[i] == "half")
    default: path = a[i]
    }
    i += 1
}
guard let path else { die("no .metal file given") }

guard let dev = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
let src = try String(contentsOfFile: path, encoding: .utf8)
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: MTLCompileOptions()) }
catch { die("compile: \(error)") }
guard let name = lib.functionNames.first, let fn = lib.makeFunction(name: name) else {
    die("no kernel function")
}

var refl: MTLComputePipelineReflection?
let pso: MTLComputePipelineState
do { pso = try dev.makeComputePipelineState(function: fn, options: [.bindingInfo], reflection: &refl) }
catch { die("pipeline: \(error)") }
guard let bindings = refl?.bindings else { die("no reflection") }

// Asked for more threads than the pipeline allows: say so rather than silently
// dispatching a different shape than the one named.
if tcount > pso.maxTotalThreadsPerThreadgroup {
    die("\(name): asked for \(tcount) threads, the pipeline allows \(pso.maxTotalThreadsPerThreadgroup)")
}

enum Role { case input(Int), output(Int), scalar(String) }
var roles: [(Int, Role)] = [], nIn = 0, nOut = 0
// Which buffer indices hold f16. The generator is the same either way; only the width
// of what lands in the buffer differs, which is what `quantize_inputs` mirrors on the
// reference side so both are given identical numbers.
var halfIdx = Set<Int>()
for b in bindings {
    guard let bb = b as? MTLBufferBinding else { continue }
    let elem = bb.bufferPointerType?.elementType ?? bb.bufferDataType
    if elem == .uint {
        roles.append((b.index, .scalar(b.name)))
    } else if elem == .float {
        if b.access == .readOnly { roles.append((b.index, .input(nIn))); nIn += 1 }
        else { roles.append((b.index, .output(nOut))); nOut += 1 }
    } else if elem == .char || elem == .uchar {
        // The ggml-derived f16 kernels take `device const char*` with BYTE strides, so
        // reflection cannot tell f16 data from any other bytes. The CALLER has to say,
        // and without `--dtype half` this stays refused rather than guessed at -- filling
        // a byte buffer with f32 misreads every element and still looks plausible.
        if !halfIO {
            die("\(name): buffer \(b.index) '\(b.name)' is a byte pointer and no --dtype was given")
        }
        if b.access == .readOnly { roles.append((b.index, .input(nIn))); nIn += 1 }
        else { roles.append((b.index, .output(nOut))); nOut += 1 }
        halfIdx.insert(b.index)
    } else if elem == .half {
        // Half buffers are filled and read AS half. They were refused rather than
        // guessed at while nothing drove them, because filling a half buffer with f32
        // misreads every element and still produces plausible-looking numbers.
        if b.access == .readOnly { roles.append((b.index, .input(nIn))); nIn += 1 }
        else { roles.append((b.index, .output(nOut))); nOut += 1 }
        halfIdx.insert(b.index)
    } else {
        die("\(name): buffer \(b.index) '\(b.name)' has element type \(elem.rawValue), unhandled")
    }
}
if nIn != inSizes.count { die("\(name): takes \(nIn) input buffer(s), \(inSizes.count) --in given") }
if nOut != outSizes.count { die("\(name): writes \(nOut) buffer(s), \(outSizes.count) --out given") }

let q = dev.makeCommandQueue()!
let cb = q.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)

var outBufs: [(Int, MTLBuffer, Int, Bool)] = []
for (idx, role) in roles {
    switch role {
    case .input(let k):
        let v = gen(k, inSizes[k])
        let buf: MTLBuffer
        if halfIdx.contains(idx) {
            let h = v.map { Float16($0) }
            buf = dev.makeBuffer(bytes: h, length: max(4, h.count * 2), options: .storageModeShared)!
        } else {
            buf = dev.makeBuffer(bytes: v, length: max(4, v.count * 4), options: .storageModeShared)!
        }
        enc.setBuffer(buf, offset: 0, index: idx)
    case .output(let k):
        let width = halfIdx.contains(idx) ? 2 : 4
        let n = max(4, outSizes[k] * width)
        let buf = dev.makeBuffer(length: n, options: .storageModeShared)!
        memset(buf.contents(), 0, n)
        enc.setBuffer(buf, offset: 0, index: idx)
        outBufs.append((k, buf, outSizes[k], halfIdx.contains(idx)))
    case .scalar(let nm):
        guard var v = scalars[nm] else { die("\(name): wants a scalar named '\(nm)', none given") }
        enc.setBytes(&v, length: 4, index: idx)
    }
}

enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: tcount, height: 1, depth: 1))
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { die("execution: \(e)") }

for (_, buf, n, isHalf) in outBufs.sorted(by: { $0.0 < $1.0 }) {
    if isHalf {
        let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
        for j in 0..<n { print(String(format: "%.9g", Float(p[j]))) }
    } else {
        let p = buf.contents().bindMemory(to: Float.self, capacity: n)
        for j in 0..<n { print(String(format: "%.9g", p[j])) }
    }
}
"#;

fn which(tool: &str) -> Option<String> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .ok()?;
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !p.is_empty()).then_some(p)
}

/// One input, one output: elementwise ops and the row reductions.
fn kernel_mlir(op: &str, rows: usize, cols: usize, out_cols: usize) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %o = llvm.mlir.constant({out_cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %o) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// A `rows x cols` matrix times a `cols`-long vector, into `rows` values.
///
/// The vector is the SECOND buffer, so the runner fills it from generator index 1 --
/// the index `reference_output` uses for it. Filling both from index 0 would compare
/// against a different problem than the one dispatched.
fn matvec_mlir(rows: usize, cols: usize) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %one = llvm.mlir.constant(1 : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %b = llvm.call @__tile_load_f32(%arg1, %one, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_matvec_f32(%a, %a, %b, %r, %c) \
         : (i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %one) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// `(m x k) . (k x n)`.
///
/// The right-hand matrix is the SECOND buffer, so the runner fills it from generator
/// index 1 -- the index `reference_output` uses. Filling both from index 0 would compare
/// against a different problem, and for a symmetric operation it would still agree.
fn matmul_mlir(m: usize, k: usize, n: usize) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
         \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
         \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) \
         : (i32, i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg2, %y, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// Two `rows x cols` inputs, elementwise, into one output of the same shape.
///
/// `min` and `max` are the only ops here that read two buffers of the SAME size, which
/// is the shape a swapped operand hides in best: min(a,b) == min(b,a). The reference
/// picks the second buffer from generator index 1, so the values differ even though the
/// operation does not care about order -- which is what makes the comparison mean
/// something rather than merely agree.
fn binary_mlir(op: &str, rows: usize, cols: usize) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %b = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_f32(%a, %a, %b, %r, %c) \
         : (i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// RMS-normalise each row, with the epsilon carried in the CALL.
///
/// The eps is deliberately not 1e-6. `Shape::rms_eps_from_mlir`'s own comment records
/// why: the emitter once hardcoded 1e-6 and the reference hardcoded 1e-6 to match, so
/// the two constants agreed with each other and checked nothing -- change the kernel's
/// eps and the comparison would have gone on passing. A value neither side can have
/// baked in makes the agreement mean something, and the reference reads it back out of
/// this MLIR rather than being told it twice.
fn rms_norm_mlir(rows: usize, cols: usize, eps: f32) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %e = llvm.mlir.constant({eps:e} : f32) : f32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_rms_norm_f32(%a, %a, %e, %r, %c) \
         : (i32, i32, f32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// The f16 form of a kernel. `ty` is the intrinsic suffix and the load/store width.
///
/// f16 is not a second spelling of the f32 path. Metal's emitter picks a DIFFERENT kernel
/// family for it: the elementwise ops become the ggml-derived shape, `device const char*`
/// with byte strides `ne0`/`nb_src`/`nb_dst`, while matvec and matmul become GEMM-shaped
/// kernels taking `half*` and M/K/N. Both are exercised here.
fn f16_mlir(op: &str, kind: F16Kind) -> String {
    match kind {
        F16Kind::Elementwise { rows, cols } => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
             attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_{op}_f16(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        ),
        F16Kind::Reduce { rows, cols } => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
             attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %o = llvm.mlir.constant(1 : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_{op}_f16(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg1, %y, %r, %o) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        ),
        F16Kind::Binary { rows, cols } => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f16(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_{op}_f16(%a, %a, %b, %r, %c) \
             : (i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg2, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        ),
        F16Kind::RmsNorm { rows, cols } => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
             attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %e = llvm.mlir.constant(2.500000e-02 : f32) : f32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_rms_norm_f16(%a, %a, %e, %r, %c) \
             : (i32, i32, f32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        ),
        F16Kind::Matvec { rows, cols } => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %one = llvm.mlir.constant(1 : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f16(%arg1, %one, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_matvec_f16(%a, %a, %b, %r, %c) \
             : (i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg2, %y, %r, %one) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        ),
        F16Kind::Matmul { m, k, n } => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
             \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
             \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f16(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_matmul_f16(%a, %a, %b, %m, %k, %n) \
             : (i32, i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg2, %y, %m, %n) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        ),
    }
}

#[derive(Clone, Copy)]
enum F16Kind {
    Elementwise {
        rows: usize,
        cols: usize,
    },
    Reduce {
        rows: usize,
        cols: usize,
    },
    Binary {
        rows: usize,
        cols: usize,
    },
    RmsNorm {
        rows: usize,
        cols: usize,
    },
    /// Built but not driven: see the note in the f16 sweep and backlog #023. Kept so
    /// the shape is ready the moment that ABI question is answered, and so the form
    /// the kernel actually wants is written down rather than remembered.
    #[allow(dead_code)]
    Matvec {
        rows: usize,
        cols: usize,
    },
    Matmul {
        m: usize,
        k: usize,
        n: usize,
    },
}

/// Judge an f16 result the way the run path does, rather than with a number chosen here.
///
/// `error_budget` returns a summation bound for the operations that sum, and `None` for
/// the rest -- for which the relative tolerance can never be tighter than what the
/// kernel's own buffer type can represent. Both come from `run.rs`; this only assembles
/// them, so the gate and the tool cannot disagree about what "correct" means.
fn f16_compare(
    label: String,
    got: Vec<f32>,
    want: &[f32],
    op: RefOp,
    shape: Shape,
    bad: &mut Vec<String>,
) {
    if got.len() != want.len() {
        bad.push(format!(
            "{label}: wrote {} values, {} were due",
            got.len(),
            want.len()
        ));
        return;
    }
    let tol = match error_budget(op, shape, "half") {
        Some(b) => Tolerance::Summation(b),
        None => Tolerance::Fixed {
            abs: 1e-5,
            rel: 1e-4_f32.max(unit_roundoff("half")),
        },
    };
    let acc = Accuracy::of(&got, want);
    if acc.compared == 0 {
        bad.push(format!(
            "{label}: nothing was compared -- every pair was skipped as non-finite on both sides"
        ));
        return;
    }
    if !tol.admits(&acc) {
        // Every field `within` actually looks at: "allowed 1.000e-5" alone hid
        // that the failure was the REL bound (4.88e-4), not the abs one.
        bad.push(format!(
            "{label}: max_abs {:.3e} max_rel {:.3e} near_zero_abs {:.3e} non_finite {} \
             over {} compared; {}",
            acc.max_abs,
            acc.max_rel,
            acc.max_abs_near_zero,
            acc.non_finite,
            acc.compared,
            match &tol {
                Tolerance::Fixed { abs, rel } => format!("allowed abs {abs:.3e} rel {rel:.3e}"),
                Tolerance::Summation(b) => format!("allowed abs {:.3e} (summation)", b.guaranteed),
            }
        ));
    }
}

#[test]
fn msl_computes_what_the_reference_computes() {
    let Some(swiftc) = which("swiftc") else {
        eprintln!("emulate_msl: skipped, no swiftc");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emulate-msl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let runner_src = dir.join("mslrun.swift");
    std::fs::write(&runner_src, RUNNER).expect("write runner");
    let runner = dir.join("mslrun");
    let built = Command::new(&swiftc)
        .arg("-O")
        .arg(&runner_src)
        .arg("-o")
        .arg(&runner)
        .output()
        .expect("swiftc runs");
    if !built.status.success() {
        // No Metal SDK on this box. Say what happened rather than passing quietly.
        eprintln!(
            "emulate_msl: skipped, the runner did not build — {}",
            String::from_utf8_lossy(&built.stderr)
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or("(no error line)")
        );
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }

    // Multi-row shapes are the point: a per-row defect is invisible at one row. 129 and
    // 33 are deliberately not multiples of the threadgroup sizes below.
    let shapes = [(1usize, 64usize), (1, 256), (4, 256), (3, 129), (8, 64)];
    // A power of two, an even non-power, and an odd one -- the emitted fold claims all
    // three are correct.
    let tcounts = [256usize, 96, 33];
    // `reduces` = one value per row rather than one per element.
    let ops: &[(&str, RefOp, bool)] = &[
        ("exp", RefOp::Exp, false),
        ("neg", RefOp::Neg, false),
        ("abs", RefOp::Abs, false),
        ("sigmoid", RefOp::Sigmoid, false),
        ("relu", RefOp::Relu, false),
        ("sqrt", RefOp::Sqrt, false),
        ("tanh", RefOp::Tanh, false),
        ("rsqrt", RefOp::Rsqrt, false),
        ("silu", RefOp::Silu, false),
        ("softplus", RefOp::Softplus, false),
        // log of a negative is NaN, and the generator spans -2.0 to 2.0, so most of
        // this kernel's output IS NaN. That is exactly the case the comparison used
        // to discard silently -- see the note on `f32::max` in `compare`.
        ("log", RefOp::Log, false),
        ("softmax", RefOp::Softmax, false),
        ("reduce_sum", RefOp::ReduceSum, true),
        ("reduce_max", RefOp::ReduceMax, true),
        ("absmax", RefOp::Absmax, true),
    ];

    let (mut checked, mut refused) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    // A closure so the elementwise/reduction sweep and the matvec sweep compare the
    // same way: same tolerance, same NaN and infinity handling, same message.
    let compare = |label: String, got: Vec<f32>, want: &[f32], bad: &mut Vec<String>| {
        if got.len() != want.len() {
            bad.push(format!(
                "{label}: wrote {} values, {} were due",
                got.len(),
                want.len()
            ));
            return;
        }
        let mut worst = 0.0f32;
        // Pairs that were actually COMPARED, as opposed to skipped for agreeing on
        // being NaN. `log` of the generator is NaN over roughly half its range, and a
        // kernel that returned NaN everywhere would match the reference everywhere it
        // was looked at -- by being skipped everywhere. That is a pass which checked
        // nothing, and it is the element-level twin of the `checked >= 100` guard below.
        let mut compared = 0usize;
        for (g, w) in got.iter().zip(want.iter()) {
            if (g.is_nan() && w.is_nan()) || (g.is_infinite() && w.is_infinite()) {
                continue;
            }
            compared += 1;
            // `f32::max` RETURNS THE OTHER OPERAND when one side is NaN, so folding a
            // one-sided NaN straight into `worst` DISCARDS it: a kernel writing NaN
            // where the reference says 0.5 left worst at 0 and passed. Hoist it to
            // infinity so the disagreement survives the fold.
            let d = (g - w).abs() / w.abs().max(1.0);
            worst = if d.is_nan() {
                f32::INFINITY
            } else {
                worst.max(d)
            };
        }
        if compared == 0 {
            bad.push(format!(
                "{label}: every one of {} pairs was skipped as NaN or infinite on both \
                 sides -- this comparison checked nothing",
                want.len()
            ));
            return;
        }
        if worst > 1e-5 {
            bad.push(format!(
                "{label}: max rel {worst:.2e} over {compared}/{} compared",
                want.len()
            ));
        }
    };

    // Every scalar name the emitted kernels use. The VALUES come from `Shape::scalar`,
    // the same mapping the device paths bind, rather than from arithmetic repeated here:
    // its own comments record a matvec whose `num_elements` was bound to the output size
    // and so looped over a sliver of each row. A name this shape cannot answer is simply
    // not passed, and the runner then refuses the kernel that wanted it -- which is a
    // finding, not a default.
    const SCALAR_NAMES: &[&str] = &[
        "num_elements",
        "M",
        "N",
        "K",
        "m",
        "n",
        "k",
        // The ggml-derived f16 kernels spell the row width `ne0` and take BYTE strides.
        // `Shape::scalar_for` knows both spellings and multiplies by the element width,
        // which is the whole reason it takes one.
        "ne0",
        "nb_src",
        "nb_dst",
        "nb0",
        "nb1",
    ];

    let run_kernel = |metal: &std::path::Path,
                      groups: usize,
                      tcount: usize,
                      ins: &[usize],
                      out: usize,
                      shape: Shape,
                      dtype: &str|
     -> Result<Vec<f32>, String> {
        let mut cmd = Command::new(&runner);
        cmd.arg(metal);
        for n in ins {
            cmd.args(["--in", &n.to_string()]);
        }
        cmd.args(["--out", &out.to_string()])
            .args(["--groups", &groups.to_string()])
            .args(["--tcount", &tcount.to_string()])
            .args(["--dtype", dtype]);
        let width: u32 = if dtype == "half" { 2 } else { 4 };
        for name in SCALAR_NAMES {
            if let Some(v) = shape.scalar_for(name, width) {
                cmd.args(["--scalar", &format!("{name}={v}")]);
            }
        }
        let r = cmd.output().map_err(|e| e.to_string())?;
        if !r.status.success() {
            return Err(String::from_utf8_lossy(&r.stderr).trim().to_string());
        }
        Ok(String::from_utf8_lossy(&r.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect())
    };

    for &(rows, cols) in &shapes {
        for (name, op, reduces) in ops {
            let src = dir.join(format!("{name}_{rows}x{cols}.mlir"));
            let out_cols = if *reduces { 1 } else { cols };
            std::fs::write(&src, kernel_mlir(name, rows, cols, out_cols)).expect("write");
            // A FRESH path per kernel: a refused emission must not leave the previous
            // one to be compared against this op's reference.
            let metal = dir.join(format!("{name}_{rows}x{cols}.metal"));
            let _ = std::fs::remove_file(&metal);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "msl", "-o"])
                .arg(&metal)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !metal.exists() {
                refused += 1; // a refusal is an answer
                continue;
            }
            let shape = if *reduces {
                Shape::RowReduce { rows, cols }
            } else {
                Shape::Rows { rows, cols }
            };
            let want = reference_output(*op, shape, 1e-6, "float");
            for &tc in &tcounts {
                let label = format!("{name} {rows}x{cols} @{tc}t");
                match run_kernel(&metal, rows, tc, &[rows * cols], want.len(), shape, "float") {
                    Ok(got) => {
                        checked += 1;
                        compare(label, got, &want, &mut bad);
                    }
                    Err(e) => bad.push(format!("{label}: {e}")),
                }
            }
        }
    }

    // min and max: two inputs of equal size, the last elementwise reference nothing drove.
    for (name, op) in [("min", RefOp::Min), ("max", RefOp::Max)] {
        for &(rows, cols) in &shapes {
            let src = dir.join(format!("{name}_{rows}x{cols}.mlir"));
            std::fs::write(&src, binary_mlir(name, rows, cols)).expect("write");
            let metal = dir.join(format!("{name}_{rows}x{cols}.metal"));
            let _ = std::fs::remove_file(&metal);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "msl", "-o"])
                .arg(&metal)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !metal.exists() {
                refused += 1;
                continue;
            }
            let shape = Shape::Rows2 { rows, cols };
            let want = reference_output(op, shape, 1e-6, "float");
            for &tc in &tcounts {
                let label = format!("{name} {rows}x{cols} @{tc}t");
                let ins = [rows * cols, rows * cols];
                match run_kernel(&metal, rows, tc, &ins, want.len(), shape, "float") {
                    Ok(got) => {
                        checked += 1;
                        compare(label, got, &want, &mut bad);
                    }
                    Err(e) => bad.push(format!("{label}: {e}")),
                }
            }
        }
    }

    // rms_norm: the last reference with nothing driving it, and the only one whose
    // parameter travels in the call rather than in the shape.
    for &(rows, cols) in &shapes {
        let eps = 2.5e-2f32;
        let mlir = rms_norm_mlir(rows, cols, eps);
        // Read it back the way the emitter does. If this ever returns None or a
        // different number, the reference and the kernel are no longer being given the
        // same eps and every rms_norm comparison below is meaningless.
        let from_source = Shape::rms_eps_from_mlir(&mlir);
        assert_eq!(
            from_source,
            Some(eps),
            "rms_eps_from_mlir did not read back the eps this kernel was built with"
        );
        let src = dir.join(format!("rms_norm_{rows}x{cols}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let metal = dir.join(format!("rms_norm_{rows}x{cols}.metal"));
        let _ = std::fs::remove_file(&metal);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "msl", "-o"])
            .arg(&metal)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !metal.exists() {
            refused += 1;
            continue;
        }
        let shape = Shape::Rows { rows, cols };
        let want = reference_output(RefOp::RmsNorm, shape, from_source.unwrap(), "float");
        for &tc in &tcounts {
            let label = format!("rms_norm {rows}x{cols} @{tc}t");
            match run_kernel(&metal, rows, tc, &[rows * cols], want.len(), shape, "float") {
                Ok(got) => {
                    checked += 1;
                    compare(label, got, &want, &mut bad);
                }
                Err(e) => bad.push(format!("{label}: {e}")),
            }
        }
    }

    // matvec: driven by nothing until now, because the three shimmable backends do not
    // lower it.
    for &(rows, cols) in &[(8usize, 16usize), (4, 4), (3, 129)] {
        let src = dir.join(format!("matvec_{rows}x{cols}.mlir"));
        std::fs::write(&src, matvec_mlir(rows, cols)).expect("write");
        let metal = dir.join(format!("matvec_{rows}x{cols}.metal"));
        let _ = std::fs::remove_file(&metal);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "msl", "-o"])
            .arg(&metal)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !metal.exists() {
            refused += 1;
            continue;
        }
        let shape = Shape::Matvec { rows, cols };
        let want = reference_output(RefOp::Matvec, shape, 1e-6, "float");
        for &tc in &tcounts {
            let label = format!("matvec {rows}x{cols} @{tc}t");
            match run_kernel(
                &metal,
                rows,
                tc,
                &[rows * cols, cols],
                want.len(),
                shape,
                "float",
            ) {
                Ok(got) => {
                    checked += 1;
                    compare(label, got, &want, &mut bad);
                }
                Err(e) => bad.push(format!("{label}: {e}")),
            }
        }
    }

    // matmul: driven on the shims already, but never on hardware. It is the one shape
    // here that binds M, N and K as well as num_elements -- and it is still one
    // threadgroup per output ROW, so the dispatch is the same as everything above.
    for &(m, k, n) in &[
        (8usize, 16usize, 4usize),
        (4, 4, 4),
        (16, 8, 32),
        (3, 129, 5),
    ] {
        let src = dir.join(format!("matmul_{m}x{k}x{n}.mlir"));
        std::fs::write(&src, matmul_mlir(m, k, n)).expect("write");
        let metal = dir.join(format!("matmul_{m}x{k}x{n}.metal"));
        let _ = std::fs::remove_file(&metal);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "msl", "-o"])
            .arg(&metal)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !metal.exists() {
            refused += 1;
            continue;
        }
        let shape = Shape::Matmul { m, k, n };
        let want = reference_output(RefOp::Matmul, shape, 1e-6, "float");
        for &tc in &tcounts {
            let label = format!("matmul {m}x{k}x{n} @{tc}t");
            match run_kernel(&metal, m, tc, &[m * k, k * n], want.len(), shape, "float") {
                Ok(got) => {
                    checked += 1;
                    compare(label, got, &want, &mut bad);
                }
                Err(e) => bad.push(format!("{label}: {e}")),
            }
        }
    }

    // ── f16 ──────────────────────────────────────────────────────────────────────
    //
    // A whole precision that nothing had measured. It is not a second spelling of the
    // sweeps above: Metal picks a DIFFERENT kernel family for f16, and the tolerance that
    // judges it cannot be the f32 one.
    //
    // The tolerance is NOT chosen here. `error_budget` and `unit_roundoff` already derive
    // it from the format and from what the operation sums, and their comments record what
    // choosing costs: "the msl f16 matmul -- which is CORRECT, and errs by 2.18e-2 because
    // `half * half` rounds each product before the f32 accumulator ever sees it -- was told
    // 'the KERNEL disagrees with torch, look at the lowering'". So this reuses the run
    // path's own `Tolerance` rather than restating the rule and getting it subtly different.
    //
    // Both sides see identical inputs: the runner writes `Float16(gen(i))` and
    // `reference_output(.., "half")` calls `quantize_inputs`. That was #010 -- an f16
    // kernel compared against f32 numbers it was never given, with part of every reported
    // gap being that rounding rather than the lowering.
    for &(rows, cols) in &shapes {
        let cases: &[(&str, RefOp, F16Kind, Shape)] = &[
            (
                "exp",
                RefOp::Exp,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "neg",
                RefOp::Neg,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "abs",
                RefOp::Abs,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "sigmoid",
                RefOp::Sigmoid,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "relu",
                RefOp::Relu,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "sqrt",
                RefOp::Sqrt,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "log",
                RefOp::Log,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "tanh",
                RefOp::Tanh,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "silu",
                RefOp::Silu,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "softmax",
                RefOp::Softmax,
                F16Kind::Elementwise { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "rms_norm",
                RefOp::RmsNorm,
                F16Kind::RmsNorm { rows, cols },
                Shape::Rows { rows, cols },
            ),
            (
                "reduce_sum",
                RefOp::ReduceSum,
                F16Kind::Reduce { rows, cols },
                Shape::RowReduce { rows, cols },
            ),
            (
                "reduce_max",
                RefOp::ReduceMax,
                F16Kind::Reduce { rows, cols },
                Shape::RowReduce { rows, cols },
            ),
            (
                "absmax",
                RefOp::Absmax,
                F16Kind::Reduce { rows, cols },
                Shape::RowReduce { rows, cols },
            ),
            (
                "min",
                RefOp::Min,
                F16Kind::Binary { rows, cols },
                Shape::Rows2 { rows, cols },
            ),
            (
                "max",
                RefOp::Max,
                F16Kind::Binary { rows, cols },
                Shape::Rows2 { rows, cols },
            ),
        ];
        for (name, op, kind, shape) in cases {
            let mlir = f16_mlir(name, *kind);
            let eps = Shape::rms_eps_from_mlir(&mlir).unwrap_or(1e-6);
            let stem = format!("f16_{name}_{rows}x{cols}");
            let src = dir.join(format!("{stem}.mlir"));
            std::fs::write(&src, &mlir).expect("write");
            let metal = dir.join(format!("{stem}.metal"));
            let _ = std::fs::remove_file(&metal);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "msl", "-o"])
                .arg(&metal)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !metal.exists() {
                refused += 1;
                continue;
            }
            let want = reference_output(*op, *shape, eps, "half");
            let ins: Vec<usize> = match kind {
                F16Kind::Binary { .. } => vec![rows * cols, rows * cols],
                _ => vec![rows * cols],
            };
            for &tc in &tcounts {
                let label = format!("f16 {name} {rows}x{cols} @{tc}t");
                match run_kernel(&metal, rows, tc, &ins, want.len(), *shape, "half") {
                    Ok(got) => {
                        checked += 1;
                        f16_compare(label, got, &want, *op, *shape, &mut bad);
                    }
                    Err(e) => bad.push(format!("{label}: {e}")),
                }
            }
        }
    }

    // f16 matvec is NOT driven, and the reason is backlog #023 rather than effort.
    //
    // The f16 kernel reads its operands in the OPPOSITE order from the f32 one emitted for
    // the same intrinsic -- `p0[i] * p1[base + i]` where f32 has `p0[base + i] * p1[i]` --
    // so driven through this harness's ABI it reads buffer 1 at offsets up to
    // `(rows-1)*cols` while that buffer holds only `cols` elements. It ran, read out of
    // bounds, and returned a max relative error of exactly 1.0.
    //
    // That is a DECISION, not a typo: `mlir_to_msl.rs` declares this kernel as
    // `activation(f32), weight(f16), output(f32)`, a DS4 convention. Either the harness is
    // calling it wrongly or the lowering disagrees with the intrinsic, and driving it under
    // a swapped ABI here would HIDE the disagreement rather than record it. The f32 matvec
    // is driven at three shapes and three threadgroup sizes and is unaffected.

    for &(m, k, n) in &[
        (8usize, 16usize, 4usize),
        (4, 4, 4),
        (16, 8, 32),
        (3, 129, 5),
    ] {
        let mlir = f16_mlir("matmul", F16Kind::Matmul { m, k, n });
        let shape = Shape::Matmul { m, k, n };
        let stem = format!("f16_matmul_{m}x{k}x{n}");
        let src = dir.join(format!("{stem}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let metal = dir.join(format!("{stem}.metal"));
        let _ = std::fs::remove_file(&metal);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "msl", "-o"])
            .arg(&metal)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !metal.exists() {
            refused += 1;
            continue;
        }
        let want = reference_output(RefOp::Matmul, shape, 1e-6, "half");
        for &tc in &tcounts {
            let label = format!("f16 matmul {m}x{k}x{n} @{tc}t");
            match run_kernel(&metal, m, tc, &[m * k, k * n], want.len(), shape, "half") {
                Ok(got) => {
                    checked += 1;
                    f16_compare(label, got, &want, RefOp::Matmul, shape, &mut bad);
                }
                Err(e) => bad.push(format!("{label}: {e}")),
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emulate_msl: {checked} kernels run on the GPU and compared, {refused} refused");
    for b in bad.iter().take(12) {
        eprintln!("  {b}");
    }
    // Guard the vacuous pass.
    assert!(
        checked >= 100,
        "only {checked} kernels ran -- msl is refusing nearly everything and this test \
         would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} Metal kernels disagree with the reference. This is not a syntax complaint: \
         the kernel was compiled by xcrun metal, dispatched to this machine's GPU, and \
         the numbers it wrote back are different.",
        bad.len()
    );
}

/// Every operation the harness can produce a reference for is driven on the GPU above.
///
/// This is a gate on the gate. The sweep's own `checked >= 100` guard proves that a lot of
/// kernels ran; it cannot notice that an operation with a perfectly good reference is
/// absent from the lists entirely -- which is how `matvec`, `matmul`, `log`, `min`, `max`
/// and `rms_norm` all came to have a reference and nothing comparing against it, some of
/// them for the whole life of this harness.
///
/// A `match` rather than a list of names, so the check is made by the COMPILER: adding a
/// variant to `RefOp` stops this file compiling until someone states which sweep drives it.
/// A variant that genuinely cannot be driven says so in its arm with the reason, and the
/// assertion below fails until that reason is written down.
#[test]
fn every_reference_operation_is_driven_on_the_gpu() {
    fn driven_by(op: RefOp) -> &'static str {
        match op {
            RefOp::Exp
            | RefOp::Neg
            | RefOp::Abs
            | RefOp::Sigmoid
            | RefOp::Relu
            | RefOp::Sqrt
            | RefOp::Tanh
            | RefOp::Rsqrt
            | RefOp::Silu
            | RefOp::Softplus
            | RefOp::Log
            | RefOp::Softmax => "the elementwise sweep",
            RefOp::ReduceSum | RefOp::ReduceMax | RefOp::Absmax => "the reduction sweep",
            RefOp::Min | RefOp::Max => "the two-input sweep",
            RefOp::Matvec => "the matvec sweep",
            RefOp::Matmul => "the matmul sweep",
            RefOp::RmsNorm => "the rms_norm sweep",
        }
    }

    // Every variant, so the loop below actually visits them. The `match` above is what
    // catches an ADDITION; this list is what catches one being quietly dropped from the
    // sweeps, and the count is asserted so the list cannot rot either.
    const ALL: &[RefOp] = &[
        RefOp::Matmul,
        RefOp::Softmax,
        RefOp::Exp,
        RefOp::Sigmoid,
        RefOp::Relu,
        RefOp::Sqrt,
        RefOp::Log,
        RefOp::Neg,
        RefOp::Abs,
        RefOp::Tanh,
        RefOp::Rsqrt,
        RefOp::Silu,
        RefOp::Softplus,
        RefOp::ReduceMax,
        RefOp::ReduceSum,
        RefOp::Min,
        RefOp::Max,
        RefOp::Matvec,
        RefOp::RmsNorm,
        RefOp::Absmax,
    ];
    assert_eq!(
        ALL.len(),
        20,
        "RefOp gained or lost a variant; add it to ALL and to a sweep"
    );

    let undriven: Vec<String> = ALL
        .iter()
        .filter(|op| driven_by(**op).is_empty())
        .map(|op| format!("{op:?}"))
        .collect();
    assert!(
        undriven.is_empty(),
        "these operations have a reference and nothing on the GPU compares against it: {}",
        undriven.join(", ")
    );
}

/// Which operations are driven in f16, and why the rest are not.
///
/// The f32 set is closed by `every_reference_operation_is_driven_on_the_gpu`, and f16 needs
/// its own statement because Metal lowers a DIFFERENT set in it and through a different
/// kernel family. Without this, "f16 is measured now" would be true of sixteen operations
/// and silent about the other four, which is the kind of half-claim `emulator_coverage.rs`
/// exists to prevent.
///
/// Exhaustive, so an operation added to `RefOp` cannot compile until someone says what f16
/// does with it.
#[test]
fn every_reference_operation_states_what_f16_does_with_it() {
    /// Empty means driven; a non-empty string is the reason it is not, and must be a
    /// property or a filed issue rather than a plan.
    fn not_driven_because(op: RefOp) -> &'static str {
        match op {
            RefOp::Exp
            | RefOp::Neg
            | RefOp::Abs
            | RefOp::Sigmoid
            | RefOp::Relu
            | RefOp::Sqrt
            | RefOp::Log
            | RefOp::Tanh
            | RefOp::Silu
            | RefOp::Softmax
            | RefOp::RmsNorm
            | RefOp::ReduceSum
            | RefOp::ReduceMax
            | RefOp::Absmax
            | RefOp::Min
            | RefOp::Max
            | RefOp::Matmul => "",
            RefOp::Rsqrt => "msl has no __tile_rsqrt_f16 arm; the f32 one is driven",
            RefOp::Matvec => {
                "backlog #023: the f16 kernel reads its operands in the opposite order from \
                 the f32 one and reads out of bounds under this ABI"
            }
            RefOp::Softplus => "msl has no __tile_softplus_f16 arm",
        }
    }

    // Every variant, so the loop visits them; the match above is what catches an addition.
    const ALL: &[RefOp] = &[
        RefOp::Matmul,
        RefOp::Softmax,
        RefOp::Exp,
        RefOp::Sigmoid,
        RefOp::Relu,
        RefOp::Sqrt,
        RefOp::Log,
        RefOp::Neg,
        RefOp::Abs,
        RefOp::Tanh,
        RefOp::Rsqrt,
        RefOp::Silu,
        RefOp::Softplus,
        RefOp::ReduceMax,
        RefOp::ReduceSum,
        RefOp::Min,
        RefOp::Max,
        RefOp::Matvec,
        RefOp::RmsNorm,
        RefOp::Absmax,
    ];
    assert_eq!(
        ALL.len(),
        20,
        "RefOp changed; say what f16 does with the new one"
    );

    let driven = ALL
        .iter()
        .filter(|op| not_driven_because(**op).is_empty())
        .count();
    // A floor rather than an equality: driving MORE is never the thing to fail on, and an
    // equality here would have to be edited every time #023 or an emitter arm is settled.
    assert!(
        driven >= 17,
        "only {driven} operations are driven in f16; the sweep has lost coverage"
    );
    for op in ALL {
        let why = not_driven_because(*op);
        if !why.is_empty() {
            eprintln!("f16 skips {op:?}: {why}");
        }
    }
}
