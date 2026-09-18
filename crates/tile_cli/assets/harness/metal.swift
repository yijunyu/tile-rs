// tile-rs run harness — Metal.
//
// Compiles the emitted kernel at runtime, dispatches it, and reports BOTH the values it
// produced and how long the device took. It does not judge either: `tile` owns the
// reference and the comparison, so the numerics live in one place and a second harness
// for another target is a transcription rather than a reimplementation.
//
// Timing uses the command buffer's own GPUStartTime/GPUEndTime. That is device time for
// the dispatch, not wall-clock around it: wall-clock would fold in encoding, driver
// submission and this process's scheduling, and report them as kernel cost.
//
// The buffer layout is passed in rather than assumed. It used to be fixed at "one input,
// one output, one length", which is why a matmul could not be run at all (backlog #002).
// `tile` reads the real layout off the emitted signature and hands it over here.
//
// Protocol on stdout: `#` lines are facts for the caller to parse, everything else is one
// output value per line.
import Metal
import Foundation

func die(_ m: String, _ code: Int32) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(code)
}

// argv: <source> <entry> <dtype> <iters> <grid> <threads> <in-counts,csv> <out-count> <scalars,csv>
let a = CommandLine.arguments
guard a.count >= 10,
      let iters = Int(a[4]), let grid = Int(a[5]), let threads = Int(a[6]),
      let outCount = Int(a[8])
else {
    die("usage: harness <src> <entry> <dtype> <iters> <grid> <threads> <ins> <out> <scalars>", 2)
}
let dtype = a[3]
// argv[10], optional: comma-separated threadgroup widths to sweep. Present means "measure
// these and print #sweep lines"; absent means the single timed run below.
let sweepWidths: [Int] = a.count > 10 && !a[10].isEmpty
    ? a[10].split(separator: ",").compactMap { Int($0) }
    : []
let inCounts = a[7].split(separator: ",").compactMap { Int($0) }
let scalars = a[9].isEmpty ? [] : a[9].split(separator: ",").compactMap { UInt32($0) }
guard !inCounts.isEmpty else { die("no input buffers", 2) }

guard let dev = MTLCreateSystemDefaultDevice() else { die("no Metal device", 6) }
let src = try String(contentsOfFile: a[1], encoding: .utf8)
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) }
catch { die("compile: \(error)", 1) }
guard let fn = lib.makeFunction(name: a[2]) else { die("no entry point \(a[2])", 1) }
let pipe = try dev.makeComputePipelineState(function: fn)

// The same rule `tile` and the torch script use, so all three compute over identical
// data. Buffer b is offset by 7 so two inputs are never the same matrix — one tensor
// used twice agrees with a transposed kernel and nobody finds out.
func inputValues(_ b: Int, _ n: Int, _ seed: Int = 0) -> [Float] {
    // The `* 1e-4` ramp: without it the sequence is periodic with period 17 and every
    // 32-element window contains the maximum, so a reduction that covers one SIMD group
    // agrees exactly with one that covers the row. See run.rs::input_values_for.
    //
    // The CONTROL arm (seed != 0) also scales and offsets, and not only reseeds. The seed
    // alone shifts which residue lands where -- it PERMUTES the row -- so a max or absmax
    // over 17 or more elements sees the same set and returns nearly the same number. The
    // residual difference comes only from the `i * 1e-4` ramp, which at f16 precision near
    // 2.25 is below one ulp: the control arm saw byte-identical output and reported that
    // the kernel ignored its inputs, for a kernel that was reading them correctly.
    //
    // Scaling moves every value by far more than any format's ulp, so an op that reads its
    // input cannot produce the same answer. The range stays modest -- about [-2.6, 3.2] --
    // so `exp` does not overflow f16.
    let k: Float = seed == 0 ? 1.0 : 1.37
    let o: Float = seed == 0 ? 0.0 : 0.11
    return (0..<n).map {
        (Float(($0 + 7 * b + seed) % 17) * 0.25 - 2.0) * k + o + Float($0) * 1e-4
    }
}

func fill(_ buf: MTLBuffer, _ v: [Float], _ dtype: String) {
    if dtype == "half" {
        let p = buf.contents().bindMemory(to: Float16.self, capacity: v.count)
        for i in 0..<v.count { p[i] = Float16(v[i]) }
    } else {
        let p = buf.contents().bindMemory(to: Float.self, capacity: v.count)
        for i in 0..<v.count { p[i] = v[i] }
    }
}

func readOut(_ buf: MTLBuffer, _ n: Int, _ dtype: String) -> [Float] {
    if dtype == "half" {
        let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
        return (0..<n).map { Float(p[$0]) }
    }
    let p = buf.contents().bindMemory(to: Float.self, capacity: n)
    return (0..<n).map { p[$0] }
}

let elemSize = (dtype == "half") ? 2 : 4
var buffers: [MTLBuffer] = []
for (b, n) in inCounts.enumerated() {
    let buf = dev.makeBuffer(length: n * elemSize, options: .storageModeShared)!
    fill(buf, inputValues(b, n), dtype)
    buffers.append(buf)
}
let outBuf = dev.makeBuffer(length: outCount * elemSize, options: .storageModeShared)!
buffers.append(outBuf)
for s in scalars {
    var v = s
    buffers.append(dev.makeBuffer(bytes: &v, length: 4, options: .storageModeShared)!)
}

let q = dev.makeCommandQueue()!
let tg = min(threads, pipe.maxTotalThreadsPerThreadgroup)

func dispatchAt(_ width: Int) -> Double {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.dispatchThreadgroups(MTLSize(width: grid, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { die("dispatch: \(e)", 1) }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6   // microseconds
}

func dispatchOnce() -> Double { dispatchAt(tg) }

// A SWEEP, when asked: measure several threadgroup widths in one process.
//
// `-O4` used to invoke this harness once per candidate, so a six-width sweep paid for six
// compilations of the same Swift file and six Metal pipeline builds. The widths differ
// only in how the dispatch is launched, so one process can measure them all.
if !sweepWidths.isEmpty {
    print("#device \(dev.name)")
    for w in sweepWidths {
        let width = min(w, pipe.maxTotalThreadsPerThreadgroup)
        // Each width gets its own warmup: residency and the shader cache are per
        // configuration, and carrying one width's warm state into the next would make
        // whichever ran second look faster.
        for _ in 0..<max(3, iters / 10) { _ = dispatchAt(width) }
        var ts: [Double] = []
        for _ in 0..<iters { ts.append(dispatchAt(width)) }
        ts.sort()
        print(String(format: "#sweep %d %.4f", w, ts[ts.count / 2]))
    }
    exit(0)
}

// Warm up before measuring: the first dispatch pays for shader compilation and residency,
// and reporting it as the kernel's cost would overstate it several-fold.
let warmup = max(3, iters / 10)
for _ in 0..<warmup { _ = dispatchOnce() }

var times: [Double] = []
times.reserveCapacity(iters)
for _ in 0..<iters { times.append(dispatchOnce()) }

print("#device \(dev.name)")
print("#threadgroup \(tg)")
print("#warmup \(warmup)")
for t in times { print(String(format: "#us %.4f", t)) }

let values = readOut(outBuf, outCount, dtype)

// The control arm. Re-fill the inputs from a DIFFERENT seed and dispatch once more: the
// output must change. If it does not, this harness is not actually moving data to the
// device and back, and every number above is describing something other than this
// dispatch -- an empty or stale read-back scores a perfect match against any reference,
// so absence of evidence renders as evidence of agreement.
//
// The rule is the PICO session's, from a board rig that reported the PREVIOUS model's
// output and timings for a model that had failed to load: every comparison run must
// include one arm that MUST disagree, and if it agrees the harness is broken, not the
// kernel. Cheap here -- one dispatch, no recompilation.
for (b, n) in inCounts.enumerated() {
    fill(buffers[b], inputValues(b, n, 1), dtype)
}
_ = dispatchOnce()
let control = readOut(outBuf, outCount, dtype)

var s = ""
for v in values { s += String(format: "%.6e\n", v) }
for v in control { s += String(format: "#c %.6e\n", v) }
print(s, terminator: "")
