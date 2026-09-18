//! Does NPU/iGPU co-execution actually beat host-only wall clock?
//!
//! Earlier concurrency work proved overlap is real (1.95x vs sequential) but
//! co-execution still lost to host-only, because one NPU expert cost ~590 us
//! against ~101 us for a host expert -- the NPU was the critical path.
//!
//! The vectorised fused expert costs ~137 us. Against a *debug* host build this
//! shows 1.29x, but that is an artifact: in release, six host experts cost 30 us
//! total, so the NPU loses by 4.5x. At 64x64 an expert is 24.6 KFLOP against an
//! 80 us dispatch floor -- three orders of magnitude too small.
//!
//! Kept as a correctness test for the concurrent path and as a record of the
//! measurement. The real comparison needs realistic dimensions and the iGPU as
//! the partner engine; see coexec_scale_hw.rs.
#![cfg(feature = "aie")]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::time::Instant;

#[link(name = "aie_shim")]
extern "C" {
    fn aie_open(index: c_uint, xclbin: *const c_char, kernel: *const c_char) -> *mut c_void;
    fn aie_close(h: *mut c_void);
    fn aie_bo_alloc(h: *mut c_void, bytes: usize, argno: c_int, flags: c_int) -> *mut c_void;
    fn aie_bo_free(b: *mut c_void);
    fn aie_bo_write(b: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn aie_bo_read(b: *mut c_void, dst: *mut c_void, bytes: usize) -> c_int;
    fn aie_run_start(h: *mut c_void, opcode: c_uint, insts: *mut c_void, words: c_uint,
                     bos: *const *mut c_void, n: c_int) -> *mut c_void;
    fn aie_run_wait(r: *mut c_void) -> c_int;
}

const D: usize = 64;
const WCAT: usize = 3 * D * D;
const OPCODE: c_uint = 3;

struct FusedExpert { h: *mut c_void, insts: *mut c_void, words: c_uint,
                     bw: *mut c_void, bx: *mut c_void, by: *mut c_void }

impl FusedExpert {
    fn new() -> Option<Self> {
        let xc = std::env::var("AIE_EXPERT")
            .unwrap_or_else(|_| "/data/aie-kernels/expert.xclbin".into());
        let ib = std::fs::read(xc.replace(".xclbin", ".insts.bin")).ok()?;
        let cx = CString::new(xc).ok()?; let ck = CString::new("").ok()?;
        let h = unsafe { aie_open(0, cx.as_ptr(), ck.as_ptr()) };
        if h.is_null() { return None; }
        let insts = unsafe { aie_bo_alloc(h, ib.len(), 1, 1) };
        unsafe { aie_bo_write(insts, ib.as_ptr() as *const c_void, ib.len()) };
        Some(Self { h, insts, words: (ib.len()/4) as c_uint,
            bw: unsafe { aie_bo_alloc(h, WCAT*4, 3, 0) },
            bx: unsafe { aie_bo_alloc(h, D*4,    4, 0) },
            by: unsafe { aie_bo_alloc(h, D*4,    5, 0) } })
    }
    /// Upload the concatenated weights once; they stay resident for later runs.
    fn upload_weights(&self, wg: &[f32], wu: &[f32], wd: &[f32]) {
        let mut cat = Vec::with_capacity(WCAT);
        cat.extend_from_slice(wg); cat.extend_from_slice(wu); cat.extend_from_slice(wd);
        unsafe { aie_bo_write(self.bw, cat.as_ptr() as *const c_void, WCAT*4) };
    }
    fn start(&self, x: &[f32]) -> *mut c_void {
        unsafe { aie_bo_write(self.bx, x.as_ptr() as *const c_void, D*4) };
        let bos = [self.bw, self.bx, self.by];
        unsafe { aie_run_start(self.h, OPCODE, self.insts, self.words, bos.as_ptr(), 3) }
    }
    fn join(&self, r: *mut c_void, w_e: f32) -> Vec<f32> {
        assert_eq!(0, unsafe { aie_run_wait(r) });
        let mut y = vec![0f32; D];
        unsafe { aie_bo_read(self.by, y.as_mut_ptr() as *mut c_void, D*4) };
        for v in y.iter_mut() { *v *= w_e; }     // y is linear in w_e
        y
    }
}
impl Drop for FusedExpert {
    fn drop(&mut self) { unsafe {
        aie_bo_free(self.bw); aie_bo_free(self.bx); aie_bo_free(self.by);
        aie_bo_free(self.insts); aie_close(self.h);
    }}
}

fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..D).map(|j| (0..D).map(|i| w[j*D+i]*x[i]).sum()).collect()
}
fn cpu_expert(x: &[f32], wg: &[f32], wu: &[f32], wd: &[f32], w_e: f32) -> Vec<f32> {
    const C: f32 = 10.0;
    let g = mv(wg,x); let u = mv(wu,x);
    let mid: Vec<f32> = g.iter().zip(u.iter())
        .map(|(&a,&b)| { let a=a.min(C); (a/(1.0+(-a).exp()))*b.clamp(-C,C)*w_e }).collect();
    mv(wd,&mid)
}


fn npu_expert_weights(npu: &FusedExpert, wg: &[f32], wu: &[f32], wd: &[f32]) {
    // the kernel's gate accumulator feeds exp2 directly, so W_gate is pre-scaled
    let mut wgs: Vec<f32> = wg.to_vec();
    for v in wgs.iter_mut() { *v *= -std::f32::consts::LOG2_E; }
    npu.upload_weights(&wgs, wu, wd);
}

#[test]
fn coexecution_beats_host_only() {
    std::env::set_var("AIE_EXPERT", "/data/aie-kernels/expertv.xclbin");
    let npu = match FusedExpert::new() { Some(n)=>n, None=>{ eprintln!("skip: no expertv.xclbin"); return; } };

    const N: usize = 6;                       // experts routed for this token
    let g = |i: usize| ((i as f32*12.9898).sin()*43758.547).fract() - 0.5;
    let x: Vec<f32> = (0..D).map(g).collect();
    let wt: Vec<f32> = (0..N).map(|e| 0.1 + 0.05*e as f32).collect();
    let wg: Vec<Vec<f32>> = (0..N).map(|e| (0..D*D).map(|i| g(i+7+e*31)*0.1).collect()).collect();
    let wu: Vec<Vec<f32>> = (0..N).map(|e| (0..D*D).map(|i| g(i+101+e*31)*0.1).collect()).collect();
    let wd: Vec<Vec<f32>> = (0..N).map(|e| (0..D*D).map(|i| g(i+907+e*31)*0.1).collect()).collect();
    // NPU indexes weights as [in][out]
    let t = |w: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; D*D];
        for j in 0..D { for i in 0..D { o[i*D+j] = w[j*D+i]; } }
        o
    };
    let tg: Vec<Vec<f32>> = wg.iter().map(|w| t(w)).collect();
    let tu: Vec<Vec<f32>> = wu.iter().map(|w| t(w)).collect();
    let td: Vec<Vec<f32>> = wd.iter().map(|w| t(w)).collect();

    let host_sum = |lo: usize, hi: usize| -> Vec<f32> {
        let mut acc = vec![0f32; D];
        for e in lo..hi {
            let y = cpu_expert(&x, &wg[e], &wu[e], &wd[e], wt[e]);
            for (a,v) in acc.iter_mut().zip(y.iter()) { *a += v; }
        }
        acc
    };

    const IT: usize = 50;
    // baseline: every expert on the host
    for _ in 0..5 { std::hint::black_box(host_sum(0, N)); }
    let t0 = Instant::now();
    for _ in 0..IT { std::hint::black_box(host_sum(0, N)); }
    let t_host = t0.elapsed().as_secs_f64()/IT as f64;

    let want = host_sum(0, N);
    println!("host-only ({N} experts)      {:8.1} us", t_host*1e6);

    let mut best = (0usize, f64::MAX);
    for k in 1..=3usize {
        // k experts on the NPU (run back to back), N-k on the host, overlapped
        let run = || -> Vec<f32> {
            let mut acc = vec![0f32; D];
            let first = npu.start(&x);                 // NPU expert 0 in flight
            let mut pending = Some((first, 0usize));
            let mut host = None;
            for i in 0..k {
                let (r, e) = pending.take().unwrap();
                if i == 0 { host = Some(host_sum(k, N)); }   // host works while NPU runs
                let y = npu.join(r, wt[e]);
                for (a,v) in acc.iter_mut().zip(y.iter()) { *a += v; }
                if i+1 < k {
                    npu_expert_weights(&npu, &tg[i+1], &tu[i+1], &td[i+1]);
                    pending = Some((npu.start(&x), i+1));
                }
            }
            for (a,v) in acc.iter_mut().zip(host.unwrap().iter()) { *a += v; }
            acc
        };
        npu_expert_weights(&npu, &tg[0], &tu[0], &td[0]);
        for _ in 0..5 { std::hint::black_box(run()); }
        npu_expert_weights(&npu, &tg[0], &tu[0], &td[0]);
        let got = run();
        let mut max_rel = 0f32;
        for (a,b) in got.iter().zip(want.iter()) {
            assert!(a.is_finite(), "non-finite output at k={k}");
            let r = (a-b).abs()/b.abs().max(1e-3);
            if r > max_rel { max_rel = r; }
        }
        let t1 = Instant::now();
        for _ in 0..IT { npu_expert_weights(&npu, &tg[0], &tu[0], &td[0]); std::hint::black_box(run()); }
        let t_co = t1.elapsed().as_secs_f64()/IT as f64;
        println!("  {k} on NPU / {} on host   {:8.1} us   {:.2}x   max_rel {max_rel:e}",
                 N-k, t_co*1e6, t_host/t_co);
        if t_co < best.1 { best = (k, t_co); }
    }
    let ratio = t_host/best.1;
    println!("best split: {} on NPU -> {ratio:.2}x vs host-only", best.0);
    if ratio > 1.0 {
        println!("NOTE: only meaningful in a release build; a debug host core is ~30x slower");
    } else {
        println!("expected at 64x64: 24.6 KFLOP per expert against an 80 us dispatch floor.");
        println!("A release host core does an expert in ~5 us; the NPU needs ~137 us.");
        println!("Co-execution needs realistic expert dimensions, and the iGPU -- not a CPU");
        println!("core -- as the partner engine.");
    }
}
