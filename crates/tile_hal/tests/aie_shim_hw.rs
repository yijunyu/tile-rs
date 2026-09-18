//! Gate D1: execute an xclbin on the NPU from Rust.
//!
//! XRT's C API only exposes the legacy Alveo path, which XDNA2 rejects
//! ("load_axlf: Operation not supported"). The NPU needs register_xclbin +
//! hw_context, which are C++-only, so a small C++ shim (`aie_shim.cpp`) wraps
//! them behind an opaque C surface. This test drives that surface from Rust.
#![cfg(feature = "aie")]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};

#[link(name = "aie_shim")]
extern "C" {
    fn aie_open(index: c_uint, xclbin: *const c_char, kernel: *const c_char) -> *mut c_void;
    fn aie_close(h: *mut c_void);
    fn aie_bo_alloc(h: *mut c_void, bytes: usize, argno: c_int, flags: c_int) -> *mut c_void;
    fn aie_bo_free(b: *mut c_void);
    fn aie_bo_write(b: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn aie_bo_read(b: *mut c_void, dst: *mut c_void, bytes: usize) -> c_int;
    fn aie_run(h: *mut c_void, opcode: c_uint, insts: *mut c_void, words: c_uint,
               bos: *const *mut c_void, n: c_int) -> c_int;
}

const N: usize = 1024;
const OPCODE: c_uint = 3;

#[test]
fn aie_vecadd_from_rust() {
    let xclbin = std::env::var("AIE_XCLBIN")
        .unwrap_or_else(|_| "/data/aie-kernels/vecadd.xclbin".into());
    let insts_path = xclbin.replace(".xclbin", ".insts.bin");
    let insts = std::fs::read(&insts_path).expect("instruction stream");

    let cx = CString::new(xclbin).unwrap();
    let ck = CString::new("").unwrap();      // empty -> first kernel in the xclbin
    let h = unsafe { aie_open(0, cx.as_ptr(), ck.as_ptr()) };
    assert!(!h.is_null(), "aie_open failed");

    // arg 1 is the instruction stream, args 3.. are data buffers
    let bi = unsafe { aie_bo_alloc(h, insts.len(), 1, 1) };
    assert!(!bi.is_null(), "insts bo");
    assert_eq!(0, unsafe { aie_bo_write(bi, insts.as_ptr() as *const c_void, insts.len()) });

    let a: Vec<f32> = (0..N).map(|i| i as f32 / N as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 2.0 * i as f32 / N as f32).collect();
    let nbytes = N * 4;

    let ba = unsafe { aie_bo_alloc(h, nbytes, 3, 0) };
    let bb = unsafe { aie_bo_alloc(h, nbytes, 4, 0) };
    let bc = unsafe { aie_bo_alloc(h, nbytes, 5, 0) };
    assert!(!ba.is_null() && !bb.is_null() && !bc.is_null());
    unsafe {
        aie_bo_write(ba, a.as_ptr() as *const c_void, nbytes);
        aie_bo_write(bb, b.as_ptr() as *const c_void, nbytes);
    }

    let bos = [ba, bb, bc];
    let rc = unsafe {
        aie_run(h, OPCODE, bi, (insts.len() / 4) as c_uint, bos.as_ptr(), 3)
    };
    assert_eq!(rc, 0, "aie_run returned {rc}");

    let mut got = vec![0f32; N];
    assert_eq!(0, unsafe { aie_bo_read(bc, got.as_mut_ptr() as *mut c_void, nbytes) });

    let mut max_abs = 0f32;
    for i in 0..N {
        max_abs = max_abs.max((got[i] - (a[i] + b[i])).abs());
    }
    println!("vecadd on NPU from Rust: max |diff| = {max_abs:e}");
    assert!(max_abs < 1e-5, "wrong result: max_abs={max_abs}");

    unsafe {
        aie_bo_free(ba); aie_bo_free(bb); aie_bo_free(bc); aie_bo_free(bi);
        aie_close(h);
    }
}
