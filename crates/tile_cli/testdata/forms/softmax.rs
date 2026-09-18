// A tile-rs kernel. `.rs` is RESERVED: this file is never sniffed, which is why it is
// safe for it to mention __global__ and kernel void in a comment without being mistaken
// for CUDA or Metal.
//
// It is also a REAL kernel, not a sketch: `tile testdata/forms/softmax.rs -o k.metal`
// lowers it through the codegen backend and emits Metal. The three crate attributes are
// not decoration -- tile_std is `#![no_core]`, so a crate that depends on it must be too.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::tile::{__tile_load_f32, __tile_softmax_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn softmax_1d(input: GmView<1, 1024, f32>, output: GmViewMut<1, 1024, f32>) {
    unsafe {
        let t = __tile_load_f32(input.as_ptr(), 1, 1024);
        let s = __tile_softmax_f32(0, t, 1, 1024);
        __tile_store_f32(output.as_mut_ptr(), s, 1, 1024);
    }
}
