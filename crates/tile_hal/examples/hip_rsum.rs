//! Run one tile-rs-generated reduce_sum kernel on AMD gfx1151 through the HIP HAL.
use std::ffi::c_void;
use tile_hal::buffer::DeviceBuffer;
use tile_hal::device::Device;
use tile_hal::hip::HipDevice;
use tile_hal::kernel::KernelLauncher;
use tile_hal::stream::Stream;

fn main() -> tile_hal::error::HalResult<()> {
    let dev = HipDevice::new(0)?;
    println!("device: {}", dev.name());
    let stream = dev.create_stream()?;
    let mut l = dev.create_launcher()?;
    l.load(std::path::Path::new("/tmp/reduce_sum.code"), tile_hal::kernel::KernelMode::Vector)?;

    let n = 1024usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut in_buf = dev.alloc_from_slice(&input, &stream)?;
    let mut out_buf = dev.alloc::<f32>(n)?;
    let mut in_ptr = in_buf.as_raw_mut_ptr();
    let mut out_ptr = out_buf.as_raw_mut_ptr();
    let grid = tile_hal::kernel::LaunchGrid { blocks: 1, threads_per_block: 256, shared_mem_bytes: 0 };
    let mut args = vec![&mut in_ptr as *mut _ as *mut std::ffi::c_void, &mut out_ptr as *mut _ as *mut std::ffi::c_void];
    unsafe {
        l.launch("_Z4rsumPKfPf", grid, &mut args, &stream)?;
    }
    stream.synchronize()?;
    let out = out_buf.copy_to_host(&stream)?;
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = input.iter().map(|&x| (x - max).exp()).sum();
    let expected: Vec<f32> = input.iter().map(|&x| (x - max).exp() / sum).collect();
    let bad: Vec<usize> = out[..256].iter().enumerate().filter_map(|(i, v)| {
        let warp_start = i & !31;
        let exp: f32 = input[warp_start..warp_start + 32].iter().sum();
        if (v - exp).abs() > 1e-3 { Some(i) } else { None }
    }).collect();
    let ok = bad.is_empty();
    println!("rsum ok = {}", ok);
    Ok(())
}
