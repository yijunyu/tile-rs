//! Ascend NPU backend for the HAL.
//!
//! Wraps `ascend_rs` types behind the HAL traits. Enabled by the `ascend` feature.

use std::ffi::c_void;
use std::path::Path;

use ascend_rs::acl::Acl;
use ascend_rs::context::AclContext;
use ascend_rs::kernel::{KernelLoader, KernelMagic};
use ascend_rs::memory::core as acl_mem;
use ascend_rs::memory::{DeviceBuffer as AclDeviceBuffer};
use ascend_rs::stream::AclStream;

use crate::backend::DynDevice;
use crate::buffer::DeviceRepr;
use crate::device::{Device, MemInfo};
use crate::error::{HalError, HalResult};
use crate::kernel::{KernelLauncher, KernelMode, LaunchGrid};
use crate::stream::Stream;

// ── Stream ──

/// Ascend stream wrapping `AclStream`.
pub struct AscendStream {
    // We own a raw stream pointer since AclStream has lifetime constraints
    // that we manage ourselves via the RAII wrapper.
    raw: ascend_sys::core::aclrtStream,
}

impl AscendStream {
    fn new() -> HalResult<Self> {
        unsafe {
            let mut raw: ascend_sys::core::aclrtStream = std::ptr::null_mut();
            let ret = ascend_sys::core::aclrtCreateStream(&mut raw);
            if ret != 0 {
                return Err(HalError::StreamError(format!("aclrtCreateStream failed: {}", ret)));
            }
            Ok(Self { raw })
        }
    }

    /// Get the raw stream pointer for FFI calls.
    pub fn raw(&self) -> ascend_sys::core::aclrtStream {
        self.raw
    }
}

impl Stream for AscendStream {
    fn synchronize(&self) -> HalResult<()> {
        unsafe {
            let ret = ascend_sys::core::aclrtSynchronizeStream(self.raw);
            if ret != 0 {
                return Err(HalError::StreamError(format!("aclrtSynchronizeStream failed: {}", ret)));
            }
        }
        Ok(())
    }
}

unsafe impl Send for AscendStream {}

impl Drop for AscendStream {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::core::aclrtDestroyStream(self.raw);
        }
    }
}

// ── Buffer ──

/// Ascend device buffer wrapping raw device memory.
pub struct AscendBuffer<T: DeviceRepr> {
    ptr: *mut T,
    count: usize,
}

impl<T: DeviceRepr> AscendBuffer<T> {
    /// Get the raw device pointer.
    pub fn raw_ptr(&self) -> *mut T {
        self.ptr
    }
}

impl<T: DeviceRepr> crate::buffer::DeviceBuffer<T> for AscendBuffer<T> {
    fn len(&self) -> usize {
        self.count
    }

    fn copy_from_host(&mut self, src: &[T], _stream: &dyn Stream) -> HalResult<()> {
        assert!(src.len() <= self.count, "source slice larger than buffer");
        unsafe {
            let size = src.len() * std::mem::size_of::<T>();
            let ret = ascend_sys::core::aclrtMemcpy(
                self.ptr as *mut c_void,
                size,
                src.as_ptr() as *const c_void,
                size,
                ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_HOST_TO_DEVICE,
            );
            if ret != 0 {
                return Err(HalError::MemoryError(format!("H2D memcpy failed: {}", ret)));
            }
        }
        Ok(())
    }

    fn copy_to_host(&self, _stream: &dyn Stream) -> HalResult<Vec<T>> {
        let mut dst = vec![unsafe { std::mem::zeroed::<T>() }; self.count];
        unsafe {
            let size = self.count * std::mem::size_of::<T>();
            let ret = ascend_sys::core::aclrtMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                size,
                self.ptr as *const c_void,
                size,
                ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_DEVICE_TO_HOST,
            );
            if ret != 0 {
                return Err(HalError::MemoryError(format!("D2H memcpy failed: {}", ret)));
            }
        }
        Ok(dst)
    }

    fn as_raw_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }

    fn as_raw_mut_ptr(&mut self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}

unsafe impl<T: DeviceRepr> Send for AscendBuffer<T> {}

impl<T: DeviceRepr> Drop for AscendBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::core::aclrtFree(self.ptr as *mut c_void);
        }
    }
}

// ── Kernel Launcher ──

/// Ascend kernel launcher wrapping `KernelLoader`.
pub struct AscendLauncher {
    loader: Option<KernelLoader>,
}

impl KernelLauncher for AscendLauncher {
    type Stream = AscendStream;

    fn load(&mut self, path: &Path, mode: KernelMode) -> HalResult<()> {
        let magic = match mode {
            KernelMode::Vector => KernelMagic::Vector,
            KernelMode::Matrix => KernelMagic::Cube,
            KernelMode::General => KernelMagic::AiCore,
        };
        // Detect .so vs .o
        let loader = if path.extension().map_or(false, |e| e == "so") {
            KernelLoader::from_shared_lib(path)
        } else {
            KernelLoader::from_bin_path_with_magic(path, magic)
        };
        self.loader = Some(loader.map_err(|e| HalError::KernelError(format!("{:?}", e)))?);
        Ok(())
    }

    unsafe fn launch(
        &self,
        name: &str,
        grid: LaunchGrid,
        args: &mut [*mut c_void],
        stream: &Self::Stream,
    ) -> HalResult<()> {
        let loader = self
            .loader
            .as_ref()
            .ok_or_else(|| HalError::KernelError("no kernel loaded".into()))?;
        let kernel = loader
            .get_kernel(name)
            .map_err(|e| HalError::KernelError(format!("{:?}", e)))?;

        // Reconstruct an AclStream from the raw pointer for the launch API
        let acl_stream = unsafe { AclStream::from_raw(stream.raw()) };
        let result = unsafe { kernel.launch(grid.blocks, &acl_stream, args) };
        // Don't drop the AclStream (we don't own it)
        std::mem::forget(acl_stream);

        result.map_err(|e| HalError::KernelError(format!("{:?}", e)))
    }
}

// ── Device ──

/// Ascend NPU device.
pub struct AscendDevice {
    ordinal: u32,
    name: String,
}

impl AscendDevice {
    /// Initialize the Ascend runtime and select the given device.
    ///
    /// This calls `aclInit` and `aclrtSetDevice` internally.
    pub fn new(ordinal: u32) -> HalResult<Self> {
        unsafe {
            let ret = ascend_sys::core::aclInit(std::ptr::null());
            // aclInit returns 0 on success, or if already initialized
            if ret != 0 {
                // Check if it's "already initialized" (error code 100000)
                if ret != 100000 {
                    return Err(HalError::DeviceError(format!("aclInit failed: {}", ret)));
                }
            }
            let ret = ascend_sys::core::aclrtSetDevice(ordinal as i32);
            if ret != 0 {
                return Err(HalError::DeviceError(format!(
                    "aclrtSetDevice({}) failed: {}",
                    ordinal, ret
                )));
            }
        }

        let soc = std::env::var("ACLRS_SOC_VERSION").unwrap_or_else(|_| "Ascend910B".into());
        Ok(Self {
            ordinal,
            name: soc,
        })
    }
}

impl Device for AscendDevice {
    type Stream = AscendStream;
    type Buffer<T: DeviceRepr> = AscendBuffer<T>;
    type Launcher = AscendLauncher;

    fn ordinal(&self) -> u32 {
        self.ordinal
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn create_stream(&self) -> HalResult<AscendStream> {
        AscendStream::new()
    }

    fn alloc<T: DeviceRepr>(&self, count: usize) -> HalResult<AscendBuffer<T>> {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let size = count * std::mem::size_of::<T>();
            let ret = ascend_sys::core::aclrtMalloc(
                &mut ptr,
                size,
                ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE_FIRST,
            );
            if ret != 0 {
                return Err(HalError::MemoryError(format!("aclrtMalloc failed: {}", ret)));
            }
            Ok(AscendBuffer {
                ptr: ptr as *mut T,
                count,
            })
        }
    }

    fn alloc_from_slice<T: DeviceRepr>(
        &self,
        data: &[T],
        _stream: &AscendStream,
    ) -> HalResult<AscendBuffer<T>> {
        let mut buf = self.alloc(data.len())?;
        // Use synchronous H2D copy
        unsafe {
            let size = data.len() * std::mem::size_of::<T>();
            let ret = ascend_sys::core::aclrtMemcpy(
                buf.ptr as *mut c_void,
                size,
                data.as_ptr() as *const c_void,
                size,
                ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_HOST_TO_DEVICE,
            );
            if ret != 0 {
                return Err(HalError::MemoryError(format!("H2D memcpy failed: {}", ret)));
            }
        }
        Ok(buf)
    }

    fn mem_info(&self) -> HalResult<MemInfo> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let ret = ascend_sys::core::aclrtGetMemInfo(
                ascend_sys::core::aclrtMemAttr_ACL_DDR_MEM,
                &mut free,
                &mut total,
            );
            if ret != 0 {
                return Err(HalError::DeviceError(format!("aclrtGetMemInfo failed: {}", ret)));
            }
            Ok(MemInfo { free, total })
        }
    }

    fn create_launcher(&self) -> HalResult<AscendLauncher> {
        Ok(AscendLauncher { loader: None })
    }
}

impl DynDevice for AscendDevice {
    fn ordinal(&self) -> u32 {
        Device::ordinal(self)
    }

    fn name(&self) -> &str {
        Device::name(self)
    }

    fn mem_info(&self) -> HalResult<MemInfo> {
        Device::mem_info(self)
    }

    fn alloc_f32(&self, count: usize) -> HalResult<Box<dyn crate::buffer::DeviceBuffer<f32>>> {
        let buf = Device::alloc::<f32>(self, count)?;
        Ok(Box::new(buf))
    }
}
