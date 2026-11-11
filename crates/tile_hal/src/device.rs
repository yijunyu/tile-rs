use crate::buffer::{DeviceBuffer, DeviceRepr};
use crate::error::HalResult;
use crate::kernel::KernelLauncher;
use crate::stream::Stream;

/// Memory usage information for a device.
#[derive(Debug, Clone, Copy)]
pub struct MemInfo {
    /// Free memory in bytes.
    pub free: usize,
    /// Total memory in bytes.
    pub total: usize,
}

impl MemInfo {
    pub fn used(&self) -> usize {
        self.total - self.free
    }
}

/// A compute device (NPU, GPU, AIE tile, etc.).
///
/// This is the central entry point for interacting with a hardware accelerator.
/// From a device you can create streams, allocate memory, and load kernels.
pub trait Device: Send + Sync {
    /// The concrete stream type for this device.
    type Stream: Stream;
    /// The concrete buffer type for this device.
    type Buffer<T: DeviceRepr>: DeviceBuffer<T>;
    /// The concrete kernel launcher for this device.
    type Launcher: KernelLauncher<Stream = Self::Stream>;

    /// Device ordinal (0-based).
    fn ordinal(&self) -> u32;

    /// Human-readable device name (e.g., "Ascend910B2", "NVIDIA A100").
    fn name(&self) -> &str;

    /// Create an execution stream on this device.
    fn create_stream(&self) -> HalResult<Self::Stream>;

    /// Allocate `count` elements of type `T` on device memory (uninitialized).
    fn alloc<T: DeviceRepr>(&self, count: usize) -> HalResult<Self::Buffer<T>>;

    /// Allocate and copy data from a host slice to device memory.
    fn alloc_from_slice<T: DeviceRepr>(
        &self,
        data: &[T],
        stream: &Self::Stream,
    ) -> HalResult<Self::Buffer<T>>;

    /// Query device memory usage.
    fn mem_info(&self) -> HalResult<MemInfo>;

    /// Create a kernel launcher that can load and dispatch kernels.
    fn create_launcher(&self) -> HalResult<Self::Launcher>;
}
