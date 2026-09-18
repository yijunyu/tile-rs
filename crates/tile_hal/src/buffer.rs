use crate::error::HalResult;
use crate::stream::Stream;

/// Marker trait for types that can be transferred to/from device memory.
///
/// # Safety
/// The type must be `Copy` and have no pointers or references
/// (i.e., it must be safe to memcpy to device memory).
pub unsafe trait DeviceRepr: Copy + Send + Sync + 'static {}

unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}
unsafe impl DeviceRepr for f16 {}
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}

/// Use a newtype since f16 isn't stable.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct f16(pub u16);

/// A typed, owned allocation in device memory.
///
/// Implementations are responsible for freeing the allocation on drop.
pub trait DeviceBuffer<T: DeviceRepr>: Send {
    /// Number of elements in the buffer.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy data from a host slice into this device buffer.
    ///
    /// The stream orders the transfer — the host slice must remain valid
    /// until the stream is synchronized.
    fn copy_from_host(&mut self, src: &[T], stream: &dyn Stream) -> HalResult<()>;

    /// Copy data from this device buffer to a host-side Vec.
    fn copy_to_host(&self, stream: &dyn Stream) -> HalResult<Vec<T>>;

    /// Return a raw pointer suitable for passing to kernel launch args.
    fn as_raw_ptr(&self) -> *const std::ffi::c_void;

    /// Return a mutable raw pointer for kernel launch args.
    fn as_raw_mut_ptr(&mut self) -> *mut std::ffi::c_void;
}
