use crate::error::HalResult;

/// An ordered execution queue on a device.
///
/// Tasks submitted to the same stream execute in FIFO order.
/// Tasks on different streams may execute concurrently.
pub trait Stream: Send {
    /// Block the host thread until all tasks in this stream complete.
    fn synchronize(&self) -> HalResult<()>;
}
