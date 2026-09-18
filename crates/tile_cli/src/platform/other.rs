//! Hosts with no vendor probe of their own (Windows, the BSDs). The tool still runs,
//! still reports its triple, and still defaults to the CPU `linalg` bridge — which is
//! the point of having a fallback that is always runnable.

use super::{Accel, Sdk};

pub fn accelerators() -> Vec<Accel> {
    Vec::new()
}

pub fn sdks() -> Vec<Sdk> {
    Vec::new()
}
