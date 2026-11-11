//! Linux probes. ROCm and Ryzen AI are DIFFERENT families (see `Accel`), Ascend comes
//! from `npu-smi` in the shared path, and the CANN toolkit is found by its env var.
//!
//! Every parser here is pure and fixture-tested in `platform::mod`'s sibling tests, so
//! this file is checkable from a Mac with `cargo check --target x86_64-unknown-linux-gnu`.

use super::{probe, Accel, Sdk};

pub fn accelerators() -> Vec<Accel> {
    let mut v = Vec::new();
    if let Some(out) = probe("rocm-smi", &["--showproductname"]) {
        if let Some(a) = parse_rocm_smi(&out) {
            v.push(a);
        }
    }
    if let Some(out) = probe("xrt-smi", &["examine"]) {
        if let Some(a) = parse_xrt_smi(&out) {
            v.push(a);
        }
    }
    v
}

pub fn sdks() -> Vec<Sdk> {
    let mut v = Vec::new();
    if let Ok(home) = std::env::var("ASCEND_TOOLKIT_HOME") {
        if !home.is_empty() {
            v.push(Sdk {
                family: "ascend",
                name: "CANN",
                version: std::env::var("ASCEND_VERSION").ok(),
            });
        }
    }
    if probe("glslangValidator", &["--version"]).is_some() {
        v.push(Sdk {
            family: "vulkan",
            name: "glslang",
            version: None,
        });
    }
    v
}

/// A ROCm GPU. Recorded as `amd-gpu`, which has NO tile-rs target — the CLI says so
/// rather than silently emitting Ryzen AI NPU source for it.
pub fn parse_rocm_smi(out: &str) -> Option<Accel> {
    let name = out
        .lines()
        .find_map(|l| l.split("Card series:").nth(1))
        .map(|s| s.trim().to_string())?;
    Some(Accel {
        family: "amd-gpu",
        device: name,
        via: "rocm-smi",
    })
}

/// A Ryzen AI NPU, which DOES have a target (`aie`).
pub fn parse_xrt_smi(out: &str) -> Option<Accel> {
    if !out.contains("NPU") && !out.contains("Ryzen AI") {
        return None;
    }
    let name = out
        .lines()
        .find(|l| l.contains("Ryzen AI") || l.contains("NPU"))
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "Ryzen AI NPU".to_string());
    Some(Accel {
        family: "amd-npu",
        device: name,
        via: "xrt-smi",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_gpu_is_amd_gpu_not_amd_npu() {
        let out = "GPU[0]\t\t: Card series:\t\tRadeon RX 7900 XTX\n";
        let a = parse_rocm_smi(out).expect("parsed");
        assert_eq!(a.family, "amd-gpu");
        assert!(a.device.contains("7900"));
    }

    #[test]
    fn ryzen_ai_is_amd_npu() {
        let a = parse_xrt_smi("Device: Ryzen AI NPU (Phoenix)\n").expect("parsed");
        assert_eq!(a.family, "amd-npu");
    }

    #[test]
    fn an_unrelated_xrt_device_is_not_an_npu() {
        assert!(parse_xrt_smi("Device: Alveo U250\n").is_none());
    }
}
