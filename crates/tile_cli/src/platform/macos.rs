//! macOS probes. Apple GPUs are found through `system_profiler`; the Metal toolchain
//! through `xcrun`. No Objective-C, no linking — this is `Command` and string parsing.

use super::{probe, Accel, Sdk};

pub fn accelerators() -> Vec<Accel> {
    let mut v = Vec::new();
    if let Some(out) = probe("system_profiler", &["SPDisplaysDataType"]) {
        if let Some(a) = parse_system_profiler(&out) {
            v.push(a);
        }
    }
    v
}

pub fn sdks() -> Vec<Sdk> {
    let mut v = Vec::new();
    if probe("xcrun", &["--find", "metal"]).is_some() {
        let version = probe("xcrun", &["metal", "--version"]).and_then(|o| parse_metal_version(&o));
        v.push(Sdk {
            family: "apple-gpu",
            name: "metal",
            version,
        });
    }
    v
}

/// `system_profiler SPDisplaysDataType` prints "Chipset Model: Apple M1 Ultra".
pub fn parse_system_profiler(out: &str) -> Option<Accel> {
    let model = out
        .lines()
        .find_map(|l| l.trim().strip_prefix("Chipset Model:"))
        .map(|s| s.trim().to_string())?;
    Some(Accel {
        family: "apple-gpu",
        device: model,
        via: "system_profiler",
    })
}

pub fn parse_metal_version(out: &str) -> Option<String> {
    out.lines()
        .find(|l| l.contains("metal"))
        .and_then(|l| {
            l.split_whitespace()
                .find(|t| t.chars().next().is_some_and(|c| c.is_ascii_digit()))
        })
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chipset_model_is_extracted() {
        let out = "Graphics/Displays:\n\n    Apple M1 Ultra:\n\n      Chipset Model: Apple M1 Ultra\n      Type: GPU\n";
        let a = parse_system_profiler(out).expect("parsed");
        assert_eq!(a.family, "apple-gpu");
        assert_eq!(a.device, "Apple M1 Ultra");
    }

    #[test]
    fn a_machine_with_no_gpu_section_yields_nothing() {
        assert!(parse_system_profiler("Graphics/Displays:\n\n").is_none());
    }
}
