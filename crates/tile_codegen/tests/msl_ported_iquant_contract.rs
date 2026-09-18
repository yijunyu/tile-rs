//! The ported i-quant matvecs and the simdgroup count their table staging
//! requires.
//!
//! These four kernels are byte-for-byte ports of upstream llama.cpp, so their
//! bodies are not ours to parameterize: 107 fidelity tests exist to keep them
//! identical. What they carry instead is an unstated precondition. Each stages
//! a lookup table into threadgroup memory by handing every lane one equal,
//! contiguous run of entries, with the run length written as a literal. The run
//! length times a threadgroup's lanes has to equal the table, which happens at
//! exactly two simdgroups and nowhere else.
//!
//! That matters because the simdgroup count is a Metal function constant: the
//! host picks it at pipeline creation, and ours picks it from the model
//! dimension over the range 1 to 8. A kernel launched at any other count reads
//! table entries that were never written, or writes past the table, and Metal
//! faults on neither.
//!
//! This gate holds the emitted text and the recorded obligation to each other,
//! so neither can drift without the other saying so.
#![cfg(feature = "emitters")]

fn contract_table() -> String {
    let p = format!(
        "{}/../../benchmarks/ds4_msl/emitted/ported_contracts.rs",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn emitter_source() -> String {
    let p = format!(
        "{}/../rustc_codegen_tile/src/mlir_to_msl_canned.rs",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// kernel row, emitter, table entries, run length per lane.
const STAGED: [(&str, &str, u32, u32); 4] = [
    ("kernel_mul_mv_iq2_xxs_f32_impl", "emit_mul_mv_iq2_xxs_f32_ggml", 256, 4),
    ("kernel_mul_mv_iq2_xs_f32_impl", "emit_mul_mv_iq2_xs_f32_ggml", 512, 8),
    ("kernel_mul_mv_iq3_xxs_f32_impl", "emit_mul_mv_iq3_xxs_f32_ggml", 256, 4),
    ("kernel_mul_mv_iq3_s_f32_impl", "emit_mul_mv_iq3_s_f32_ggml", 512, 8),
];

fn body_of(src: &str, emitter: &str) -> String {
    let sig = format!("pub(super) fn {emitter}(out: &mut String) {{");
    let i = src.find(&sig).unwrap_or_else(|| panic!("no emitter {emitter}"));
    let j = src[i..].find("\n}\n").unwrap() + i;
    src[i..j].to_string()
}

#[test]
fn every_staged_table_charges_the_count_its_run_length_implies() {
    let table = contract_table();
    let src = emitter_source();
    for (kernel, emitter, entries, run) in STAGED {
        // The run length is still what the contract was derived from.
        let body = body_of(&src, emitter);
        assert!(
            body.contains(&format!("\"        int nval = {run};\"")),
            "{emitter} no longer stages {run} entries per lane; the contract row \
             for {kernel} was derived from that number"
        );
        // The count is derived, not asserted: a run length that does not divide
        // the table spans no whole number of simdgroups and cannot be charged.
        let lanes = run * 32;
        assert_eq!(
            entries % lanes,
            0,
            "{kernel}: runs of {run} cover {lanes} entries per simdgroup, which \
             does not divide the {entries} the table holds"
        );
        let want = entries / lanes;
        assert_eq!(want, 2, "{kernel}: arithmetic changed under the test");
        let row = table
            .find(&format!("kernel: {kernel:?}"))
            .unwrap_or_else(|| panic!("{kernel} has no contract row"));
        let rest = &table[row..];
        let end = rest.find("KernelContract {").unwrap_or(rest.len());
        assert!(
            rest[..end].contains(&format!("Obligation::SimdgroupsExactly {{ n: {want},")),
            "{kernel} does not charge exactly {want} simdgroups:\n{}",
            &rest[..end]
        );
    }
}

#[test]
fn a_kernel_without_fixed_run_staging_is_not_charged() {
    let table = contract_table();
    // q8_0 is the same family and the same shell, and stages no table.
    let src = emitter_source();
    let body = body_of(&src, "emit_mul_mv_q8_0_f32_ggml");
    assert!(!body.contains("int nval ="), "q8_0 unexpectedly stages a table");
    assert!(
        !table.contains("kernel_mul_mv_q8_0_f32_impl"),
        "a kernel with no fixed-run staging picked up the obligation anyway"
    );
}

/// The contract is only worth having if something reads it. This records what
/// currently does, so a later reader can tell enforcement from documentation.
#[test]
fn the_obligation_reaches_the_generated_table_with_its_reason() {
    let table = contract_table();
    assert!(
        table.contains("SimdgroupsExactly { n: u32, table: &'static str, because: &'static str }"),
        "the generated enum lost the variant"
    );
    for (kernel, _, _, _) in STAGED {
        let at = table.find(&format!("kernel: {kernel:?}")).unwrap();
        let rest = &table[at..];
        let end = rest.find("KernelContract {").unwrap_or(rest.len());
        let row = &rest[..end];
        assert!(row.contains("migrated: false"), "{kernel}: obligations are not fully known, say so");
        assert!(
            row.contains("because: \"the ") && row.len() > 300,
            "{kernel}: the row carries no usable reason:\n{row}"
        );
    }
}
