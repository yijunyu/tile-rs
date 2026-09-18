// Built with tpc-clang for the Gaudi TPC.
#include "kernel_config.h"

void main(tensor in, tensor out) {
    float64 v = v_f32_ld_tnsr_b(in);
    v_f32_st_tnsr(out, v_exp_f32(v));
}
