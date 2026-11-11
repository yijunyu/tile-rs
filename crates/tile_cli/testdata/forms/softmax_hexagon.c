// Qualcomm HVX intrinsics.
#include <hexagon_types.h>

void softmax(const float* in, float* out, int n) {
    HVX_Vector v = hvx_load(in);
    hvx_store(out, hvx_exp(v));
}
