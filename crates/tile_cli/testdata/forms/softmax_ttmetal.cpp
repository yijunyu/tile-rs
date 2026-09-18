#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    cb_wait_front(tt::CB::c_in0, 1);
    exp_tile_init();
    cb_pop_front(tt::CB::c_in0, 1);
}
}
