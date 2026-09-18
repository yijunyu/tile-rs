// C++ shim exposing the XRT hw_context flow to C/Rust.
//
// XRT's C API only offers the legacy Alveo/PL path (xrtDeviceLoadXclbinFile),
// which XDNA2 rejects with "load_axlf: Operation not supported". The NPU needs
// register_xclbin + hw_context, which exist only in the C++ API. This wraps
// that in a small opaque C surface so tile_hal can stay pure Rust FFI.
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstring>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

struct AieCtx {
    xrt::device dev;
    xrt::xclbin xclbin;
    xrt::hw_context ctx;
    xrt::kernel kern;
};

extern "C" {

void* aie_open(unsigned index, const char* xclbin_path, const char* kernel_name) {
    try {
        auto* c = new AieCtx{};
        c->dev = xrt::device(index);
        c->xclbin = xrt::xclbin(std::string(xclbin_path));
        c->dev.register_xclbin(c->xclbin);
        c->ctx = xrt::hw_context(c->dev, c->xclbin.get_uuid());
        std::string name = kernel_name && *kernel_name ? kernel_name : "";
        if (name.empty()) {
            auto ks = c->xclbin.get_kernels();
            if (ks.empty()) { delete c; return nullptr; }
            name = ks[0].get_name();
        }
        c->kern = xrt::kernel(c->ctx, name);
        return c;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "aie_open: %s\n", e.what());
        return nullptr;
    }
}

void aie_close(void* h) { delete static_cast<AieCtx*>(h); }

int aie_group_id(void* h, int argno) {
    try { return static_cast<AieCtx*>(h)->kern.group_id(argno); }
    catch (...) { return -1; }
}

// flags: 0 = host_only (data buffers), 1 = cacheable (instruction stream)
void* aie_bo_alloc(void* h, size_t bytes, int argno, int flags) {
    try {
        auto* c = static_cast<AieCtx*>(h);
        auto grp = c->kern.group_id(argno);
        auto f = flags ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only;
        return new xrt::bo(c->dev, bytes, f, grp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "aie_bo_alloc: %s\n", e.what());
        return nullptr;
    }
}

void aie_bo_free(void* b) { delete static_cast<xrt::bo*>(b); }

int aie_bo_write(void* b, const void* src, size_t bytes) {
    try {
        auto* bo = static_cast<xrt::bo*>(b);
        bo->write(src, bytes, 0);
        bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return 0;
    } catch (...) { return -1; }
}

int aie_bo_read(void* b, void* dst, size_t bytes) {
    try {
        auto* bo = static_cast<xrt::bo*>(b);
        bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo->read(dst, bytes, 0);
        return 0;
    } catch (...) { return -1; }
}

// (opcode, insts, insts_len_words, data buffers...) -- the convention confirmed
// on hardware from Python.
int aie_run(void* h, unsigned opcode, void* insts, unsigned insts_words,
            void* const* bos, int n_bos) {
    try {
        auto* c = static_cast<AieCtx*>(h);
        auto run = xrt::run(c->kern);
        int a = 0;
        run.set_arg(a++, opcode);
        run.set_arg(a++, *static_cast<xrt::bo*>(insts));
        run.set_arg(a++, insts_words);
        for (int i = 0; i < n_bos; i++)
            run.set_arg(a++, *static_cast<xrt::bo*>(bos[i]));
        run.start();
        auto st = run.wait();
        return st == ERT_CMD_STATE_COMPLETED ? 0 : (int)st;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "aie_run: %s\n", e.what());
        return -1;
    }
}


// --- asynchronous form -----------------------------------------------------
// aie_run above blocks. To overlap NPU work with host/iGPU work the dispatch
// must be split: start returns a live run handle, wait joins it.

void* aie_run_start(void* h, unsigned opcode, void* insts, unsigned insts_words,
                    void* const* bos, int n_bos) {
    try {
        auto* c = static_cast<AieCtx*>(h);
        auto* run = new xrt::run(c->kern);
        int a = 0;
        run->set_arg(a++, opcode);
        run->set_arg(a++, *static_cast<xrt::bo*>(insts));
        run->set_arg(a++, insts_words);
        for (int i = 0; i < n_bos; i++)
            run->set_arg(a++, *static_cast<xrt::bo*>(bos[i]));
        run->start();
        return run;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "aie_run_start: %s\n", e.what());
        return nullptr;
    }
}

int aie_run_wait(void* r) {
    try {
        auto* run = static_cast<xrt::run*>(r);
        auto st = run->wait();
        delete run;
        return st == ERT_CMD_STATE_COMPLETED ? 0 : (int)st;
    } catch (...) { return -1; }
}

} // extern "C"

// Per-dispatch cost was dominated by rebuilding the command: aie_run_start
// news an xrt::run and re-sets every argument each time. For a fixed set of
// buffers only the doorbell needs ringing, so prepare once and re-execute.
extern "C" void* aie_run_prepare(void* h, unsigned opcode, void* insts,
                                 unsigned insts_words, void* const* bos, int n_bos) {
    try {
        auto* c = static_cast<AieCtx*>(h);
        auto* run = new xrt::run(c->kern);
        int a = 0;
        run->set_arg(a++, opcode);
        run->set_arg(a++, *static_cast<xrt::bo*>(insts));
        run->set_arg(a++, insts_words);
        for (int i = 0; i < n_bos; i++)
            run->set_arg(a++, *static_cast<xrt::bo*>(bos[i]));
        return run;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "aie_run_prepare: %s\n", e.what());
        return nullptr;
    }
}

extern "C" int aie_run_exec(void* r) {
    try {
        auto* run = static_cast<xrt::run*>(r);
        run->start();
        auto st = run->wait();
        return st == ERT_CMD_STATE_COMPLETED ? 0 : (int)st;
    } catch (...) { return -1; }
}

extern "C" int aie_run_exec_start(void* r) {
    try { static_cast<xrt::run*>(r)->start(); return 0; } catch (...) { return -1; }
}

extern "C" int aie_run_exec_wait(void* r) {
    try {
        auto st = static_cast<xrt::run*>(r)->wait();
        return st == ERT_CMD_STATE_COMPLETED ? 0 : (int)st;
    } catch (...) { return -1; }
}

extern "C" void aie_run_release(void* r) { delete static_cast<xrt::run*>(r); }

// ── Mapped-view safety ───────────────────────────────────────────────────────
//
// XRT can allocate a LOCAL COPY of an argument buffer when the compute unit's
// bank connectivity does not match the allocation, warning only:
//
//   [XRT] WARNING: ... the argument is allocated in bank 0, the compute unit is
//   connected to bank 65537. Allocating local copy of argument buffer in
//   connected bank.
//
// HONEST LIMITS OF THIS CHECK. It was written after a real bug in which reading
// a kernel's output through a cached mapped pointer returned stale data on every
// call but the first, and switching to aie_bo_read fixed it. The shadow-copy
// story above is the obvious explanation, but it is NOT established: an isolated
// reproduction with the same access pattern and the same buffer count (28 output
// BOs across 4 prepared runs, cached map pointers) shows the mapped view, a
// fresh map, and aie_bo_read all agreeing. So:
//
//   * aie_bo_view_diverged catches a mapping that has visibly diverged. It did
//     NOT catch the bug that motivated it.
//   * The guard that DID catch that bug was numerical: recompute a sample of the
//     kernel's output on the CPU and compare, on a call OTHER than the first.
//     The first call was correct; that is what made it survive review.
//
// Treat these as a cheap necessary condition, not a proof of soundness.

extern "C" void* aie_bo_map(void* b) {
    try { return static_cast<xrt::bo*>(b)->map<void*>(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "aie_bo_map: %s\n", e.what());
        return nullptr;
    }
}

extern "C" int aie_bo_sync_to_dev(void* b) {
    try { static_cast<xrt::bo*>(b)->sync(XCL_BO_SYNC_BO_TO_DEVICE); return 0; }
    catch (...) { return -1; }
}

extern "C" int aie_bo_sync_from_dev(void* b) {
    try { static_cast<xrt::bo*>(b)->sync(XCL_BO_SYNC_BO_FROM_DEVICE); return 0; }
    catch (...) { return -1; }
}

// Does the mapped view disagree with XRT's own read path?
//   1 = diverged (a shadow copy exists; the mapping is NOT a valid view)
//   0 = agrees
//  -1 = could not tell
//
// ONLY MEANINGFUL AFTER A DISPATCH HAS WRITTEN THE BUFFER. A host-only round
// trip agrees whether or not a shadow exists -- that was measured on a buffer
// which was in fact shadowed, so it proves nothing. Run the kernel, then call.
extern "C" int aie_bo_view_diverged(void* b, size_t bytes) {
    try {
        auto* bo = static_cast<xrt::bo*>(b);
        void* m = bo->map<void*>();
        if (!m || bytes == 0) return -1;
        bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<unsigned char> via_read(bytes);
        bo->read(via_read.data(), bytes, 0);
        return std::memcmp(m, via_read.data(), bytes) != 0 ? 1 : 0;
    } catch (...) { return -1; }
}

// Map, but only if the view has been shown to agree. Returns nullptr and says
// why otherwise, so a stale-read bug surfaces as a failure instead of as
// plausible-looking numbers.
extern "C" void* aie_bo_map_checked(void* b, size_t bytes) {
    const int diverged = aie_bo_view_diverged(b, bytes);
    if (diverged == 0) return aie_bo_map(b);
    std::fprintf(stderr,
                 "aie_bo_map_checked: REFUSING to hand out a mapped pointer -- "
                 "%s. Use aie_bo_read/aie_bo_write instead.\n",
                 diverged == 1 ? "the mapping disagrees with XRT's read path, so "
                                 "XRT is keeping a local copy of this buffer"
                               : "could not verify the mapping");
    return nullptr;
}
