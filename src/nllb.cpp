#include "nllb.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cctype>

#ifndef _WIN32
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define NLLB_HAVE_MMAP 1
#else
#  define NLLB_HAVE_MMAP 0
#endif

// --------------------------------------------------------------------------
// metadata helpers
// --------------------------------------------------------------------------

namespace {

bool kv_u32(const gguf_context * g, const char * key, int32_t & dst) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    dst = (int32_t) gguf_get_val_u32(g, id);
    return true;
}

bool kv_f32(const gguf_context * g, const char * key, float & dst) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    dst = gguf_get_val_f32(g, id);
    return true;
}

// Wall clock in milliseconds, for nllb_step_timing. steady_clock rather than
// system_clock: the numbers are differences, and a clock that can be stepped
// backwards by NTP has no business measuring a 40 ms decoder step.
using clk = std::chrono::steady_clock;

double ms_between(const clk::time_point & a, const clk::time_point & b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

ggml_tensor * need(ggml_context * ctx, const std::string & name, std::string & err) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t && err.empty()) err = "missing tensor: " + name;
    return t;
}

double nllb_now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

// Map the whole file read-only. Returns nullptr and leaves err untouched on
// any failure - the caller falls back to reading, which is always correct.
void * nllb_mmap_file(const std::string & path, size_t & out_size) {
#if NLLB_HAVE_MMAP
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;

    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return nullptr; }

    const double t_open = nllb_now_ms();
    void * addr = ::mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    // The mapping outlives the descriptor: mmap holds its own reference to the
    // file, so closing fd here leaks nothing and keeps the model struct simple.
    ::close(fd);
    if (addr == MAP_FAILED) return nullptr;
    const double t_map = nllb_now_ms() - t_open;

    // MADV_WILLNEED IS OFF BY DEFAULT, AND THAT IS THE POINT OF THIS COMMENT.
    //
    // The obvious thing to do here is tell the kernel we will read the whole
    // file, so it can fault ahead instead of one page at a time. On Linux that
    // is an asynchronous readahead hint and costs nothing. On Darwin it is
    // SYNCHRONOUS - XNU pages the range in before returning - so it performs
    // exactly the eager load mmap exists to avoid, and measurably slower than
    // a plain read: 783 ms of "map" against 395 ms for read() of the same
    // 1.235 GB f16 file, with the subsequent fault-in then reporting a
    // fictitious 32 GB/s because the pages were already resident.
    //
    // So it is opt-in, and NLLB_LOAD_TRACE=1 prices it.
    double t_advise = 0.0;
    if (::getenv("NLLB_MADVISE")) {
        const double t0 = nllb_now_ms();
#  ifdef POSIX_MADV_WILLNEED
        ::posix_madvise(addr, (size_t) st.st_size, POSIX_MADV_WILLNEED);
#  elif defined(MADV_WILLNEED)
        ::madvise(addr, (size_t) st.st_size, MADV_WILLNEED);
#  endif
        t_advise = nllb_now_ms() - t0;
    }

    if (::getenv("NLLB_LOAD_TRACE")) {
        std::fprintf(stderr,
            "[load-trace] open+fstat+mmap %7.2f ms   madvise %7.2f ms%s\n",
            t_map, t_advise, ::getenv("NLLB_MADVISE") ? "" : " (off)");
    }

    out_size = (size_t) st.st_size;
    return addr;
#else
    (void) path; (void) out_size;
    return nullptr;
#endif
}

void nllb_munmap_file(void * addr, size_t size) {
#if NLLB_HAVE_MMAP
    if (addr) ::munmap(addr, size);
#else
    (void) addr; (void) size;
#endif
}


// --------------------------------------------------------------------------
// execution helpers (step 13)
//
// Every graph in this file follows the same five moves, and they are the whole
// of the port:
//
//   1. build the graph in a context created with no_alloc = true - it holds
//      tensor STRUCTS and the graph, never data;
//   2. mark every tensor the host writes with ggml_set_input and every tensor
//      the host reads back with ggml_set_output;
//   3. ggml_backend_sched_alloc_graph: split the graph across backends and let
//      gallocr place every intermediate in the compute buffer;
//   4. ggml_backend_tensor_set the inputs, compute;
//   5. ggml_backend_tensor_get the outputs.
//
// Step 2 is not bookkeeping. gallocr reuses a tensor's memory as soon as its
// last consumer has run, so a hidden state that is not marked as an output is
// overwritten by something later in the graph and reads back as garbage - on
// the CPU backend as much as on a GPU. The bump arena kept every intermediate
// alive, which is why the old code could read any of them afterwards.
// --------------------------------------------------------------------------

// Nodes a graph may have, and the scheduler's hash-set size. The 12+12 600M
// decoder with every hidden state expanded is well under 1000; the headroom is
// for the parked batching and beam-search work.
constexpr int NLLB_GRAPH_NODES = 8192;

// The graph context holds tensor structs and the graph - ~3.4 MB of metadata
// at 8192 nodes, most of it never touched. Measured on the first draft: a
// fresh malloc of it per cached step cost ~100 minor faults and ~0.2 ms, the
// same page-fault bill trap 17 found for the old data arena. So the caller
// passes a buffer that lives as long as the model (one-shot graphs) or the
// cache (steps), and it is reused.
ggml_context * nllb_graph_ctx(std::vector<unsigned char> & meta) {
    const size_t need = ggml_tensor_overhead() * NLLB_GRAPH_NODES
                      + ggml_graph_overhead_custom(NLLB_GRAPH_NODES, false);
    if (meta.size() < need) meta.resize(need);
    ggml_init_params ip = {};
    ip.mem_size   = meta.size();
    ip.mem_buffer = meta.data();
    ip.no_alloc   = true;
    return ggml_init(ip);
}

ggml_backend_sched * nllb_new_sched(const nllb_model & m) {
    // The CPU must be LAST: the scheduler assigns graph inputs to the last
    // backend and uses it as the fallback for unsupported ops.
    ggml_backend_t b[2];
    int n = 0;
    if (!m.on_cpu) b[n++] = m.backend;
    b[n++] = m.backend_cpu;
    // parallel = false: one copy of the inputs, no pipelining across splits.
    // op_offload = false: an op whose weights live on the CPU stays on the CPU.
    return ggml_backend_sched_new(b, nullptr, n, NLLB_GRAPH_NODES,
                                  /*parallel=*/false, /*op_offload=*/false);
}

size_t nllb_sched_bytes(ggml_backend_sched * s) {
    size_t total = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(s); ++i) {
        total += ggml_backend_sched_get_buffer_size(s, ggml_backend_sched_get_backend(s, i));
    }
    return total;
}

bool nllb_alloc_graph(ggml_backend_sched * s, ggml_cgraph * gf, const char * what,
                      std::string & err) {
    ggml_backend_sched_reset(s);
    if (!ggml_backend_sched_alloc_graph(s, gf)) {
        err = std::string("ggml_backend_sched_alloc_graph failed (") + what + ")";
        return false;
    }
    // NLLB_ARENA_TRACE=1: what every graph really needs, per backend, and how
    // many splits it took. More than one split on a GPU backend means some
    // op fell back to the CPU, with a copy in and a copy out around it - that
    // is correct and slow, and this line is where you find out it happened.
    if (::getenv("NLLB_ARENA_TRACE")) {
        std::fprintf(stderr, "[arena] %-16s nodes %5d  splits %2d  compute",
                     what, ggml_graph_n_nodes(gf), ggml_backend_sched_get_n_splits(s));
        for (int i = 0; i < ggml_backend_sched_get_n_backends(s); ++i) {
            ggml_backend_t b = ggml_backend_sched_get_backend(s, i);
            std::fprintf(stderr, "  %s %.2f MiB", ggml_backend_name(b),
                         ggml_backend_sched_get_buffer_size(s, b) / (1024.0 * 1024.0));
        }
        std::fprintf(stderr, "\n");
    }
    return true;
}

bool nllb_compute_graph(const nllb_model & m, ggml_backend_sched * s, ggml_cgraph * gf,
                        int n_threads, const char * what, std::string & err) {
    ggml_backend_cpu_set_n_threads(m.backend_cpu, n_threads > 0 ? n_threads : 1);
    const ggml_status st = ggml_backend_sched_graph_compute(s, gf);
    if (st != GGML_STATUS_SUCCESS) {
        err = std::string("graph compute failed (") + what + "), status " +
              std::to_string((int) st);
        return false;
    }
    return true;
}

void nllb_get_f32(const ggml_tensor * t, std::vector<float> & dst, size_t n) {
    dst.resize(n);
    ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
}

// The causal mask as DATA, [n_kv, Tq]: row j is query position n_past + j and
// may see keys 0..n_past + j. 0 where visible, -inf where not.
//
// Why not ggml_diag_mask_inf, which the CPU path used since step 3: Metal
// does not implement it, so under ggml_backend_sched every decoder layer
// would bounce kq to the CPU and back - two copies and a sync per layer per
// step. soft_max_ext with a mask is what every backend implements.
//
// It is BIT-IDENTICAL to diag_mask_inf on the CPU, and that was checked in
// the kernel before it was measured: ggml_compute_forward_soft_max_f32 does
// wp = kq * scale, then wp += slope * mask with slope = 1 when max_bias = 0.
// With scale = 1 and a mask of exactly 0.0, both are exact no-ops on every
// finite float; -inf is -inf either way. nllb-dump-states --diff against the
// pre-port build is the measurement.
//
// Trap 13 is preserved exactly: an offset too LARGE still masks nothing at
// Tq == 1. The chunked passes in tests/test_cache.cpp remain the only test of
// this function's n_past arithmetic.
void nllb_causal_mask(std::vector<float> & m, int n_kv, int Tq, int n_past) {
    m.assign((size_t) n_kv * Tq, 0.0f);
    for (int j = 0; j < Tq; ++j) {
        for (int i = n_past + j + 1; i < n_kv; ++i) {
            m[(size_t) j * n_kv + i] = -INFINITY;
        }
    }
}

std::string nllb_lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// Resolve the user's backend request to a device. nullptr means the CPU.
//
// THE NAMES ARE NOT THE MARKETING NAMES, and assuming they were is what made
// `--backend metal` fail on the first M1 run while `--backend auto` worked:
// ggml's Metal registry calls itself "MTL" and its devices "MTL0", "MTL1".
// So a word like "metal" is matched as a FAMILY - against the registry name
// and the device name, through the alias table below - and an exact device
// name is matched first, because that is the unambiguous spelling
// --list-backends prints.
bool nllb_pick_device(const std::string & want_in, ggml_backend_dev_t & dev,
                      std::string & err) {
    const std::string want = nllb_lower(want_in);
    dev = nullptr;
    if (want == "cpu") return true;

    // Pass 1: an exact device name, as --list-backends prints it ("MTL0").
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (nllb_lower(ggml_backend_dev_name(d)) != want) continue;
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU) { dev = nullptr; return true; }
        if (t == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            // BLAS and friends are accelerators that run alongside the CPU
            // backend, not instead of it; nothing here schedules them.
            err = "device '" + want_in + "' is an accelerator, not a backend "
                  "the graphs can run on";
            return false;
        }
        dev = d;
        return true;
    }

    // Pass 2: the first GPU, either any of them ("auto"/"gpu") or one whose
    // registry or device name matches the family asked for.
    const bool any_gpu = want == "auto" || want == "gpu";
    // Family -> the spellings ggml actually uses. "metal" is the one that bit;
    // the others are here so that naming a backend ggml grows later works
    // without another round trip.
    static const struct { const char * family; const char * spellings[3]; } aliases[] = {
        { "metal",  { "mtl",    "metal",  nullptr } },
        { "vulkan", { "vk",     "vulkan", nullptr } },
        { "cuda",   { "cuda",   nullptr,  nullptr } },
        { "rocm",   { "rocm",   "hip",    nullptr } },
        { "sycl",   { "sycl",   nullptr,  nullptr } },
        { "opencl", { "opencl", "ocl",    nullptr } },
    };
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (any_gpu) { dev = d; return true; }

        const std::string name = nllb_lower(ggml_backend_dev_name(d));
        const std::string reg  = nllb_lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)));
        bool match = name.find(want) != std::string::npos ||
                     reg.find(want)  != std::string::npos;
        for (const auto & a : aliases) {
            if (match || want != a.family) continue;
            for (const char * sp : a.spellings) {
                if (!sp) break;
                if (name.find(sp) != std::string::npos || reg.find(sp) != std::string::npos) {
                    match = true;
                    break;
                }
            }
        }
        if (match) { dev = d; return true; }
    }

    if (want == "auto") return true;   // no GPU in this build: the CPU fallback

    // Name every device rather than only saying no: the failure this replaced
    // reported "metal is not available" on a machine whose device list began
    // with MTL0, and the list was the thing that gave the bug away.
    std::string have;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        have += std::string(have.empty() ? "" : ", ") + ggml_backend_dev_name(d) +
                " (" + ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)) + ")";
    }
    err = "backend '" + want_in + "' is not available in this build (devices: " + have +
          "). Name one of those, or 'gpu', or 'cpu'";
    return false;
}

} // namespace

nllb_model::~nllb_model() {
    // Order matters twice over. The scheduler and the weight buffer reference
    // the backends, so they go before them. And when mapped, the weight
    // buffer WRAPS the mapping without owning it (buffer_from_host_ptr), and
    // ctx_w's tensors point into it - so buffer and context go first and the
    // mapping last. Freeing the map first leaves dangling ->data behind.
    if (sched)       { ggml_backend_sched_free(sched);        sched = nullptr; }
    if (weights_buf) { ggml_backend_buffer_free(weights_buf); weights_buf = nullptr; }
    if (gguf)        { gguf_free(gguf);  gguf  = nullptr; }
    if (ctx_w)       { ggml_free(ctx_w); ctx_w = nullptr; }
    if (backend && backend != backend_cpu) ggml_backend_free(backend);
    if (backend_cpu) ggml_backend_free(backend_cpu);
    backend = backend_cpu = nullptr;
    if (mapping) { nllb_munmap_file(mapping, mapping_size);
                   mapping = nullptr; mapping_size = 0; mapped = false; }
}

// --------------------------------------------------------------------------
// load
// --------------------------------------------------------------------------

std::vector<std::string> nllb_list_backends() {
    std::vector<std::string> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char * type = "?";
        switch (ggml_backend_dev_type(d)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   type = "cpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type = "gpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type = "igpu";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accel - not schedulable here"; break;
            default: break;
        }
        out.push_back(std::string(ggml_backend_dev_name(d)) + " - " +
                      ggml_backend_dev_description(d) + " (" + type + ")");
    }
    return out;
}

namespace {

// Read every weight from the file into tensors that already live in a backend
// buffer. The destination may be device memory, so this goes through
// ggml_backend_tensor_set with a bounded host staging buffer rather than
// through ->data - the copy gguf_init_from_file(no_alloc = false) used to do
// straight into host memory is exactly the pattern a device backend forbids.
bool nllb_read_weights(const std::string & path, gguf_context * g, ggml_context * ctx,
                       std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<char> stage((size_t) 64 * 1024 * 1024);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) { std::fclose(f); err = std::string("tensor vanished: ") + name; return false; }
        const size_t nb = ggml_nbytes(t);
#ifdef _WIN32
        if (_fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0) {
#else
        if (fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0) {
#endif
            std::fclose(f); err = std::string("seek failed for ") + name; return false;
        }
        for (size_t done = 0; done < nb; ) {
            const size_t n = std::min(stage.size(), nb - done);
            if (std::fread(stage.data(), 1, n, f) != n) {
                std::fclose(f);
                err = std::string("short read in tensor ") + name + " - truncated file?";
                return false;
            }
            ggml_backend_tensor_set(t, stage.data(), done, n);
            done += n;
        }
    }
    std::fclose(f);
    return true;
}

} // namespace

bool nllb_model_load(const std::string & path, nllb_model & model, std::string & err,
                     bool use_mmap, const std::string & backend) {
    // ---- backends first: they decide where the weights may live ----------
    // (Skipped on the read-fallback recursion below, which keeps them.)
    if (!model.backend_cpu) {
        std::string want = backend;
        if (want.empty()) {
            const char * env = ::getenv("NLLB_BACKEND");
            want = env && *env ? env : "cpu";
        }
        ggml_backend_dev_t dev = nullptr;
        if (!nllb_pick_device(want, dev, err)) return false;

        model.backend_cpu = ggml_backend_cpu_init();
        if (!model.backend_cpu) { err = "ggml_backend_cpu_init failed"; return false; }
        model.backend = dev ? ggml_backend_dev_init(dev, nullptr) : model.backend_cpu;
        if (!model.backend) {
            err = std::string("failed to initialise backend ") + ggml_backend_dev_name(dev);
            return false;
        }
        model.on_cpu       = model.backend == model.backend_cpu;
        model.backend_name = ggml_backend_name(model.backend);
    }

    // Try the mapping FIRST. gguf is ALWAYS parsed with no_alloc = true now:
    // it creates the ggml_tensor structs with correct name, type and ne and
    // reads no bytes, and the bytes are then attached in one of two ways
    // below - wrapped in place (mapped) or copied into a backend buffer (read).
    size_t map_size = 0;
    void * map_addr = use_mmap ? nllb_mmap_file(path, map_size) : nullptr;

    ggml_context * ctx_data = nullptr;
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx      = &ctx_data;

    const bool trace = ::getenv("NLLB_LOAD_TRACE") != nullptr;
    const double t_parse0 = nllb_now_ms();
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        if (map_addr) nllb_munmap_file(map_addr, map_size);
        err = "gguf_init_from_file failed: " + path;
        return false;
    }
    if (trace) {
        std::fprintf(stderr, "[load-trace] gguf parse (metadata only)  %7.2f ms\n",
                     nllb_now_ms() - t_parse0);
    }

    model.gguf  = g;
    model.ctx_w = ctx_data;

    ggml_backend_dev_t pdev = ggml_backend_get_device(model.backend);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(pdev, &props);

    if (map_addr && !props.caps.buffer_from_host_ptr) {
        // The device cannot wrap host memory, so mapping buys nothing: the
        // bytes have to be copied into device memory either way.
        if (trace) std::fprintf(stderr, "[load-trace] %s cannot wrap host memory - reading\n",
                                model.backend_name.c_str());
        nllb_munmap_file(map_addr, map_size);
        map_addr = nullptr;
    }

    if (map_addr) {
        // Wrap the whole mapping in ONE backend buffer and place every tensor
        // at its offset inside it. On the CPU this is the step-10 mmap path
        // unchanged in effect; on Metal it is zero-copy too, because Apple
        // silicon's GPU reads the same unified memory - the mapped file becomes
        // an MTLBuffer and nothing is uploaded.
        const double   t_fix0   = nllb_now_ms();
        const size_t   data_off = gguf_get_data_offset(g);
        const int64_t  n_tensor = gguf_get_n_tensors(g);
        const size_t   align    = gguf_get_alignment(g);
        char * const   base     = (char *) map_addr;

        // ggml's kernels want their rows aligned. mmap hands back a page-
        // aligned base and GGUF pads data_off and every tensor offset to
        // `align`, so the sum is aligned iff align is - refuse rather than
        // hand a misaligned pointer to a SIMD kernel and crash somewhere else.
        bool ok = align >= 16 && (align & (align - 1)) == 0 && data_off % align == 0;

        size_t max_tensor = 0;
        for (int64_t i = 0; ok && i < n_tensor; ++i) {
            ggml_tensor * t = ggml_get_tensor(ctx_data, gguf_get_tensor_name(g, i));
            if (!t) { ok = false; break; }
            const size_t off = gguf_get_tensor_offset(g, i);
            if (data_off + off + ggml_nbytes(t) > map_size) { ok = false; break; }
            max_tensor = std::max(max_tensor, ggml_nbytes(t));
        }

        ggml_backend_buffer_t buf = nullptr;
        if (ok) {
            buf = ggml_backend_dev_buffer_from_host_ptr(pdev, map_addr, map_size, max_tensor);
            ok = buf != nullptr;
        }
        for (int64_t i = 0; ok && i < n_tensor; ++i) {
            ggml_tensor * t = ggml_get_tensor(ctx_data, gguf_get_tensor_name(g, i));
            ok = ggml_backend_tensor_alloc(buf, t, base + data_off + gguf_get_tensor_offset(g, i))
                 == GGML_STATUS_SUCCESS;
        }

        if (!ok) {
            // Nothing is half-done: drop the whole attempt and redo the load
            // by reading. A file we cannot map is not a file we cannot use.
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(g);
            ggml_free(ctx_data);
            model.gguf = nullptr; model.ctx_w = nullptr;
            nllb_munmap_file(map_addr, map_size);
            return nllb_model_load(path, model, err, /*use_mmap =*/ false, backend);
        }

        model.weights_buf      = buf;
        model.weights_host_ptr = true;
        model.mapping          = map_addr;
        model.mapping_size     = map_size;
        model.mapped           = true;

        if (trace) {
            std::fprintf(stderr,
                "[load-trace] wrap mapping in a %s buffer, %lld tensors %7.2f ms\n",
                ggml_backend_buffer_name(buf), (long long) n_tensor, nllb_now_ms() - t_fix0);
        }
    } else {
        const double t_read0 = nllb_now_ms();
        model.weights_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                ctx_data, ggml_backend_get_default_buffer_type(model.backend));
        if (!model.weights_buf) {
            err = "could not allocate the weights on " + model.backend_name;
            return false;
        }
        if (!nllb_read_weights(path, g, ctx_data, err)) return false;
        if (trace) {
            std::fprintf(stderr, "[load-trace] read weights into a %s buffer %7.2f ms\n",
                         ggml_backend_buffer_name(model.weights_buf), nllb_now_ms() - t_read0);
        }
    }
    // The scheduler puts an op on the backend that holds its weights, and it
    // only recognises a buffer as holding weights by this flag.
    ggml_backend_buffer_set_usage(model.weights_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    {
        const int64_t id = gguf_find_key(g, "general.architecture");
        const char * arch = id >= 0 ? gguf_get_val_str(g, id) : "";
        if (std::strcmp(arch, "nllb") != 0) {
            err = std::string("expected general.architecture=nllb, got '") + arch + "'";
            return false;
        }
    }

    nllb_hparams & hp = model.hp;
    bool ok = true;
    ok &= kv_u32(g, "nllb.embedding_length",              hp.d_model);
    ok &= kv_u32(g, "nllb.encoder.attention.head_count",  hp.n_head);
    ok &= kv_u32(g, "nllb.encoder.block_count",           hp.n_layer);
    ok &= kv_u32(g, "nllb.encoder.feed_forward_length",   hp.n_ff);
    if (!ok) { err = "missing required nllb.* metadata"; return false; }

    kv_u32(g, "nllb.vocab_size",       hp.n_vocab);
    kv_u32(g, "nllb.context_length",   hp.n_ctx);
    kv_u32(g, "nllb.position_offset",  hp.pos_offset);
    kv_f32(g, "nllb.attention.layer_norm_epsilon", hp.eps);
    if (!kv_f32(g, "nllb.embedding_scale", hp.embd_scale)) {
        hp.embd_scale = std::sqrt((float) hp.d_model);
    }
    kv_u32(g, "tokenizer.ggml.padding_token_id", hp.pad_id);
    kv_u32(g, "tokenizer.ggml.bos_token_id",     hp.bos_id);
    kv_u32(g, "tokenizer.ggml.eos_token_id",     hp.eos_id);
    // NLLB's decoder_start_token_id is eos, not bos. Read it if the converter
    // wrote it; fall back to eos rather than to the more obvious-looking bos.
    hp.dec_start_id = hp.eos_id;
    kv_u32(g, "nllb.decoder_start_token_id", hp.dec_start_id);

    // Decoder metadata is absent in --encoder-only files; that is not an error.
    model.has_decoder = kv_u32(g, "nllb.decoder.block_count", hp.n_dec_layer);
    if (model.has_decoder) {
        bool dok = true;
        dok &= kv_u32(g, "nllb.decoder.attention.head_count", hp.n_dec_head);
        dok &= kv_u32(g, "nllb.decoder.feed_forward_length",  hp.n_dec_ff);
        if (!dok) { err = "decoder.block_count present but decoder.* incomplete";
                    return false; }
    }

    if (hp.d_model % hp.n_head != 0) {
        err = "d_model not divisible by n_head";
        return false;
    }
    if (model.has_decoder && hp.n_dec_head > 0 && hp.d_model % hp.n_dec_head != 0) {
        err = "d_model not divisible by decoder n_head";
        return false;
    }

    model.tok_embd   = need(ctx_data, "token_embd.weight",     err);
    model.pos_embd   = need(ctx_data, "position_embd.weight",  err);
    model.out_norm_w = need(ctx_data, "enc.output_norm.weight", err);
    model.out_norm_b = need(ctx_data, "enc.output_norm.bias",   err);

    model.enc.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        nllb_enc_layer & l = model.enc[i];
        l.attn_norm_w = need(ctx_data, p + "attn_norm.weight", err);
        l.attn_norm_b = need(ctx_data, p + "attn_norm.bias",   err);
        l.wq = need(ctx_data, p + "attn_q.weight",   err);
        l.bq = need(ctx_data, p + "attn_q.bias",     err);
        l.wk = need(ctx_data, p + "attn_k.weight",   err);
        l.bk = need(ctx_data, p + "attn_k.bias",     err);
        l.wv = need(ctx_data, p + "attn_v.weight",   err);
        l.bv = need(ctx_data, p + "attn_v.bias",     err);
        l.wo = need(ctx_data, p + "attn_out.weight", err);
        l.bo = need(ctx_data, p + "attn_out.bias",   err);
        l.ffn_norm_w = need(ctx_data, p + "ffn_norm.weight", err);
        l.ffn_norm_b = need(ctx_data, p + "ffn_norm.bias",   err);
        l.w_up   = need(ctx_data, p + "ffn_up.weight",   err);
        l.b_up   = need(ctx_data, p + "ffn_up.bias",     err);
        l.w_down = need(ctx_data, p + "ffn_down.weight", err);
        l.b_down = need(ctx_data, p + "ffn_down.bias",   err);
    }
    if (model.has_decoder) {
        model.dec_out_norm_w = need(ctx_data, "dec.output_norm.weight", err);
        model.dec_out_norm_b = need(ctx_data, "dec.output_norm.bias",   err);
        // Tied lm_head is the NLLB default, so a missing output.weight is
        // expected rather than a failure.
        model.output_w = ggml_get_tensor(ctx_data, "output.weight");

        model.dec.resize(hp.n_dec_layer);
        for (int i = 0; i < hp.n_dec_layer; ++i) {
            const std::string p = "dec.blk." + std::to_string(i) + ".";
            nllb_dec_layer & l = model.dec[i];
            l.attn_norm_w = need(ctx_data, p + "attn_norm.weight", err);
            l.attn_norm_b = need(ctx_data, p + "attn_norm.bias",   err);
            l.wq = need(ctx_data, p + "attn_q.weight",   err);
            l.bq = need(ctx_data, p + "attn_q.bias",     err);
            l.wk = need(ctx_data, p + "attn_k.weight",   err);
            l.bk = need(ctx_data, p + "attn_k.bias",     err);
            l.wv = need(ctx_data, p + "attn_v.weight",   err);
            l.bv = need(ctx_data, p + "attn_v.bias",     err);
            l.wo = need(ctx_data, p + "attn_out.weight", err);
            l.bo = need(ctx_data, p + "attn_out.bias",   err);

            l.cross_norm_w = need(ctx_data, p + "cross_attn_norm.weight", err);
            l.cross_norm_b = need(ctx_data, p + "cross_attn_norm.bias",   err);
            l.cwq = need(ctx_data, p + "cross_attn_q.weight",   err);
            l.cbq = need(ctx_data, p + "cross_attn_q.bias",     err);
            l.cwk = need(ctx_data, p + "cross_attn_k.weight",   err);
            l.cbk = need(ctx_data, p + "cross_attn_k.bias",     err);
            l.cwv = need(ctx_data, p + "cross_attn_v.weight",   err);
            l.cbv = need(ctx_data, p + "cross_attn_v.bias",     err);
            l.cwo = need(ctx_data, p + "cross_attn_out.weight", err);
            l.cbo = need(ctx_data, p + "cross_attn_out.bias",   err);

            l.ffn_norm_w = need(ctx_data, p + "ffn_norm.weight", err);
            l.ffn_norm_b = need(ctx_data, p + "ffn_norm.bias",   err);
            l.w_up   = need(ctx_data, p + "ffn_up.weight",   err);
            l.b_up   = need(ctx_data, p + "ffn_up.bias",     err);
            l.w_down = need(ctx_data, p + "ffn_down.weight", err);
            l.b_down = need(ctx_data, p + "ffn_down.bias",   err);
        }
    }

    if (!err.empty()) return false;

    // The position table must cover n_ctx + pos_offset rows, because token j
    // reads row j + pos_offset. If the converter wrote a short table this is
    // where you find out, rather than by reading garbage at long sequence
    // lengths.
    if (model.pos_embd->ne[1] < hp.n_ctx + hp.pos_offset) {
        err = "position_embd has " + std::to_string(model.pos_embd->ne[1]) +
              " rows, need " + std::to_string(hp.n_ctx + hp.pos_offset);
        return false;
    }

    model.sched = nllb_new_sched(model);
    if (!model.sched) { err = "ggml_backend_sched_new failed"; return false; }
    return true;
}

// --------------------------------------------------------------------------
// encoder graph
// --------------------------------------------------------------------------

bool nllb_encode(const nllb_model & model,
                 const std::vector<int32_t> & ids,
                 int n_threads,
                 bool want_hidden_states,
                 nllb_encode_result & out,
                 std::string & err) {
    const nllb_hparams & hp = model.hp;
    const int T  = (int) ids.size();
    const int D  = hp.d_model;
    const int H  = hp.n_head;
    const int HD = hp.head_dim();

    if (T <= 0)                      { err = "empty input";        return false; }
    if (T > hp.n_ctx && hp.n_ctx > 0){ err = "sequence too long";  return false; }

    // No arena formula any more: the context holds structs and the graph, and
    // gallocr sizes the compute buffer from the graph itself. Only the hidden
    // states the caller asked for are kept alive (ggml_set_output); everything
    // else is reused as soon as its last consumer has run.
    ggml_context * ctx = nllb_graph_ctx(model.graph_meta);
    if (!ctx) { err = "ggml_init failed"; return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, NLLB_GRAPH_NODES, false);

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(inp_ids, "inp_ids");
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_ids);
    ggml_set_input(inp_pos);

    // HF: position_ids = cumsum(ne_pad) * ne_pad + padding_idx, and the table
    // was built with offset 2 == padding_idx + 1. For an unpadded sequence
    // this is simply j + 2 - but write it the general way so padded batches
    // stay correct.
    std::vector<int32_t> pos(T);
    {
        int32_t run = 0;
        for (int j = 0; j < T; ++j) {
            const bool is_pad = (ids[j] == hp.pad_id);
            if (!is_pad) ++run;
            pos[j] = is_pad ? hp.pad_id : run + hp.pad_id;
        }
    }

    // embeddings: token * sqrt(d_model), then + position (position NOT scaled)
    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_ids);
    cur = ggml_scale(ctx, cur, hp.embd_scale);
    cur = ggml_add(ctx, cur, ggml_get_rows(ctx, model.pos_embd, inp_pos));

    std::vector<ggml_tensor *> hs;
    hs.reserve(hp.n_layer + 1);

    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < hp.n_layer; ++il) {
        const nllb_enc_layer & l = model.enc[il];
        hs.push_back(cur);   // HF hidden_states[il] == input to block il

        // ---- self-attention (pre-layernorm) ----
        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.attn_norm_w), l.attn_norm_b);

        // HF scales Q, not the logits. Same result, but keeping the placement
        // means intermediate dumps line up when bisecting.
        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, h), l.bq);
        Q = ggml_scale(ctx, Q, attn_scale);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, h), l.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, h), l.bv);

        Q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, Q, HD, H, T), 0, 2, 1, 3));  // (HD, T, H)
        K = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, K, HD, H, T), 0, 2, 1, 3));  // (HD, T, H)
        V = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, V, HD, H, T), 1, 2, 0, 3));  // (T, HD, H)

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);               // (T, T, H)
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);             // (HD, T, H)

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // (HD, H, T)
        cur = ggml_reshape_2d(ctx, kqv, D, T);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, cur), l.bo);
        cur = ggml_add(ctx, cur, residual);

        // ---- feed forward ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.ffn_norm_w), l.ffn_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_up, h), l.b_up);
        h = ggml_relu(ctx, h);                                    // NLLB is relu
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_down, h), l.b_down);
        cur = ggml_add(ctx, h, residual);
    }

    cur = ggml_norm(ctx, cur, hp.eps);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, model.out_norm_w), model.out_norm_b);
    ggml_set_name(cur, "enc_out");
    ggml_set_output(cur);
    hs.push_back(cur);   // HF hidden_states[n_layer] == post-final-norm output

    ggml_build_forward_expand(gf, cur);
    if (want_hidden_states) {
        for (ggml_tensor * t : hs) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
    }

    ggml_backend_sched * sched = model.sched;
    bool ok = nllb_alloc_graph(sched, gf, "encoder", err);
    if (ok) {
        ggml_backend_tensor_set(inp_ids, ids.data(), 0, (size_t) T * sizeof(int32_t));
        ggml_backend_tensor_set(inp_pos, pos.data(), 0, (size_t) T * sizeof(int32_t));
        ok = nllb_compute_graph(model, sched, gf, n_threads, "encoder", err);
    }
    if (ok) {
        out.n_tokens = T;
        out.d_model  = D;
        nllb_get_f32(cur, out.output, (size_t) T * D);

        out.hidden_states.clear();
        if (want_hidden_states) {
            out.hidden_states.resize(hs.size());
            for (size_t i = 0; i < hs.size(); ++i) {
                nllb_get_f32(hs[i], out.hidden_states[i], (size_t) T * D);
            }
        }
    }

    ggml_backend_sched_reset(sched);
    ggml_free(ctx);
    return ok;
}

// --------------------------------------------------------------------------
// decoder graph
//
// Teacher-forced: the whole target sequence in one pass, causally masked, no
// KV cache. Cross-attention K and V are computed once per block from the
// frozen encoder output - they do not depend on the decoder step, which is
// exactly why they are the first thing worth caching once this is green.
// --------------------------------------------------------------------------

bool nllb_decode(const nllb_model & model,
                 const nllb_encode_result & enc,
                 const std::vector<int32_t> & dec_ids,
                 const std::vector<int32_t> & src_ids,
                 int n_threads,
                 bool want_hidden_states,
                 int logits_mode,
                 nllb_decode_result & out,
                 std::string & err,
                 int pos_mode) {
    const nllb_hparams & hp = model.hp;
    const int Td = (int) dec_ids.size();
    const int Ts = enc.n_tokens;
    const int D  = hp.d_model;
    const int H  = hp.n_dec_head;
    const int HD = hp.dec_head_dim();
    const int L  = hp.n_dec_layer;

    if (!model.has_decoder)          { err = "GGUF has no decoder (converted with --encoder-only?)"; return false; }
    if (Td <= 0)                     { err = "empty decoder input";  return false; }
    if (Ts <= 0)                     { err = "empty encoder output"; return false; }
    if (enc.d_model != D)            { err = "encoder output d_model mismatch"; return false; }
    if ((int) enc.output.size() != Ts * D) { err = "encoder output size mismatch"; return false; }
    if (Td > hp.n_ctx && hp.n_ctx > 0)     { err = "target sequence too long"; return false; }

    // Cross-attention here is unmasked, exactly as the encoder's self-attention
    // is. That is correct for a single unpadded source and wrong for a padded
    // batch, so refuse the padded case loudly instead of returning plausible
    // numbers. Lifting this means a mask tensor into ggml_soft_max_ext on the
    // cross-attention logits, and goldens that actually exercise padding.
    for (int32_t id : src_ids) {
        if (id == hp.pad_id) {
            err = "padded source sequences are not supported yet "
                  "(cross-attention is unmasked); see nllb_decode";
            return false;
        }
    }

    const bool want_logits = logits_mode != NLLB_LOGITS_NONE;
    const int  logit_rows  = logits_mode == NLLB_LOGITS_LAST ? 1 : Td;
    // (The logits are sized from the lm_head tensor below, not from
    // nllb.vocab_size - that key is optional metadata and defaults to 0.)
    ggml_context * ctx = nllb_graph_ctx(model.graph_meta);
    if (!ctx) { err = "ggml_init failed (decoder)"; return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, NLLB_GRAPH_NODES, false);

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Td);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Td);
    ggml_set_name(inp_ids, "dec_ids");
    ggml_set_name(inp_pos, "dec_pos");
    ggml_set_input(inp_ids);
    ggml_set_input(inp_pos);

    // Position ids. The two rules below agree on any sequence without pad_id in
    // it, which is every golden case here, and disagree as soon as one appears.
    //
    //   NLLB_POS_CUMSUM   cumsum(mask) * mask + pad_id
    //                     what HF computes in a forward pass with use_cache=
    //                     False, i.e. what the teacher-forced goldens contain.
    //                     A pad shifts every later position down by one.
    //
    //   NLLB_POS_ABSOLUTE (j + 1) + pad_id for a non-pad token, pad_id for a
    //                     pad. What HF computes inside generate(): with a cache
    //                     it feeds one token with past_key_values_length = j,
    //                     and its formula is (cumsum(mask) + n_past) * mask +
    //                     pad_id where the cumsum restarts every step - so
    //                     n_past is inside the mask and earlier pads do not
    //                     shift anything.
    //
    // This is HF being inconsistent with itself, not a choice made here. The
    // caller picks the rule that matches the reference it is being diffed
    // against; see nllb_pos_mode in nllb.h.
    std::vector<int32_t> pos(Td);
    {
        int32_t * p = pos.data();
        int32_t run = 0;
        for (int j = 0; j < Td; ++j) {
            const bool is_pad = (dec_ids[j] == hp.pad_id);
            if (!is_pad) ++run;
            const int32_t idx = (pos_mode == NLLB_POS_ABSOLUTE) ? (j + 1) : run;
            p[j] = is_pad ? hp.pad_id : idx + hp.pad_id;
        }
    }

    // Encoder output as a graph input: ne = [D, Ts], matching the row-major
    // [Ts, D] buffer nllb_encode returns.
    ggml_tensor * enc_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Ts);
    ggml_set_name(enc_t, "enc_out");
    ggml_set_input(enc_t);

    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_ids);
    cur = ggml_scale(ctx, cur, hp.embd_scale);
    cur = ggml_add(ctx, cur, ggml_get_rows(ctx, model.pos_embd, inp_pos));

    std::vector<ggml_tensor *> hs;
    hs.reserve(L + 1);

    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < L; ++il) {
        const nllb_dec_layer & l = model.dec[il];
        hs.push_back(cur);   // HF decoder_hidden_states[il] == input to block il

        // ---- causal self-attention ----
        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.attn_norm_w), l.attn_norm_b);

        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, h), l.bq);
        Q = ggml_scale(ctx, Q, attn_scale);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, h), l.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, h), l.bv);

        Q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, Q, HD, H, Td), 0, 2, 1, 3));  // (HD, Td, H)
        K = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, K, HD, H, Td), 0, 2, 1, 3));  // (HD, Td, H)
        V = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, V, HD, H, Td), 1, 2, 0, 3));  // (Td, HD, H)

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                // (Td_k, Td_q, H)
        // The one structural difference from the encoder. n_past is 0 because
        // this is the un-cached path.
        //
        // STILL ggml_diag_mask_inf, deliberately, although the cached step
        // moved to a mask tensor (nllb_causal_mask) because Metal does not
        // implement this op. This path is the REFERENCE the cache gate diffs
        // the cached step against, and the first draft of step 13 routed both
        // through the one helper - which made tests/test_cache.cpp blind to
        // any bug in it: a mask offset one too large broke both sides
        // identically and the gate passed (tests/corrupt_backend.py,
        // `mask-too-large`). Trap 44 again: a control that shares the defect
        // is not a control. Keeping ggml's own op here keeps two independent
        // implementations of the mask. The price is that on Metal this op runs
        // on the CPU fallback - in the reference path only, and it is the one
        // place the fallback is exercised on purpose.
        kq = ggml_diag_mask_inf(ctx, kq, 0);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);              // (HD, Td, H)

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // (HD, H, Td)
        cur = ggml_reshape_2d(ctx, kqv, D, Td);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, cur), l.bo);
        cur = ggml_add(ctx, cur, residual);

        // ---- cross-attention against the frozen encoder output ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.cross_norm_w), l.cross_norm_b);

        // Q from the decoder (Td rows), K and V from the encoder (Ts rows).
        // Mixing these up produces a graph that builds, runs, and is wrong.
        ggml_tensor * cQ = ggml_add(ctx, ggml_mul_mat(ctx, l.cwq, h), l.cbq);
        cQ = ggml_scale(ctx, cQ, attn_scale);
        ggml_tensor * cK = ggml_add(ctx, ggml_mul_mat(ctx, l.cwk, enc_t), l.cbk);
        ggml_tensor * cV = ggml_add(ctx, ggml_mul_mat(ctx, l.cwv, enc_t), l.cbv);

        cQ = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cQ, HD, H, Td), 0, 2, 1, 3));  // (HD, Td, H)
        cK = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cK, HD, H, Ts), 0, 2, 1, 3));  // (HD, Ts, H)
        cV = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cV, HD, H, Ts), 1, 2, 0, 3));  // (Ts, HD, H)

        ggml_tensor * ckq = ggml_mul_mat(ctx, cK, cQ);              // (Ts, Td, H)
        // No causal mask here: every target position may see the whole source.
        ckq = ggml_soft_max_ext(ctx, ckq, nullptr, 1.0f, 0.0f);
        ggml_tensor * ckqv = ggml_mul_mat(ctx, cV, ckq);            // (HD, Td, H)

        ckqv = ggml_cont(ctx, ggml_permute(ctx, ckqv, 0, 2, 1, 3)); // (HD, H, Td)
        cur = ggml_reshape_2d(ctx, ckqv, D, Td);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.cwo, cur), l.cbo);
        cur = ggml_add(ctx, cur, residual);

        // ---- feed forward ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.ffn_norm_w), l.ffn_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_up, h), l.b_up);
        h = ggml_relu(ctx, h);                                     // NLLB is relu
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_down, h), l.b_down);
        cur = ggml_add(ctx, h, residual);
    }

    cur = ggml_norm(ctx, cur, hp.eps);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, model.dec_out_norm_w), model.dec_out_norm_b);
    ggml_set_name(cur, "dec_out");
    ggml_set_output(cur);
    hs.push_back(cur);   // HF decoder_hidden_states[n_dec_layer]

    // lm_head, tied to token_embd unless the checkpoint untied it. ne of the
    // embedding is [D, vocab], so mul_mat against [D, Td] gives [vocab, Td] -
    // row-major logits[t][v], no transpose anywhere.
    //
    // NLLB_LOGITS_LAST projects only the final row. Greedy decoding reads
    // nothing else, and on the 600M this is a [256206, 1024] x [1024, Td]
    // matmul - the dominant cost of the whole graph at Td > 1.
    ggml_tensor * logits = nullptr;
    if (want_logits) {
        ggml_tensor * lm_in = cur;
        if (logits_mode == NLLB_LOGITS_LAST && Td > 1) {
            lm_in = ggml_cont(ctx, ggml_view_2d(ctx, cur, D, 1, cur->nb[1],
                                                (size_t)(Td - 1) * cur->nb[1]));
        }
        logits = ggml_mul_mat(ctx, model.lm_head(), lm_in);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    ggml_build_forward_expand(gf, cur);
    if (want_hidden_states) {
        for (ggml_tensor * t : hs) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
    }

    ggml_backend_sched * sched = model.sched;
    bool ok = nllb_alloc_graph(sched, gf, "decoder", err);
    if (ok) {
        ggml_backend_tensor_set(inp_ids, dec_ids.data(), 0, (size_t) Td * sizeof(int32_t));
        ggml_backend_tensor_set(inp_pos, pos.data(),     0, (size_t) Td * sizeof(int32_t));
        ggml_backend_tensor_set(enc_t, enc.output.data(), 0, (size_t) Ts * D * sizeof(float));
        ok = nllb_compute_graph(model, sched, gf, n_threads, "decoder", err);
    }
    if (ok) {
        out.n_tokens = Td;
        out.d_model  = D;
        out.n_vocab  = (int32_t) model.lm_head()->ne[1];
        nllb_get_f32(cur, out.output, (size_t) Td * D);

        out.logits.clear();
        out.n_logit_rows = 0;
        if (want_logits) {
            out.n_logit_rows = logit_rows;
            nllb_get_f32(logits, out.logits, (size_t) logit_rows * out.n_vocab);
        }

        out.hidden_states.clear();
        if (want_hidden_states) {
            out.hidden_states.resize(hs.size());
            for (size_t i = 0; i < hs.size(); ++i) {
                nllb_get_f32(hs[i], out.hidden_states[i], (size_t) Td * D);
            }
        }
    }

    ggml_backend_sched_reset(sched);
    ggml_free(ctx);
    return ok;
}

// --------------------------------------------------------------------------
// KV cache
//
// Storage is in the shape attention consumes, not the shape the projections
// produce. The un-cached path spends a permute + ggml_cont on K and V every
// block; doing that here would copy the entire history once per block per step
// and give back most of what the cache is for. So the write does the permute
// once, into the slot for the new token, and the read is a plain view.
//
//   k_self [HD, n_ctx, H]   token t at row t of each head slice
//   v_self [n_ctx, HD, H]   transposed, because mul_mat wants (n_kv, HD, H)
//
// Both halves live in one backend buffer the cache owns, on the model's primary
// backend. The per-step graph gets a context of its own and refers to these
// tensors across contexts, exactly as it already does for the model weights.
// --------------------------------------------------------------------------

nllb_kv_cache::~nllb_kv_cache() {
    if (sched) { ggml_backend_sched_free(sched);  sched = nullptr; }
    if (buf)   { ggml_backend_buffer_free(buf);   buf   = nullptr; }
    if (ctx)   { ggml_free(ctx); ctx = nullptr; }
}

void nllb_kv_cache_reset(nllb_kv_cache & cache) {
    cache.n_past = 0;
}

namespace {
void nllb_kv_cache_release(nllb_kv_cache & cache) {
    if (cache.sched) { ggml_backend_sched_free(cache.sched); cache.sched = nullptr; }
    if (cache.buf)   { ggml_backend_buffer_free(cache.buf);  cache.buf   = nullptr; }
    if (cache.ctx)   { ggml_free(cache.ctx); cache.ctx = nullptr; }
    cache.k_self.clear();  cache.v_self.clear();
    cache.k_cross.clear(); cache.v_cross.clear();
}
} // namespace

bool nllb_kv_cache_init(const nllb_model & model,
                        const nllb_encode_result & enc,
                        int n_ctx_cache,
                        int n_threads,
                        nllb_kv_cache & cache,
                        std::string & err,
                        int kv_type) {
    const nllb_hparams & hp = model.hp;
    const int D  = hp.d_model;
    const int H  = hp.n_dec_head;
    const int HD = hp.dec_head_dim();
    const int L  = hp.n_dec_layer;
    const int Ts = enc.n_tokens;

    if (!model.has_decoder) { err = "GGUF has no decoder (converted with --encoder-only?)"; return false; }
    if (n_ctx_cache <= 0)   { err = "kv cache capacity must be > 0"; return false; }
    if (Ts <= 0)            { err = "empty encoder output"; return false; }
    if (enc.d_model != D)   { err = "encoder output d_model mismatch"; return false; }
    if ((int) enc.output.size() != Ts * D) { err = "encoder output size mismatch"; return false; }
    if (hp.n_ctx > 0 && n_ctx_cache > hp.n_ctx) {
        err = "kv cache capacity " + std::to_string(n_ctx_cache) +
              " exceeds the model's n_ctx " + std::to_string(hp.n_ctx);
        return false;
    }
    if (kv_type != NLLB_KV_F32 && kv_type != NLLB_KV_F16) {
        err = "unknown kv cache type " + std::to_string(kv_type);
        return false;
    }
    // The only place the type is turned into a ggml type.
    const ggml_type kvt = kv_type == NLLB_KV_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    nllb_kv_cache_release(cache);

    // Four tensors per layer, as STRUCTS in a no_alloc context; the bytes are
    // one backend buffer on the model's primary backend, so on Metal the cache
    // lives where the attention that reads it runs. The self half is sized by
    // capacity, the cross half by the source length - which is why a new
    // sentence needs a new init and not a reset.
    ggml_init_params cp = {};
    cp.mem_size   = (size_t) ggml_tensor_overhead() * (4 * L + 16);
    cp.mem_buffer = nullptr;
    cp.no_alloc   = true;
    cache.ctx = ggml_init(cp);
    if (!cache.ctx) { err = "ggml_init failed (kv cache)"; return false; }

    cache.kv_type = kv_type;
    cache.n_ctx   = n_ctx_cache;
    cache.n_past  = 0;
    cache.n_src   = Ts;
    cache.d_model = D;
    cache.n_head  = H;
    cache.n_layer = L;
    cache.model   = &model;

    cache.k_self.resize(L);  cache.v_self.resize(L);
    cache.k_cross.resize(L); cache.v_cross.resize(L);
    for (int il = 0; il < L; ++il) {
        cache.k_self[il]  = ggml_new_tensor_3d(cache.ctx, kvt, HD, n_ctx_cache, H);
        cache.v_self[il]  = ggml_new_tensor_3d(cache.ctx, kvt, n_ctx_cache, HD, H);
        cache.k_cross[il] = ggml_new_tensor_3d(cache.ctx, kvt, HD, Ts, H);
        cache.v_cross[il] = ggml_new_tensor_3d(cache.ctx, kvt, Ts, HD, H);
    }
    cache.buf = ggml_backend_alloc_ctx_tensors_from_buft(
            cache.ctx, ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.buf) {
        err = "kv cache allocation failed on " + model.backend_name;
        nllb_kv_cache_release(cache);
        return false;
    }
    // Deliberately NOT cleared. The step-5 cache lived in a fresh malloc and
    // read as whatever the allocator handed back; nothing reads a slot before
    // it is written (the mask hides every slot >= n_kv, and the views never
    // reach past n_kv), and the [arena] and `full` passes in the cache gate
    // would say so if that ever stopped being true.
    ggml_backend_buffer_set_usage(cache.buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    cache.sched = nllb_new_sched(model);
    if (!cache.sched) { err = "ggml_backend_sched_new failed (kv cache)";
                        nllb_kv_cache_release(cache); return false; }

    // ---- fill the cross-attention half: one graph, once per source ----
    ggml_context * ctx = nllb_graph_ctx(model.graph_meta);
    if (!ctx) { err = "ggml_init failed (cross-attention cache build)";
                nllb_kv_cache_release(cache); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, NLLB_GRAPH_NODES, false);

    ggml_tensor * enc_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Ts);
    ggml_set_name(enc_t, "enc_out");
    ggml_set_input(enc_t);

    for (int il = 0; il < L; ++il) {
        const nllb_dec_layer & l = model.dec[il];

        // Identical arithmetic to the un-cached path's cK/cV, including the
        // bias and the absence of the attention scale (that goes on Q). If this
        // ever stops matching, tests/test_cache.cpp reports it as a
        // cross-attention divergence at layer il rather than as wrong ids.
        ggml_tensor * cK = ggml_add(ctx, ggml_mul_mat(ctx, l.cwk, enc_t), l.cbk);
        ggml_tensor * cV = ggml_add(ctx, ggml_mul_mat(ctx, l.cwv, enc_t), l.cbv);

        cK = ggml_permute(ctx, ggml_reshape_3d(ctx, cK, HD, H, Ts), 0, 2, 1, 3); // (HD, Ts, H)
        cV = ggml_permute(ctx, ggml_reshape_3d(ctx, cV, HD, H, Ts), 1, 2, 0, 3); // (Ts, HD, H)

        // A ggml_cpy INTO a pre-allocated tensor: the scheduler runs it on the
        // backend that owns the cache buffer, so the write lands in the cache
        // and not in a temporary copy of it.
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cK, cache.k_cross[il]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cV, cache.v_cross[il]));
    }

    bool ok = nllb_alloc_graph(model.sched, gf, "cross-kv build", err);
    if (ok) {
        ggml_backend_tensor_set(enc_t, enc.output.data(), 0, (size_t) Ts * D * sizeof(float));
        ok = nllb_compute_graph(model, model.sched, gf, n_threads, "cross-attention cache", err);
    }
    ggml_backend_sched_reset(model.sched);
    ggml_free(ctx);
    if (!ok) { nllb_kv_cache_release(cache); return false; }
    return true;
}

bool nllb_decode_cached(const nllb_model & model,
                        nllb_kv_cache & cache,
                        const std::vector<int32_t> & new_ids,
                        int n_threads,
                        bool want_hidden_states,
                        int logits_mode,
                        nllb_decode_result & out,
                        std::string & err,
                        nllb_step_timing * timing) {
    const clk::time_point t_enter = clk::now();
    if (timing) *timing = nllb_step_timing();

    const nllb_hparams & hp = model.hp;
    const int Tn = (int) new_ids.size();
    const int D  = hp.d_model;
    const int H  = hp.n_dec_head;
    const int HD = hp.dec_head_dim();
    const int L  = hp.n_dec_layer;

    if (!model.has_decoder) { err = "GGUF has no decoder (converted with --encoder-only?)"; return false; }
    if (!cache.ctx || !cache.buf) { err = "kv cache is not initialised (call nllb_kv_cache_init)"; return false; }
    if (cache.model != &model) { err = "kv cache was built for a different nllb_model"; return false; }
    if (Tn <= 0)            { err = "empty decoder input"; return false; }
    if (cache.n_layer != L || cache.d_model != D || cache.n_head != H) {
        err = "kv cache was built for a different model shape";
        return false;
    }
    const int n_past = cache.n_past;
    const int n_kv   = n_past + Tn;
    if (n_kv > cache.n_ctx) {
        err = "kv cache is full: " + std::to_string(n_past) + " + " +
              std::to_string(Tn) + " > capacity " + std::to_string(cache.n_ctx);
        return false;
    }

    const bool want_logits = logits_mode != NLLB_LOGITS_NONE;
    const int  logit_rows  = logits_mode == NLLB_LOGITS_LAST ? 1 : Tn;

    // The step's compute allocator. Reused: gallocr keeps its buffer and grows
    // it only when this step's graph genuinely needs more. Not reused (the
    // [arena] pass): a fresh scheduler for this step alone, freed below.
    ggml_backend_sched * sched = cache.sched;
    ggml_backend_sched * fresh = nullptr;
    if (!cache.reuse_scratch || !sched) {
        fresh = nllb_new_sched(model);
        if (!fresh) { err = "ggml_backend_sched_new failed (cached decoder)"; return false; }
        sched = fresh;
    }

    ggml_context * ctx = nllb_graph_ctx(cache.graph_meta);
    if (!ctx) { if (fresh) ggml_backend_sched_free(fresh);
                err = "ggml_init failed (cached decoder)"; return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, NLLB_GRAPH_NODES, false);

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tn);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tn);
    ggml_set_name(inp_ids, "dec_ids");
    ggml_set_name(inp_pos, "dec_pos");
    ggml_set_input(inp_ids);
    ggml_set_input(inp_pos);

    // NLLB_POS_ABSOLUTE, and only that. HF's cached rule is
    // (cumsum(mask) + n_past) * mask + pad_id with the cumsum restarting each
    // step, which for a token at absolute index j collapses to j + 1 + pad_id -
    // or pad_id if the token IS pad. The un-cached CUMSUM rule cannot be
    // evaluated from n_past alone (it depends on how many pads the prefix
    // contained), and reproducing it here would mean matching HF's *other*
    // path, so it is not offered. See nllb_pos_mode in nllb.h.
    std::vector<int32_t> pos(Tn);
    for (int i = 0; i < Tn; ++i) {
        const int32_t j = n_past + i;
        pos[i] = (new_ids[i] == hp.pad_id) ? hp.pad_id : j + 1 + hp.pad_id;
    }

    // Query row j is absolute position n_past + j and may see keys
    // 0..n_past + j. At Tn == 1 the correct offset masks nothing, and the error
    // this hides is one-sided: an offset that is too LARGE still masks nothing
    // and is invisible to single-token generation, while one that is too small
    // masks a real key and is caught immediately (trap 13). That asymmetry is
    // why tests/test_cache.cpp feeds the same prefix in chunks as well.
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, Tn);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);
    std::vector<float> mask;
    nllb_causal_mask(mask, n_kv, Tn, n_past);

    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_ids);
    cur = ggml_scale(ctx, cur, hp.embd_scale);
    cur = ggml_add(ctx, cur, ggml_get_rows(ctx, model.pos_embd, inp_pos));

    std::vector<ggml_tensor *> hs;
    hs.reserve(L + 1);

    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < L; ++il) {
        const nllb_dec_layer & l = model.dec[il];
        hs.push_back(cur);

        // ---- causal self-attention over cache + new tokens ----
        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.attn_norm_w), l.attn_norm_b);

        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, h), l.bq);
        Q = ggml_scale(ctx, Q, attn_scale);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, h), l.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, h), l.bv);

        Q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, Q, HD, H, Tn), 0, 2, 1, 3));   // (HD, Tn, H)

        ggml_tensor * kc = cache.k_self[il];
        ggml_tensor * vc = cache.v_self[il];

        // Write the new rows in the layout the read expects. ggml_cpy handles
        // the strided destination, which is what buys the read below being a
        // view rather than a copy of the whole history.
        ggml_tensor * Kp = ggml_permute(ctx,
                ggml_reshape_3d(ctx, K, HD, H, Tn), 0, 2, 1, 3);    // (HD, Tn, H)
        ggml_tensor * Vp = ggml_permute(ctx,
                ggml_reshape_3d(ctx, V, HD, H, Tn), 1, 2, 0, 3);    // (Tn, HD, H)

        ggml_tensor * k_dst = ggml_view_3d(ctx, kc, HD, Tn, H,
                kc->nb[1], kc->nb[2], (size_t) n_past * kc->nb[1]);
        ggml_tensor * v_dst = ggml_view_3d(ctx, vc, Tn, HD, H,
                vc->nb[1], vc->nb[2], (size_t) n_past * vc->nb[0]);

        // ORDER MATTERS (trap 15). These copies have no data dependency on the
        // reads below, so only graph order makes the write happen first: every
        // backend walks gf->nodes in insertion order, and
        // ggml_build_forward_expand appends depth-first. What protects this
        // graph is that NOTHING downstream is expanded during the layer loop -
        // `cur` and the logits enter the graph once, at the end - so the copies
        // are the only nodes inserted here and land ahead of every read.
        //
        // Step 13 adds a reader of the node order that may REORDER it: Metal's
        // graph_optimize moves nodes to raise concurrency. It is memory-range
        // aware - it will not move a node that reads a range an earlier node
        // writes - so the cpy and the mul_mat over the overlapping cache view
        // keep their order. The chunked passes of the cache gate are what would
        // say otherwise; GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1 turns it off to
        // bisect.
        //
        // The rule to keep: do not expand anything that reads the cache before
        // these two lines. A per-layer debug hook is exactly the
        // innocent-looking change that would.
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kp, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vp, v_dst));

        ggml_tensor * Kv = ggml_view_3d(ctx, kc, HD, n_kv, H,
                kc->nb[1], kc->nb[2], 0);                           // (HD, n_kv, H)
        ggml_tensor * Vv = ggml_view_3d(ctx, vc, n_kv, HD, H,
                vc->nb[1], vc->nb[2], 0);                           // (n_kv, HD, H)

        ggml_tensor * kq = ggml_mul_mat(ctx, Kv, Q);                // (n_kv, Tn, H)
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, Vv, kq);              // (HD, Tn, H)

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));   // (HD, H, Tn)
        cur = ggml_reshape_2d(ctx, kqv, D, Tn);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, cur), l.bo);
        cur = ggml_add(ctx, cur, residual);

        // ---- cross-attention: K and V come straight out of the cache ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.cross_norm_w), l.cross_norm_b);

        ggml_tensor * cQ = ggml_add(ctx, ggml_mul_mat(ctx, l.cwq, h), l.cbq);
        cQ = ggml_scale(ctx, cQ, attn_scale);
        cQ = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cQ, HD, H, Tn), 0, 2, 1, 3));  // (HD, Tn, H)

        ggml_tensor * ckq = ggml_mul_mat(ctx, cache.k_cross[il], cQ);  // (Ts, Tn, H)
        // Still unmasked: every target position may see the whole source.
        ckq = ggml_soft_max_ext(ctx, ckq, nullptr, 1.0f, 0.0f);
        ggml_tensor * ckqv = ggml_mul_mat(ctx, cache.v_cross[il], ckq); // (HD, Tn, H)

        ckqv = ggml_cont(ctx, ggml_permute(ctx, ckqv, 0, 2, 1, 3)); // (HD, H, Tn)
        cur = ggml_reshape_2d(ctx, ckqv, D, Tn);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.cwo, cur), l.cbo);
        cur = ggml_add(ctx, cur, residual);

        // ---- feed forward ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.ffn_norm_w), l.ffn_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_up, h), l.b_up);
        h = ggml_relu(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_down, h), l.b_down);
        cur = ggml_add(ctx, h, residual);
    }

    cur = ggml_norm(ctx, cur, hp.eps);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, model.dec_out_norm_w), model.dec_out_norm_b);
    ggml_set_name(cur, "dec_out");
    ggml_set_output(cur);
    hs.push_back(cur);

    ggml_tensor * logits = nullptr;
    if (want_logits) {
        ggml_tensor * lm_in = cur;
        if (logits_mode == NLLB_LOGITS_LAST && Tn > 1) {
            lm_in = ggml_cont(ctx, ggml_view_2d(ctx, cur, D, 1, cur->nb[1],
                                                (size_t)(Tn - 1) * cur->nb[1]));
        }
        logits = ggml_mul_mat(ctx, model.lm_head(), lm_in);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    ggml_build_forward_expand(gf, cur);
    if (want_hidden_states) {
        for (ggml_tensor * t : hs) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
    }

    const clk::time_point t_build1 = clk::now();
    if (timing) {
        timing->build   = ms_between(t_enter, t_build1);
        timing->n_nodes = ggml_graph_n_nodes(gf);
        timing->scratch_reused = fresh == nullptr;
    }

    bool ok = nllb_alloc_graph(sched, gf, "cached step", err);
    const clk::time_point t_alloc1 = clk::now();
    if (timing) {
        timing->alloc = ms_between(t_build1, t_alloc1);
        timing->arena = nllb_sched_bytes(sched);
    }
    if (ok) {
        ggml_backend_tensor_set(inp_ids, new_ids.data(), 0, (size_t) Tn * sizeof(int32_t));
        ggml_backend_tensor_set(inp_pos, pos.data(),     0, (size_t) Tn * sizeof(int32_t));
        ggml_backend_tensor_set(kq_mask, mask.data(),    0, mask.size() * sizeof(float));
        ok = nllb_compute_graph(model, sched, gf, n_threads, "cached decoder", err);
    }
    const clk::time_point t_compute1 = clk::now();
    if (timing) timing->compute = ms_between(t_alloc1, t_compute1);

    if (ok) {
        out.n_tokens = Tn;
        out.d_model  = D;
        out.n_vocab  = (int32_t) model.lm_head()->ne[1];
        nllb_get_f32(cur, out.output, (size_t) Tn * D);

        out.logits.clear();
        out.n_logit_rows = 0;
        if (want_logits) {
            out.n_logit_rows = logit_rows;
            nllb_get_f32(logits, out.logits, (size_t) logit_rows * out.n_vocab);
        }

        out.hidden_states.clear();
        if (want_hidden_states) {
            out.hidden_states.resize(hs.size());
            for (size_t i = 0; i < hs.size(); ++i) {
                nllb_get_f32(hs[i], out.hidden_states[i], (size_t) Tn * D);
            }
        }
    }

    const clk::time_point t_read1 = clk::now();
    if (timing) timing->readback = ms_between(t_compute1, t_read1);

    ggml_backend_sched_reset(sched);
    if (fresh) ggml_backend_sched_free(fresh);
    ggml_free(ctx);

    if (timing) {
        timing->release = ms_between(t_read1, clk::now());
        timing->total   = ms_between(t_enter, clk::now());
    }
    if (!ok) return false;

    // Only now, after the graph really ran: a caller that retries after a
    // failure must not find the cache half-advanced.
    cache.n_past = n_kv;
    return true;
}

// --------------------------------------------------------------------------
// greedy generation
//
// One encoder pass, then one decoder pass per emitted token over the whole
// prefix. No KV cache: step t rebuilds and recomputes everything steps
// 0..t-1 already computed, which is O(n^2) in the target length and exactly
// what the cached path will be diffed against.
//
// Parity with HF generate(num_beams=1, do_sample=False):
//   - the sequence starts at [decoder_start_token_id] and that token counts as
//     position 0, so max_new_tokens caps the result at 1 + max_new_tokens;
//   - forced_bos_token_id replaces the argmax at the FIRST generated position
//     (HF does it by masking every other logit to -inf; same outcome, and the
//     unforced argmax is kept in the trace so a divergence is still readable);
//   - eos terminates and is included in the output;
//   - ties go to the lower token id, which is what torch.argmax does;
//   - positions follow NLLB_POS_ABSOLUTE, because generate() decodes with a
//     cache and HF's cached position rule is not the same as its un-cached one
//     once a pad_id appears in the prefix. See nllb_pos_mode in nllb.h.
// --------------------------------------------------------------------------

bool nllb_generate(const nllb_model & model,
                   const std::vector<int32_t> & src_ids,
                   const nllb_generate_params & params,
                   int n_threads,
                   nllb_generate_result & out,
                   std::string & err) {
    const nllb_hparams & hp = model.hp;

    if (!model.has_decoder) {
        err = "GGUF has no decoder (converted with --encoder-only?)";
        return false;
    }
    if (src_ids.empty())            { err = "empty source"; return false; }
    if (params.max_new_tokens <= 0) { err = "max_new_tokens must be > 0"; return false; }

    // Same restriction as nllb_decode, checked up front so the caller is told
    // before an encoder pass is spent on it.
    for (int32_t id : src_ids) {
        if (id == hp.pad_id) {
            err = "padded source sequences are not supported yet "
                  "(cross-attention is unmasked); see nllb_decode";
            return false;
        }
    }

    const int32_t eos   = params.eos_id       >= 0 ? params.eos_id       : hp.eos_id;
    const int32_t start = params.dec_start_id >= 0 ? params.dec_start_id : hp.dec_start_id;

    // These arrive from a manifest file and from CLI flags, and go straight
    // into ggml_get_rows as indices. Out of range, ggml aborts the process on
    // an assert; a CI run deserves a message instead of a stack trace.
    const int32_t n_vocab = (int32_t) model.lm_head()->ne[1];
    const int32_t embd_rows = (int32_t) model.tok_embd->ne[1];
    const int32_t id_limit = n_vocab < embd_rows ? n_vocab : embd_rows;
    struct { const char * what; int32_t id; } checked[] = {
        { "decoder_start_token_id", start },
        { "eos_id",                 eos   },
        { "forced_bos_id",          params.forced_bos_id },
    };
    for (const auto & c : checked) {
        if (c.id < -1 || c.id >= id_limit) {
            err = std::string(c.what) + " = " + std::to_string(c.id) +
                  " is out of range [0, " + std::to_string(id_limit) + ")";
            return false;
        }
    }
    if (start < 0) { err = "decoder_start_token_id must be >= 0"; return false; }
    if (eos   < 0) { err = "eos_id must be >= 0"; return false; }

    out = nllb_generate_result();
    out.n_src = (int32_t) src_ids.size();
    out.stop  = NLLB_STOP_MAX_NEW;
    out.used_cache = params.use_cache;

    nllb_encode_result enc;
    if (!nllb_encode(model, src_ids, n_threads, false, enc, err)) return false;

    out.ids.reserve((size_t) params.max_new_tokens + 1);
    out.ids.push_back(start);

    // The cached path needs its capacity up front: decoder_start plus every
    // token that could be emitted. Clamped to the model's n_ctx, so running out
    // of cache and running out of context are the same event with the same
    // stop reason rather than two different-looking failures.
    nllb_kv_cache cache;
    if (params.use_cache) {
        int cap = params.max_new_tokens + 1;
        if (hp.n_ctx > 0 && cap > hp.n_ctx) cap = hp.n_ctx;
        if (!nllb_kv_cache_init(model, enc, cap, n_threads, cache, err,
                                params.kv_type)) return false;
    }

    for (int step = 0; step < params.max_new_tokens; ++step) {
        // nllb_decode accepts Td == n_ctx, so the prefix may still be decoded
        // when it has exactly n_ctx tokens; only a longer one must be refused.
        if (hp.n_ctx > 0 && (int) out.ids.size() > hp.n_ctx) {
            out.stop = NLLB_STOP_CONTEXT;
            break;
        }

        nllb_decode_result dec;
        if (params.use_cache) {
            // Step 0 feeds [decoder_start]; every later step feeds only the
            // token just emitted, with the rest of the prefix already in the
            // cache. cache.n_past is therefore the absolute index of that
            // token, which is exactly what the position rule needs.
            const std::vector<int32_t> feed(1, out.ids.back());
            if ((int) cache.n_past != (int) out.ids.size() - 1) {
                err = "kv cache out of step with the generated prefix: n_past=" +
                      std::to_string(cache.n_past) + ", expected " +
                      std::to_string((int) out.ids.size() - 1);
                return false;
            }
            if (!nllb_decode_cached(model, cache, feed, n_threads,
                                    /*want_hidden_states=*/false,
                                    NLLB_LOGITS_LAST, dec, err)) {
                return false;
            }
        } else {
            // NLLB_POS_ABSOLUTE, not the teacher-forced default: the goldens
            // come from generate(), which decodes incrementally. The two rules
            // differ only if the model emits pad_id, and then this is the one
            // HF used - and the one the cached branch above has no choice about.
            if (!nllb_decode(model, enc, out.ids, src_ids, n_threads,
                             /*want_hidden_states=*/false, NLLB_LOGITS_LAST,
                             dec, err, NLLB_POS_ABSOLUTE)) {
                return false;
            }
        }
        if (dec.n_logit_rows != 1 || (int) dec.logits.size() != dec.n_vocab) {
            err = "decoder returned unexpected logits shape";
            return false;
        }

        // argmax plus runner-up, in one pass. Strict > keeps the first index on
        // a tie, matching torch.argmax.
        const float * p = dec.logits.data();
        const int n = dec.n_vocab;
        int best = 0, second = -1;
        for (int i = 1; i < n; ++i) {
            if (p[i] > p[best]) { second = best; best = i; }
            else if (second < 0 || p[i] > p[second]) second = i;
        }

        nllb_generate_step s;
        s.argmax = best;
        s.logit  = p[best];
        s.margin = second >= 0 ? p[best] - p[second] : 0.0f;
        s.forced = (step == 0 && params.forced_bos_id >= 0);
        s.token  = s.forced ? params.forced_bos_id : best;

        out.ids.push_back(s.token);
        out.steps.push_back(s);

        if (s.token == eos) {
            out.stop    = NLLB_STOP_EOS;
            out.hit_eos = true;
            break;
        }
    }

    return true;
}
