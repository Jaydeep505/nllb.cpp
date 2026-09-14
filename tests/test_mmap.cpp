// mmap'ed weights vs read weights. No goldens, no torch, no ref/ directory.
//
//   ./build/bin/nllb-test-mmap models/nllb-600M-f16.gguf [-v]
//
// WHAT THIS ASSERTS, and why it is the whole assertion.
// Mapping the file changes WHERE the weight bytes live, not what they are.
// The weights are never written, so if every tensor is byte-identical between
// the two paths then everything downstream - encoder, decoder, cache, logits,
// ids, printed text - is identical BY CONSTRUCTION, and no amount of decoding
// could show otherwise. So this gate compares the bytes and stops.
//
// That puts it in the same family as the graph-arena reuse in step 6, which
// also asserted bit-identity rather than measuring a difference, and it earns
// its place for the same reason: an allocation change that is "obviously"
// equivalent is exactly the kind that is silently wrong. Three ways this can
// break, all of which produce plausible numbers rather than a crash:
//
//   1. an off-by-one in the offset arithmetic, so a tensor reads its
//      neighbour's tail - the shapes still match and the model still runs;
//   2. gguf_get_data_offset not accounted for, so every tensor is shifted by
//      the size of the metadata block;
//   3. the mapping freed before the context, leaving ->data dangling - which
//      reads as whatever the allocator put there next.
//
// It also prints the load time of each path, because that is the only reason
// to do this at all.

#include "nllb.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> [-v]\n"
            "  Loads the same file twice - mapped and read - and asserts every\n"
            "  weight tensor is byte-identical. Needs no goldens.\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    bool verbose = false;
    for (int i = 2; i < argc; ++i) if (!std::strcmp(argv[i], "-v")) verbose = true;

    std::string err;

    nllb_model mapped;
    const double t0 = now_ms();
    if (!nllb_model_load(path, mapped, err, /*use_mmap =*/ true)) {
        std::fprintf(stderr, "load failed (mmap): %s\n", err.c_str());
        return 2;
    }
    const double t_map = now_ms() - t0;

    // THE NUMBER THAT MAKES THE COMPARISON HONEST.
    //
    // mmap does not remove the cost of getting the weights into memory, it
    // DEFERS it: the call returns in microseconds and every page is faulted in
    // on first touch, which for a translator means during the first token.
    // Timing the map alone and printing "1580x faster" would be true and
    // useless. So touch one byte per page across every weight, and report that
    // separately - the sum is what the read path pays up front.
    //
    // This also has to happen BEFORE the read load, or the read would be timed
    // against a page cache this loop had already warmed.
    double t_fault = 0.0;
    {
        volatile unsigned char sink = 0;
        const long page = 4096;   // a lower bound on any real page size, so
                                  // this over-touches rather than skipping one
        const double tf = now_ms();
        for (int64_t i = 0; i < gguf_get_n_tensors(mapped.gguf); ++i) {
            ggml_tensor * t = ggml_get_tensor(mapped.ctx_w,
                                              gguf_get_tensor_name(mapped.gguf, i));
            // Host pointers only when the backend buffer WRAPS the mapping
            // (step 13): true on the CPU and on Metal, and then ->data is the
            // address inside the mapping. Anything else is a device handle.
            if (!t || !t->data || !mapped.weights_host_ptr) continue;
            const unsigned char * p = (const unsigned char *) t->data;
            const size_t nb = ggml_nbytes(t);
            for (size_t off = 0; off < nb; off += (size_t) page) sink ^= p[off];
            if (nb) sink ^= p[nb - 1];
        }
        t_fault = now_ms() - tf;
        (void) sink;
    }

    nllb_model readin;
    const double t1 = now_ms();
    if (!nllb_model_load(path, readin, err, /*use_mmap =*/ false)) {
        std::fprintf(stderr, "load failed (read): %s\n", err.c_str());
        return 2;
    }
    const double t_read = now_ms() - t1;

    std::printf("model:  %s\n", path.c_str());
    std::printf("backend: %s (weights %s)\n", mapped.backend_name.c_str(),
                mapped.weights_host_ptr ? "zero-copy: the backend buffer wraps the mapping"
                                        : "copied");
    std::printf("mapped: %s\n\n",
                mapped.mapped ? "yes" : "NO - fell back to read, see below");

    // A fallback is not a failure of the model, but it IS a failure of this
    // gate: there is nothing left to compare, and reporting PASS would say
    // "mmap is correct here" about a run in which mmap never happened. The
    // step-4 SKIP lesson.
    if (!mapped.mapped) {
        std::fprintf(stderr,
            "the mmap path fell back to reading, so the two loads are the same\n"
            "code and this comparison is vacuous. Either the platform has no\n"
            "mmap, or the file's general.alignment does not permit mapping.\n");
        return 1;
    }

    const int64_t n = gguf_get_n_tensors(mapped.gguf);
    if (n != gguf_get_n_tensors(readin.gguf)) {
        std::fprintf(stderr, "tensor counts differ - not the same file\n");
        return 1;
    }

    int64_t n_bad = 0, n_ok = 0;
    size_t  total = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(mapped.gguf, i);
        ggml_tensor * a = ggml_get_tensor(mapped.ctx_w, name);
        ggml_tensor * b = ggml_get_tensor(readin.ctx_w, name);
        if (!a || !b) {
            std::printf("  %-40s MISSING in %s\n", name, a ? "read" : "mmap");
            ++n_bad; continue;
        }
        if (a->type != b->type || std::memcmp(a->ne, b->ne, sizeof(a->ne)) != 0) {
            std::printf("  %-40s type/shape differs\n", name);
            ++n_bad; continue;
        }
        const size_t nb = ggml_nbytes(a);
        total += nb;
        // Through the backend, not through ->data: since step 13 the read
        // path copies the weights into a backend buffer, which on a GPU is
        // device memory that the host cannot dereference.
        std::vector<unsigned char> va(nb), vb(nb);
        ggml_backend_tensor_get(a, va.data(), 0, nb);
        ggml_backend_tensor_get(b, vb.data(), 0, nb);
        if (std::memcmp(va.data(), vb.data(), nb) != 0) {
            // Locate the first differing byte: an offset error usually puts it
            // at 0, a size error near the end.
            size_t k = 0;
            const unsigned char * pa = va.data();
            const unsigned char * pb = vb.data();
            while (k < nb && pa[k] == pb[k]) ++k;
            std::printf("  %-40s DIFFERS at byte %zu of %zu (%s)\n",
                        name, k, nb,
                        k == 0 ? "offset is wrong" : "size or tail is wrong");
            ++n_bad; continue;
        }
        ++n_ok;
        if (verbose) {
            std::printf("  %-40s %-6s %8zu B  identical\n",
                        name, ggml_type_name(a->type), nb);
        }
    }

    std::printf("\n%lld tensors, %.3f GB, %lld identical, %lld differing\n",
                (long long) n, total / 1e9, (long long) n_ok, (long long) n_bad);
    const double t_map_total = t_map + t_fault;
    std::printf("\nload, mmap:  %8.1f ms map+parse  + %8.1f ms faulting every page in\n"
                "                                    = %8.1f ms before the weights are usable\n",
                t_map, t_fault, t_map_total);
    std::printf("load, read:  %8.1f ms, all of it up front\n", t_read);
    std::printf("\n  QUOTE THE TOTAL, NOT THE MAP. %.1f ms vs %.1f ms is %.2fx.\n"
                "  The %.1f ms map+parse figure on its own is real but means only\n"
                "  that the cost MOVED: with mmap the pages fault in during the\n"
                "  first token, so a translator's time-to-first-token barely\n"
                "  changes while `load:` collapses. What mmap actually removes is\n"
                "  the COPY - the read path pulls the blob out of the page cache\n"
                "  into fresh anonymous pages, and the map path never does.\n",
                t_map_total, t_read,
                t_read / (t_map_total > 0.0 ? t_map_total : 1.0), t_map);
    std::printf("  Neither figure includes nllb_vocab_load_from_gguf, so unlike\n"
                "  nllb-cli's `load:` line the tokenizer build is in neither.\n"
                "  Deriving 514098 merges is pure CPU and happens either way -\n"
                "  once the copy is gone, THAT is what the load time becomes.\n");
    std::printf("  Run it twice. The first run measures the SSD; the second\n"
                "  measures memcpy against a warm page cache, which is the\n"
                "  steady-state number a user actually sees.\n");

    if (n_bad) { std::printf("\nMISMATCH\n"); return 1; }
    std::printf("\nevery weight byte is identical between the two paths - the\n"
                "mapping changes where the bytes live, not what they are, so\n"
                "everything downstream is unchanged by construction.\nPASS\n");
    return 0;
}
