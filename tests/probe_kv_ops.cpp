// nllb-probe-kv-ops: the cached decoder step's four KV-cache ops, one at a
// time, on a device backend against the CPU backend. No model, no goldens.
//
// Written in step 17 because the cache gate went red on Vulkan (lavapipe) with
// `full = exact` but `step` and `chunk` wrong from layer 1 and [arena]
// differing - i.e. a single-token (Tn <= 8) self-attention on the cache VIEWS
// reads the wrong bytes, and a full-length pass does not. That localises it to
// one of four ops, and each is tested here in isolation at the cache's real
// strides, for Tn = 1, 3 and 9 (either side of the mat-vec / mat-mat switch
// ggml's GPU backends make at 8 columns - trap 48).
//
//   nllb-probe-kv-ops [backend]          default: the first GPU
//
// Exit 0 unless some op is WRONG (>= 1e-2 relative, i.e. read the wrong
// rows). ~1e-3 is reported as "half": the device ran the matmul in f16 (trap
// 48), which is a precision fact, not a bug. The table says which op broke.
//
// Measured on Vulkan (lavapipe, ggml 7840aab): "mul_mat(K view, q)" is BAD at
// every Tn and n_past, "mul_mat(cont(K), q)" is ok - which is the workaround
// src/nllb.cpp applies on Vulkan. When this row turns ok, ggml has fixed it.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * name;
    // builds the op from the provided cache tensors and inputs; returns output
    std::function<ggml_tensor *(ggml_context *, ggml_tensor * kc, ggml_tensor * vc,
                                ggml_tensor * x, int n_past, int Tn)> build;
    bool reads_cache_after_write;   // output is the cache itself (a cpy)
};

// Run `c` on `backend` with deterministic inputs; return the bytes of the
// output (for a cpy: the WHOLE cache tensor, so a write to the wrong slot shows).
std::vector<float> run(ggml_backend_t backend, const Case & c, int HD, int H, int n_ctx,
                       int n_past, int Tn) {
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, n_ctx, H);   // K cache
    ggml_tensor * vc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ctx, HD, H);   // V cache, transposed
    // x is sized for the largest consumer: K/V rows (HD*H*Tn) or q (HD, Tn, H)
    // or kq (n_kv, Tn, H).
    const int n_kv = n_past + Tn;
    const int64_t nx = std::max<int64_t>((int64_t) HD * H * Tn, (int64_t) n_kv * Tn * H);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);

    ggml_tensor * out = c.build(ctx, kc, vc, x, n_past, Tn);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> v(ggml_nelements(t));
        for (float & f : v) f = U(rng);
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };
    fill(kc); fill(vc); fill(x);

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> r(ggml_nelements(out));
    if (ggml_is_contiguous(out)) {
        ggml_backend_tensor_get(out, r.data(), 0, r.size() * sizeof(float));
    } else {   // a view of the cache: read the whole cache it lives in
        ggml_tensor * base = out->view_src ? out->view_src : out;
        r.resize(ggml_nelements(base));
        ggml_backend_tensor_get(base, r.data(), 0, r.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string want = argc > 1 ? argv[1] : "";
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        // Case-insensitive, device OR registry name: the first draft compared
        // "vulkan" against "Vulkan0" case-sensitively, found nothing, and the
        // CI step reading its table passed on an EMPTY table. Trap 49 in a
        // tool written the same week as the note about trap 49.
        auto lower = [](std::string x) { for (char & ch : x) ch = (char) std::tolower((unsigned char) ch); return x; };
        const std::string w = lower(want);
        const std::string n = lower(ggml_backend_dev_name(d));
        const std::string r = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)));
        if (w.empty() || n.find(w) != std::string::npos || r.find(w) != std::string::npos) { dev = d; break; }
    }
    if (!dev) { std::fprintf(stderr, "no device backend matching '%s'\n", want.c_str()); return 2; }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::printf("device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    const int HD = 64, H = 4, n_ctx = 64;

    std::vector<Case> cases = {
        { "cpy K -> k_dst view", [&](ggml_context * ctx, ggml_tensor * kc, ggml_tensor *, ggml_tensor * x, int n_past, int Tn) {
              ggml_tensor * K  = ggml_view_1d(ctx, x, (int64_t) HD * H * Tn, 0);
              ggml_tensor * Kp = ggml_permute(ctx, ggml_reshape_3d(ctx, K, HD, H, Tn), 0, 2, 1, 3);
              ggml_tensor * dst = ggml_view_3d(ctx, kc, HD, Tn, H, kc->nb[1], kc->nb[2], (size_t) n_past * kc->nb[1]);
              return ggml_cpy(ctx, Kp, dst); }, true },
        { "cpy V -> v_dst view", [&](ggml_context * ctx, ggml_tensor *, ggml_tensor * vc, ggml_tensor * x, int n_past, int Tn) {
              ggml_tensor * V  = ggml_view_1d(ctx, x, (int64_t) HD * H * Tn, 0);
              ggml_tensor * Vp = ggml_permute(ctx, ggml_reshape_3d(ctx, V, HD, H, Tn), 1, 2, 0, 3);
              ggml_tensor * dst = ggml_view_3d(ctx, vc, Tn, HD, H, vc->nb[1], vc->nb[2], (size_t) n_past * vc->nb[0]);
              return ggml_cpy(ctx, Vp, dst); }, true },
        { "mul_mat(K view, q)", [&](ggml_context * ctx, ggml_tensor * kc, ggml_tensor *, ggml_tensor * x, int n_past, int Tn) {
              const int n_kv = n_past + Tn;
              ggml_tensor * Kv = ggml_view_3d(ctx, kc, HD, n_kv, H, kc->nb[1], kc->nb[2], 0);
              ggml_tensor * q  = ggml_reshape_3d(ctx, ggml_view_1d(ctx, x, (int64_t) HD * Tn * H, 0), HD, Tn, H);
              return ggml_mul_mat(ctx, Kv, q); }, false },
        { "mul_mat(cont(K), q)", [&](ggml_context * ctx, ggml_tensor * kc, ggml_tensor *, ggml_tensor * x, int n_past, int Tn) {
              // what src/nllb.cpp does on Vulkan (nllb_model::kv_k_cont)
              const int n_kv = n_past + Tn;
              ggml_tensor * Kv = ggml_cont(ctx, ggml_view_3d(ctx, kc, HD, n_kv, H, kc->nb[1], kc->nb[2], 0));
              ggml_tensor * q  = ggml_reshape_3d(ctx, ggml_view_1d(ctx, x, (int64_t) HD * Tn * H, 0), HD, Tn, H);
              return ggml_mul_mat(ctx, Kv, q); }, false },
        { "mul_mat(V view, kq)", [&](ggml_context * ctx, ggml_tensor *, ggml_tensor * vc, ggml_tensor * x, int n_past, int Tn) {
              const int n_kv = n_past + Tn;
              ggml_tensor * Vv = ggml_view_3d(ctx, vc, n_kv, HD, H, vc->nb[1], vc->nb[2], 0);
              ggml_tensor * kq = ggml_reshape_3d(ctx, ggml_view_1d(ctx, x, (int64_t) n_kv * Tn * H, 0), n_kv, Tn, H);
              return ggml_mul_mat(ctx, Vv, kq); }, false },
    };

    int n_bad = 0;
    std::printf("%-22s %4s %6s  %-12s %s\n", "op", "Tn", "n_past", "max rel", "verdict");
    for (const Case & c : cases) {
        for (int Tn : { 1, 3, 9 }) {
            for (int n_past : { 0, 5, 20 }) {
                std::vector<float> a = run(cpu, c, HD, H, n_ctx, n_past, Tn);
                std::vector<float> b = run(gpu, c, HD, H, n_ctx, n_past, Tn);
                double worst = 0, amax = 0;
                size_t n_diff = 0;
                for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                    const double d = std::fabs((double) a[i] - b[i]);
                    amax = std::max(amax, (double) std::fabs(a[i]));
                    if (d > 0) ++n_diff;
                    worst = std::max(worst, d);
                }
                const double rel = amax > 0 ? worst / amax : worst;
                // Three bands, not two. A GPU backend may run a matmul's
                // inputs in half (trap 48): ~1e-3 relative is that, and is a
                // precision fact rather than a wrong read. A wrong read lands
                // at O(1), because it is a different row entirely.
                const bool same = a.size() == b.size();
                const char * v = !same ? "BAD (size)" : rel < 1e-5 ? "ok" : rel < 1e-2 ? "half (trap 48)" : "BAD";
                if (!same || rel >= 1e-2) ++n_bad;
                std::printf("%-22s %4d %6d  %-12.3e %s (%zu floats differ)\n", c.name, Tn, n_past, rel,
                            v, n_diff);
            }
        }
    }
    std::printf("%s - %d of %zu combinations disagree with the CPU\n", n_bad ? "FAIL" : "PASS", n_bad, cases.size() * 9);
    std::printf("(the raw \"K view\" row is EXPECTED to be BAD on Vulkan at ggml 7840aab - see the header)\n");
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return n_bad ? 1 : 0;
}
