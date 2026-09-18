// Diff the ggml encoder against the HF goldens in ref/.
//
//   ./build/bin/nllb-test models/nllb-600M-f32.gguf ref/ [--tol 2e-4] [-t 4]
//
// Exits non-zero on mismatch so it drops straight into CI.

#include "nllb.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool read_raw(const std::string & path, std::vector<char> & buf) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    buf.resize((size_t) n);
    return (bool) f.read(buf.data(), n);
}

bool read_f32(const std::string & path, std::vector<float> & v) {
    std::vector<char> b;
    if (!read_raw(path, b)) return false;
    v.resize(b.size() / sizeof(float));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(float));
    return true;
}

bool read_i32(const std::string & path, std::vector<int32_t> & v) {
    std::vector<char> b;
    if (!read_raw(path, b)) return false;
    v.resize(b.size() / sizeof(int32_t));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(int32_t));
    return true;
}

struct diff_t { double max_abs = 0.0; double mean_abs = 0.0; bool finite = true; };

diff_t compare(const std::vector<float> & a, const std::vector<float> & b) {
    diff_t d;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { d.finite = false; continue; }
        const double x = std::fabs((double) a[i] - (double) b[i]);
        if (x > d.max_abs) d.max_abs = x;
        sum += x;
    }
    d.mean_abs = n ? sum / (double) n : 0.0;
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <ref-dir> [--tol T] [-t threads]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';

    double tol = 2e-4;      // ggml f32 vs torch f32 across 12 layers
    int n_threads = 4;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
    }

    nllb_model model;
    std::string err;
    if (!nllb_model_load(model_path, model, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }
    std::printf("model: d_model=%d layers=%d heads=%d ffn=%d vocab=%d\n"
                "       eps=%g embd_scale=%g pos_offset=%d pad_id=%d\n\n",
                model.hp.d_model, model.hp.n_layer, model.hp.n_head, model.hp.n_ff,
                model.hp.n_vocab, model.hp.eps, model.hp.embd_scale,
                model.hp.pos_offset, model.hp.pad_id);

    std::ifstream mf(ref_dir + "manifest.txt");
    if (!mf) {
        std::fprintf(stderr, "no manifest.txt in %s\n"
            "  run: python tools/dump_reference.py <model> --out %s --per-layer\n",
            ref_dir.c_str(), ref_dir.c_str());
        return 2;
    }

    int n_cases = 0, n_fail = 0;
    double worst = 0.0;
    std::string worst_where;
    std::string line;

    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name;
        int T = 0, D = 0, n_hs = 0;
        ss >> name >> T >> D >> n_hs;
        if (name.empty()) continue;

        std::vector<int32_t> ids;
        std::vector<float>   want;
        if (!read_i32(ref_dir + name + ".ids.bin", ids) ||
            !read_f32(ref_dir + name + ".enc.bin", want)) {
            std::fprintf(stderr, "  %-20s SKIP (missing .bin)\n", name.c_str());
            continue;
        }
        if ((int) ids.size() != T || (int) want.size() != T * D) {
            std::fprintf(stderr, "  %-20s SKIP (manifest disagrees with .bin)\n",
                         name.c_str());
            continue;
        }
        if (D != model.hp.d_model) {
            std::fprintf(stderr, "  %-20s SKIP (d_model %d != model %d)\n",
                         name.c_str(), D, model.hp.d_model);
            continue;
        }

        nllb_encode_result res;
        if (!nllb_encode(model, ids, n_threads, n_hs > 0, res, err)) {
            std::fprintf(stderr, "  %-20s ERROR %s\n", name.c_str(), err.c_str());
            ++n_fail; ++n_cases;
            continue;
        }

        const diff_t d = compare(res.output, want);
        const bool ok = d.finite && d.max_abs <= tol;
        if (d.max_abs > worst) { worst = d.max_abs; worst_where = name + " (final)"; }
        ++n_cases; n_fail += !ok;

        std::printf("  %-20s T=%-3d max|d|=%.3e mean|d|=%.3e  %s\n",
                    name.c_str(), T, d.max_abs, d.mean_abs,
                    !d.finite ? "NON-FINITE" : (ok ? "ok" : "FAIL"));

        // Per-layer bisect. hidden_states[i] is the INPUT to block i for
        // i < n_layer, and the post-final-norm output at i == n_layer.
        // The first layer that goes red is where your graph diverges.
        if (!ok && n_hs > 0 && (int) res.hidden_states.size() == n_hs) {
            for (int j = 0; j < n_hs; ++j) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%s.layer_%02d.bin", name.c_str(), j);
                std::vector<float> ref_l;
                if (!read_f32(ref_dir + buf, ref_l)) continue;
                const diff_t dl = compare(res.hidden_states[j], ref_l);
                std::printf("      layer_%02d  max|d|=%.3e  %s%s\n", j, dl.max_abs,
                            dl.max_abs <= tol ? "ok" : "FAIL",
                            j == 0 ? "   (embeddings)"
                                   : (j == n_hs - 1 ? "   (after final norm)" : ""));
                if (dl.max_abs > tol) break;   // first divergence is the one that matters
            }
        }
    }

    std::printf("\n%d cases, %d failed. worst max|d| = %.3e (%s), tol = %.1e\n",
                n_cases, n_fail, worst, worst_where.c_str(), tol);
    if (n_cases == 0) { std::fprintf(stderr, "no cases ran\n"); return 2; }
    if (n_fail)       { std::printf("MISMATCH\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
