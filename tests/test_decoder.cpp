// Diff the ggml decoder against the HF goldens in ref/.
//
//   ./build/bin/nllb-test-decoder models/nllb-600M-f32.gguf ref/ [--tol 5e-5] [-t 4]
//
// Reads manifest_dec.txt, written by tools/dump_decoder_reference.py. Each case
// runs the *real* pipeline: nllb_encode on the source ids, then nllb_decode on
// the teacher-forced target ids against that encoder output. The decoder
// therefore inherits the encoder's ~1e-6 error, which is the point - it is the
// same composition the translator will do.
//
// TOLERANCES ARE RELATIVE. NLLB is pre-layernorm, so the residual stream is
// never normalised between blocks and reaches ~4000 by block 4 on the real
// 600M. One float32 ULP there is ~5e-4, so an absolute threshold tuned on the
// encoder's O(1) post-layernorm output is below what float32 can represent and
// the per-layer bisect fires on nothing but rounding. The final output and the
// logits happen to be O(1) and survive an absolute check; the intermediate
// states this test bisects through do not.
//
// Exits non-zero on mismatch so it drops straight into CI.

#include "nllb.h"

#include "ggml.h"

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

struct diff_t {
    double max_abs  = 0.0;
    double mean_abs = 0.0;
    double mag      = 0.0;   // max |reference|
    double rel      = 0.0;   // max_abs / max(mag, 1)
    bool   finite   = true;
};

// `ref` is the golden. The magnitude comes from it, never from the computed
// side - a blown-up result must not be allowed to enlarge its own denominator
// and hide underneath it.
diff_t compare(const std::vector<float> & got, const std::vector<float> & ref) {
    diff_t d;
    const size_t n = got.size() < ref.size() ? got.size() : ref.size();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { d.finite = false; continue; }
        const double x = std::fabs((double) got[i] - (double) ref[i]);
        if (x > d.max_abs) d.max_abs = x;
        const double m = std::fabs((double) ref[i]);
        if (m > d.mag) d.mag = m;
        sum += x;
    }
    d.mean_abs = n ? sum / (double) n : 0.0;
    d.rel = d.max_abs / (d.mag > 1.0 ? d.mag : 1.0);
    return d;
}

int argmax(const float * p, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return (int) best;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <ref-dir> [--tol T] [--logit-tol T] [-t threads]\n"
            "  --tol/--logit-tol are RELATIVE (max|d| / max|ref|)\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';

    // Relative. The numpy gate measures 1.0e-06 on hidden states and 1.6e-06 on
    // logits against the real 600M; ggml adds its own kernel differences on top,
    // and the encoder's ~2.4e-06 composes in as well. A real bug lands at 1e-2,
    // so there is a wide gap either side. Tighten these once the real model has
    // been measured through this path.
    double tol = 5e-5;
    double logit_tol = 1e-4;
    int n_threads = 4;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--logit-tol") && i + 1 < argc) logit_tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
    }

    nllb_model model;
    std::string err;
    if (!nllb_model_load(model_path, model, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }
    if (!model.has_decoder) {
        std::fprintf(stderr, "this GGUF has no decoder "
                             "(converted with --encoder-only?)\n");
        return 2;
    }
    std::printf("model: d_model=%d dec_layers=%d dec_heads=%d dec_ffn=%d vocab=%d\n"
                "       eps=%g embd_scale=%g dec_start=%d pad=%d eos=%d lm_head=%s\n"
                "tolerances are RELATIVE: states %.1e, logits %.1e\n\n",
                model.hp.d_model, model.hp.n_dec_layer, model.hp.n_dec_head,
                model.hp.n_dec_ff, model.hp.n_vocab, model.hp.eps,
                model.hp.embd_scale, model.hp.dec_start_id, model.hp.pad_id,
                model.hp.eos_id, model.output_w ? "untied" : "tied",
                tol, logit_tol);

    // A red wall on an f16 checkpoint means "this is f16", not "this is
    // broken", and nothing in the output said so. The tolerances above are the
    // FLOAT32 noise floor - rel ~1e-7 is arithmetic, rel ~1e-2 is a bug - and
    // f16 sits at ~8e-04, three orders above the floor and four below a real
    // defect. Every case will fail and the failure carries no information.
    //
    // Loosening the tolerance to accommodate f16 would be worse than useless:
    // it would then pass a genuine graph error of the size this gate exists to
    // catch. So the gate keeps its f32 tolerances, keeps failing, and says why.
    if (model.lm_head()->type != GGML_TYPE_F32) {
        const char * ty = ggml_type_name(model.lm_head()->type);
        std::printf(
            "!! this GGUF stores its weights as %s, and this gate's tolerances "
            "are the FLOAT32\n"
            "   noise floor. Expect every case to FAIL at roughly 1e-03, which "
            "is what %s costs\n"
            "   and NOT evidence of a bug - a real defect lands at 1e-02 and "
            "flips the argmax.\n"
            "   Do not loosen the tolerance to make this green: at 1e-03 it "
            "would also pass a\n"
            "   genuine graph error. Verify reduced precision with the gate "
            "built for it, which\n"
            "   asserts on ids rather than on distances:\n"
            "     ./build/bin/nllb-test-precision <f32.gguf> %s <ref-dir>\n\n",
            ty, ty, model_path.c_str());
    }

    std::ifstream mf(ref_dir + "manifest_dec.txt");
    if (!mf) {
        std::fprintf(stderr, "no manifest_dec.txt in %s\n"
            "  run: python tools/dump_reference.py <model> --out %s --per-layer\n"
            "  then python tools/dump_decoder_reference.py <model> --out %s --per-layer\n",
            ref_dir.c_str(), ref_dir.c_str(), ref_dir.c_str());
        return 2;
    }

    int n_cases = 0, n_fail = 0;
    double worst = 0.0, worst_logit = 0.0, worst_mag = 0.0;
    std::string worst_where;
    std::string line;

    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name;
        int Td = 0, D = 0, n_hs = 0, vocab = 0;
        ss >> name >> Td >> D >> n_hs >> vocab;
        if (name.empty()) continue;

        std::vector<int32_t> src_ids, dec_ids;
        std::vector<float>   want;
        if (!read_i32(ref_dir + name + ".ids.bin", src_ids) ||
            !read_i32(ref_dir + name + ".dec_ids.bin", dec_ids) ||
            !read_f32(ref_dir + name + ".dec.bin", want)) {
            std::fprintf(stderr, "  %-20s SKIP (missing .bin)\n", name.c_str());
            continue;
        }
        if ((int) dec_ids.size() != Td || (int) want.size() != Td * D) {
            std::fprintf(stderr, "  %-20s SKIP (manifest disagrees with .bin)\n",
                         name.c_str());
            continue;
        }
        if (D != model.hp.d_model) {
            std::fprintf(stderr, "  %-20s SKIP (d_model %d != model %d)\n",
                         name.c_str(), D, model.hp.d_model);
            continue;
        }

        // The encoder half is already proven; run it for real anyway so the
        // decoder is diffed against the composition, not against a golden
        // encoder output it would never see at runtime.
        nllb_encode_result enc;
        if (!nllb_encode(model, src_ids, n_threads, false, enc, err)) {
            std::fprintf(stderr, "  %-20s ERROR (encode) %s\n", name.c_str(), err.c_str());
            ++n_fail; ++n_cases;
            continue;
        }

        std::vector<float> want_logits;
        const bool have_logits =
            read_f32(ref_dir + name + ".logits_last.bin", want_logits) &&
            (int) want_logits.size() == vocab;

        nllb_decode_result res;
        if (!nllb_decode(model, enc, dec_ids, src_ids, n_threads,
                         n_hs > 0, have_logits, res, err)) {
            std::fprintf(stderr, "  %-20s ERROR (decode) %s\n", name.c_str(), err.c_str());
            ++n_fail; ++n_cases;
            continue;
        }

        const diff_t d = compare(res.output, want);
        bool ok = d.finite && d.rel <= tol;
        if (d.rel > worst) { worst = d.rel; worst_where = name + " (final)"; }

        // Logits at the last step, plus argmax agreement - the only property
        // greedy decoding actually depends on. A diff inside tolerance that
        // still flips the argmax is a real failure.
        double dlog = 0.0;
        bool argmax_ok = true;
        if (have_logits && !res.logits.empty()) {
            const float * got_last = res.logits.data() + (size_t)(Td - 1) * res.n_vocab;
            std::vector<float> got_vec(got_last, got_last + res.n_vocab);
            const diff_t dl = compare(got_vec, want_logits);
            dlog = dl.rel;
            if (dlog > worst_logit) worst_logit = dlog;
            argmax_ok = argmax(got_vec.data(), got_vec.size()) ==
                        argmax(want_logits.data(), want_logits.size());
            ok = ok && dl.finite && dlog <= logit_tol && argmax_ok;
        }

        ++n_cases; n_fail += !ok;

        std::printf("  %-20s Td=%-3d rel=%.3e abs=%.3e", name.c_str(),
                    Td, d.rel, d.max_abs);
        if (have_logits) {
            std::printf(" logit=%.3e argmax=%s", dlog, argmax_ok ? "same" : "DIFFERS");
        }
        std::printf("  %s\n", !d.finite ? "NON-FINITE" : (ok ? "ok" : "FAIL"));

        // Per-layer bisect. hidden_states[i] is the INPUT to decoder block i
        // for i < n_dec_layer, and the post-dec.output_norm result at the end.
        // The first layer that goes red names the broken block. If dec_layer_00
        // is red the problem is in the embeddings or the position offset; if
        // every layer is green but the logits are red, it is the lm_head.
        //
        // |ref| is printed because these are raw residual-stream values, not
        // normalised ones. Seeing 4000 next to an absolute diff of 5e-4 is what
        // tells you the diff is one ULP rather than a bug.
        if (!ok && n_hs > 0 && (int) res.hidden_states.size() == n_hs) {
            for (int j = 0; j < n_hs; ++j) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%s.dec_layer_%02d.bin", name.c_str(), j);
                std::vector<float> ref_l;
                if (!read_f32(ref_dir + buf, ref_l)) continue;
                const diff_t dl = compare(res.hidden_states[j], ref_l);
                if (dl.mag > worst_mag) worst_mag = dl.mag;
                std::printf("      dec_layer_%02d  rel=%.3e abs=%.3e |ref|max=%.1f  %s%s\n",
                            j, dl.rel, dl.max_abs, dl.mag,
                            dl.rel <= tol ? "ok" : "FAIL",
                            j == 0 ? "   (embeddings)"
                                   : (j == n_hs - 1 ? "   (after dec.output_norm)" : ""));
                if (dl.rel > tol) break;   // first divergence is the one that matters
            }
        }
    }

    std::printf("\n%d cases, %d failed. worst rel = %.3e (%s), tol = %.1e\n",
                n_cases, n_fail, worst, worst_where.c_str(), tol);
    if (worst_logit > 0.0) {
        std::printf("worst logit rel = %.3e, tol = %.1e\n", worst_logit, logit_tol);
    }
    if (n_cases == 0) { std::fprintf(stderr, "no cases ran\n"); return 2; }
    if (n_fail) {
        std::printf("MISMATCH\n");
        std::printf("  before touching the graph, run:\n"
                    "    python tests/diagnose_decoder.py <hf-dir> <gguf>\n"
                    "  it reports ULPs per state, which separates a real "
                    "divergence from the float32 noise floor\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}