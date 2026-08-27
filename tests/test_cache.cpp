// Diff the KV-cached decoder against the un-cached one. No goldens, no torch.
//
//   ./build/bin/nllb-test-cache models/nllb-600M-f32.gguf ref/ [-t 8]
//
// WHY THIS GATE DOES NOT LOOK AT HF.
// The un-cached path is already proven exact against `generate(num_beams=1,
// do_sample=False)` on the real 600M - that is what nllb-test-generate does, and
// it is the whole reason step 4 came before step 5. Diffing the cache against
// torch would test both at once and leave a failure ambiguous between "the
// cache is wrong" and "the graph is wrong". Diffing it against the un-cached
// path isolates it: everything the two share cancels, and what is left is
// exactly the cache. Correctness against HF then follows by composition, and
// nllb-test-generate is the half that keeps holding up its end.
//
// The practical consequence is that this gate needs no reference dump, no
// python and no download - only a GGUF and some source ids. It reads
// manifest_gen.txt purely to reuse the same nine sources and the same
// forced_bos / max_new settings the generation gate runs on.
//
// WHAT IS COMPARED, in order of how sharply it localises a bug:
//
//   [gen]     un-cached generate vs cached generate. Ids are exact or they are
//             not. This is the outcome anyone actually cares about, and the
//             worst possible diagnostic: a single flipped token 20 steps in
//             says nothing about which layer or which half of the cache.
//   [step]    the same prefix decoded one token at a time through the cache,
//             against one teacher-forced un-cached pass, compared per layer.
//             A cross-attention cache bug shows up at every layer from 0; a
//             self-attention bug shows up from the layer it was introduced in;
//             a position bug shows up at layer 0, in the embeddings.
//   [chunk]   the same prefix fed to the cache in chunks of more than one
//             token. This is the ONLY thing here that exercises the causal
//             mask's n_past offset: at Tn == 1 ggml_diag_mask_inf(kq, n_past)
//             masks nothing whatever n_past is, so single-token generation
//             cannot tell a correct offset from a missing one.
//
// TOLERANCES ARE RELATIVE, for the reason the decoder gate documents at
// length: NLLB is pre-layernorm, the residual stream reaches ~4000 by block 4
// on the real 600M, and one float32 ULP there is ~5e-4.
//
// BIT-IDENTITY IS REQUIRED OF `full` AND OF NOTHING ELSE, and the distinction
// is the point rather than a hedge. `full` runs the same sequence length as the
// reference, so no addition can regroup and the only difference left is the
// cache: 0 differing floats, asserted. The incremental passes run at a
// DIFFERENT length by construction, and attention at position t sums over n_kv
// terms where n_kv is that length - the extra terms carry softmax weight
// exactly 0, which is mathematically nothing and arithmetically a different
// grouping of the same additions. So they land a ULP or so away, and `grow`
// exists to show that the un-cached decoder does exactly the same thing to
// itself when you vary the length. An earlier draft of this file asserted
// bit-identity everywhere; that was wrong, and `grow` is what proved it.
//
// Exits non-zero on mismatch so it drops straight into CI.

#include "nllb.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool read_i32(const std::string & path, std::vector<int32_t> & v) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<char> b((size_t) n);
    if (!f.read(b.data(), n)) return false;
    v.resize(b.size() / sizeof(int32_t));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(int32_t));
    return true;
}

struct diff_t {
    double max_abs = 0.0;
    double mag     = 0.0;   // max |reference|
    double rel     = 0.0;   // max_abs / max(mag, 1)
    long   ndiff   = 0;     // floats that are not bit-for-bit equal
    long   ndiff_cmp = 0;   // ... of the first n_cmp only. See below.
    bool   finite  = true;
};

// `ref` is the un-cached side. As in the decoder gate, the magnitude comes from
// the reference only, so a blown-up cached result cannot enlarge its own
// denominator and hide underneath it.
//
// `n_cmp` exists because the raw ndiff of the four passes is NOT comparable
// between them. `grow`'s last iteration decodes the entire prefix, which is
// bit-for-bit the same call as the reference, so its final row and all of its
// logits are equal BY CONSTRUCTION and contribute nothing. `step` and `chunk`
// carry their accumulated drift into that same final row, so theirs do differ -
// and on the 600M the logits alone are 256206 floats, which swamps everything
// else and makes the incremental passes look an order of magnitude worse than
// the control when the relative errors are in fact identical.
//
// So the reported comparison counts only the first `n_cmp` floats: hidden-state
// rows 0..T-2, logits excluded. That population means the same thing for all
// four passes. `ndiff` keeps the complete count, which is what `full`'s
// bit-identity assertion needs.
void accumulate(diff_t & d, const float * got, const float * ref,
                size_t n, size_t n_cmp = 0) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { d.finite = false; continue; }
        if (got[i] != ref[i]) { ++d.ndiff; if (i < n_cmp) ++d.ndiff_cmp; }
        const double x = std::fabs((double) got[i] - (double) ref[i]);
        if (x > d.max_abs) d.max_abs = x;
        const double m = std::fabs((double) ref[i]);
        if (m > d.mag) d.mag = m;
    }
    d.rel = d.max_abs / (d.mag > 1.0 ? d.mag : 1.0);
}

int argmax_of(const float * p, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return (int) best;
}

double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// Feed `ids` through a fresh cache in chunks of `chunk` tokens, collecting the
// per-layer hidden state of every token and the logits of the last one.
// chunk == 1 is the generation path; chunk > 1 additionally exercises the
// causal mask offset and the multi-row cache write.
bool run_cached(const nllb_model & model, const nllb_encode_result & enc,
                const std::vector<int32_t> & ids, int chunk, int n_threads,
                std::vector<std::vector<float>> & hs_out,   // [L+1][Td*D]
                std::vector<float> & logits_last,
                std::string & err, bool reuse_scratch = true) {
    const int Td = (int) ids.size();
    const int D  = model.hp.d_model;
    const int L  = model.hp.n_dec_layer;

    nllb_kv_cache cache;
    if (!nllb_kv_cache_init(model, enc, Td, n_threads, cache, err)) return false;
    cache.reuse_scratch = reuse_scratch;

    hs_out.assign((size_t) L + 1, std::vector<float>((size_t) Td * D, 0.0f));
    logits_last.clear();

    for (int i = 0; i < Td; ) {
        const int tn = (i + chunk <= Td) ? chunk : (Td - i);
        const std::vector<int32_t> feed(ids.begin() + i, ids.begin() + i + tn);
        nllb_decode_result r;
        if (!nllb_decode_cached(model, cache, feed, n_threads,
                                /*want_hidden_states=*/true, NLLB_LOGITS_LAST,
                                r, err)) {
            return false;
        }
        if ((int) r.hidden_states.size() != L + 1) {
            err = "cached decode returned " + std::to_string(r.hidden_states.size()) +
                  " hidden states, expected " + std::to_string(L + 1);
            return false;
        }
        for (int h = 0; h <= L; ++h) {
            std::memcpy(hs_out[(size_t) h].data() + (size_t) i * D,
                        r.hidden_states[(size_t) h].data(),
                        (size_t) tn * D * sizeof(float));
        }
        if (i + tn == Td) logits_last = r.logits;
        i += tn;
    }
    return true;
}

// THE CONTROL. No cache anywhere: decode prefixes of length 1, 2, ... Td with
// the ordinary un-cached decoder and keep the LAST row of each, which is the
// row for token t. Compared against one full-length un-cached pass, this
// measures how much simply *changing the sequence length* moves the numbers -
// and it moves them, because the attention output at position t is a sum over
// n_kv terms and n_kv is the sequence length. In the full pass the extra terms
// carry softmax weight 0, which is mathematically nothing and arithmetically a
// different grouping of the same additions.
//
// This is what the un-cached generation loop already did at every step in step
// 4, so it is not new error - it is the error that was always there, now
// measured. Without it, the cache would be blamed for it.
bool run_uncached_growing(const nllb_model & model, const nllb_encode_result & enc,
                          const std::vector<int32_t> & ids,
                          const std::vector<int32_t> & src_ids, int n_threads,
                          std::vector<std::vector<float>> & hs_out,
                          std::vector<float> & logits_last,
                          std::string & err) {
    const int Td = (int) ids.size();
    const int D  = model.hp.d_model;
    const int L  = model.hp.n_dec_layer;

    hs_out.assign((size_t) L + 1, std::vector<float>((size_t) Td * D, 0.0f));
    logits_last.clear();

    for (int t = 0; t < Td; ++t) {
        const std::vector<int32_t> prefix(ids.begin(), ids.begin() + t + 1);
        nllb_decode_result r;
        if (!nllb_decode(model, enc, prefix, src_ids, n_threads,
                         /*want_hidden_states=*/true, NLLB_LOGITS_LAST,
                         r, err, NLLB_POS_ABSOLUTE)) {
            return false;
        }
        for (int h = 0; h <= L; ++h) {
            std::memcpy(hs_out[(size_t) h].data() + (size_t) t * D,
                        r.hidden_states[(size_t) h].data() + (size_t) t * D,
                        (size_t) D * sizeof(float));
        }
        if (t == Td - 1) logits_last = r.logits;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <ref-dir> [-t threads] [--tol T] [--logit-tol T]\n"
            "       [--chunk N] [--forced-bos ID] [--max-new N] [-v]\n"
            "  the reference is the UN-CACHED path, not HF. --tol/--logit-tol are\n"
            "  relative (max|d| / max|uncached|); both sides should in fact be\n"
            "  bit-identical, which the summary reports separately.\n"
            "  --chunk: tokens per cached call in the [chunk] comparison (default 3).\n"
            "           This is what exercises the causal mask's n_past offset; at\n"
            "           chunk 1 the mask is vacuous and cannot be wrong.\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';

    int    n_threads  = 4;
    int    chunk      = 3;
    int    forced_bos = -2;    // -2 = take it from the manifest
    int    max_new    = -1;
    double tol        = 5e-5;
    double logit_tol  = 1e-4;
    bool   verbose    = false;
    for (int i = 3; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "-t")           && i + 1 < argc) n_threads  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chunk")      && i + 1 < argc) chunk      = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--forced-bos") && i + 1 < argc) forced_bos = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-new")    && i + 1 < argc) max_new    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol")        && i + 1 < argc) tol        = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--logit-tol")  && i + 1 < argc) logit_tol  = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v"))                           verbose    = true;
    }
    if (chunk < 1) chunk = 1;

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
    std::printf("model: d_model=%d dec_layers=%d dec_heads=%d vocab=%d "
                "dec_start=%d pad=%d eos=%d lm_head=%s\n"
                "reference: the UN-CACHED path. tolerances relative: states %.1e, "
                "logits %.1e. chunk=%d\n\n",
                model.hp.d_model, model.hp.n_dec_layer, model.hp.n_dec_head,
                model.hp.n_vocab, model.hp.dec_start_id, model.hp.pad_id,
                model.hp.eos_id, model.output_w ? "untied" : "tied",
                tol, logit_tol, chunk);

    std::ifstream mf(ref_dir + "manifest_gen.txt");
    if (!mf) {
        std::fprintf(stderr,
            "no manifest_gen.txt in %s\n"
            "  this gate compares two C++ paths and needs no goldens, but it does\n"
            "  need source ids and the same generation settings the other gate\n"
            "  uses. Regenerate:\n"
            "    python tools/dump_reference.py         <model> --out %s --per-layer\n"
            "    python tools/dump_decoder_reference.py <model> --out %s --per-layer --greedy\n",
            ref_dir.c_str(), ref_dir.c_str(), ref_dir.c_str());
        return 2;
    }

    const int L = model.hp.n_dec_layer;
    const int D = model.hp.d_model;

    int n_cases = 0, n_fail = 0, n_skip = 0, n_lines = 0;
    double worst_state = 0.0, worst_logit = 0.0, worst_grow = 0.0;
    long   total_ndiff = 0;
    long   arena_ndiff = 0;
    std::string worst_where;
    double t_uncached = 0.0, t_cached = 0.0;
    int    n_tok_timed = 0;
    std::string line;

    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name;
        int Tg = 0, m_bos = -1, m_max_new = 0, m_start = -1;
        const bool have_start = (bool) (ss >> name >> Tg >> m_bos >> m_max_new >> m_start);
        if (name.empty()) continue;
        ++n_lines;
        if (!have_start) {
            std::printf("  %-20s SKIP (manifest_gen.txt line is malformed)\n", name.c_str());
            ++n_skip;
            continue;
        }

        std::vector<int32_t> src_ids;
        if (!read_i32(ref_dir + name + ".ids.bin", src_ids)) {
            std::printf("  %-20s SKIP (missing .ids.bin)\n", name.c_str());
            ++n_skip;
            continue;
        }

        ++n_cases;
        bool case_ok = true;

        nllb_generate_params gp;
        gp.max_new_tokens = max_new > 0 ? max_new : m_max_new;
        gp.forced_bos_id  = forced_bos != -2 ? forced_bos : m_bos;
        gp.dec_start_id   = m_start;

        // ---- [gen] the outcome: same ids out of both paths? ----
        nllb_generate_result A, B;
        gp.use_cache = false;
        auto t0 = std::chrono::steady_clock::now();
        if (!nllb_generate(model, src_ids, gp, n_threads, A, err)) {
            std::fprintf(stderr, "  %-20s ERROR (un-cached) %s\n", name.c_str(), err.c_str());
            ++n_fail; continue;
        }
        const double dt_un = ms_since(t0);

        gp.use_cache = true;
        t0 = std::chrono::steady_clock::now();
        if (!nllb_generate(model, src_ids, gp, n_threads, B, err)) {
            std::fprintf(stderr, "  %-20s ERROR (cached) %s\n", name.c_str(), err.c_str());
            ++n_fail; continue;
        }
        const double dt_ca = ms_since(t0);
        t_uncached += dt_un;
        t_cached   += dt_ca;
        n_tok_timed += (int) A.ids.size();

        if (!B.used_cache) {
            std::printf("  %-20s FAIL: use_cache=true but the result says it did "
                        "not run the cached path\n", name.c_str());
            case_ok = false;
        }

        size_t k = 0;
        const size_t n_common = A.ids.size() < B.ids.size() ? A.ids.size() : B.ids.size();
        while (k < n_common && A.ids[k] == B.ids[k]) ++k;
        const bool ids_same = A.ids.size() == B.ids.size() && k == A.ids.size();

        if (!ids_same) {
            case_ok = false;
            std::printf("  %-20s IDS DIFFER: un-cached %zu tokens, cached %zu\n",
                        name.c_str(), A.ids.size(), B.ids.size());
            if (k >= n_common) {
                std::printf("      prefixes agree for all %zu common tokens; only the "
                            "length differs.\n"
                            "      -> a stopping-rule difference between the two "
                            "loops, not a cache arithmetic bug.\n", n_common);
            } else {
                std::printf("      first difference at position %zu (step %zu): "
                            "un-cached %d, cached %d\n",
                            k, k - 1, A.ids[k], B.ids[k]);
                if (k - 1 < A.steps.size() && k - 1 < B.steps.size()) {
                    std::printf("      margins there: un-cached %.6e, cached %.6e\n"
                                "      -> a margin near float noise means the cache is "
                                "arithmetically fine and one\n"
                                "         rounding difference tipped a coin toss; a "
                                "wide margin means it is not.\n"
                                "         Either way read the [step] line below, which "
                                "localises it to a layer.\n",
                                A.steps[k - 1].margin, B.steps[k - 1].margin);
                }
            }
        }

        // Per-step trace agreement, independent of the ids: the unforced argmax
        // is what the loop actually decided, and comparing it catches a cache
        // that produces the right token for the wrong reason.
        const size_t n_steps = A.steps.size() < B.steps.size() ? A.steps.size() : B.steps.size();
        for (size_t s = 0; s < n_steps; ++s) {
            if (A.steps[s].argmax != B.steps[s].argmax) {
                std::printf("  %-20s step %zu: unforced argmax differs, "
                            "un-cached %d (margin %.3e) vs cached %d\n",
                            name.c_str(), s, A.steps[s].argmax,
                            A.steps[s].margin, B.steps[s].argmax);
                case_ok = false;
                break;
            }
        }

        // ---- [step] / [chunk] per-layer bisect on the agreed prefix ----
        //
        // Teacher-force the whole emitted sequence through the un-cached
        // decoder once, then reproduce it through the cache twice: one token
        // per call, and `chunk` tokens per call. NLLB_POS_ABSOLUTE on the
        // un-cached side, because that is the rule the cache implements and
        // the only one it can implement - diffing against NLLB_POS_CUMSUM here
        // would report a position-rule disagreement as a cache bug, which is
        // precisely the trap this step was warned about.
        // Truncate to n_ctx before teacher-forcing. When generation stops on
        // NLLB_STOP_CONTEXT the last id was emitted but never fed back in, so
        // the sequence is one longer than anything nllb_decode will accept -
        // and the bisect would report "target sequence too long" for all nine
        // cases while the generation comparison above had in fact succeeded.
        std::vector<int32_t> prefix = A.ids;
        if (model.hp.n_ctx > 0 && (int) prefix.size() > model.hp.n_ctx) {
            prefix.resize((size_t) model.hp.n_ctx);
        }
        nllb_encode_result enc;
        if (!nllb_encode(model, src_ids, n_threads, false, enc, err)) {
            std::fprintf(stderr, "  %-20s ERROR (encode) %s\n", name.c_str(), err.c_str());
            ++n_fail; continue;
        }

        nllb_decode_result ref;
        if (!nllb_decode(model, enc, prefix, src_ids, n_threads,
                         /*want_hidden_states=*/true, NLLB_LOGITS_LAST,
                         ref, err, NLLB_POS_ABSOLUTE)) {
            std::fprintf(stderr, "  %-20s ERROR (un-cached decode) %s\n",
                         name.c_str(), err.c_str());
            ++n_fail; continue;
        }

        // Four passes. `full` is held to BIT-IDENTITY and everything else to a
        // tolerance, for a reason worth stating plainly: `full` is the same
        // sequence length as the reference, so nothing about the arithmetic
        // changes and the only difference left is the cache itself. The moment
        // the length changes - which is the entire point of an incremental
        // decoder - float addition regroups, and `grow` measures exactly how
        // much of that happens with no cache in sight.
        enum { P_FULL = 0, P_STEP, P_CHUNK, P_GROW, P_N };
        struct pass_t { const char * tag; int chunk; bool cached; bool exact; };
        const pass_t passes[P_N] = {
            { "full",  (int) prefix.size(), true,  true  },
            { "step",  1,                   true,  false },
            { "chunk", chunk,               true,  false },
            { "grow",  0,                   false, false },
        };

        double pass_rel[P_N]   = { 0.0, 0.0, 0.0, 0.0 };
        long   pass_ndiff[P_N] = { 0, 0, 0, 0 };   // complete count
        long   pass_cmp[P_N]   = { 0, 0, 0, 0 };   // comparable count, see accumulate()
        bool   pass_ran[P_N]   = { false, false, false, false };

        // Hidden-state rows 0..T-2: the floats every pass computes the same way,
        // and the only population whose counts mean the same thing across all
        // four. At T == 1 it is empty and the comparison is simply not reported.
        const size_t n_cmp_rows = prefix.size() > 1 ? prefix.size() - 1 : 0;
        const size_t n_cmp      = n_cmp_rows * (size_t) D;
        const long   n_cmp_pop  = (long) n_cmp * (L + 1);

        for (int pi = 0; pi < P_N; ++pi) {
            const pass_t & pass = passes[pi];
            // chunk == 1 or chunk >= T would just repeat step/full.
            if (pi == P_CHUNK &&
                (pass.chunk <= 1 || pass.chunk >= (int) prefix.size())) continue;

            std::vector<std::vector<float>> hs;
            std::vector<float> lg;
            const bool ok = pass.cached
                ? run_cached(model, enc, prefix, pass.chunk, n_threads, hs, lg, err)
                : run_uncached_growing(model, enc, prefix, src_ids, n_threads, hs, lg, err);
            if (!ok) {
                std::printf("  %-20s [%s] ERROR %s\n", name.c_str(), pass.tag, err.c_str());
                case_ok = false;
                continue;
            }
            pass_ran[pi] = true;

            int bad_layer = -1;
            for (int h = 0; h <= L; ++h) {
                diff_t d;
                accumulate(d, hs[(size_t) h].data(), ref.hidden_states[(size_t) h].data(),
                           (size_t) prefix.size() * D, n_cmp);
                pass_ndiff[pi] += d.ndiff;
                pass_cmp[pi]   += d.ndiff_cmp;
                if (d.rel > pass_rel[pi]) pass_rel[pi] = d.rel;
                const bool bad = !d.finite || d.rel > tol ||
                                 (pass.exact && d.ndiff != 0);
                if (bad && bad_layer < 0) {
                    bad_layer = h;
                    std::printf("  %-20s [%s] FIRST BAD LAYER %d: rel=%.3e "
                                "max|d|=%.3e over |ref| up to %.3e, %ld floats differ%s\n",
                                name.c_str(), pass.tag, h, d.rel, d.max_abs, d.mag,
                                d.ndiff, d.finite ? "" : "  (NON-FINITE)");
                    if (pass.exact && d.finite && d.rel <= tol) {
                        std::printf("      -> this pass feeds the whole prefix in ONE "
                                    "cached call, so it is the same graph as the\n"
                                    "         un-cached reference with K/V routed "
                                    "through the cache. It must be bit-identical.\n"
                                    "         A small non-zero difference here means "
                                    "the cache changed the arithmetic:\n"
                                    "         a wrong layout, a stale slot, or a "
                                    "read that overlaps the write.\n");
                    } else {
                        std::printf("      -> layer 0 is the embeddings: suspect the "
                                    "position rule, not the cache tensors.\n"
                                    "         layer 0 clean and every later layer bad: "
                                    "cross-attention K/V.\n"
                                    "         clean up to layer %d then bad: "
                                    "self-attention K/V, from that block on.\n", h);
                    }
                    case_ok = false;
                }
            }

            // Logits of the final position: the only number greedy reads.
            if (lg.size() != ref.logits.size()) {
                std::printf("  %-20s [%s] logits shape differs: %zu vs %zu\n",
                            name.c_str(), pass.tag, lg.size(), ref.logits.size());
                case_ok = false;
            } else if (!lg.empty()) {
                diff_t dl;
                accumulate(dl, lg.data(), ref.logits.data(), lg.size());
                pass_ndiff[pi] += dl.ndiff;
                if (dl.rel > worst_logit) worst_logit = dl.rel;
                const int a_ref = argmax_of(ref.logits.data(), ref.logits.size());
                const int a_got = argmax_of(lg.data(), lg.size());
                if (!dl.finite || dl.rel > logit_tol || a_ref != a_got ||
                    (pass.exact && dl.ndiff != 0)) {
                    std::printf("  %-20s [%s] LOGITS: rel=%.3e argmax %d vs %d, "
                                "%ld floats differ%s\n",
                                name.c_str(), pass.tag, dl.rel, a_ref, a_got,
                                dl.ndiff, dl.finite ? "" : "  (NON-FINITE)");
                    case_ok = false;
                }
            }
        }

        // ---- [arena] does reusing the graph arena change the answer? ----
        //
        // This one does NOT compare against the un-cached path, and that is the
        // point. nllb_kv_cache::scratch keeps one ggml arena across steps
        // instead of mallocing a fresh one per token, so the graph now runs in
        // memory the PREVIOUS step wrote. A fresh mmap reads as zero; a reused
        // arena reads as whatever was there. Any op that reads graph memory it
        // has not written therefore goes from silently seeing 0.0 to seeing
        // last step's values - a class of bug that allocating fresh step was
        // accidentally papering over.
        //
        // None of the four passes above can see that. `full` is a single call
        // per cache, so its arena is fresh no matter what this flag says, and
        // `step` and `chunk` are held to a tolerance that a small stale read
        // could hide under. So: run the identical stepping pass twice, once
        // with reuse and once without, and hold the two to BIT-IDENTITY. The
        // arithmetic is the same graph on the same inputs; the arena is the
        // only difference, and an allocator has no business changing a float.
        {
            std::vector<std::vector<float>> hs_on, hs_off;
            std::vector<float> lg_on, lg_off;
            const bool ok_on  = run_cached(model, enc, prefix, 1, n_threads,
                                           hs_on, lg_on, err, /*reuse=*/true);
            const bool ok_off = ok_on && run_cached(model, enc, prefix, 1, n_threads,
                                                    hs_off, lg_off, err, /*reuse=*/false);
            if (!ok_off) {
                std::printf("  %-20s [arena] ERROR %s\n", name.c_str(), err.c_str());
                case_ok = false;
            } else {
                long ndiff = 0;
                int  bad_layer = -1;
                for (int h = 0; h <= L; ++h) {
                    diff_t d;
                    accumulate(d, hs_on[(size_t) h].data(), hs_off[(size_t) h].data(),
                               (size_t) prefix.size() * D);
                    ndiff += d.ndiff;
                    if (d.ndiff && bad_layer < 0) bad_layer = h;
                }
                diff_t dl;
                if (lg_on.size() == lg_off.size() && !lg_on.empty()) {
                    accumulate(dl, lg_on.data(), lg_off.data(), lg_on.size());
                    ndiff += dl.ndiff;
                }
                if (ndiff != 0) {
                    std::printf("  %-20s [arena] %ld floats differ between a REUSED "
                                "and a FRESH graph arena, first at layer %d\n"
                                "      -> the two runs are the same graph on the same "
                                "inputs; only the allocation differs.\n"
                                "         A difference means something reads graph "
                                "memory it did not write, and was\n"
                                "         reading the zeros a fresh mmap happens to "
                                "hand back. Set reuse_scratch = false\n"
                                "         to confirm, then find the read.\n",
                                name.c_str(), ndiff, bad_layer);
                    case_ok = false;
                }
                arena_ndiff += ndiff;
            }
        }

        if (pass_rel[P_STEP] > worst_state) {
            worst_state = pass_rel[P_STEP];
            worst_where = name;
        }
        total_ndiff += pass_ndiff[P_FULL];
        if (pass_rel[P_GROW] > worst_grow) worst_grow = pass_rel[P_GROW];

        std::printf("  %-20s T=%-3zu ids=%-4s full=%s step=%.2e chunk=%.2e "
                    "grow=%.2e  %5.0f->%5.0fms (%.1fx)  %s\n",
                    name.c_str(), A.ids.size(), ids_same ? "same" : "DIFF",
                    pass_ndiff[P_FULL] == 0 ? "exact" : "DIFFERS",
                    pass_rel[P_STEP],
                    pass_ran[P_CHUNK] ? pass_rel[P_CHUNK] : 0.0,
                    pass_rel[P_GROW],
                    dt_un, dt_ca, dt_ca > 0.0 ? dt_un / dt_ca : 0.0,
                    case_ok ? "ok" : "FAIL");
        if (verbose && n_cmp_pop > 0) {
            // Comparable population only - hidden-state rows 0..T-2, logits
            // excluded. Counting everything would put grow at an unfair
            // advantage: its final row and its logits are the reference itself.
            // See accumulate() for the full reasoning.
            std::printf("      differing floats over %d layers x %zu rows "
                        "(logits and the final row excluded, see accumulate):\n"
                        "        step=%ld  chunk=%ld  grow=%ld   of %ld  "
                        "[full=%ld over everything, and must be 0]\n",
                        L + 1, n_cmp_rows,
                        pass_cmp[P_STEP], pass_cmp[P_CHUNK], pass_cmp[P_GROW],
                        n_cmp_pop, pass_ndiff[P_FULL]);
        }
        if (!case_ok) ++n_fail;
    }

    std::printf("\n%d of %d manifest cases ran, %d failed.\n", n_cases, n_lines, n_fail);
    if (n_cases) {
        std::printf("worst incremental state rel = %.3e (%s), worst logit rel = %.3e\n",
                    worst_state, worst_where.empty() ? "-" : worst_where.c_str(),
                    worst_logit);
        std::printf("control: un-cached decoding of GROWING prefixes moves the same "
                    "states by %.3e\n"
                    "  with no cache involved at all. Read the two together: the "
                    "cache is only responsible\n"
                    "  for whatever the first number has that the second does not.\n",
                    worst_grow);
        if (total_ndiff == 0) {
            std::printf("one cached call over the whole prefix is BIT-IDENTICAL to "
                        "the un-cached pass -\n"
                        "  0 differing floats. Same sequence length, so the "
                        "arithmetic cannot regroup, and\n"
                        "  the cache's layout, writes and reads are therefore exact "
                        "rather than merely close.\n");
        } else {
            std::printf("%ld floats differ in the SAME-LENGTH comparison. That one is "
                        "supposed to be exact:\n"
                        "  identical arithmetic, K/V merely routed through the cache. "
                        "A non-zero count here is\n"
                        "  a cache bug however small it looks. Do not ship on the "
                        "tolerance being green.\n", total_ndiff);
        }
        if (arena_ndiff == 0) {
            std::printf("reusing the graph arena across steps is BIT-IDENTICAL to "
                        "allocating a fresh one -\n"
                        "  0 differing floats. Nothing in the graph reads memory it "
                        "did not write, which is\n"
                        "  what makes keeping the arena safe rather than merely "
                        "faster.\n");
        } else {
            std::printf("%ld floats differ between a reused and a fresh graph arena. "
                        "See the [arena] lines.\n", arena_ndiff);
        }
        if (t_cached > 0.0) {
            std::printf("generation wall clock over all cases: un-cached %.0f ms, "
                        "cached %.0f ms (%.2fx), %d tokens\n",
                        t_uncached, t_cached, t_uncached / t_cached, n_tok_timed);
        }
    }
    if (n_cases == 0) { std::fprintf(stderr, "no cases ran\n"); return 2; }
    if (n_skip) {
        std::printf("%d case%s SKIPPED - the manifest or the .ids.bin files are "
                    "incomplete.\n"
                    "  regenerate them; a gate that silently shrinks is worse than "
                    "one that fails.\n",
                    n_skip, n_skip == 1 ? "" : "s");
        return 1;
    }
    if (n_fail) { std::printf("MISMATCH\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
