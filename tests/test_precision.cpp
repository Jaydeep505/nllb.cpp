// Diff one model against itself at a different precision. No torch.
//
//   ./build/bin/nllb-test-precision models/nllb-600M-f32.gguf
//                                   models/nllb-600M-f16.gguf ref/ [-t 8] [-v]
//
// WHY THE DECODER GATE CANNOT DO THIS JOB.
// tests/test_decoder.cpp holds hidden states to a RELATIVE tolerance chosen as
// the float32 noise floor - `rel ~1e-7` is arithmetic, `rel ~1e-2` is a bug.
// f16 lands at ~8e-04, three orders above that floor and four below a real bug,
// so pointing that gate at an f16 checkpoint produces a wall of red that means
// "this is f16" and reads as "this is broken". Loosening its tolerance to 1e-03
// would be worse: it would then pass a genuine graph error, which is precisely
// what the handoff means by "stay on f32 for verification".
//
// So f16 gets its own gate, and it asks a different question. Not "how far did
// the numbers move" - they moved, that is what f16 is - but:
//
//   [gen]     do the two precisions emit the SAME TOKEN IDS? Exact, no
//             tolerance. This is the only thing a user of the translator can
//             observe, and it is the assertion.
//   [state]   per-layer divergence, REPORTED and not gated. It exists to
//             localise a real bug if [gen] ever fails: f16 noise is spread
//             evenly across layers, a bug is not.
//   [margin]  the one that matters, and the reason this file exists.
//
// THE MARGIN AUDIT.
// The project's whole case for f16 has been two aggregate numbers next to each
// other: "the smallest greedy margin is 1.822e-02, f16's error is 8.9e-04, so
// there is 20x of headroom". That compares the WORST margin against the WORST
// error as if they occurred at the same decision, which nobody checked. It is
// an argument, not a measurement.
//
// This gate measures it per decision. Teacher-force the reference's own output
// through both models with logits at every position, and at each position t:
//
//     margin_t = top1 - top2                    (of the REFERENCE logits)
//     err_t    = max |candidate - reference|    (over that row)
//
// The argmax at t cannot change while, for EVERY other token i,
//
//     (logit[best] - logit[i])  >  err[best] + err[i]
//
// - the winner can only fall by its own error and a rival can only rise by
// its own. The gate reports the tightest such ratio over the whole row, which
// is exact rather than conservative: an earlier draft used the row's MAXIMUM
// error for both terms, which is an upper bound but overstates the error at
// the two tokens that are actually competing, and it cried wolf. Note the
// quantifier is over every token, not just the runner-up: a distant token with
// a large error can in principle overtake, and only checking top2 would miss
// it.
//
// Positions failing that test are counted and reported: they are decisions that
// AGREED BUT DID NOT HAVE TO, and a suite that only checks ids cannot see them.
// On the deep synthetic fixture there are many - its margins are genuinely
// below f16's error and the ids match by luck. That fixture therefore cannot
// support an f16 claim, however green it looks, and this gate is what says so.
//
// EXACT ID EQUALITY IS AN f16 ASSERTION, NOT A PRECISION ASSERTION (trap 30).
// It is what the paragraphs above describe, it held for f16 on the real 600M,
// and it is spent at Q8_0: one real-model greedy decision has already flipped
// while the translation stayed good. The fix is not a looser tolerance - there
// is no tolerance here to loosen, and trap 19 says what happens when you invent
// one. The fix is a DIFFERENT ASSERTION, and it has to be asked for by name:
//
//   [topk]  the reference's argmax must still be inside the candidate's top k.
//
// That is a property of the arithmetic rather than of the outcome, which is
// exactly what ids are not. A quantised model is allowed to reorder a near-tie
// - that is what "near-tie" means - but it is not allowed to move the winner
// out of the top few, because doing so is no longer rounding, it is a different
// model. k is small and fixed (default 4) so the claim stays falsifiable: at
// k = the vocabulary size it would assert nothing.
//
// --allow-id-drift is what switches the assertion over. WITHOUT it nothing
// changes: ids must match exactly, and every f16 number in claude/handoff.md
// reproduces. WITH it, id drift is reported and [topk] is what fails the run.
// The flag is deliberately not a default: a run that permits drift has to say
// so, so that a green line always names the claim it is green about.
//
// Read [topk] together with [margin]. They answer different questions and it is
// possible to pass one and fail the other:
//   - headroom > 1 says this decision COULD NOT have flipped. It is the strong
//     statement and it is the one that collapses under quantisation.
//   - rank 0 says it DID NOT flip; rank < k says that even where it flipped,
//     the two models still agree about which tokens are plausible.
// A run with headroom 0.06x and every rank 0 is a model whose decisions are all
// coin tosses that happened to land the same way. Say that, rather than PASS.
//
// THE KV CACHE IS THE OTHER PRECISION KNOB, and `--kv f32:f16` measures it with
// the same machinery. The weights' precision is a property of the file; the KV
// cache's is a property of the run, so that comparison passes the SAME file
// twice and differs only in nllb_kv_type. It also changes which decoder path
// the teacher-forced logits come from: with both sides at f32 KV the gate uses
// the un-cached nllb_decode, exactly as it always has, so every recorded number
// still reproduces; when the two KV types differ it uses nllb_decode_cached on
// both sides, because the un-cached path has no KV cache to be f16.
//
// Exit status: non-zero if the active id assertion fails, if any decision's
// rank reaches k, or if --strict and any decision was inside the error band.
// Unsafe-but-agreeing decisions are a loud warning by default rather than a
// failure, because they are a property of the model and the inputs rather than
// of the code.

#include "nllb.h"

#include "ggml.h"

#include <algorithm>
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

// top1 - top2 over one logit row, and the winning index. Strict `>` keeps the
// first maximum on a tie, matching torch.argmax and nllb_generate.
void margin_of(const float * p, int n, double & margin, int & best) {
    best = 0;
    int second = -1;
    for (int i = 1; i < n; ++i) {
        if (p[i] > p[best])                        { second = best; best = i; }
        else if (second < 0 || p[i] > p[second])   { second = i; }
    }
    margin = second >= 0 ? (double) p[best] - (double) p[second] : 0.0;
}

// How much room this decision had, exactly: the smallest value of
//   (ref[best] - ref[i]) / (err[best] + err[i])   over every i != best.
// Above 1.0 the argmax provably cannot change at this precision; at or below
// it, it could have. Returns +inf when the candidate reproduces the row bit
// for bit (no error anywhere), which is a real outcome on short rows.
double headroom_of(const float * pr, const float * pc, int n, int best) {
    const double e_best = std::fabs((double) pr[best] - (double) pc[best]);
    double worst = 1e30;
    for (int i = 0; i < n; ++i) {
        if (i == best) continue;
        const double gap   = (double) pr[best] - (double) pr[i];
        const double denom = e_best + std::fabs((double) pr[i] - (double) pc[i]);
        if (denom <= 0.0) continue;
        const double r = gap / denom;
        if (r < worst) worst = r;
    }
    return worst;
}

// Where the reference's winner sits in the CANDIDATE's ordering. 0 means the
// candidate agrees; k or more fails the [topk] assertion.
//
// The tie rule is the project's everywhere else (trap 11): equal logits are
// ordered by ascending id, because torch.argmax returns the first maximum and
// nllb_generate's strict `>` does the same. So the rank is the count of tokens
// that beat `best` outright plus the count that tie with it and sort earlier.
// Getting that wrong would report rank 1 for a decision the candidate would in
// fact have made, which is a gate lying in the direction that matters.
int rank_of(const float * pc, int n, int best) {
    const float v = pc[best];
    int r = 0;
    for (int i = 0; i < n; ++i) {
        if (pc[i] > v)                  ++r;
        else if (pc[i] == v && i < best) ++r;
    }
    return r;
}

const char * head_type(const nllb_model & m) {
    return ggml_type_name(m.lm_head()->type);
}

bool same_shape(const nllb_model & a, const nllb_model & b, std::string & err) {
    const nllb_hparams & x = a.hp;
    const nllb_hparams & y = b.hp;
    if (x.d_model != y.d_model || x.n_layer != y.n_layer ||
        x.n_dec_layer != y.n_dec_layer || x.n_head != y.n_head ||
        x.n_dec_head != y.n_dec_head || x.n_dec_ff != y.n_dec_ff ||
        a.lm_head()->ne[1] != b.lm_head()->ne[1]) {
        err = "the two GGUFs are not the same model at different precisions - "
              "shapes differ";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <reference.gguf> <candidate.gguf> <ref-dir> [-t threads]\n"
            "       [--forced-bos ID] [--max-new N] [--strict] [-v]\n"
            "       [--allow-id-drift] [--topk K] [--kv f32:f16] [--backend A:B]\n"
            "  Both files must be the SAME model converted at different\n"
            "  precisions - typically f32 and f16. The reference is whichever\n"
            "  one the project trusts; the candidate is the one being argued for.\n"
            "  --strict makes a decision inside the error band a FAILURE rather\n"
            "  than a warning.\n"
            "\n"
            "  --allow-id-drift  ids may move; the assertion becomes [topk].\n"
            "                    Required at Q8_0 and below, where exact id\n"
            "                    equality is spent (trap 30). Without it the ids\n"
            "                    are still the assertion and f16 runs are\n"
            "                    unchanged.\n"
            "  --topk K          k for [topk] (default 4). The reference's argmax\n"
            "                    must stay within the candidate's top K.\n"
            "  --kv A:B          KV cache type for reference:candidate, each f32\n"
            "                    or f16 (default f32:f32). With the two different,\n"
            "                    PASS THE SAME FILE TWICE - the weights are the\n"
            "                    constant and the cache is the variable. The\n"
            "                    teacher-forced logits then come from the CACHED\n"
            "                    decoder on both sides, because the un-cached one\n"
            "                    has no KV cache to be f16.\n"
            "  --backend A:B     backend for reference:candidate (step 13), e.g.\n"
            "                    cpu:metal. PASS THE SAME FILE TWICE: the weights\n"
            "                    are the constant and the KERNELS are the variable.\n"
            "                    This is the cross-backend gate - exact ids plus\n"
            "                    the margin audit, the same claim f16 makes.\n",
            argv[0]);
        return 2;
    }
    const std::string ref_path  = argv[1];
    const std::string cand_path = argv[2];
    std::string ref_dir = argv[3];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';

    int  n_threads  = 4;
    int  forced_bos = -2;
    int  max_new    = -1;
    int  topk       = 4;
    bool strict     = false;
    bool verbose    = false;
    bool allow_drift = false;
    int  kv_ref = NLLB_KV_F32, kv_cand = NLLB_KV_F32;
    std::string be_ref, be_cand;   // "" -> $NLLB_BACKEND, else cpu
    for (int i = 4; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "-t")           && i + 1 < argc) n_threads  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--forced-bos") && i + 1 < argc) forced_bos = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-new")    && i + 1 < argc) max_new    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--topk")       && i + 1 < argc) topk       = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--allow-id-drift"))             allow_drift = true;
        else if (!std::strcmp(argv[i], "--strict"))                     strict     = true;
        else if (!std::strcmp(argv[i], "-v"))                           verbose    = true;
        else if (!std::strcmp(argv[i], "--backend")    && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t colon = spec.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "--backend wants A:B, e.g. cpu:metal, got %s\n", spec.c_str());
                return 2;
            }
            be_ref  = spec.substr(0, colon);
            be_cand = spec.substr(colon + 1);
        }
        else if (!std::strcmp(argv[i], "--kv")         && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t colon = spec.find(':');
            const std::string a = colon == std::string::npos ? spec : spec.substr(0, colon);
            const std::string b = colon == std::string::npos ? spec : spec.substr(colon + 1);
            auto parse = [](const std::string & t, int & out) {
                if      (t == "f32") { out = NLLB_KV_F32; return true; }
                else if (t == "f16") { out = NLLB_KV_F16; return true; }
                return false;
            };
            if (!parse(a, kv_ref) || !parse(b, kv_cand)) {
                std::fprintf(stderr, "--kv wants f32 or f16 on each side, got %s\n",
                             spec.c_str());
                return 2;
            }
        }
    }
    if (topk < 1) topk = 1;
    // Both sides cached, or neither. The un-cached decoder has no KV cache, so
    // "f32 KV" there is not a configuration - it is the absence of one - and
    // routing one side through the cache and not the other would fold trap 16's
    // 4e-07 regrouping into a number this gate would then report as f16's cost.
    const bool kv_compare = (kv_ref != kv_cand);

    nllb_model ref, cand;
    std::string err;
    if (!nllb_model_load(ref_path, ref, err, true, be_ref)) {
        std::fprintf(stderr, "load failed (reference): %s\n", err.c_str());
        return 2;
    }
    if (!nllb_model_load(cand_path, cand, err, true, be_cand)) {
        std::fprintf(stderr, "load failed (candidate): %s\n", err.c_str());
        return 2;
    }
    if (!ref.has_decoder || !cand.has_decoder) {
        std::fprintf(stderr, "one of these GGUFs has no decoder\n");
        return 2;
    }
    if (!same_shape(ref, cand, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    // Two backends that resolved to the same device would be the same code on
    // the same bytes - vacuous, and the step-4 SKIP lesson says a vacuous
    // comparison must not print PASS.
    const bool be_compare = ref.backend_name != cand.backend_name;
    if (!be_ref.empty() && !be_compare) {
        std::fprintf(stderr,
            "--backend %s:%s resolved to %s on both sides, so there is nothing to\n"
            "compare. (Is the GPU backend in this build? nllb-cli --list-backends)\n",
            be_ref.c_str(), be_cand.c_str(), ref.backend_name.c_str());
        return 2;
    }
    if (std::strcmp(head_type(ref), head_type(cand)) == 0 && !kv_compare && !be_compare) {
        std::fprintf(stderr,
            "both files store the lm_head as %s and --kv is f32:f32, so this\n"
            "comparison has nothing to measure. Convert one with --outtype f16,\n"
            "pass --kv f32:f16 to measure the CACHE's precision, or --backend\n"
            "cpu:metal to measure the KERNELS.\n",
            head_type(ref));
        return 2;
    }

    const int L       = ref.hp.n_dec_layer;
    const int D       = ref.hp.d_model;
    const int n_vocab = (int) ref.lm_head()->ne[1];

    const char * kv_name[2] = { "f32", "f16" };
    std::printf("reference: %s (lm_head %s, KV %s, backend %s)\n"
                "candidate: %s (lm_head %s, KV %s, backend %s)\n"
                "model: d_model=%d dec_layers=%d dec_heads=%d vocab=%d\n"
                "teacher-forced logits from: %s\n\n",
                ref_path.c_str(),  head_type(ref),  kv_name[kv_ref],  ref.backend_name.c_str(),
                cand_path.c_str(), head_type(cand), kv_name[kv_cand], cand.backend_name.c_str(),
                D, L, ref.hp.n_dec_head, n_vocab,
                kv_compare ? "nllb_decode_cached (both sides - --kv differs)"
                           : "nllb_decode, un-cached (the historical path)");
    if (allow_drift) {
        std::printf("THE ASSERTION IS [topk]: the reference's argmax must stay "
                    "inside the candidate's\n"
                    "top %d. Id drift is REPORTED, not gated (--allow-id-drift). "
                    "That is the right\n"
                    "assertion below f16 and the wrong one above it - at f16 the "
                    "ids do match exactly,\n"
                    "and asserting less than the truth is how a regression gets "
                    "through. Drop the flag\n"
                    "for anything you expect to be exact.\n\n", topk);
    } else {
        std::printf("the assertion is EXACT ID EQUALITY. state divergence is "
                    "reported,\n"
                    "not gated - f16 moves the numbers, that is what f16 is. the "
                    "line\n"
                    "to read is [margin]: a decision is safe only while its margin\n"
                    "exceeds twice the logit error at that same position.\n"
                    "[topk] runs too, at k=%d, and is also an assertion. Pass\n"
                    "--allow-id-drift to make it the ONLY one.\n\n", topk);
    }

    std::ifstream mf(ref_dir + "manifest_gen.txt");
    if (!mf) {
        std::fprintf(stderr, "no manifest_gen.txt in %s\n", ref_dir.c_str());
        return 2;
    }

    int    n_cases = 0, n_lines = 0, n_skip = 0, n_fail = 0;
    long   total_decisions = 0, total_unsafe = 0;
    long   total_flips = 0, total_out_of_topk = 0;
    int    worst_rank = 0;
    std::string worst_rank_where;
    // rank_hist[r] for r < 8, rank_hist[8] = everything worse. A histogram
    // rather than a maximum because "one decision at rank 6" and "forty at
    // rank 1" are different findings and a single worst-case hides which.
    long   rank_hist[9] = {};
    int    n_ids_diff = 0;
    double worst_logit_err = 0.0, worst_state_rel = 0.0;
    double tightest_ratio  = 1e30;
    std::string tightest_where;
    std::string line;

    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name;
        int Tg = 0, m_bos = -1, m_max_new = 0, m_start = -1;
        const bool ok_line = (bool) (ss >> name >> Tg >> m_bos >> m_max_new >> m_start);
        if (name.empty()) continue;
        ++n_lines;
        if (!ok_line) {
            std::printf("  %-20s SKIP (malformed manifest line)\n", name.c_str());
            ++n_skip; continue;
        }

        std::vector<int32_t> src_ids;
        if (!read_i32(ref_dir + name + ".ids.bin", src_ids)) {
            std::printf("  %-20s SKIP (missing .ids.bin)\n", name.c_str());
            ++n_skip; continue;
        }
        ++n_cases;
        bool case_ok = true;

        nllb_generate_params gp;
        gp.max_new_tokens = max_new > 0 ? max_new : m_max_new;
        gp.forced_bos_id  = forced_bos != -2 ? forced_bos : m_bos;
        gp.dec_start_id   = m_start;
        gp.use_cache      = true;

        // ---- [gen] the only thing a user can observe ----
        nllb_generate_params gp_ref = gp, gp_cand = gp;
        gp_ref.kv_type  = kv_ref;
        gp_cand.kv_type = kv_cand;
        nllb_generate_result A, B;
        if (!nllb_generate(ref,  src_ids, gp_ref,  n_threads, A, err) ||
            !nllb_generate(cand, src_ids, gp_cand, n_threads, B, err)) {
            std::fprintf(stderr, "  %-20s ERROR %s\n", name.c_str(), err.c_str());
            ++n_fail; continue;
        }

        size_t k = 0;
        const size_t n_common = A.ids.size() < B.ids.size() ? A.ids.size() : B.ids.size();
        while (k < n_common && A.ids[k] == B.ids[k]) ++k;
        const bool ids_same = A.ids.size() == B.ids.size() && k == A.ids.size();
        if (!ids_same) {
            ++n_ids_diff;
            // Under --allow-id-drift this is data, not a defect: it is exactly
            // what the run went looking for. [topk] below is what can fail.
            if (!allow_drift) case_ok = false;
            std::printf("  %-20s IDS DIFFER: reference %zu tokens, candidate %zu\n",
                        name.c_str(), A.ids.size(), B.ids.size());
            if (k < n_common) {
                std::printf("      first difference at position %zu (step %zu): "
                            "%d vs %d\n", k, k - 1, A.ids[k], B.ids[k]);
                if (k - 1 < A.steps.size()) {
                    std::printf("      the reference's margin there was %.6e - "
                                "compare it with the logit error below.\n"
                                "      margin >> error means a real divergence; "
                                "margin <= 2x error means the\n"
                                "      precision simply decided a coin toss, and "
                                "the fix is not in the graph.\n",
                                A.steps[k - 1].margin);
                }
            }
        }

        // ---- teacher-force the REFERENCE's output through both, all logits ----
        std::vector<int32_t> prefix = A.ids;
        if (ref.hp.n_ctx > 0 && (int) prefix.size() > ref.hp.n_ctx) {
            prefix.resize((size_t) ref.hp.n_ctx);
        }
        nllb_encode_result e_ref, e_cand;
        nllb_decode_result d_ref, d_cand;
        // One prefix, all logits, both models. Which decoder does it is decided
        // ONCE for both sides - see kv_compare's definition. When the KV types
        // match this is nllb_decode, byte for byte the call every earlier
        // measurement in claude/handoff.md was taken with.
        auto teacher_force = [&](const nllb_model & m, const nllb_encode_result & e,
                                 int kvt, nllb_decode_result & out) -> bool {
            if (!kv_compare) {
                return nllb_decode(m, e, prefix, src_ids, n_threads, true,
                                   NLLB_LOGITS_ALL, out, err, NLLB_POS_ABSOLUTE);
            }
            nllb_kv_cache c;
            if (!nllb_kv_cache_init(m, e, (int) prefix.size(), n_threads, c, err, kvt)) {
                return false;
            }
            // The whole prefix in ONE call, which is test_cache.cpp's `full`
            // pass: same sequence length as the un-cached reference, so nothing
            // regroups and the only thing that moved is the KV type.
            return nllb_decode_cached(m, c, prefix, n_threads, true,
                                      NLLB_LOGITS_ALL, out, err);
        };
        if (!nllb_encode(ref,  src_ids, n_threads, false, e_ref,  err) ||
            !nllb_encode(cand, src_ids, n_threads, false, e_cand, err) ||
            !teacher_force(ref,  e_ref,  kv_ref,  d_ref) ||
            !teacher_force(cand, e_cand, kv_cand, d_cand)) {
            std::fprintf(stderr, "  %-20s ERROR (teacher-forced) %s\n",
                         name.c_str(), err.c_str());
            ++n_fail; continue;
        }

        // ---- [state] reported, not gated ----
        double case_state_rel = 0.0;
        int    worst_layer    = -1;
        for (int h = 0; h <= L; ++h) {
            double max_abs = 0.0, mag = 0.0;
            const float * a = d_ref.hidden_states[(size_t) h].data();
            const float * b = d_cand.hidden_states[(size_t) h].data();
            const size_t n = (size_t) prefix.size() * D;
            for (size_t i = 0; i < n; ++i) {
                const double d = std::fabs((double) a[i] - (double) b[i]);
                if (d > max_abs) max_abs = d;
                const double m = std::fabs((double) a[i]);
                if (m > mag) mag = m;
            }
            const double rel = max_abs / (mag > 1.0 ? mag : 1.0);
            if (rel > case_state_rel) { case_state_rel = rel; worst_layer = h; }
        }
        if (case_state_rel > worst_state_rel) worst_state_rel = case_state_rel;

        // ---- [margin] per decision ----
        //
        // Row t of the teacher-forced logits is the distribution the loop saw
        // when it chose token t+1. Only rows 0..T-2 correspond to a decision;
        // the last row's token was never fed back in.
        const int rows = d_ref.n_logit_rows;
        long case_decisions = 0, case_unsafe = 0;
        long case_flips = 0, case_out_of_topk = 0;
        int  case_worst_rank = 0, case_worst_rank_pos = -1;
        double case_logit_err = 0.0, case_min_margin = 1e30, case_min_ratio = 1e30;
        int    case_worst_pos = -1;

        for (int t = 0; t + 1 < rows; ++t) {
            const float * pr = d_ref.logits.data()  + (size_t) t * n_vocab;
            const float * pc = d_cand.logits.data() + (size_t) t * n_vocab;

            double e = 0.0;
            for (int i = 0; i < n_vocab; ++i) {
                const double d = std::fabs((double) pr[i] - (double) pc[i]);
                if (d > e) e = d;
            }
            double m; int best_r;
            margin_of(pr, n_vocab, m, best_r);
            int best_c; double dummy;
            margin_of(pc, n_vocab, dummy, best_c);

            ++case_decisions;
            if (e > case_logit_err)  case_logit_err  = e;
            if (m < case_min_margin) case_min_margin = m;

            // ---- [topk] ----
            const int rk = rank_of(pc, n_vocab, best_r);
            rank_hist[rk < 8 ? rk : 8]++;
            if (rk > 0) ++case_flips;
            if (rk >= topk) ++case_out_of_topk;
            if (rk > case_worst_rank) { case_worst_rank = rk; case_worst_rank_pos = t; }

            const double ratio = headroom_of(pr, pc, n_vocab, best_r);
            if (ratio < case_min_ratio) { case_min_ratio = ratio; case_worst_pos = t; }
            if (ratio <= 1.0) {
                ++case_unsafe;
                if (verbose) {
                    std::printf("      pos %-4d margin %.3e  row err %.3e  "
                                "headroom %.2fx  argmax %s\n",
                                t, m, e, ratio,
                                best_r == best_c ? "agreed" : "FLIPPED");
                }
            }
        }
        total_decisions   += case_decisions;
        total_unsafe      += case_unsafe;
        total_flips       += case_flips;
        total_out_of_topk += case_out_of_topk;
        if (case_worst_rank > worst_rank) {
            worst_rank = case_worst_rank;
            worst_rank_where = name + " pos " + std::to_string(case_worst_rank_pos);
        }
        if (case_out_of_topk) case_ok = false;   // [topk] is an assertion in both modes
        if (case_logit_err > worst_logit_err) worst_logit_err = case_logit_err;
        if (case_min_ratio < tightest_ratio) {
            tightest_ratio = case_min_ratio;
            tightest_where = name + " pos " + std::to_string(case_worst_pos);
        }

        std::printf("  %-20s T=%-3zu ids=%-4s  state rel=%.2e (layer %d)  "
                    "logit err=%.2e  min margin=%.2e  headroom=%.1fx  "
                    "unsafe=%ld/%ld  argmax moved=%ld (worst rank %d)  "
                    "out-of-top%d=%ld  %s\n",
                    name.c_str(), A.ids.size(), ids_same ? "same" : "DIFF",
                    case_state_rel, worst_layer, case_logit_err,
                    case_min_margin == 1e30 ? 0.0 : case_min_margin,
                    case_min_ratio == 1e30 ? 0.0 : case_min_ratio,
                    case_unsafe, case_decisions,
                    case_flips, case_worst_rank,
                    topk, case_out_of_topk,
                    case_ok ? "ok" : "FAIL");
        if (!case_ok) ++n_fail;
    }

    std::printf("\n%d of %d manifest cases ran, %d failed.\n",
                n_cases, n_lines, n_fail);
    if (n_cases == 0) { std::fprintf(stderr, "no cases ran\n"); return 2; }

    std::printf("worst state divergence = %.3e relative (reported, NOT gated -\n"
                "  f16 moves the numbers by construction; this localises a bug "
                "if the ids ever differ)\n"
                "worst logit error      = %.3e\n"
                "tightest headroom      = %.2fx at %s  (>1 means the argmax\n"
                "  provably could not change; it is a ratio of the "
                "winner's lead to the error\n  that could close it)\n",
                worst_state_rel, worst_logit_err,
                tightest_ratio == 1e30 ? 0.0 : tightest_ratio,
                tightest_where.empty() ? "-" : tightest_where.c_str());

    // ------------------------------------------------------------- [topk]
    std::printf("\n[topk] where the reference's argmax lands in the candidate's "
                "ordering, over all\n"
                "  %ld decisions. This is the assertion that survives "
                "quantisation: ids are an\n"
                "  OUTCOME and reorder on a coin toss, the ranking is the "
                "arithmetic itself.\n",
                total_decisions);
    for (int r = 0; r < 9; ++r) {
        if (!rank_hist[r]) continue;
        const double pct = total_decisions
            ? 100.0 * (double) rank_hist[r] / (double) total_decisions : 0.0;
        if (r < 8) std::printf("    rank %d  %8ld  %6.2f%%%s\n", r, rank_hist[r], pct,
                               r == 0 ? "   (the candidate agreed)" : "");
        else       std::printf("    rank >=8 %7ld  %6.2f%%\n", rank_hist[r], pct);
    }
    std::printf("  worst rank = %d at %s;  %ld of %ld decisions moved the argmax; "
                "%ld reached rank >= %d\n",
                worst_rank, worst_rank_where.empty() ? "-" : worst_rank_where.c_str(),
                total_flips, total_decisions, total_out_of_topk, topk);
    if (n_ids_diff) {
        std::printf("  %d case%s emitted different ids%s\n", n_ids_diff,
                    n_ids_diff == 1 ? "" : "s",
                    allow_drift ? " - reported, not gated (--allow-id-drift)."
                                : " - and that IS gated: see the FAILs above.");
    }
    if (total_out_of_topk == 0 && total_unsafe > 0) {
        // MEASURED, and it is the most important caveat this gate carries.
        // Round-tripping token_embd through Q4_0 on the shape fixture - a
        // relative weight perturbation of 8e-02, far past anything a shippable
        // mix would do - collapses the tightest headroom from 5.73x to 0.26x
        // and puts one decision inside the error band, while leaving ALL 96
        // ranks at 0. [topk] does not notice. It is a much blunter instrument
        // than the margin audit, and it is a REPLACEMENT for exact id equality
        // rather than a replacement for [margin].
        std::printf("  !! READ THIS TOGETHER WITH [margin], and do not ship on "
                    "[topk] alone. Every\n"
                    "  decision stayed inside the top %d - but %ld of %ld had "
                    "headroom <= 1, so they\n"
                    "  could have gone the other way and did not. That is "
                    "'the reordering happened to\n"
                    "  be harmless on these inputs', not 'the reordering cannot "
                    "happen'. [topk] is a\n"
                    "  coarse instrument by construction: a perturbation big "
                    "enough to take the\n"
                    "  headroom from 5.7x to 0.26x has been measured to move "
                    "ZERO ranks. Use --strict\n"
                    "  to make an unsafe decision a failure, and say which of "
                    "the two claims you are\n"
                    "  making.\n", topk, total_unsafe, total_decisions);
    }

    if (total_unsafe == 0) {
        std::printf("\nall %ld greedy decisions had headroom > 1.\n"
                    "  Not one of them could have gone the other way at this "
                    "precision. That is a\n"
                    "  stronger statement than 'the ids matched', which is also "
                    "true but is compatible\n"
                    "  with having been lucky.\n", total_decisions);
    } else {
        std::printf("\n!! %ld of %ld greedy decisions had headroom <= 1 - the "
                    "argmax COULD have flipped.\n"
                    "  The ids above may still all match - they do unless a case "
                    "says IDS DIFFER - but\n"
                    "  those decisions agreed without having to. Re-run with -v "
                    "to list them.\n"
                    "  This is a property of the model and these inputs, not of "
                    "the code: it means\n"
                    "  THIS FIXTURE CANNOT SUPPORT A CLAIM ABOUT %s, however "
                    "green it looks.\n"
                    "  Use a checkpoint whose margins clear the error, and say "
                    "which one the claim rests on.\n",
                    total_unsafe, total_decisions, head_type(cand));
    }

    if (n_skip) {
        std::printf("%d case%s SKIPPED - regenerate the manifest.\n",
                    n_skip, n_skip == 1 ? "" : "s");
        return 1;
    }
    if (n_fail) {
        std::printf("MISMATCH\n");
        return 1;
    }
    if (strict && total_unsafe)  { std::printf("STRICT: unsafe decisions present\n"); return 1; }
    // A PASS has to name the claim it is passing, because the two modes make
    // very different ones and a bare "PASS" in a log six months from now is
    // indistinguishable between them.
    if (allow_drift) {
        std::printf("PASS - [topk] held at k=%d. NOT a claim of identical ids: "
                    "%d case%s drifted.\n",
                    topk, n_ids_diff, n_ids_diff == 1 ? "" : "s");
    } else {
        std::printf("PASS - exact ids AND [topk] at k=%d.\n", topk);
    }
    return 0;
}
