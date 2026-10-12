// Diff ggml greedy generation against the HF goldens in ref/.
//
//   ./build/bin/nllb-test-generate models/nllb-600M-f32.gguf ref/ [-t 8]
//
// Reads manifest_gen.txt, written by scripts/dump_decoder_reference.py --greedy.
// Each case runs the whole pipeline for real - nllb_encode on the source ids,
// then nllb_generate looping nllb_decode - and compares the emitted id
// sequence against `generate(num_beams=1, do_sample=False)` byte for byte.
//
// WHY THIS IS AN EXACT COMPARISON AND THE OTHER GATES ARE NOT.
// The decoder gate measures float distance because floats are what it
// compares. Generation emits token ids, and an id is either right or wrong;
// there is no tolerance to set. The float error shows up here as a *risk*
// rather than a number: greedy only depends on which logit is largest, so an
// error of 1e-6 changes nothing unless the top two logits are within 1e-6 of
// each other, at which point the sequences fork and never rejoin.
//
// So a divergence is reported with the margin that produced it:
//
//   - margin of whole logits  -> the graph disagrees with HF. A real bug.
//   - margin near the float32 noise floor -> a coin toss that landed the other
//     way. The graph is fine and the golden is not reproducible at this
//     precision. This is the generation-level analogue of the ULP report in
//     tests/diagnose_decoder.py.
//
// The summary prints the smallest margin seen across all cases even when
// everything passes, because that number is the headroom - it is what says
// whether f16 or a quantised lm_head can be adopted without forking the output.
//
// HF's generate() runs with use_cache=True, so the goldens come from an
// incrementally-decoded run while this gate re-decodes the full prefix every
// step. Those are the same computation up to rounding - the margin question
// above - with ONE exception that is not rounding at all: HF's position rule
// differs between the cached and un-cached paths once a pad_id appears in the
// prefix. nllb_generate therefore decodes with NLLB_POS_ABSOLUTE, the cached
// rule, while the teacher-forced decoder gate uses NLLB_POS_CUMSUM, the
// un-cached one. See nllb_pos_mode in nllb.h; a fixture that emits pad is the
// only thing that tells them apart.
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

void print_ids(const char * label, const std::vector<int32_t> & v,
               size_t from, size_t upto) {
    std::printf("      %-6s", label);
    for (size_t i = from; i < v.size() && i < upto; ++i) std::printf(" %d", v[i]);
    if (v.size() > upto) std::printf(" ...");
    std::printf("\n");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <ref-dir> [-t threads] [--forced-bos ID]\n"
            "       [--max-new N] [--tie-floor F] [--check-modes] [--use-cache] [-v]\n"
            "  --forced-bos/--max-new override manifest_gen.txt for every case\n"
            "  --use-cache: run the SAME comparison through the KV-cached path.\n"
            "               nllb-test-cache already proves cached == un-cached; this\n"
            "               closes the loop against HF directly instead of by\n"
            "               composition, and costs one flag.\n"
            "  --tie-floor: below this logit margin a divergence is reported as a\n"
            "               near-tie rather than a disagreement (default 1e-3)\n"
            "  --check-modes: also prove NLLB_LOGITS_LAST matches the last row of\n"
            "               NLLB_LOGITS_ALL, i.e. that this gate and the decoder\n"
            "               gate read the same numbers\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';

    int   n_threads    = 4;
    int   forced_bos   = -2;     // -2 = take it from the manifest
    int   max_new      = -1;
    float tie_floor    = 1e-3f;
    bool  verbose      = false;
    bool  check_modes  = false;
    bool  use_cache    = false;
    for (int i = 3; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "-t")           && i + 1 < argc) n_threads  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--forced-bos") && i + 1 < argc) forced_bos = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-new")    && i + 1 < argc) max_new    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tie-floor")  && i + 1 < argc) tie_floor  = (float) std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v"))                           verbose    = true;
        else if (!std::strcmp(argv[i], "--check-modes"))                check_modes = true;
        else if (!std::strcmp(argv[i], "--use-cache"))                  use_cache   = true;
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
    std::printf("model: d_model=%d dec_layers=%d dec_heads=%d vocab=%d "
                "dec_start=%d eos=%d lm_head=%s\n"
                "greedy: exact id comparison, no tolerance. tie floor %.1e\n"
                "path:   %s\n\n",
                model.hp.d_model, model.hp.n_dec_layer, model.hp.n_dec_head,
                model.hp.n_vocab, model.hp.dec_start_id, model.hp.eos_id,
                model.output_w ? "untied" : "tied", tie_floor,
                use_cache ? "KV-CACHED (nllb_generate use_cache=true)"
                          : "un-cached (re-decodes the whole prefix each step)");

    std::ifstream mf(ref_dir + "manifest_gen.txt");
    if (!mf) {
        std::fprintf(stderr,
            "no manifest_gen.txt in %s\n"
            "  it carries forced_bos_token_id and max_new_tokens, which are\n"
            "  *inputs* to generation - reading them off the golden instead\n"
            "  would let the test pass by construction. Regenerate:\n"
            "    python scripts/dump_reference.py         <model> --out %s --per-layer\n"
            "    python scripts/dump_decoder_reference.py <model> --out %s --per-layer --greedy\n",
            ref_dir.c_str(), ref_dir.c_str(), ref_dir.c_str());
        return 2;
    }

    // Skips are failures, not silence. A missing or stale .gen.bin used to
    // print to stderr and drop the case, so deleting eight of nine goldens
    // still exited 0 with "1 cases, 0 failed. PASS" - a partial regeneration
    // or a renamed prompt would shrink the gate to nothing and CI would never
    // notice.
    int n_cases = 0, n_fail = 0, n_tie = 0, n_skip = 0, n_lines = 0;
    float min_margin = 0.0f;
    bool  have_min = false;
    std::string min_where;
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
            std::printf("  %-20s SKIP (manifest_gen.txt line has %d fields, "
                        "expected 5: name Tg forced_bos max_new dec_start)\n",
                        name.c_str(), 4);
            ++n_skip;
            continue;
        }

        std::vector<int32_t> src_ids, want;
        if (!read_i32(ref_dir + name + ".ids.bin", src_ids) ||
            !read_i32(ref_dir + name + ".gen.bin", want)) {
            std::printf("  %-20s SKIP (missing .bin)\n", name.c_str());
            ++n_skip;
            continue;
        }
        if ((int) want.size() != Tg) {
            std::printf("  %-20s SKIP (manifest says Tg=%d, gen.bin has %zu)\n",
                        name.c_str(), Tg, want.size());
            ++n_skip;
            continue;
        }

        nllb_generate_params gp;
        gp.max_new_tokens = max_new > 0 ? max_new : m_max_new;
        gp.forced_bos_id  = forced_bos != -2 ? forced_bos : m_bos;
        gp.dec_start_id   = m_start;
        gp.use_cache      = use_cache;

        nllb_generate_result got;
        if (!nllb_generate(model, src_ids, gp, n_threads, got, err)) {
            std::fprintf(stderr, "  %-20s ERROR %s\n", name.c_str(), err.c_str());
            ++n_fail; ++n_cases;
            continue;
        }

        // Headroom: the closest any step came to flipping. Tracked on passing
        // cases too - it is the number that decides whether f16 is safe.
        for (size_t s = 0; s < got.steps.size(); ++s) {
            if (got.steps[s].forced) continue;   // forced positions cannot flip
            const float m = got.steps[s].margin;
            if (!have_min || m < min_margin) {
                have_min = true; min_margin = m;
                min_where = name + " step " + std::to_string(s);
            }
        }

        // First position where the two sequences part company. The prefixes are
        // identical up to it by construction, so this really is one decision.
        size_t k = 0;
        const size_t n_common = got.ids.size() < want.size() ? got.ids.size() : want.size();
        while (k < n_common && got.ids[k] == want[k]) ++k;

        const bool same = got.ids.size() == want.size() && k == want.size();
        bool modes_bad = false;   // --check-modes findings, kept separate from
                                  // `same` so the id diagnosis is not printed
                                  // for a failure that is not about ids
        ++n_cases;

        // Two assumptions the rest of this file rests on, both checked here
        // rather than asserted in prose:
        //
        // 1. The generate loop reads logits through NLLB_LOGITS_LAST while the
        //    decoder gate reads them through NLLB_LOGITS_ALL. If those stop
        //    agreeing, the two gates are silently measuring different
        //    computations and a green decoder gate implies nothing about
        //    generation. The shapes reach different ggml kernels, so this is
        //    exactly what a backend or vocab change can move.
        // 2. The per-step trace (argmax, margin) is what a failure gets
        //    diagnosed with, and what the f16 headroom claim is built on - so
        //    it is recomputed here from an independent full row and compared.
        //
        // A differing argmax fails the case. Differing floats with the same
        // argmax only warn: on a new backend that is plausible and harmless,
        // while a flipped argmax never is.
        if (check_modes && want.size() > 1) {
            std::vector<int32_t> prefix(want.begin(), want.end() - 1);
            nllb_encode_result enc;
            nllb_decode_result all_r, last_r;
            if (nllb_encode(model, src_ids, n_threads, false, enc, err) &&
                nllb_decode(model, enc, prefix, src_ids, n_threads, false,
                            NLLB_LOGITS_ALL, all_r, err, NLLB_POS_ABSOLUTE) &&
                nllb_decode(model, enc, prefix, src_ids, n_threads, false,
                            NLLB_LOGITS_LAST, last_r, err, NLLB_POS_ABSOLUTE)) {
                const float * la = all_r.logits.data() +
                                   (size_t)(all_r.n_logit_rows - 1) * all_r.n_vocab;
                double mx = 0.0; long ndiff = 0;
                int am_a = 0, am_b = 0, second = -1;
                for (int i = 0; i < all_r.n_vocab; ++i) {
                    const double dd = std::fabs((double) la[i] - (double) last_r.logits[i]);
                    if (dd > mx) mx = dd;
                    if (la[i] != last_r.logits[i]) ++ndiff;
                    if (la[i] > la[am_a]) { second = am_a; am_a = i; }
                    else if (second < 0 || la[i] > la[second]) second = i;
                    if (last_r.logits[i] > last_r.logits[am_b]) am_b = i;
                }
                const bool modes_ok = (am_a == am_b);
                std::printf("  %-20s [modes] last row of ALL vs LAST: "
                            "max|d|=%.3e differing=%ld/%d argmax=%s%s\n",
                            name.c_str(), mx, ndiff, all_r.n_vocab,
                            modes_ok ? "same" : "DIFFERS",
                            (modes_ok && ndiff) ? "  (WARN: not bit-identical)" : "");
                if (!modes_ok) modes_bad = true;

                // Cross-check the trace for this prefix's step. Only meaningful
                // when the sequences matched, since otherwise the loop took a
                // different path to get here.
                const size_t step = want.size() - 2;
                if (same && step < got.steps.size()) {
                    const float mg = second >= 0 ? la[am_a] - la[second] : 0.0f;
                    const bool trace_ok = got.steps[step].forced ||
                        (got.steps[step].argmax == am_a &&
                         std::fabs(got.steps[step].margin - mg) <= 1e-6f * (std::fabs(mg) + 1.0f));
                    if (!trace_ok) {
                        std::printf("  %-20s [modes] TRACE MISMATCH at step %zu: "
                                    "recorded argmax=%d margin=%.6e, "
                                    "recomputed argmax=%d margin=%.6e\n",
                                    name.c_str(), step, got.steps[step].argmax,
                                    got.steps[step].margin, am_a, mg);
                        modes_bad = true;
                    }
                }
            }
        }

        static const char * const STOP[] = { "eos ", "cap ", "ctx " };
        std::printf("  %-20s Tg=%-3d got=%-3zu %s", name.c_str(), Tg,
                    got.ids.size(),
                    STOP[got.stop >= 0 && got.stop <= 2 ? got.stop : 1]);

        if (same && !modes_bad) {
            std::printf(" ok");
            if (verbose && !got.steps.empty()) {
                float mm = 0.0f; bool first = true;
                for (const auto & s : got.steps) {
                    if (s.forced) continue;
                    if (first || s.margin < mm) { mm = s.margin; first = false; }
                }
                std::printf("   min margin %.3e", mm);
            }
            std::printf("\n");
            continue;
        }

        ++n_fail;

        if (same) {
            // Ids matched; what failed is one of the --check-modes invariants,
            // already printed above. Do not run the id diagnosis - it would
            // describe a divergence that does not exist.
            std::printf("  FAIL (ids match; see the [modes] line above)\n");
            continue;
        }

        std::printf("  MISMATCH\n");

        if (k == 0) {
            // Position 0 is the decoder_start token, which no logit decided.
            // Reachable when manifest_gen.txt and the GGUF disagree about
            // decoder_start_token_id - and NLLB's is eos (2), not bos, which is
            // exactly the value people guess wrong.
            std::printf("      position 0 differs: got %d, want %d\n"
                        "      -> that is the decoder_start token, not a "
                        "prediction. manifest_gen.txt and the GGUF's\n"
                        "         nllb.decoder_start_token_id disagree. For NLLB "
                        "it is eos (2), not bos.\n",
                        got.ids.empty() ? -1 : got.ids[0], want[0]);
        } else if (k >= n_common) {
            // Same tokens all the way, different length: one side stopped
            // early. Almost always a stopping-rule difference (eos handling,
            // max_new_tokens counting) rather than a numeric one.
            std::printf("      prefixes agree for all %zu common tokens, "
                        "lengths differ: got %zu, want %zu\n"
                        "      -> stopping rule, not arithmetic. Check eos "
                        "handling and how max_new_tokens is counted.\n",
                        n_common, got.ids.size(), want.size());
        } else {
            const int32_t got_tok  = got.ids[k];
            const int32_t want_tok = want[k];
            // k-1 emitted tokens precede position k, so step index is k-1.
            const size_t  step     = k - 1;
            const bool forced_here = step < got.steps.size() && got.steps[step].forced;
            std::printf("      diverges at position %zu (step %zu): got %d, want %d\n",
                        k, step, got_tok, want_tok);

            if (forced_here) {
                // The logits did not decide this position, so pricing them
                // would be misleading. This is a configuration mismatch.
                std::printf("      -> FORCED position. forced_bos_id is %d here "
                            "and the golden used %d, so nothing about the graph\n"
                            "         is under test yet. Fix the target language "
                            "code (manifest_gen.txt / --forced-bos) and rerun.\n",
                            got_tok, want_tok);
                print_ids("want", want, 0, k + 6);
                print_ids("got",  got.ids, 0, k + 6);
                continue;
            }

            // Re-decode the shared prefix to price the disagreement: how much
            // did our logits prefer our token over HF's? That single number
            // separates a broken graph from a coin toss.
            std::vector<int32_t> prefix(want.begin(), want.begin() + k);
            nllb_encode_result enc;
            nllb_decode_result dec;
            if (nllb_encode(model, src_ids, n_threads, false, enc, err) &&
                nllb_decode(model, enc, prefix, src_ids, n_threads, false,
                            NLLB_LOGITS_LAST, dec, err) &&
                want_tok >= 0 && want_tok < dec.n_vocab &&
                got_tok  >= 0 && got_tok  < dec.n_vocab) {
                const float lg = dec.logits[got_tok];
                const float lw = dec.logits[want_tok];
                const float gap = lg - lw;
                std::printf("      logit(got)=%.6f logit(want)=%.6f gap=%.3e\n",
                            lg, lw, gap);
                if (std::fabs(gap) < tie_floor) {
                    ++n_tie;
                    std::printf("      -> NEAR TIE (gap < %.1e). float32 noise "
                                "decided this, not the graph. The rest of the\n"
                                "         sequence is meaningless once it forks; "
                                "compare the decoder gate before suspecting a bug.\n",
                                tie_floor);
                } else {
                    std::printf("      -> DECISIVE (gap >= %.1e). The graph "
                                "disagrees with HF. Run the decoder gate:\n"
                                "         ./build/bin/nllb-test-decoder %s %s\n",
                                tie_floor, model_path.c_str(), ref_dir.c_str());
                }
            }
            print_ids("want", want, 0, k + 6);
            print_ids("got",  got.ids, 0, k + 6);
        }
    }

    std::printf("\n%d of %d manifest cases ran, %d failed", n_cases, n_lines, n_fail);
    if (n_tie) std::printf(" (%d of them near-ties, i.e. float noise)", n_tie);
    std::printf(".\n");
    if (have_min) {
        std::printf("smallest logit margin over all steps: %.3e (%s)\n"
                    "  that is the headroom before a rounding change forks the "
                    "output; f16 measures ~9e-4 of error.\n",
                    min_margin, min_where.c_str());
    }
    if (n_cases == 0) { std::fprintf(stderr, "no cases ran\n"); return 2; }
    if (n_skip) {
        // A skip is a hole in the gate, not a neutral event: the run is green
        // for the cases it looked at and blind to the rest.
        std::printf("%d case%s SKIPPED - the goldens are incomplete or stale.\n"
                    "  regenerate them; a gate that silently shrinks is worse "
                    "than one that fails.\n",
                    n_skip, n_skip == 1 ? "" : "s");
        return 1;
    }
    if (n_fail) { std::printf("MISMATCH\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
