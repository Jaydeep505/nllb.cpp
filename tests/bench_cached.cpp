// Where does a cached decode step actually spend its time?
//
//   ./build/bin/nllb-bench <model.gguf> [-t 8] [--src 24] [--steps 64]
//
// Step 5 ended with an arithmetic argument rather than a measurement: fitting
// total = n*L + d*n^2/2 to two un-cached cases gave a fixed per-step cost of
// ~12 ms, the cached path runs at ~38, and the 22 ms gap was attributed to
// nllb_decode_cached building a fresh ~72 MB ggml arena every token. That is a
// plausible story told entirely from weight sizes. This tool is the clock.
//
// It reports four things, and the third and fourth are the ones that decide
// whether ggml_gallocr is worth writing:
//
//   1. The phase split of one cached step - arena, graph build, compute,
//      readback, free. Anything not in `compute` is overhead by definition.
//   2. Minor page faults per step, from getrusage. A fresh 72 MB arena that is
//      genuinely being faulted in costs ~18000 of them at 4 KiB pages, and that
//      is a number you can check rather than infer.
//   3. The same step with the lm_head switched off (NLLB_LOGITS_NONE). The
//      difference is what the 1.05 GB output projection costs, which is the
//      part no graph-allocation change can touch.
//   4. The same step run against a WARM arena. ggml_init with mem_buffer =
//      nullptr mallocs but does not touch, so first-touch faults land inside
//      ggml_graph_compute and are invisible in the phase split above. Running
//      the identical graph a second time through a reused context prices them
//      directly - and that is the ceiling on what a reused graph could win,
//      measured before writing one.
//
// The weights are irrelevant here, so this runs happily on
// tests/make_shape_model.py output. What matters is the SHAPE: on the 600M the
// lm_head is 256206 x 1024 and dominates. A tiny fixture answers a different
// question.

#include "nllb.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/resource.h>
#  define HAVE_RUSAGE 1
#endif

namespace {

double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

long minor_faults() {
#ifdef HAVE_RUSAGE
    rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) return ru.ru_minflt;
#endif
    return -1;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Per-phase samples over every timed step.
//
// THE MEDIAN IS THE HEADLINE AND THE MEAN IS THE FOOTNOTE, which is the
// opposite of the first version of this file and the reason it misreported.
// A decode step has a long right tail from things that are not the code:
// first-touch of a 2.3 GB model file, and a fanless laptop throttling partway
// through 64 steps that each stream a gigabyte. Those inflate the mean without
// changing the typical step at all. Measured on an M2 Air, the arena
// comparison below read "saves 2.09 ms/step (5.3%)" from the means while the
// medians were a dead heat - a conclusion invented entirely by outliers.
struct stat_t {
    double sum = 0.0;
    std::vector<double> each;
    void add(double x) { sum += x; each.push_back(x); }
    double mean() const { return each.empty() ? 0.0 : sum / (double) each.size(); }
    double med()  const { return median(each); }
    double p90()  const {
        if (each.empty()) return 0.0;
        std::vector<double> v = each;
        std::sort(v.begin(), v.end());
        return v[(size_t) ((double) (v.size() - 1) * 0.9)];
    }
};

struct run_t {
    stat_t alloc, build, compute, readback, release, total;
    long   faults = 0;
    size_t arena  = 0;
    int    nodes  = 0;
    // The number every other line is a share of, and the one to quote.
    double typical() const { return total.med(); }
};

void report(const char * tag, const run_t & r, int steps) {
    const double t = r.typical();
    auto pct = [&](double x) { return t > 0.0 ? 100.0 * x / t : 0.0; };
    std::printf("  %-14s %7.2f ms/step (median)   arena %.0f MiB, %d nodes, "
                "%ld minor faults/step\n",
                tag, t, (double) r.arena / (1024.0 * 1024.0), r.nodes,
                steps ? r.faults / steps : 0);
    std::printf("      alloc %6.2f (%4.1f%%)   build %6.2f (%4.1f%%)   "
                "compute %7.2f (%4.1f%%)   readback %5.2f (%4.1f%%)   "
                "free %5.2f (%4.1f%%)\n",
                r.alloc.med(), pct(r.alloc.med()),
                r.build.med(), pct(r.build.med()),
                r.compute.med(), pct(r.compute.med()),
                r.readback.med(), pct(r.readback.med()),
                r.release.med(), pct(r.release.med()));
    // The spread is printed, not hidden, because it is the evidence for
    // preferring the median. mean >> median means outliers; if the three
    // numbers below are close together, the mean would have been fine.
    std::printf("      spread: median %.2f  mean %.2f  p90 %.2f  max %.2f%s\n",
                r.total.med(), r.total.mean(), r.total.p90(),
                r.total.each.empty() ? 0.0
                    : *std::max_element(r.total.each.begin(), r.total.each.end()),
                r.total.mean() > 1.15 * r.total.med()
                    ? "   <- mean inflated by outliers, read the median" : "");
}

// Untimed steps before a measured loop, so the timed ones are not paying to
// fault in the model file. Without this the first block of the profile carries
// ~2 GB of first-touch and its mean runs 30% above its median.
bool warmup(const nllb_model & model, const nllb_encode_result & enc,
            int n_warm, int n_threads, int logits_mode, std::string & err) {
    nllb_kv_cache cache;
    if (!nllb_kv_cache_init(model, enc, n_warm + 1, n_threads, cache, err)) return false;
    for (int s = 0; s < n_warm; ++s) {
        nllb_decode_result out;
        const std::vector<int32_t> feed(1, 4 + s);
        if (!nllb_decode_cached(model, cache, feed, n_threads, false,
                                logits_mode, out, err)) return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> [-t threads] [--src N] [--steps N]\n"
            "  --src   source length fed to the encoder (default 24)\n"
            "  --steps cached decode steps to time (default 64)\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    int n_threads = 4, n_src = 24, n_steps = 64;
    for (int i = 2; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "-t")      && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--src")   && i + 1 < argc) n_src     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) n_steps   = std::atoi(argv[++i]);
    }

    nllb_model model;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!nllb_model_load(model_path, model, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }
    const double t_load = ms_since(t0);
    if (!model.has_decoder) {
        std::fprintf(stderr, "this GGUF has no decoder\n");
        return 2;
    }

    const nllb_hparams & hp = model.hp;
    const int n_vocab = (int) model.lm_head()->ne[1];
    std::printf("model: d_model=%d enc=%d dec=%d heads=%d ffn=%d vocab=%d "
                "n_ctx=%d  (loaded in %.0f ms)\n",
                hp.d_model, hp.n_layer, hp.n_dec_layer, hp.n_dec_head,
                hp.n_dec_ff, n_vocab, hp.n_ctx, t_load);
    // ggml_nbytes, not dims x 4. An f16 checkpoint halves these, and the whole
    // point of the f16 change is that the step is bandwidth-bound - so a byte
    // count that assumed f32 would report the same figure for a model running
    // twice as fast, which is the one number here that must not be wrong.
    {
        const ggml_tensor * lm = model.lm_head();
        size_t dec_bytes = 0;
        for (const auto & l : model.dec) {
            for (const ggml_tensor * t : { l.wq, l.wk, l.wv, l.wo,
                                           l.cwq, l.cwk, l.cwv, l.cwo,
                                           l.w_up, l.w_down }) {
                if (t) dec_bytes += ggml_nbytes(t);
            }
        }
        std::printf("lm_head is %d x %d %s = %.2f GiB read per step; decoder "
                    "weights are %.2f GiB (%s)\n",
                    n_vocab, hp.d_model, ggml_type_name(lm->type),
                    (double) ggml_nbytes(lm) / (1 << 30),
                    (double) dec_bytes / (1 << 30),
                    model.dec.empty() ? "-" : ggml_type_name(model.dec[0].wq->type));
    }
    std::printf("threads=%d src=%d steps=%d\n\n", n_threads, n_src, n_steps);

    // Ids only have to be in range: this measures time, not text.
    std::vector<int32_t> src_ids(n_src);
    for (int i = 0; i < n_src; ++i) src_ids[i] = 4 + (i * 7919) % (n_vocab - 8);

    nllb_encode_result enc;
    t0 = std::chrono::steady_clock::now();
    if (!nllb_encode(model, src_ids, n_threads, false, enc, err)) {
        std::fprintf(stderr, "encode failed: %s\n", err.c_str());
        return 2;
    }
    std::printf("encoder pass over %d tokens: %.0f ms (once per sentence)\n",
                n_src, ms_since(t0));

    // One pass of each configuration. `logits_mode` is the only difference
    // between the first two, so their gap is the lm_head and nothing else.
    struct cfg_t { const char * tag; int logits; };
    const cfg_t cfgs[] = {
        { "with lm_head", NLLB_LOGITS_LAST },
        { "no lm_head",   NLLB_LOGITS_NONE },
    };

    std::printf("\n[1] cached step, phase split\n");
    // Page the weights in first. These steps are not timed; without them the
    // timed ones carry the first touch of a 2.3 GB file and the means come out
    // ~30%% above the medians for reasons that have nothing to do with the code.
    if (!warmup(model, enc, 8, n_threads, NLLB_LOGITS_LAST, err)) {
        std::fprintf(stderr, "warm-up failed: %s\n", err.c_str());
        return 2;
    }
    double with_head = 0.0, without_head = 0.0;
    for (const cfg_t & c : cfgs) {
        nllb_kv_cache cache;
        if (!nllb_kv_cache_init(model, enc, n_steps + 1, n_threads, cache, err)) {
            std::fprintf(stderr, "cache init failed: %s\n", err.c_str());
            return 2;
        }
        // The step-5 baseline: a fresh arena every step. Section [2] prices
        // the alternative; this section has to be the code as it stood.
        cache.reuse_scratch = false;
        run_t r;
        const long f0 = minor_faults();
        for (int s = 0; s < n_steps; ++s) {
            const std::vector<int32_t> feed(1, 4 + (s * 104729) % (n_vocab - 8));
            nllb_decode_result out;
            nllb_step_timing tm;
            if (!nllb_decode_cached(model, cache, feed, n_threads, false,
                                    c.logits, out, err, &tm)) {
                std::fprintf(stderr, "cached decode failed at step %d: %s\n", s, err.c_str());
                return 2;
            }
            r.alloc.add(tm.alloc);       r.build.add(tm.build);
            r.compute.add(tm.compute);   r.readback.add(tm.readback);
            r.release.add(tm.release);   r.total.add(tm.total);
            r.arena = tm.arena;          r.nodes = tm.n_nodes;
        }
        const long f1 = minor_faults();
        r.faults = (f0 >= 0 && f1 >= 0) ? f1 - f0 : 0;
        report(c.tag, r, n_steps);
        (std::strcmp(c.tag, "no lm_head") == 0 ? without_head : with_head) = r.typical();
    }
    std::printf("  -> the lm_head accounts for %.2f ms of the %.2f ms step "
                "(%.0f%%), medians throughout.\n"
                "     No allocation change touches that. It is one matmul against "
                "a %d x %d matrix.\n",
                with_head - without_head, with_head,
                with_head > 0.0 ? 100.0 * (with_head - without_head) / with_head : 0.0,
                n_vocab, hp.d_model);

    // ------------------------------------------------------------------
    // [2] the identical step, cold arena vs reused arena.
    //
    // ggml_init(mem_buffer = nullptr) mallocs but does not touch, so the cost
    // of faulting a fresh ~80 MB arena in does not appear under `alloc` - it
    // appears inside ggml_graph_compute, indistinguishable from arithmetic.
    // nllb_kv_cache::reuse_scratch hands ggml a buffer this process has already
    // written to, changing that and nothing else, so the gap between the two
    // runs below IS the first-touch bill.
    //
    // The two are INTERLEAVED rather than run in sequence. Run back to back,
    // the second loop inherits whatever the first did to the allocator and the
    // caches, and a thermal or scheduling drift over a few seconds lands
    // entirely on one of them. Alternating puts both under the same drift.
    // ------------------------------------------------------------------
    std::printf("\n[2] the SAME cached step, cold arena vs reused arena "
                "(interleaved)\n");
    {
        nllb_kv_cache cold, warm;
        if (!nllb_kv_cache_init(model, enc, n_steps + 2, n_threads, cold, err) ||
            !nllb_kv_cache_init(model, enc, n_steps + 2, n_threads, warm, err)) {
            std::fprintf(stderr, "cache init failed: %s\n", err.c_str());
            return 2;
        }
        cold.reuse_scratch = false;
        warm.reuse_scratch = true;

        run_t rc, rw;
        long f_cold = 0, f_warm = 0;
        for (int s = 0; s < n_steps; ++s) {
            const std::vector<int32_t> feed(1, 4 + (s * 104729) % (n_vocab - 8));
            // Same ids into both, so the arithmetic is identical and only the
            // arena differs. Alternate which one goes first as well, so neither
            // is systematically the one that warms the shared weights.
            const bool cold_first = (s % 2) == 0;
            for (int which = 0; which < 2; ++which) {
                const bool do_cold = (which == 0) == cold_first;
                nllb_kv_cache & kc = do_cold ? cold : warm;
                run_t & r = do_cold ? rc : rw;
                long & fa = do_cold ? f_cold : f_warm;

                nllb_decode_result out;
                nllb_step_timing tm;
                const long f0 = minor_faults();
                if (!nllb_decode_cached(model, kc, feed, n_threads, false,
                                        NLLB_LOGITS_LAST, out, err, &tm)) {
                    std::fprintf(stderr, "cached decode failed: %s\n", err.c_str());
                    return 2;
                }
                const long f1 = minor_faults();
                if (f0 >= 0 && f1 >= 0) fa += f1 - f0;
                r.alloc.add(tm.alloc);       r.build.add(tm.build);
                r.compute.add(tm.compute);   r.readback.add(tm.readback);
                r.release.add(tm.release);   r.total.add(tm.total);
                r.arena = tm.arena;          r.nodes = tm.n_nodes;
            }
        }
        rc.faults = f_cold;
        rw.faults = f_warm;
        report("cold arena", rc, n_steps);
        report("reused arena", rw, n_steps);

        // MEDIANS, not means. The two arms differ by well under a millisecond
        // on some machines, which is inside the noise a single throttled step
        // introduces - so a mean-based difference here is not a measurement,
        // it is whichever arm happened to catch the outlier. An earlier version
        // of this line used means and reported a 5.3% saving on an M2 Air where
        // the medians were a dead heat.
        const double win  = rc.typical() - rw.typical();
        const double base = rc.typical();
        std::printf("  -> reusing the arena: %+.2f ms/step of %.2f (%+.1f%%) by "
                    "median, and %ld fewer minor faults/step.\n",
                    win, base, base > 0.0 ? 100.0 * win / base : 0.0,
                    n_steps ? (rc.faults - rw.faults) / n_steps : 0);
        // Half a percent of a 36 ms step is not distinguishable from noise on a
        // laptop, and saying so is more useful than printing a number with two
        // decimal places and letting the reader believe all of them.
        const double noise = 0.005 * base;
        if (win < noise && win > -noise) {
            std::printf("     That is inside the noise floor (+-%.2f ms here), so "
                        "on THIS machine arena reuse is a wash on time.\n"
                        "     It still removes the faults and the per-step malloc; "
                        "it is just not a speed-up you can measure.\n", noise);
        } else if (win <= -noise) {
            std::printf("     Reuse came out SLOWER here, which should not happen "
                        "and means the run was disturbed. Re-run.\n");
        }
        std::printf("     Either way, compare it with the %.2f ms the lm_head "
                    "costs: allocation is not where the step is.\n",
                    with_head - without_head);

        const double out_cold = rc.alloc.med() + rc.build.med()
                              + rc.readback.med() + rc.release.med();
        const double out_warm = rw.alloc.med() + rw.build.med()
                              + rw.readback.med() + rw.release.med();
        std::printf("  -> everything outside compute - alloc + build + readback "
                    "+ free - is %.2f ms/step cold, %.2f warm\n"
                    "     (%.1f%% and %.1f%% of the step). That is the WHOLE of "
                    "what ggml_gallocr plus a reused graph\n"
                    "     could remove, and it is bounded by a number you can "
                    "read off this line rather than\n"
                    "     inferred from weight sizes - which is how step 5 got "
                    "it wrong by an order of magnitude.\n",
                    out_cold, out_warm,
                    base > 0.0 ? 100.0 * out_cold / base : 0.0,
                    base > 0.0 ? 100.0 * out_warm / base : 0.0);
    }

    // ------------------------------------------------------------------
    // [3] the un-cached step at the same prefix length, for scale.
    // ------------------------------------------------------------------
    std::printf("\n[3] un-cached decode of the same growing prefix, for scale\n");
    {
        std::vector<int32_t> prefix;
        prefix.reserve(n_steps);
        for (int s = 0; s < n_steps; ++s) prefix.push_back(4 + (s * 104729) % (n_vocab - 8));
        const int probes[] = { 1, n_steps / 4, n_steps / 2, n_steps };
        for (int p : probes) {
            if (p < 1 || p > (int) prefix.size()) continue;
            const std::vector<int32_t> take(prefix.begin(), prefix.begin() + p);
            nllb_decode_result out;
            auto t1 = std::chrono::steady_clock::now();
            if (!nllb_decode(model, enc, take, src_ids, n_threads, false,
                             NLLB_LOGITS_LAST, out, err, NLLB_POS_ABSOLUTE)) {
                std::fprintf(stderr, "un-cached decode failed: %s\n", err.c_str());
                return 2;
            }
            std::printf("  prefix %-4d %8.2f ms\n", p, ms_since(t1));
        }
    }

    return 0;
}
