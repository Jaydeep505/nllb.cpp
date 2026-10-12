// nllb-test-batch: the batched path against the single-sequence path.
//
// Two C++ paths, no goldens, no torch, no ref/ directory: sources are seeded
// random ids, so this runs on the real checkpoint and on any shape fixture.
// Like nllb-test-cache it compares two implementations reading the same
// weights, so a corrupt weight cancels - this gate owns "does batching change
// the answer", not "is the model right".
//
//   [plan]   nllb_plan_batches: every input once, both budgets respected     exact, no model
//   [b1]     a batch of ONE vs nllb_generate: ids, stop, every step's logit
//            and margin                                                      BIT-IDENTICAL
//   [equal]  N rows of EQUAL length vs N single calls, every step's logits   BIT-IDENTICAL
//   [perm]   the padded batch with its rows REVERSED                         BIT-IDENTICAL
//   [mask]   the padded batch with a DIFFERENT FILLER behind the mask        BIT-IDENTICAL
//   [pad1]   one row, padded, NO batch: the control for [pad]                measured, never gated
//   [pad]    rows of different length vs single calls: ids exact, the
//            unpadded row bit-identical, padded rows within tolerance
//   [eos]    an extra row that stops at step 0 changes nothing for the others BIT-IDENTICAL
//
// Why the split: a PADDED row cannot be bit-identical to its
// unpadded single call - padding raises the softmax length and regroups the
// same real additions. So the bit-identity claims are made where they are true
// (equal lengths, the unpadded row, reordering, the filler) and only the padded
// rows get a tolerance. [mask] is the mask assertion and has no threshold at
// all: nothing in the graph may read the filler, so changing it must not move
// one bit - and a leaking mask moves a whole real token's worth.
//
// [pad1] is a control for the BATCH DIMENSION, not for the MASK: it builds a
// mask of its own, so a broken mask inflates it too. [pad]'s padded
// rows are therefore held to an ABSOLUTE tolerance, never to "a few times the
// control".
//
// Bit-identity AGAINST THE SINGLE PATH is a CPU claim. On a GPU the batched
// step picks a different matmul kernel for B columns than for 1 (measured on
// the M1: mul_mv at B=1, mul_mv_ext at B=2..5, tiled mul_mm above 8 - which
// also rounds its inputs to half). Different kernels sum in a
// different order, so [b1], [equal] and the unpadded row of [pad] are held to
// a tolerance there instead, and say so:
//   GPU, rows <= 8   1e-5 relative: f32 reordering lands ~3e-7 on the M1, a
//                    mask bug lands ~1e-2
//   GPU, rows  > 8   the padded-row tolerance: the mm kernel's half inputs
// [perm], [mask] and [eos] compare batched against batched - same kernels on
// both sides - and stay BIT-IDENTICAL on every backend. --strict demands
// bit-identity everywhere (to see the raw kernel difference).

#include "nllb.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct opts {
    std::string model;
    std::string backend;          // "" -> $NLLB_BACKEND or cpu (the library default)
    int  n_threads = 4;
    int  rows      = 4;
    int  max_new   = 12;
    int  seed      = 1234;
    bool kv_f16    = false;
    bool verbose   = false;
    bool strict    = false;       // bit-identity vs the single path even on a GPU
    std::vector<int> bench;       // --bench 1,2,4,8: time the decoder step, skip the gate
    int  bench_src   = 24;
    int  bench_steps = 32;
};

int g_fail = 0;

void verdict(const char * pass, bool ok, const std::string & msg) {
    std::printf("[%s]%*s %s  %s\n", pass, (int) (7 - std::strlen(pass)), "", ok ? "ok  " : "FAIL", msg.c_str());
    if (!ok) ++g_fail;
}

// logits per step for one sequence: [steps][n_vocab]
using trace = std::vector<std::vector<float>>;

struct cmp_stat {
    size_t n_diff = 0;
    double max_abs = 0, max_ref = 0;
    double rel() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
};

void cmp_into(cmp_stat & st, const float * a, const float * b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (a[i] != b[i]) st.n_diff++;
        if (d > st.max_abs) st.max_abs = d;
        const double r = std::fabs((double) b[i]);
        if (r > st.max_ref) st.max_ref = r;
    }
}

char fmtbuf[256];
const char * fmt_stat(const cmp_stat & s) {
    if (s.n_diff == 0) std::snprintf(fmtbuf, sizeof fmtbuf, "0 differing floats");
    else std::snprintf(fmtbuf, sizeof fmtbuf, "%zu differ, rel %.3e", s.n_diff, s.rel());
    return fmtbuf;
}

// Teacher-forced single path: feed ids[0..n-2] through the cache one at a time.
bool single_trace(const nllb_model & m, const std::vector<int32_t> & src, const std::vector<int32_t> & ids,
                  const opts & o, trace & out, std::string & err) {
    nllb_encode_result enc;
    if (!nllb_encode(m, src, o.n_threads, false, enc, err)) return false;
    nllb_kv_cache cache;
    if (!nllb_kv_cache_init(m, enc, (int) ids.size(), o.n_threads, cache, err,
                            o.kv_f16 ? NLLB_KV_F16 : NLLB_KV_F32)) return false;
    out.clear();
    for (size_t t = 0; t + 1 < ids.size(); ++t) {
        nllb_decode_result d;
        if (!nllb_decode_cached(m, cache, std::vector<int32_t>(1, ids[t]), o.n_threads,
                                false, NLLB_LOGITS_LAST, d, err)) return false;
        out.push_back(d.logits);
    }
    return true;
}

// Teacher-forced batched path. Row b is fed ids[b][t] while it lasts, then pad.
bool batch_trace(const nllb_model & m, const std::vector<std::vector<int32_t>> & srcs,
                 const std::vector<std::vector<int32_t>> & ids, const opts & o,
                 std::vector<trace> & out, std::string & err, int pad_to = 0, int32_t filler = -1) {
    const int B = (int) srcs.size();
    nllb_encode_batch_result enc;
    if (!nllb_encode_batch(m, srcs, o.n_threads, enc, err, pad_to, filler)) return false;
    size_t steps = 0;
    for (const auto & v : ids) steps = std::max(steps, v.size() - 1);
    nllb_kv_cache cache;
    if (!nllb_kv_cache_init_batch(m, enc, (int) steps + 1, o.n_threads, cache, err,
                                  o.kv_f16 ? NLLB_KV_F16 : NLLB_KV_F32)) return false;
    out.assign(B, trace());
    std::vector<int32_t> feed(B);
    for (size_t t = 0; t < steps; ++t) {
        for (int b = 0; b < B; ++b) feed[b] = t + 1 < ids[b].size() ? ids[b][t] : m.hp.pad_id;
        nllb_decode_result d;
        if (!nllb_decode_cached_batch(m, cache, feed, o.n_threads, false, d, err)) return false;
        for (int b = 0; b < B; ++b) {
            if (t + 1 < ids[b].size()) {
                out[b].emplace_back(d.logits.begin() + (size_t) b * d.n_vocab,
                                    d.logits.begin() + (size_t) (b + 1) * d.n_vocab);
            }
        }
    }
    return true;
}

cmp_stat cmp_trace(const trace & a, const trace & b) {
    cmp_stat st;
    if (a.size() != b.size()) { st.n_diff = (size_t) -1; st.max_abs = INFINITY; return st; }
    for (size_t t = 0; t < a.size(); ++t) {
        if (a[t].size() != b[t].size()) { st.n_diff = (size_t) -1; st.max_abs = INFINITY; return st; }
        cmp_into(st, a[t].data(), b[t].data(), a[t].size());
    }
    return st;
}

std::vector<int32_t> random_src(std::mt19937 & rng, int len, int32_t lo, int32_t hi, int32_t pad, int32_t eos) {
    std::uniform_int_distribution<int32_t> d(lo, hi);
    std::vector<int32_t> s;
    for (int j = 0; j < len - 1; ++j) {
        int32_t id;
        do { id = d(rng); } while (id == pad || id == eos);
        s.push_back(id);
    }
    s.push_back(eos);   // NLLB sources end on eos
    return s;
}

void usage(const char * a0) {
    std::fprintf(stderr,
        "usage: %s MODEL.gguf [-t N] [--backend NAME] [--rows N] [--max-new N] [--seed N] [--kv-f16] [--strict] [-v]\n"
        "       %s MODEL.gguf --bench 1,2,4,8 [--src 24] [--steps 32] [-t N] [--backend NAME]\n"
        "  Batched vs single-sequence decoding. Seeded random sources; no ref/ needed.\n"
        "  --bench times ONE batched decoder step (the only thing batching changes) and\n"
        "  reports ms/step and ms/token. Read ms/token: the ratio hides where the floor is.\n", a0, a0);
}


// ---- --bench: the decoder step at several batch sizes ---------------------------
int run_bench(const nllb_model & model, const opts & o) {
    std::string err;
    const nllb_hparams & hp = model.hp;
    const int32_t n_vocab = (int32_t) std::min<int64_t>(model.lm_head()->ne[1], model.tok_embd->ne[1]);
    std::mt19937 rng(o.seed);
    std::printf("bench: %s | backend %s | src %d tokens | %d timed steps | %d threads\n",
                o.model.c_str(), model.backend_name.c_str(), o.bench_src, o.bench_steps, o.n_threads);
    std::printf("%4s %12s %12s %8s %10s\n", "B", "ms/step", "ms/token", "vs B=1", "arena MiB");
    double base = 0;
    for (int B : o.bench) {
        if (B < 1) continue;
        std::vector<std::vector<int32_t>> srcs;
        for (int b = 0; b < B; ++b) srcs.push_back(random_src(rng, o.bench_src, 4, n_vocab - 1, hp.pad_id, hp.eos_id));
        nllb_encode_batch_result enc;
        nllb_kv_cache cache;
        if (!nllb_encode_batch(model, srcs, o.n_threads, enc, err) ||
            !nllb_kv_cache_init_batch(model, enc, o.bench_steps + 4, o.n_threads, cache, err,
                                      o.kv_f16 ? NLLB_KV_F16 : NLLB_KV_F32)) {
            std::fprintf(stderr, "B=%d: %s\n", B, err.c_str()); return 2;
        }
        std::vector<double> t;
        size_t arena = 0;
        std::vector<int32_t> feed(B, hp.dec_start_id);
        for (int s = 0; s < o.bench_steps + 2; ++s) {          // 2 warm-up steps
            nllb_decode_result d;
            nllb_step_timing tm;
            const auto t0 = std::chrono::steady_clock::now();
            if (!nllb_decode_cached_batch(model, cache, feed, o.n_threads, false, d, err, &tm)) {
                std::fprintf(stderr, "B=%d step %d: %s\n", B, s, err.c_str()); return 2;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (s >= 2) t.push_back(ms);
            arena = tm.arena;
            for (int b = 0; b < B; ++b) feed[b] = 4 + (s * 31 + b * 17) % (n_vocab - 5);
        }
        std::sort(t.begin(), t.end());
        const double med = t[t.size() / 2];
        if (base == 0) base = med / B;
        std::printf("%4d %12.2f %12.2f %7.2fx %10.1f\n", B, med, med / B, base / (med / B), arena / (1024.0 * 1024.0));
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { usage(argv[0]); std::exit(2); } return argv[++i]; };
        if      (a == "-t" || a == "--threads") o.n_threads = std::atoi(next());
        else if (a == "--backend")              o.backend   = next();
        else if (a == "--rows")                 o.rows      = std::atoi(next());
        else if (a == "--max-new")              o.max_new   = std::atoi(next());
        else if (a == "--seed")                 o.seed      = std::atoi(next());
        else if (a == "--kv-f16")               o.kv_f16    = true;
        else if (a == "-v")                     o.verbose   = true;
        else if (a == "--strict")               o.strict    = true;
        else if (a == "--bench") {
            std::string l = next(); size_t p0 = 0;
            while (p0 <= l.size()) {
                size_t q = l.find(',', p0);
                if (q == std::string::npos) q = l.size();
                if (q > p0) o.bench.push_back(std::atoi(l.substr(p0, q - p0).c_str()));
                p0 = q + 1;
            }
        }
        else if (a == "--src")                  o.bench_src   = std::atoi(next());
        else if (a == "--steps")                o.bench_steps = std::atoi(next());
        else if (a == "-h" || a == "--help")    { usage(argv[0]); return 0; }
        else if (o.model.empty() && a[0] != '-') o.model = a;
        else { usage(argv[0]); return 2; }
    }
    if (o.model.empty() || o.rows < 2 || o.max_new < 2) { usage(argv[0]); return 2; }
    if (!o.bench.empty()) {
        nllb_model bm;
        std::string berr;
        if (!nllb_model_load(o.model, bm, berr, true, o.backend)) { std::fprintf(stderr, "%s\n", berr.c_str()); return 2; }
        return run_bench(bm, o);
    }

    // ---- [plan]: no model needed --------------------------------------------
    {
        std::mt19937 rng(o.seed);
        std::uniform_int_distribution<int> d(5, 400);
        std::vector<int32_t> lens(40);
        for (auto & l : lens) l = d(rng);
        lens.push_back(1000);                 // one source over any sane budget
        bool ok = true;
        std::string why;
        const int cases[][2] = { {1, 0}, {4, 0}, {8, 0}, {8, 512}, {32, 1024} };
        std::string summary;
        for (const auto & c : cases) {
            const auto groups = nllb_plan_batches(lens, c[0], c[1]);
            std::vector<int> seen(lens.size(), 0);
            long padding = 0;
            for (const auto & g : groups) {
                int32_t w = 0; long real = 0;
                for (int32_t i : g) { seen[i]++; w = std::max(w, lens[i]); real += lens[i]; }
                padding += (long) w * (long) g.size() - real;
                if ((int) g.size() > c[0]) { ok = false; why = "group over max_seqs"; }
                if (c[1] > 0 && g.size() > 1 && (long) w * (long) g.size() > c[1]) { ok = false; why = "group over max_tokens"; }
            }
            for (int s : seen) if (s != 1) { ok = false; why = "an input planned zero or two times"; }
            char b[96];
            std::snprintf(b, sizeof b, " %d/%d:%zu groups,%ld pad", c[0], c[1], groups.size(), padding);
            summary += b;
        }
        verdict("plan", ok, ok ? "max_seqs/max_tokens ->" + summary : why);
    }

    nllb_model model;
    std::string err;
    if (!nllb_model_load(o.model, model, err, true, o.backend)) {
        std::fprintf(stderr, "model load failed: %s\n", err.c_str());
        return 2;
    }
    if (!model.has_decoder) { std::fprintf(stderr, "GGUF has no decoder\n"); return 2; }
    const nllb_hparams & hp = model.hp;
    const int32_t n_vocab = (int32_t) std::min<int64_t>(model.lm_head()->ne[1], model.tok_embd->ne[1]);
    const bool f32 = !o.kv_f16 && model.tok_embd->type == GGML_TYPE_F32;
    // Absolute tolerance on a PADDED row's relative logit error. Measured, not
    // derived: padding noise on the shape fixture is ~1e-7 at f32; the real
    // 600M at f16 reached 2.2e-3. The closest mask bug in
    // tests/corrupt_batch.py lands orders of magnitude above either.
    const double pad_tol = f32 ? 5e-5 : 5e-3;
    // Single-path comparisons on a GPU: see the header. 0 means bit-identity.
    const bool   gpu      = !model.on_cpu;
    const double kern_tol = (!gpu || o.strict) ? 0.0 : (o.rows > 8 ? pad_tol : 1e-5);
    auto vs_single = [&](const cmp_stat & st, std::string & note) -> bool {
        if (st.n_diff == 0) return true;
        if (kern_tol > 0 && st.rel() <= kern_tol) {
            char b[96];
            std::snprintf(b, sizeof b, " [%s kernel selection, within %.0e]", model.backend_name.c_str(), kern_tol);
            note = b;
            return true;
        }
        return false;
    };

    std::printf("model: %s | backend %s | d_model %d | dec %d layers | vocab %d | %s | rows %d | max_new %d\n",
                o.model.c_str(), model.backend_name.c_str(), hp.d_model, hp.n_dec_layer, n_vocab,
                f32 ? "f32" : "f16/quant (tolerance widened)", o.rows, o.max_new);
    if (gpu) {
        if (kern_tol > 0) std::printf("vs single path: GPU - held to %.0e, not bit-identity (kernel selection, see header; --strict to demand it)\n", kern_tol);
        else              std::printf("vs single path: GPU, --strict - bit-identity demanded\n");
    }

    std::mt19937 rng(o.seed);
    const int32_t lo = 4, hi = n_vocab - 1;
    const int32_t forced = std::min<int32_t>(10, n_vocab - 1);   // a stand-in "target language"

    nllb_generate_params gp;
    gp.max_new_tokens = o.max_new;
    gp.forced_bos_id  = forced;
    gp.use_cache      = true;
    gp.kv_type        = o.kv_f16 ? NLLB_KV_F16 : NLLB_KV_F32;

    auto gen_single = [&](const std::vector<int32_t> & src, nllb_generate_result & r) {
        if (!nllb_generate(model, src, gp, o.n_threads, r, err)) {
            std::fprintf(stderr, "nllb_generate failed: %s\n", err.c_str()); std::exit(2);
        }
    };

    // ---- [b1] ----------------------------------------------------------------
    {
        const auto src = random_src(rng, 9, lo, hi, hp.pad_id, hp.eos_id);
        nllb_generate_result s;
        gen_single(src, s);
        nllb_generate_batch_result b;
        bool ok = nllb_generate_batch(model, { src }, {}, gp, o.n_threads, b, err);
        std::string msg;
        if (!ok) msg = "nllb_generate_batch failed: " + err;
        else {
            const auto & r = b.seq[0];
            // ids, stop and argmax are exact on every backend; the logits are
            // bit-identical on the CPU and within kern_tol on a GPU.
            ok = r.ids == s.ids && r.stop == s.stop && r.steps.size() == s.steps.size();
            bool exact = true;
            for (size_t i = 0; ok && i < r.steps.size(); ++i) {
                ok = r.steps[i].argmax == s.steps[i].argmax;
                const bool same = r.steps[i].logit == s.steps[i].logit && r.steps[i].margin == s.steps[i].margin;
                if (!same) {
                    exact = false;
                    const double scale = std::max(1.0, (double) std::fabs(s.steps[i].logit));
                    if (!(kern_tol > 0 && std::fabs((double) r.steps[i].logit - s.steps[i].logit) <= kern_tol * scale)) ok = false;
                }
            }
            msg = std::to_string(s.ids.size()) + " ids, stop " + std::to_string(s.stop) +
                  (!ok   ? " - DIFFERS from nllb_generate"
                   : exact ? ", every step's logit and margin bit-identical"
                           : ", ids exact; logits within kernel tolerance [" + model.backend_name + " kernel selection]");
        }
        verdict("b1", ok, msg);
    }

    // ---- [equal] -------------------------------------------------------------
    {
        const int L = 11;
        std::vector<std::vector<int32_t>> srcs, ids;
        std::vector<trace> ref;
        for (int b = 0; b < o.rows; ++b) {
            srcs.push_back(random_src(rng, L, lo, hi, hp.pad_id, hp.eos_id));
            nllb_generate_result g; gen_single(srcs.back(), g);
            ids.push_back(g.ids);
            trace t;
            if (!single_trace(model, srcs.back(), g.ids, o, t, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
            ref.push_back(t);
        }
        std::vector<trace> got;
        if (!batch_trace(model, srcs, ids, o, got, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        cmp_stat all;
        for (int b = 0; b < o.rows; ++b) {
            cmp_stat st = cmp_trace(got[b], ref[b]);
            if (o.verbose) std::printf("         row %d: %s\n", b, fmt_stat(st));
            all.n_diff += st.n_diff; all.max_abs = std::max(all.max_abs, st.max_abs); all.max_ref = std::max(all.max_ref, st.max_ref);
        }
        std::string note;
        const bool ok = vs_single(all, note);
        verdict("equal", ok,
                std::to_string(o.rows) + " rows of length " + std::to_string(L) + " vs single calls: " + fmt_stat(all) + note);
    }

    // ---- the padded set: [pad1] [pad] [perm] [mask] [eos] ----------------------
    std::vector<int> lens;
    for (int b = 0; b < o.rows; ++b) lens.push_back(3 + (b * 7) % 17);   // 3, 10, 17, 7, ...
    int longest = 0;
    for (int b = 0; b < o.rows; ++b) if (lens[b] > lens[longest]) longest = b;
    std::vector<std::vector<int32_t>> srcs, ids;
    std::vector<nllb_generate_result> single_gen;
    std::vector<trace> ref;
    for (int b = 0; b < o.rows; ++b) {
        srcs.push_back(random_src(rng, lens[b], lo, hi, hp.pad_id, hp.eos_id));
        nllb_generate_result g; gen_single(srcs.back(), g);
        single_gen.push_back(g);
        ids.push_back(g.ids);
        trace t;
        if (!single_trace(model, srcs.back(), g.ids, o, t, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        ref.push_back(t);
    }
    const int Ts = lens[longest];

    std::vector<trace> got;
    if (!batch_trace(model, srcs, ids, o, got, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }

    // [pad1]: each padded row alone, padded to Ts, B = 1.
    std::vector<double> ctl(o.rows, 0.0);
    {
        double worst = 0;
        for (int b = 0; b < o.rows; ++b) {
            if (lens[b] == Ts) continue;
            std::vector<trace> one;
            if (!batch_trace(model, { srcs[b] }, { ids[b] }, o, one, err, Ts)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
            ctl[b] = cmp_trace(one[0], ref[b]).rel();
            worst = std::max(worst, ctl[b]);
        }
        char m[160];
        std::snprintf(m, sizeof m, "padding alone, no batch dimension: worst rel %.3e  (control, not gated)", worst);
        std::printf("[pad1]   --    %s\n", m);
    }

    // [pad]
    {
        bool ok = true;
        double worst_pad = 0;
        std::string unpadded_msg;
        for (int b = 0; b < o.rows; ++b) {
            cmp_stat st = cmp_trace(got[b], ref[b]);
            if (lens[b] == Ts) {
                std::string note;
                if (!vs_single(st, note)) ok = false;
                unpadded_msg = std::string("unpadded row: ") + fmt_stat(st) + note;
            } else {
                worst_pad = std::max(worst_pad, st.rel());
                if (!(st.rel() <= pad_tol)) ok = false;
            }
            if (o.verbose) std::printf("         row %d len %2d: %s  (control %.3e)\n", b, lens[b], fmt_stat(st), ctl[b]);
        }
        // ids through the real batched generator, against nllb_generate.
        nllb_generate_batch_result gb;
        if (!nllb_generate_batch(model, srcs, {}, gp, o.n_threads, gb, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        int ids_same = 0;
        float min_margin = INFINITY;
        for (int b = 0; b < o.rows; ++b) {
            if (gb.seq[b].ids == single_gen[b].ids && gb.seq[b].stop == single_gen[b].stop) ids_same++;
            for (const auto & s : single_gen[b].steps) if (!s.forced) min_margin = std::min(min_margin, s.margin);
        }
        if (ids_same != o.rows) ok = false;
        char m[256];
        std::snprintf(m, sizeof m, "%s; padded rows worst rel %.3e (tol %.0e); ids %d/%d exact (min margin %.3e); slots %lld kept %lld",
                      unpadded_msg.c_str(), worst_pad, pad_tol, ids_same, o.rows, (double) min_margin > 0 ? min_margin : 0.0f,
                      (long long) gb.token_slots, (long long) gb.tokens_kept);
        verdict("pad", ok, m);
    }

    // [perm]
    {
        std::vector<std::vector<int32_t>> rs(srcs.rbegin(), srcs.rend()), ri(ids.rbegin(), ids.rend());
        std::vector<trace> rev;
        if (!batch_trace(model, rs, ri, o, rev, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        cmp_stat all;
        for (int b = 0; b < o.rows; ++b) {
            cmp_stat st = cmp_trace(rev[o.rows - 1 - b], got[b]);
            all.n_diff += st.n_diff; all.max_abs = std::max(all.max_abs, st.max_abs); all.max_ref = std::max(all.max_ref, st.max_ref);
        }
        verdict("perm", all.n_diff == 0, "same padded rows, reversed: " + std::string(fmt_stat(all)));
    }

    // [mask]
    {
        const int32_t other = std::min<int32_t>(7, n_vocab - 1) == hp.pad_id ? 8 : std::min<int32_t>(7, n_vocab - 1);
        std::vector<trace> alt;
        if (!batch_trace(model, srcs, ids, o, alt, err, 0, other)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        cmp_stat all;
        for (int b = 0; b < o.rows; ++b) {
            cmp_stat st = cmp_trace(alt[b], got[b]);
            all.n_diff += st.n_diff; all.max_abs = std::max(all.max_abs, st.max_abs); all.max_ref = std::max(all.max_ref, st.max_ref);
        }
        verdict("mask", all.n_diff == 0,
                "filler " + std::to_string(hp.pad_id) + " -> " + std::to_string(other) + " behind the masks: " + fmt_stat(all));
    }

    // [eos]: append a row whose forced token IS eos, so it stops at step 0.
    {
        nllb_generate_batch_result base, with;
        bool ok = nllb_generate_batch(model, srcs, {}, gp, o.n_threads, base, err);
        auto srcs2 = srcs;
        srcs2.push_back(random_src(rng, 5, lo, hi, hp.pad_id, hp.eos_id));
        std::vector<int32_t> f(srcs2.size(), forced);
        f.back() = hp.eos_id;
        ok = ok && nllb_generate_batch(model, srcs2, f, gp, o.n_threads, with, err);
        std::string msg;
        if (!ok) msg = "generate failed: " + err;
        else {
            for (int b = 0; b < o.rows && ok; ++b) {
                const auto & x = base.seq[b], & y = with.seq[b];
                ok = x.ids == y.ids && x.stop == y.stop && x.steps.size() == y.steps.size();
                for (size_t i = 0; ok && i < x.steps.size(); ++i) ok = x.steps[i].logit == y.steps[i].logit;
            }
            const auto & e = with.seq.back();
            const bool stopped = e.stop == NLLB_STOP_EOS && e.ids.size() == 2;
            ok = ok && stopped;
            msg = std::string("extra row stopped at step 0: ") + (stopped ? "yes" : "NO") +
                  "; the running rows' ids and step logits " + (ok ? "bit-identical" : "DIFFER");
        }
        verdict("eos", ok, msg);
    }

    if (g_fail) { std::printf("FAIL - %d pass(es)\n", g_fail); return 1; }
    std::printf("PASS\n");
    return 0;
}
