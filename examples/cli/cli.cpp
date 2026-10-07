// nllb-cli - text in, translation out. The last mile.
//
//   ./build/bin/nllb-cli -m models/nllb-600M-f16.gguf
//       --src eng_Latn --tgt deu_Latn "Hello, world!"
//
// (that is one shell line; the backslash is left out because a trailing
// backslash inside a // comment is a line continuation and -Wcomment says so)
//
// WHY THIS IS NOT JUST GLUE, and why it gets its own gate.
//
// Every gate in this repo stops at ids. tests/test-tokenizer.cpp proves
// text -> ids. The decoder, generate, cache and precision gates prove
// ids -> ids. NOTHING here has ever compared text to text, so this binary is
// the first thing to cross that seam, and composition of green gates is not
// evidence about it. The failure modes it can have are exactly the ones no
// existing gate can see:
//
//   * the target language code forced at the wrong position (it belongs at
//     the first GENERATED position, not in the encoder input - trap 10);
//   * --src and --tgt silently swapped, or one of them ignored;
//   * a lost or doubled leading space in detokenization (Metaspace);
//   * the eos stripped or kept in the printed string.
//
// tests/verify_translation.py is the gate for all four. See docs/cli.md.
//
// WHAT GOES WHERE. The translation, and only the translation, goes to stdout,
// one line per input. Everything else - banners, ids, timings, warnings - goes
// to stderr. That is what lets the gate compare stdout byte for byte and lets
// a user pipe the output somewhere without filtering it.
//
// SCOPE. By default each input line is an independent call: its own encoder
// pass and its own fresh KV cache. Reusing a cache across sentences would mean
// nllb_kv_cache_reset, which rewinds only the self-attention half and is
// correct exactly when the source is unchanged - nothing enforces that, so it
// is not done.
//
// --batch N (stdin only, opt-in) groups up to N lines - sorted by length,
// budgeted by --batch-tokens padded source tokens - through nllb_generate_batch,
// then prints them in INPUT order. Same one-line-per-line contract; the output
// is just no longer streamed. It never rewinds a cache either: a batched cache
// is built for its group and thrown away. OFF by default so the [stdin] pass of
// tests/verify_translation.py keeps testing what it always tested; run that gate
// with --batch N to make the batched claim (byte-identical output).

#include "nllb.h"
#include "nllb-vocab.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <io.h>
#  define nllb_isatty(fd) _isatty(fd)
#  define nllb_fileno(f)  _fileno(f)
#else
#  include <unistd.h>
#  define nllb_isatty(fd) isatty(fd)
#  define nllb_fileno(f)  fileno(f)
#endif

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m <model.gguf> [--src CODE] --tgt CODE [text]\n"
        "       %s -m <model.gguf> --list-langs\n"
        "       %s --list-backends\n"
        "\n"
        "  -m, --model PATH      GGUF converted by models/convert-nllb-to-gguf.py.\n"
        "                        It must carry the tokenizer (converted at step 8\n"
        "                        or later); an older file is refused by name.\n"
        "      --src CODE        FLORES-200 source code (default eng_Latn)\n"
        "      --tgt CODE        FLORES-200 target code, e.g. deu_Latn\n"
        "      --list-langs      print every code in this model and exit. Reads\n"
        "                        metadata only, so it does not load the weights.\n"
        "  -n, --max-new-tokens  cap on generated tokens (default %d). HF counts\n"
        "                        this from the decoder start token, so the id\n"
        "                        sequence is at most 1 + n long.\n"
        "  -t, --threads N       (default %d). CPU threads: the whole run on the\n"
        "                        CPU backend, only fallback ops on a GPU backend.\n"
        "  -b, --backend NAME    where the graphs run (default auto):\n"
        "                          auto   the GPU if this build has one, else CPU\n"
        "                          metal  Metal; FAILS rather than fall back\n"
        "                          cuda   CUDA; FAILS rather than fall back\n"
        "                          cpu    ggml's CPU backend - the reference, and\n"
        "                                 bit-identical to every pre-step-13 build\n"
        "                          or a device name from --list-backends\n"
        "                        On a GPU backend, ops it does not implement run\n"
        "                        on the CPU automatically (-v says how many splits).\n"
        "      --list-backends   print the devices this build can run on and exit\n"
        "      --no-mmap         read the weights onto the heap instead of\n"
        "                        mapping the file. Identical output either way\n"
        "                        (nllb-test-mmap asserts it); the trade is a\n"
        "                        near-zero load whose cost reappears as page\n"
        "                        faults during the first tokens, against one\n"
        "                        sequential read up front. MEASURE, do not assume.\n"
        "      --batch N         stdin only: translate up to N lines together (one\n"
        "                        batched decoder step for all of them). Lines are\n"
        "                        grouped by length and printed in INPUT order once\n"
        "                        their group finishes. Default 1 = off.\n"
        "      --batch-tokens N  cap a group's PADDED source tokens (default 2048)\n"
        "      --no-batch        the default, spelled out\n"
        "      --no-cache        re-decode the whole prefix every step. Same ids,\n"
        "                        O(n^2) instead of O(n) - see docs/cache.md.\n"
        "      --kv-f16          store the cached K and V at f16 instead of f32.\n"
        "                        Halves the cache and the bytes attention reads,\n"
        "                        and SPENDS HEADROOM: K and V are activations on a\n"
        "                        residual stream that reaches ~4200 and is never\n"
        "                        renormalised. Off by default. Run\n"
        "                        nllb-test-precision --kv f32:f16 on YOUR pair\n"
        "                        before trusting it.\n"
        "  -v, --verbose         ids, pieces and timings, on stderr\n"
        "  -h, --help\n"
        "\n"
        "With no [text] argument, reads stdin and translates ONE LINE AT A TIME,\n"
        "writing one line of output per line of input. Pass \"\" to translate the\n"
        "empty string. Only the translation goes to stdout.\n",
        argv0, argv0, argv0, 256, 4);
}

// One line of input -> one line of output. Everything that can go wrong here
// is a per-line failure, not a fatal one: a line that is not valid UTF-8 must
// not silently shift every later line's output up by one.
bool translate_one(const nllb_model & model, const nllb_vocab & vocab,
                   const std::string & text,
                   const std::string & src_lang, int32_t tgt_id,
                   const nllb_generate_params & base, int n_threads,
                   bool verbose, std::string & out, std::string & err) {
    const double t0 = now_ms();

    std::vector<int32_t> src_ids;
    if (!nllb_vocab_encode(vocab, text, src_lang, src_ids, err)) return false;

    const double t1 = now_ms();

    nllb_generate_params gp = base;
    gp.forced_bos_id = tgt_id;

    nllb_generate_result gen;
    if (!nllb_generate(model, src_ids, gp, n_threads, gen, err)) return false;

    const double t2 = now_ms();

    // skip_special drops the decoder start token, the forced target code and
    // the trailing eos. Those three are the whole reason this is not
    // `join(pieces)` - printing them is one of the four failure modes above.
    out = nllb_vocab_decode(vocab, gen.ids, /*skip_special=*/true);

    const double t3 = now_ms();

    if (verbose) {
        // Stable, parseable prefixes: tests/verify_translation.py reads these
        // back on a mismatch to say whether the ids moved or only the string
        // did. Changing them changes that bisect.
        std::fprintf(stderr, "src ids (%zu):", src_ids.size());
        for (int32_t id : src_ids) std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");

        std::fprintf(stderr, "src pieces:");
        for (int32_t id : src_ids) {
            if (id >= 0 && id < (int32_t) vocab.tokens.size()) {
                std::fprintf(stderr, " %s", vocab.tokens[id].c_str());
            }
        }
        std::fprintf(stderr, "\n");

        std::fprintf(stderr, "out ids (%zu):", gen.ids.size());
        for (int32_t id : gen.ids) std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");

        const char * stop = gen.stop == NLLB_STOP_EOS     ? "eos"
                          : gen.stop == NLLB_STOP_CONTEXT ? "ctx (ran out of n_ctx)"
                          : gen.stop == NLLB_STOP_ABORTED ? "aborted"
                                                          : "cap (max_new_tokens)";
        const size_t n_new = gen.ids.empty() ? 0 : gen.ids.size() - 1;
        std::fprintf(stderr,
            "stop: %s | %zu new tokens | cache %s\n"
            "timing: tokenize %.2f ms | generate %.1f ms (%.1f ms/token) | "
            "detokenize %.2f ms\n",
            stop, n_new, gen.used_cache ? "on" : "off",
            t1 - t0, t2 - t1, n_new ? (t2 - t1) / (double) n_new : 0.0, t3 - t2);
    }
    return true;
}


// --batch N: read every line, plan groups by length, one nllb_generate_batch
// per group, print in INPUT order. -v prints the same "src ids" / "out ids" /
// "stop" lines as translate_one, per line and in input order, so
// tests/verify_translation.py's bisect reads them unchanged.
int translate_stdin_batched(const nllb_model & model, const nllb_vocab & vocab,
                            const std::string & src_lang, int32_t tgt_id,
                            const nllb_generate_params & base, int n_threads,
                            int batch, int batch_tokens, bool verbose) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    const size_t n = lines.size();
    std::vector<std::vector<int32_t>> src(n);
    std::vector<std::string> out(n), errs(n);
    std::vector<nllb_generate_result> gen(n);
    std::vector<char> ok(n, 0);

    // A line that fails to tokenize still gets its blank output line, in its
    // place. The line correspondence is the whole contract of stdin mode.
    std::vector<int32_t> idx, lens;
    for (size_t i = 0; i < n; ++i) {
        std::string err;
        if (!nllb_vocab_encode(vocab, lines[i], src_lang, src[i], err)) { errs[i] = err; continue; }
        idx.push_back((int32_t) i);
        lens.push_back((int32_t) src[i].size());
    }

    const double t0 = now_ms();
    const auto groups = nllb_plan_batches(lens, batch, batch_tokens);
    long long slots = 0, kept = 0;
    for (const auto & g : groups) {
        std::vector<std::vector<int32_t>> srcs;
        for (int32_t k : g) srcs.push_back(src[idx[k]]);
        nllb_generate_params gp = base;
        gp.forced_bos_id = tgt_id;
        nllb_generate_batch_result r;
        std::string err;
        if (!nllb_generate_batch(model, srcs, {}, gp, n_threads, r, err)) {
            for (int32_t k : g) errs[idx[k]] = err;   // the whole group fails together
            continue;
        }
        slots += r.token_slots;
        kept  += r.tokens_kept;
        for (size_t j = 0; j < g.size(); ++j) {
            const size_t i = (size_t) idx[g[j]];
            gen[i] = r.seq[j];
            out[i] = nllb_vocab_decode(vocab, r.seq[j].ids, /*skip_special=*/true);
            ok[i]  = 1;
        }
    }
    const double t1 = now_ms();

    int n_fail = 0;
    for (size_t i = 0; i < n; ++i) {
        if (verbose && ok[i]) {
            std::fprintf(stderr, "src ids (%zu):", src[i].size());
            for (int32_t id : src[i]) std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, "\nout ids (%zu):", gen[i].ids.size());
            for (int32_t id : gen[i].ids) std::fprintf(stderr, " %d", id);
            const char * stop = gen[i].stop == NLLB_STOP_EOS     ? "eos"
                              : gen[i].stop == NLLB_STOP_CONTEXT ? "ctx (ran out of n_ctx)"
                              : gen[i].stop == NLLB_STOP_ABORTED ? "aborted"
                                                                 : "cap (max_new_tokens)";
            std::fprintf(stderr, "\nstop: %s | %zu new tokens | cache on (batched)\n",
                         stop, gen[i].ids.empty() ? (size_t) 0 : gen[i].ids.size() - 1);
        }
        if (!ok[i]) {
            std::fprintf(stderr, "line %zu: %s\n", i + 1, errs[i].c_str());
            ++n_fail;
        }
        std::printf("%s\n", ok[i] ? out[i].c_str() : "");
    }
    std::fflush(stdout);
    if (verbose) {
        std::fprintf(stderr, "%zu lines in %zu groups (batch %d, %d padded tokens), %d failed | "
                             "generate %.1f ms | decoder slots %lld, kept %lld\n",
                     n, groups.size(), batch, batch_tokens, n_fail, t1 - t0, slots, kept);
    }
    return n_fail ? 1 : 0;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string src_lang = "eng_Latn";
    std::string tgt_lang;
    std::string text;
    bool have_text  = false;
    bool list_langs = false;
    bool verbose    = false;
    bool use_mmap   = true;
    bool list_backends = false;
    // "auto", the opposite of the library's "cpu" default and for the same
    // reason use_cache differs: the library default is the configuration the
    // gates were measured on, and the product should use the GPU it has.
    std::string backend = "auto";
    bool kv_f16     = false;  // an arithmetic change, so it is opt-in. See the
                              // note on nllb_kv_type in include/nllb.h.
    bool use_cache  = true;   // see the note by nllb_generate_params::use_cache:
                              // the LIBRARY default is off so the step-4 gate
                              // keeps testing the un-cached path. A user-facing
                              // binary wants the fast path, and step 5 proved
                              // the two agree bit for bit at equal length.
    int  n_threads  = 4;
    int  max_new    = 256;
    int  batch      = 1;      // --batch N; 1 is the unbatched path, untouched
    int  batch_tokens = 2048;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        auto need = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (!std::strcmp(a, "-m") || !std::strcmp(a, "--model"))  model_path = need(a);
        else if (!std::strcmp(a, "--src"))                             src_lang   = need(a);
        else if (!std::strcmp(a, "--tgt"))                             tgt_lang   = need(a);
        else if (!std::strcmp(a, "-t") || !std::strcmp(a, "--threads")) n_threads = std::atoi(need(a));
        else if (!std::strcmp(a, "-n") || !std::strcmp(a, "--max-new-tokens")) max_new = std::atoi(need(a));
        else if (!std::strcmp(a, "--no-mmap"))                         use_mmap   = false;
        else if (!std::strcmp(a, "-b") || !std::strcmp(a, "--backend")) backend   = need(a);
        else if (!std::strcmp(a, "--list-backends"))                   list_backends = true;
        else if (!std::strcmp(a, "--no-cache"))                        use_cache  = false;
        else if (!std::strcmp(a, "--batch"))                           batch      = std::atoi(need(a));
        else if (!std::strcmp(a, "--batch-tokens"))                    batch_tokens = std::atoi(need(a));
        else if (!std::strcmp(a, "--no-batch"))                        batch      = 1;
        else if (!std::strcmp(a, "--kv-f16"))                          kv_f16     = true;
        else if (!std::strcmp(a, "--list-langs"))                      list_langs = true;
        else if (!std::strcmp(a, "-v") || !std::strcmp(a, "--verbose")) verbose   = true;
        else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help"))   { usage(argv[0]); return 0; }
        else if (a[0] == '-' && a[1] != '\0') {
            std::fprintf(stderr, "unknown option: %s\n", a);
            usage(argv[0]);
            return 2;
        }
        else if (!have_text) { text = a; have_text = true; }
        else {
            // Almost always an unquoted sentence: `nllb-cli ... Hello world`
            // arrives as two arguments. Joining them silently would translate
            // something the user did not type.
            std::fprintf(stderr,
                "unexpected extra argument: \"%s\"\n"
                "  the text must be ONE argument - quote it:\n"
                "    %s -m ... --tgt deu_Latn \"%s %s\"\n",
                a, argv[0], text.c_str(), a);
            return 2;
        }
    }

    if (list_backends) {
        for (const std::string & d : nllb_list_backends()) std::printf("%s\n", d.c_str());
        return 0;
    }
    if (model_path.empty()) { usage(argv[0]); return 2; }
    if (n_threads < 1)      { std::fprintf(stderr, "--threads must be >= 1\n"); return 2; }
    if (max_new   < 1)      { std::fprintf(stderr, "--max-new-tokens must be >= 1\n"); return 2; }
    if (batch     < 1)      { std::fprintf(stderr, "--batch must be >= 1\n"); return 2; }
    if (batch > 1 && !use_cache) {
        // Batching exists on the cached path only. Someone passing both is
        // comparing two numbers; quietly changing one of them would be the worst
        // thing this binary could do to them.
        std::fprintf(stderr, "--batch and --no-cache are mutually exclusive\n");
        return 2;
    }

    // The most common first-run failure is not having a model at all. Say how
    // to get one instead of the loader's bare "cannot open".
    if (FILE * probe = std::fopen(model_path.c_str(), "rb")) {
        std::fclose(probe);
    } else {
        std::fprintf(stderr,
            "model not found: %s\n"
            "\n"
            "download a pre-converted one (checksum-verified):\n"
            "  ./models/download-nllb-model.sh f16          # Linux / macOS\n"
            "  .\\models\\download-nllb-model.ps1 f16         # Windows\n"
            "or convert your own checkpoint: see docs/models.md\n",
            model_path.c_str());
        return 1;
    }

    std::string err;

    // --list-langs never touches the weights. nllb_vocab_load opens the GGUF
    // with no_alloc = true and reads metadata only, so this costs the same on
    // the 2.46 GB checkpoint as on a 6.7 MB vocab-only file.
    if (list_langs) {
        nllb_vocab vocab;
        if (!nllb_vocab_load(model_path, vocab, err)) {
            std::fprintf(stderr, "vocab load failed: %s\n", err.c_str());
            return 2;
        }
        const std::vector<std::string> langs = nllb_vocab_langs(vocab);
        for (const std::string & l : langs) std::printf("%s\n", l.c_str());
        std::fprintf(stderr, "%zu FLORES-200 codes\n", langs.size());
        return 0;
    }

    if (tgt_lang.empty()) {
        std::fprintf(stderr, "--tgt is required (try --list-langs)\n");
        usage(argv[0]);
        return 2;
    }

    const double t_load0 = now_ms();

    nllb_model model;
    if (!nllb_model_load(model_path, model, err, use_mmap, backend)) {
        std::fprintf(stderr, "model load failed: %s\n", err.c_str());
        return 2;
    }
    if (!model.has_decoder) {
        std::fprintf(stderr,
            "this GGUF has no decoder (converted with --encoder-only?)\n");
        return 2;
    }

    // Reuse the gguf_context the model already holds rather than opening the
    // file a second time.
    nllb_vocab vocab;
    if (!nllb_vocab_load_from_gguf(model.gguf, vocab, err)) {
        std::fprintf(stderr,
            "tokenizer load failed: %s\n"
            "  a GGUF converted before step 8 does not carry the tokenizer.\n"
            "  Reconvert:  python models/convert-nllb-to-gguf.py <snapshot> "
            "--outfile %s --outtype f16\n",
            err.c_str(), model_path.c_str());
        return 2;
    }

    const double t_load = now_ms() - t_load0;

    // Resolve both codes BEFORE any work. nllb_vocab_encode would catch a bad
    // --src on its own, but a bad --tgt would otherwise reach nllb_generate as
    // forced_bos_id = -1, which is the documented way to say "force nothing" -
    // so a typo would translate into whatever language the model felt like
    // rather than failing. That is the swap/typo failure mode, and it is worth
    // an explicit check rather than an implicit fallback.
    const int32_t tgt_id = nllb_vocab_lang_id(vocab, tgt_lang);
    if (tgt_id < 0) {
        std::fprintf(stderr,
            "unknown target language code '%s'. Expected a FLORES-200 code "
            "such as deu_Latn; --list-langs prints them all.\n", tgt_lang.c_str());
        return 2;
    }
    const int32_t src_id = nllb_vocab_lang_id(vocab, src_lang);
    if (src_id < 0) {
        std::fprintf(stderr,
            "unknown source language code '%s'. Expected a FLORES-200 code "
            "such as eng_Latn; --list-langs prints them all.\n", src_lang.c_str());
        return 2;
    }

    if (verbose) {
        std::fprintf(stderr,
            "model:  %s\n"
            "        d_model=%d enc=%d dec=%d heads=%d/%d vocab=%d n_ctx=%d "
            "lm_head=%s\n"
            "vocab:  %zu tokens, %u merges, grapheme tables from %s%s%s\n"
            "run:    %s (%d) -> %s (%d) | max_new=%d | cache %s | %d threads\n"
            "load:   %.1f ms (weights %s, %s)\n"
            "backend: %s%s\n",
            model_path.c_str(),
            model.hp.d_model, model.hp.n_layer, model.hp.n_dec_layer,
            model.hp.n_head, model.hp.n_dec_head, model.hp.n_vocab,
            model.hp.n_ctx, model.output_w ? "untied" : "tied",
            vocab.tokens.size(), vocab.n_merges,
            vocab.tables_from_file ? "THE GGUF" : "the compiled-in fallback",
            vocab.ucd_version.empty() ? "" : " UCD ",
            vocab.ucd_version.c_str(),
            src_lang.c_str(), src_id, tgt_lang.c_str(), tgt_id,
            max_new,
            use_cache ? (kv_f16 ? "on (KV f16)" : "on (KV f32)") : "off",
            n_threads, t_load,
            model.mapped ? "mmap'ed" : "read",
            model.weights_host_ptr ? "zero-copy in the backend buffer" : "copied to the backend",
            model.backend_name.c_str(),
            model.on_cpu ? "" : " (CPU fallback for unsupported ops)");
    }

    nllb_generate_params base;
    base.max_new_tokens = max_new;
    base.use_cache      = use_cache;
    base.kv_type        = kv_f16 ? NLLB_KV_F16 : NLLB_KV_F32;
    // forced_bos_id is set per call in translate_one; eos and dec_start come
    // from the model (dec_start is eos for NLLB, not bos - trap 8).

    int n_fail = 0;

    if (have_text) {
        std::string out;
        if (!translate_one(model, vocab, text, src_lang, tgt_id, base,
                           n_threads, verbose, out, err)) {
            std::fprintf(stderr, "translate failed: %s\n", err.c_str());
            return 1;
        }
        std::printf("%s\n", out.c_str());
        return 0;
    }

    if (nllb_isatty(nllb_fileno(stdin))) {
        std::fprintf(stderr,
            "no text given and stdin is a terminal. Pass the text as one "
            "quoted argument, or pipe it in.\n");
        usage(argv[0]);
        return 2;
    }

    if (batch > 1) {
        return translate_stdin_batched(model, vocab, src_lang, tgt_id, base,
                                       n_threads, batch, batch_tokens, verbose);
    }

    std::string line;
    size_t n_lines = 0;
    while (std::getline(std::cin, line)) {
        // A CRLF file would otherwise put a bare CR into the text, and the
        // precompiled charsmap maps CR LF to a single space - so the line
        // would tokenize differently from the same line typed as an argument.
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::string out;
        if (!translate_one(model, vocab, line, src_lang, tgt_id, base,
                           n_threads, verbose, out, err)) {
            // One line per line, even on failure: dropping the line would
            // shift every later translation up by one and the caller would
            // have no way to tell.
            std::fprintf(stderr, "line %zu: %s\n", n_lines + 1, err.c_str());
            std::printf("\n");
            ++n_fail;
        } else {
            std::printf("%s\n", out.c_str());
        }
        std::fflush(stdout);
        ++n_lines;
    }

    if (verbose) std::fprintf(stderr, "%zu lines, %d failed\n", n_lines, n_fail);
    return n_fail ? 1 : 0;
}
