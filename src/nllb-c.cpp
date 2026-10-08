// nllb-c.cpp - see nllb-c.h. A facade: every number it produces comes from
// the same four calls nllb-cli makes, so the CLI gate (tests/verify_translation.py)
// is evidence about this path too, line for line.
#include "nllb-c.h"

#include "nllb.h"
#include "nllb-vocab.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct nllb_ctx {
    nllb_model model;
    nllb_vocab vocab;
    int        n_threads = 4;
    std::vector<std::string> langs;
    std::atomic<bool> abort_flag{false};
};

static void set_err(char * err, size_t err_len, const std::string & msg) {
    if (!err || err_len == 0) return;
    std::snprintf(err, err_len, "%s", msg.c_str());
}

static char * dup_cstr(const std::string & s) {
    char * p = (char *) std::malloc(s.size() + 1);
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

extern "C" {

nllb_ctx_params nllb_ctx_default_params(void) {
    nllb_ctx_params p;
    p.backend   = "auto";
    p.use_mmap  = 1;
    p.n_threads = 4;
    return p;
}

nllb_translate_params nllb_translate_default_params(void) {
    nllb_translate_params p;
    p.src_lang       = "eng_Latn";
    p.tgt_lang       = "deu_Latn";
    p.max_new_tokens = 256;
    p.on_partial     = nullptr;
    p.user_data      = nullptr;
    return p;
}

nllb_ctx * nllb_ctx_init(const char * model_path, nllb_ctx_params params,
                         char * err, size_t err_len) {
    if (!model_path) { set_err(err, err_len, "model_path is NULL"); return nullptr; }

    nllb_ctx * ctx = new nllb_ctx();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    std::string e;
    const std::string backend = params.backend ? params.backend : "auto";
    if (!nllb_model_load(model_path, ctx->model, e, params.use_mmap != 0, backend)) {
        set_err(err, err_len, "model load failed: " + e);
        delete ctx;
        return nullptr;
    }
    if (!ctx->model.has_decoder) {
        set_err(err, err_len, "this GGUF has no decoder (converted with --encoder-only?)");
        delete ctx;
        return nullptr;
    }
    // The tokenizer comes out of the SAME file. A GGUF without the tokenizer
    // keys is refused here with the name of the missing key.
    if (!nllb_vocab_load_from_gguf(ctx->model.gguf, ctx->vocab, e)) {
        set_err(err, err_len, "tokenizer load failed: " + e);
        delete ctx;
        return nullptr;
    }
    ctx->langs = nllb_vocab_langs(ctx->vocab);
    return ctx;
}

void nllb_ctx_free(nllb_ctx * ctx) { delete ctx; }

const char * nllb_ctx_backend_name(const nllb_ctx * ctx) {
    return ctx ? ctx->model.backend_name.c_str() : "";
}

int nllb_ctx_weights_zero_copy(const nllb_ctx * ctx) {
    return ctx && ctx->model.mapped && ctx->model.weights_host_ptr ? 1 : 0;
}

int nllb_ctx_n_langs(const nllb_ctx * ctx) { return ctx ? (int) ctx->langs.size() : 0; }

const char * nllb_ctx_lang(const nllb_ctx * ctx, int i) {
    if (!ctx || i < 0 || i >= (int) ctx->langs.size()) return nullptr;
    return ctx->langs[(size_t) i].c_str();
}

int nllb_ctx_has_lang(const nllb_ctx * ctx, const char * code) {
    return ctx && code && nllb_vocab_lang_id(ctx->vocab, code) >= 0 ? 1 : 0;
}

void nllb_ctx_request_abort(nllb_ctx * ctx) {
    if (ctx) ctx->abort_flag.store(true, std::memory_order_relaxed);
}

void nllb_string_free(char * s) { std::free(s); }

} // extern "C"

namespace {

// State the per-token hook needs. Decoding the whole prefix each step is
// O(n) in the line length and measured at 0.03 ms for 76 tokens, so
// streaming costs nothing next to an ~18 ms decoder step.
struct stream_state {
    nllb_ctx *                ctx       = nullptr;
    const nllb_translate_params * p     = nullptr;
    const std::string *       done      = nullptr;   // finished lines, joined
    std::vector<int32_t>      ids;                   // this line, as emitted
    int                       n_tokens  = 0;
    bool                      user_stop = false;
};

bool on_token(int32_t token, void * user) {
    stream_state & st = *(stream_state *) user;
    st.ids.push_back(token);
    st.n_tokens++;

    if (st.ctx->abort_flag.load(std::memory_order_relaxed)) return false;

    if (st.p->on_partial) {
        // skip_special drops decoder_start, the forced code and eos - the same
        // call nllb-cli makes on the finished sequence, so the last partial a
        // UI sees is byte-identical to the final result.
        const std::string partial = *st.done + nllb_vocab_decode(st.ctx->vocab, st.ids, true);
        if (!st.p->on_partial(partial.c_str(), st.p->user_data)) {
            st.user_stop = true;
            return false;
        }
    }
    return true;
}

bool is_blank(const std::string & s) {
    for (char c : s) if (c != ' ' && c != '\t' && c != '\r') return false;
    return true;
}

} // namespace

extern "C" int nllb_translate(nllb_ctx * ctx, const char * text,
                              nllb_translate_params params,
                              char ** out_text, int * out_n_tokens,
                              char * err, size_t err_len) {
    if (out_text)     *out_text = nullptr;
    if (out_n_tokens) *out_n_tokens = 0;
    if (!ctx || !text || !out_text) {
        set_err(err, err_len, "nllb_translate: NULL argument");
        return NLLB_TRANSLATE_ERROR;
    }
    ctx->abort_flag.store(false, std::memory_order_relaxed);

    const char * src = params.src_lang ? params.src_lang : "";
    const char * tgt = params.tgt_lang ? params.tgt_lang : "";

    // Resolve both codes BEFORE any work, as nllb-cli does: a bad target code
    // would otherwise reach nllb_generate as forced_bos_id = -1 and the model
    // would translate into whatever language it felt like.
    const int32_t tgt_id = nllb_vocab_lang_id(ctx->vocab, tgt);
    if (tgt_id < 0) {
        set_err(err, err_len, std::string("unknown target language code '") + tgt + "'");
        return NLLB_TRANSLATE_ERROR;
    }
    if (nllb_vocab_lang_id(ctx->vocab, src) < 0) {
        set_err(err, err_len, std::string("unknown source language code '") + src + "'");
        return NLLB_TRANSLATE_ERROR;
    }

    nllb_generate_params gp;
    gp.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 256;
    gp.use_cache      = true;          // the CLI default; ids identical to un-cached
    gp.kv_type        = NLLB_KV_F32;   // f16 KV is 3% SLOWER and costs headroom
    gp.forced_bos_id  = tgt_id;
    gp.on_token       = on_token;

    std::string done;       // finished output lines, each followed by '\n'
    int  total_tokens = 0;
    bool aborted      = false;

    // Split on '\n'. A trailing newline yields a trailing empty line, which is
    // what keeps "one line in, one line out" true at the end of the text too.
    std::vector<std::string> lines;
    {
        std::string cur;
        for (const char * c = text; *c; ++c) {
            if (*c == '\n') { lines.push_back(cur); cur.clear(); }
            else            cur.push_back(*c);
        }
        lines.push_back(cur);
    }

    for (size_t li = 0; li < lines.size(); ++li) {
        std::string line = lines[li];
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const bool last = li + 1 == lines.size();

        if (ctx->abort_flag.load(std::memory_order_relaxed)) { aborted = true; break; }

        std::string out_line;
        if (!is_blank(line)) {
            std::string e;
            std::vector<int32_t> src_ids;
            if (!nllb_vocab_encode(ctx->vocab, line, src, src_ids, e)) {
                set_err(err, err_len, "tokenize failed on line " + std::to_string(li + 1) + ": " + e);
                return NLLB_TRANSLATE_ERROR;
            }

            stream_state st;
            st.ctx  = ctx;
            st.p    = &params;
            st.done = &done;
            gp.on_token_user = &st;

            nllb_generate_result gen;
            if (!nllb_generate(ctx->model, src_ids, gp, ctx->n_threads, gen, e)) {
                set_err(err, err_len, "generate failed on line " + std::to_string(li + 1) + ": " + e);
                return NLLB_TRANSLATE_ERROR;
            }
            total_tokens += st.n_tokens;
            out_line = nllb_vocab_decode(ctx->vocab, gen.ids, /*skip_special=*/true);

            if (gen.stop == NLLB_STOP_ABORTED) {
                done += out_line;
                aborted = true;
                break;
            }
        }

        done += out_line;
        if (!last) {
            done += '\n';
            // A blank line produces no tokens and therefore no hook call, so
            // report it here or the UI would lag one line behind until the
            // next sentence's first token.
            if (params.on_partial && is_blank(line) &&
                !params.on_partial(done.c_str(), params.user_data)) {
                aborted = true;
                break;
            }
        }
    }

    if (out_n_tokens) *out_n_tokens = total_tokens;
    *out_text = dup_cstr(done);
    if (!*out_text) { set_err(err, err_len, "out of memory"); return NLLB_TRANSLATE_ERROR; }
    return aborted ? NLLB_TRANSLATE_ABORTED : NLLB_TRANSLATE_OK;
}
