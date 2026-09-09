#include "nllb.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef _WIN32
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define NLLB_HAVE_MMAP 1
#else
#  define NLLB_HAVE_MMAP 0
#endif

// --------------------------------------------------------------------------
// metadata helpers
// --------------------------------------------------------------------------

namespace {

bool kv_u32(const gguf_context * g, const char * key, int32_t & dst) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    dst = (int32_t) gguf_get_val_u32(g, id);
    return true;
}

bool kv_f32(const gguf_context * g, const char * key, float & dst) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    dst = gguf_get_val_f32(g, id);
    return true;
}

// Wall clock in milliseconds, for nllb_step_timing. steady_clock rather than
// system_clock: the numbers are differences, and a clock that can be stepped
// backwards by NTP has no business measuring a 40 ms decoder step.
using clk = std::chrono::steady_clock;

double ms_between(const clk::time_point & a, const clk::time_point & b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

ggml_tensor * need(ggml_context * ctx, const std::string & name, std::string & err) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t && err.empty()) err = "missing tensor: " + name;
    return t;
}

double nllb_now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

// Map the whole file read-only. Returns nullptr and leaves err untouched on
// any failure - the caller falls back to reading, which is always correct.
void * nllb_mmap_file(const std::string & path, size_t & out_size) {
#if NLLB_HAVE_MMAP
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;

    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return nullptr; }

    const double t_open = nllb_now_ms();
    void * addr = ::mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    // The mapping outlives the descriptor: mmap holds its own reference to the
    // file, so closing fd here leaks nothing and keeps the model struct simple.
    ::close(fd);
    if (addr == MAP_FAILED) return nullptr;
    const double t_map = nllb_now_ms() - t_open;

    // MADV_WILLNEED IS OFF BY DEFAULT, AND THAT IS THE POINT OF THIS COMMENT.
    //
    // The obvious thing to do here is tell the kernel we will read the whole
    // file, so it can fault ahead instead of one page at a time. On Linux that
    // is an asynchronous readahead hint and costs nothing. On Darwin it is
    // SYNCHRONOUS - XNU pages the range in before returning - so it performs
    // exactly the eager load mmap exists to avoid, and measurably slower than
    // a plain read: 783 ms of "map" against 395 ms for read() of the same
    // 1.235 GB f16 file, with the subsequent fault-in then reporting a
    // fictitious 32 GB/s because the pages were already resident.
    //
    // So it is opt-in, and NLLB_LOAD_TRACE=1 prices it.
    double t_advise = 0.0;
    if (::getenv("NLLB_MADVISE")) {
        const double t0 = nllb_now_ms();
#  ifdef POSIX_MADV_WILLNEED
        ::posix_madvise(addr, (size_t) st.st_size, POSIX_MADV_WILLNEED);
#  elif defined(MADV_WILLNEED)
        ::madvise(addr, (size_t) st.st_size, MADV_WILLNEED);
#  endif
        t_advise = nllb_now_ms() - t0;
    }

    if (::getenv("NLLB_LOAD_TRACE")) {
        std::fprintf(stderr,
            "[load-trace] open+fstat+mmap %7.2f ms   madvise %7.2f ms%s\n",
            t_map, t_advise, ::getenv("NLLB_MADVISE") ? "" : " (off)");
    }

    out_size = (size_t) st.st_size;
    return addr;
#else
    (void) path; (void) out_size;
    return nullptr;
#endif
}

void nllb_munmap_file(void * addr, size_t size) {
#if NLLB_HAVE_MMAP
    if (addr) ::munmap(addr, size);
#else
    (void) addr; (void) size;
#endif
}

} // namespace

nllb_model::~nllb_model() {
    // Order matters: ctx_w's tensors point INTO the mapping when mapped, so
    // the context goes first and the mapping last. Freeing the map first
    // leaves dangling ->data behind for anything still holding the model.
    if (gguf)    { gguf_free(gguf);  gguf  = nullptr; }
    if (ctx_w)   { ggml_free(ctx_w); ctx_w = nullptr; }
    if (mapping) { nllb_munmap_file(mapping, mapping_size);
                   mapping = nullptr; mapping_size = 0; mapped = false; }
}

// --------------------------------------------------------------------------
// load
// --------------------------------------------------------------------------

bool nllb_model_load(const std::string & path, nllb_model & model, std::string & err,
                     bool use_mmap) {
    // Try the mapping FIRST, because it decides how gguf is asked to parse:
    // no_alloc = true creates the ggml_tensor structs with correct name, type
    // and ne but leaves ->data null and reads no bytes, which is exactly the
    // shape we need. If the map fails we fall back to no_alloc = false and
    // gguf copies the blob onto the heap as before.
    size_t map_size = 0;
    void * map_addr = use_mmap ? nllb_mmap_file(path, map_size) : nullptr;

    ggml_context * ctx_data = nullptr;
    gguf_init_params gp = {};
    gp.no_alloc = map_addr != nullptr;
    gp.ctx      = &ctx_data;

    const bool trace = ::getenv("NLLB_LOAD_TRACE") != nullptr;
    const double t_parse0 = nllb_now_ms();
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        if (map_addr) nllb_munmap_file(map_addr, map_size);
        err = "gguf_init_from_file failed: " + path;
        return false;
    }
    if (trace) {
        // With no_alloc = true this is metadata only - but "metadata only"
        // still means the tokenizer's ~7 MB of KV: 256204 token strings, the
        // charsmap and the grapheme tables. It is not free, and once the blob
        // copy is gone it is a visible share of what remains.
        std::fprintf(stderr, "[load-trace] gguf parse (%s) %7.2f ms\n",
                     gp.no_alloc ? "no_alloc, metadata only"
                                 : "reads the whole tensor blob",
                     nllb_now_ms() - t_parse0);
    }

    model.gguf  = g;
    model.ctx_w = ctx_data;

    if (map_addr) {
        // Point every tensor at its bytes inside the mapping. gguf_get_data_
        // offset is where the blob starts (padded to general.alignment) and
        // each tensor offset is relative to that.
        const double   t_fix0   = nllb_now_ms();
        const size_t   data_off = gguf_get_data_offset(g);
        const int64_t  n_tensor = gguf_get_n_tensors(g);
        const size_t   align    = gguf_get_alignment(g);
        char * const   base     = (char *) map_addr;

        // ggml's kernels want their rows aligned. mmap hands back a page-
        // aligned base and GGUF pads data_off and every tensor offset to
        // `align`, so the sum is aligned iff align is - refuse rather than
        // hand a misaligned pointer to a SIMD kernel and crash somewhere else.
        bool ok = align >= 16 && (align & (align - 1)) == 0 && data_off % align == 0;

        for (int64_t i = 0; ok && i < n_tensor; ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(ctx_data, name);
            if (!t) { ok = false; break; }

            const size_t off = gguf_get_tensor_offset(g, i);
            if (data_off + off + ggml_nbytes(t) > map_size) { ok = false; break; }
            t->data = base + data_off + off;
        }

        if (!ok) {
            // Nothing is half-done: drop the whole attempt and redo the load
            // by reading. A file we cannot map is not a file we cannot use.
            gguf_free(g);
            ggml_free(ctx_data);
            model.gguf = nullptr; model.ctx_w = nullptr;
            nllb_munmap_file(map_addr, map_size);
            return nllb_model_load(path, model, err, /*use_mmap =*/ false);
        }

        model.mapping      = map_addr;
        model.mapping_size = map_size;
        model.mapped       = true;

        if (trace) {
            std::fprintf(stderr,
                "[load-trace] pointer fixup, %lld tensors %7.2f ms\n",
                (long long) n_tensor, nllb_now_ms() - t_fix0);
        }
    }

    {
        const int64_t id = gguf_find_key(g, "general.architecture");
        const char * arch = id >= 0 ? gguf_get_val_str(g, id) : "";
        if (std::strcmp(arch, "nllb") != 0) {
            err = std::string("expected general.architecture=nllb, got '") + arch + "'";
            return false;
        }
    }

    nllb_hparams & hp = model.hp;
    bool ok = true;
    ok &= kv_u32(g, "nllb.embedding_length",              hp.d_model);
    ok &= kv_u32(g, "nllb.encoder.attention.head_count",  hp.n_head);
    ok &= kv_u32(g, "nllb.encoder.block_count",           hp.n_layer);
    ok &= kv_u32(g, "nllb.encoder.feed_forward_length",   hp.n_ff);
    if (!ok) { err = "missing required nllb.* metadata"; return false; }

    kv_u32(g, "nllb.vocab_size",       hp.n_vocab);
    kv_u32(g, "nllb.context_length",   hp.n_ctx);
    kv_u32(g, "nllb.position_offset",  hp.pos_offset);
    kv_f32(g, "nllb.attention.layer_norm_epsilon", hp.eps);
    if (!kv_f32(g, "nllb.embedding_scale", hp.embd_scale)) {
        hp.embd_scale = std::sqrt((float) hp.d_model);
    }
    kv_u32(g, "tokenizer.ggml.padding_token_id", hp.pad_id);
    kv_u32(g, "tokenizer.ggml.bos_token_id",     hp.bos_id);
    kv_u32(g, "tokenizer.ggml.eos_token_id",     hp.eos_id);
    // NLLB's decoder_start_token_id is eos, not bos. Read it if the converter
    // wrote it; fall back to eos rather than to the more obvious-looking bos.
    hp.dec_start_id = hp.eos_id;
    kv_u32(g, "nllb.decoder_start_token_id", hp.dec_start_id);

    // Decoder metadata is absent in --encoder-only files; that is not an error.
    model.has_decoder = kv_u32(g, "nllb.decoder.block_count", hp.n_dec_layer);
    if (model.has_decoder) {
        bool dok = true;
        dok &= kv_u32(g, "nllb.decoder.attention.head_count", hp.n_dec_head);
        dok &= kv_u32(g, "nllb.decoder.feed_forward_length",  hp.n_dec_ff);
        if (!dok) { err = "decoder.block_count present but decoder.* incomplete";
                    return false; }
    }

    if (hp.d_model % hp.n_head != 0) {
        err = "d_model not divisible by n_head";
        return false;
    }
    if (model.has_decoder && hp.n_dec_head > 0 && hp.d_model % hp.n_dec_head != 0) {
        err = "d_model not divisible by decoder n_head";
        return false;
    }

    model.tok_embd   = need(ctx_data, "token_embd.weight",     err);
    model.pos_embd   = need(ctx_data, "position_embd.weight",  err);
    model.out_norm_w = need(ctx_data, "enc.output_norm.weight", err);
    model.out_norm_b = need(ctx_data, "enc.output_norm.bias",   err);

    model.enc.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        nllb_enc_layer & l = model.enc[i];
        l.attn_norm_w = need(ctx_data, p + "attn_norm.weight", err);
        l.attn_norm_b = need(ctx_data, p + "attn_norm.bias",   err);
        l.wq = need(ctx_data, p + "attn_q.weight",   err);
        l.bq = need(ctx_data, p + "attn_q.bias",     err);
        l.wk = need(ctx_data, p + "attn_k.weight",   err);
        l.bk = need(ctx_data, p + "attn_k.bias",     err);
        l.wv = need(ctx_data, p + "attn_v.weight",   err);
        l.bv = need(ctx_data, p + "attn_v.bias",     err);
        l.wo = need(ctx_data, p + "attn_out.weight", err);
        l.bo = need(ctx_data, p + "attn_out.bias",   err);
        l.ffn_norm_w = need(ctx_data, p + "ffn_norm.weight", err);
        l.ffn_norm_b = need(ctx_data, p + "ffn_norm.bias",   err);
        l.w_up   = need(ctx_data, p + "ffn_up.weight",   err);
        l.b_up   = need(ctx_data, p + "ffn_up.bias",     err);
        l.w_down = need(ctx_data, p + "ffn_down.weight", err);
        l.b_down = need(ctx_data, p + "ffn_down.bias",   err);
    }
    if (model.has_decoder) {
        model.dec_out_norm_w = need(ctx_data, "dec.output_norm.weight", err);
        model.dec_out_norm_b = need(ctx_data, "dec.output_norm.bias",   err);
        // Tied lm_head is the NLLB default, so a missing output.weight is
        // expected rather than a failure.
        model.output_w = ggml_get_tensor(ctx_data, "output.weight");

        model.dec.resize(hp.n_dec_layer);
        for (int i = 0; i < hp.n_dec_layer; ++i) {
            const std::string p = "dec.blk." + std::to_string(i) + ".";
            nllb_dec_layer & l = model.dec[i];
            l.attn_norm_w = need(ctx_data, p + "attn_norm.weight", err);
            l.attn_norm_b = need(ctx_data, p + "attn_norm.bias",   err);
            l.wq = need(ctx_data, p + "attn_q.weight",   err);
            l.bq = need(ctx_data, p + "attn_q.bias",     err);
            l.wk = need(ctx_data, p + "attn_k.weight",   err);
            l.bk = need(ctx_data, p + "attn_k.bias",     err);
            l.wv = need(ctx_data, p + "attn_v.weight",   err);
            l.bv = need(ctx_data, p + "attn_v.bias",     err);
            l.wo = need(ctx_data, p + "attn_out.weight", err);
            l.bo = need(ctx_data, p + "attn_out.bias",   err);

            l.cross_norm_w = need(ctx_data, p + "cross_attn_norm.weight", err);
            l.cross_norm_b = need(ctx_data, p + "cross_attn_norm.bias",   err);
            l.cwq = need(ctx_data, p + "cross_attn_q.weight",   err);
            l.cbq = need(ctx_data, p + "cross_attn_q.bias",     err);
            l.cwk = need(ctx_data, p + "cross_attn_k.weight",   err);
            l.cbk = need(ctx_data, p + "cross_attn_k.bias",     err);
            l.cwv = need(ctx_data, p + "cross_attn_v.weight",   err);
            l.cbv = need(ctx_data, p + "cross_attn_v.bias",     err);
            l.cwo = need(ctx_data, p + "cross_attn_out.weight", err);
            l.cbo = need(ctx_data, p + "cross_attn_out.bias",   err);

            l.ffn_norm_w = need(ctx_data, p + "ffn_norm.weight", err);
            l.ffn_norm_b = need(ctx_data, p + "ffn_norm.bias",   err);
            l.w_up   = need(ctx_data, p + "ffn_up.weight",   err);
            l.b_up   = need(ctx_data, p + "ffn_up.bias",     err);
            l.w_down = need(ctx_data, p + "ffn_down.weight", err);
            l.b_down = need(ctx_data, p + "ffn_down.bias",   err);
        }
    }

    if (!err.empty()) return false;

    // The position table must cover n_ctx + pos_offset rows, because token j
    // reads row j + pos_offset. If the converter wrote a short table this is
    // where you find out, rather than by reading garbage at long sequence
    // lengths.
    if (model.pos_embd->ne[1] < hp.n_ctx + hp.pos_offset) {
        err = "position_embd has " + std::to_string(model.pos_embd->ne[1]) +
              " rows, need " + std::to_string(hp.n_ctx + hp.pos_offset);
        return false;
    }
    return true;
}

// --------------------------------------------------------------------------
// encoder graph
// --------------------------------------------------------------------------

bool nllb_encode(const nllb_model & model,
                 const std::vector<int32_t> & ids,
                 int n_threads,
                 bool want_hidden_states,
                 nllb_encode_result & out,
                 std::string & err) {
    const nllb_hparams & hp = model.hp;
    const int T  = (int) ids.size();
    const int D  = hp.d_model;
    const int H  = hp.n_head;
    const int HD = hp.head_dim();

    if (T <= 0)                      { err = "empty input";        return false; }
    if (T > hp.n_ctx && hp.n_ctx > 0){ err = "sequence too long";  return false; }

    // ggml_graph_compute_with_ctx allocates work buffers inside the context
    // too, so be generous. Every intermediate stays live (no reuse), which is
    // what lets us read per-layer states back out afterwards.
    size_t per_layer = (size_t) 32 * T * D + (size_t) 6 * T * T * H;
    size_t mem = ((size_t) hp.n_layer * per_layer + (size_t) 64 * T * D) * sizeof(float);
    mem = mem * 2 + 64u * 1024 * 1024 + (size_t) ggml_tensor_overhead() * 4096
        + ggml_graph_overhead();

    ggml_init_params ip = {};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed"; return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(inp_ids, "inp_ids");
    ggml_set_name(inp_pos, "inp_pos");
    std::memcpy(inp_ids->data, ids.data(), T * sizeof(int32_t));

    // HF: position_ids = cumsum(ne_pad) * ne_pad + padding_idx, and the table
    // was built with offset 2 == padding_idx + 1. For an unpadded sequence
    // this is simply j + 2 - but write it the general way so padded batches
    // stay correct.
    {
        int32_t * p = (int32_t *) inp_pos->data;
        int32_t run = 0;
        for (int j = 0; j < T; ++j) {
            const bool is_pad = (ids[j] == hp.pad_id);
            if (!is_pad) ++run;
            p[j] = is_pad ? hp.pad_id : run + hp.pad_id;
        }
    }

    // embeddings: token * sqrt(d_model), then + position (position NOT scaled)
    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_ids);
    cur = ggml_scale(ctx, cur, hp.embd_scale);
    cur = ggml_add(ctx, cur, ggml_get_rows(ctx, model.pos_embd, inp_pos));

    std::vector<ggml_tensor *> hs;
    hs.reserve(hp.n_layer + 1);

    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < hp.n_layer; ++il) {
        const nllb_enc_layer & l = model.enc[il];
        hs.push_back(cur);   // HF hidden_states[il] == input to block il

        // ---- self-attention (pre-layernorm) ----
        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.attn_norm_w), l.attn_norm_b);

        // HF scales Q, not the logits. Same result, but keeping the placement
        // means intermediate dumps line up when bisecting.
        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, h), l.bq);
        Q = ggml_scale(ctx, Q, attn_scale);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, h), l.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, h), l.bv);

        Q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, Q, HD, H, T), 0, 2, 1, 3));  // (HD, T, H)
        K = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, K, HD, H, T), 0, 2, 1, 3));  // (HD, T, H)
        V = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, V, HD, H, T), 1, 2, 0, 3));  // (T, HD, H)

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);               // (T, T, H)
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);             // (HD, T, H)

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // (HD, H, T)
        cur = ggml_reshape_2d(ctx, kqv, D, T);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, cur), l.bo);
        cur = ggml_add(ctx, cur, residual);

        // ---- feed forward ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.ffn_norm_w), l.ffn_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_up, h), l.b_up);
        h = ggml_relu(ctx, h);                                    // NLLB is relu
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_down, h), l.b_down);
        cur = ggml_add(ctx, h, residual);
    }

    cur = ggml_norm(ctx, cur, hp.eps);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, model.out_norm_w), model.out_norm_b);
    ggml_set_name(cur, "enc_out");
    hs.push_back(cur);   // HF hidden_states[n_layer] == post-final-norm output

    ggml_build_forward_expand(gf, cur);
    if (want_hidden_states) {
        for (ggml_tensor * t : hs) ggml_build_forward_expand(gf, t);
    }

    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        err = "ggml_graph_compute failed, status " + std::to_string((int) st);
        ggml_free(ctx);
        return false;
    }

    out.n_tokens = T;
    out.d_model  = D;
    out.output.resize((size_t) T * D);
    std::memcpy(out.output.data(), cur->data, out.output.size() * sizeof(float));

    out.hidden_states.clear();
    if (want_hidden_states) {
        out.hidden_states.resize(hs.size());
        for (size_t i = 0; i < hs.size(); ++i) {
            out.hidden_states[i].resize((size_t) T * D);
            std::memcpy(out.hidden_states[i].data(), hs[i]->data,
                        out.hidden_states[i].size() * sizeof(float));
        }
    }

    ggml_free(ctx);
    return true;
}

// --------------------------------------------------------------------------
// decoder graph
//
// Teacher-forced: the whole target sequence in one pass, causally masked, no
// KV cache. Cross-attention K and V are computed once per block from the
// frozen encoder output - they do not depend on the decoder step, which is
// exactly why they are the first thing worth caching once this is green.
// --------------------------------------------------------------------------

bool nllb_decode(const nllb_model & model,
                 const nllb_encode_result & enc,
                 const std::vector<int32_t> & dec_ids,
                 const std::vector<int32_t> & src_ids,
                 int n_threads,
                 bool want_hidden_states,
                 int logits_mode,
                 nllb_decode_result & out,
                 std::string & err,
                 int pos_mode) {
    const nllb_hparams & hp = model.hp;
    const int Td = (int) dec_ids.size();
    const int Ts = enc.n_tokens;
    const int D  = hp.d_model;
    const int H  = hp.n_dec_head;
    const int HD = hp.dec_head_dim();
    const int L  = hp.n_dec_layer;

    if (!model.has_decoder)          { err = "GGUF has no decoder (converted with --encoder-only?)"; return false; }
    if (Td <= 0)                     { err = "empty decoder input";  return false; }
    if (Ts <= 0)                     { err = "empty encoder output"; return false; }
    if (enc.d_model != D)            { err = "encoder output d_model mismatch"; return false; }
    if ((int) enc.output.size() != Ts * D) { err = "encoder output size mismatch"; return false; }
    if (Td > hp.n_ctx && hp.n_ctx > 0)     { err = "target sequence too long"; return false; }

    // Cross-attention here is unmasked, exactly as the encoder's self-attention
    // is. That is correct for a single unpadded source and wrong for a padded
    // batch, so refuse the padded case loudly instead of returning plausible
    // numbers. Lifting this means a mask tensor into ggml_soft_max_ext on the
    // cross-attention logits, and goldens that actually exercise padding.
    for (int32_t id : src_ids) {
        if (id == hp.pad_id) {
            err = "padded source sequences are not supported yet "
                  "(cross-attention is unmasked); see nllb_decode";
            return false;
        }
    }

    const bool want_logits = logits_mode != NLLB_LOGITS_NONE;
    const int  logit_rows  = logits_mode == NLLB_LOGITS_LAST ? 1 : Td;
    // Size the logits budget from the tensor that actually produces them, not
    // from nllb.vocab_size - that key is optional metadata and defaults to 0,
    // which would silently budget nothing for a [vocab, Td] result.
    const int  vocab_rows  = (int) model.lm_head()->ne[1];

    size_t per_layer = (size_t) 48 * Td * D
                     + (size_t)  8 * Td * Td * H
                     + (size_t) 12 * Td * Ts * H
                     + (size_t)  8 * Ts * D
                     + (size_t)  6 * Td * hp.n_dec_ff;
    size_t mem = ((size_t) L * per_layer + (size_t) 64 * Td * D
                  + (size_t) 2 * Ts * D) * sizeof(float);
    if (want_logits) mem += (size_t) 3 * logit_rows * vocab_rows * sizeof(float);
    mem = mem * 2 + 128u * 1024 * 1024 + (size_t) ggml_tensor_overhead() * 8192
        + ggml_graph_overhead();

    ggml_init_params ip = {};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed (decoder)"; return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Td);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Td);
    ggml_set_name(inp_ids, "dec_ids");
    ggml_set_name(inp_pos, "dec_pos");
    std::memcpy(inp_ids->data, dec_ids.data(), Td * sizeof(int32_t));

    // Position ids. The two rules below agree on any sequence without pad_id in
    // it, which is every golden case here, and disagree as soon as one appears.
    //
    //   NLLB_POS_CUMSUM   cumsum(mask) * mask + pad_id
    //                     what HF computes in a forward pass with use_cache=
    //                     False, i.e. what the teacher-forced goldens contain.
    //                     A pad shifts every later position down by one.
    //
    //   NLLB_POS_ABSOLUTE (j + 1) + pad_id for a non-pad token, pad_id for a
    //                     pad. What HF computes inside generate(): with a cache
    //                     it feeds one token with past_key_values_length = j,
    //                     and its formula is (cumsum(mask) + n_past) * mask +
    //                     pad_id where the cumsum restarts every step - so
    //                     n_past is inside the mask and earlier pads do not
    //                     shift anything.
    //
    // This is HF being inconsistent with itself, not a choice made here. The
    // caller picks the rule that matches the reference it is being diffed
    // against; see nllb_pos_mode in nllb.h.
    {
        int32_t * p = (int32_t *) inp_pos->data;
        int32_t run = 0;
        for (int j = 0; j < Td; ++j) {
            const bool is_pad = (dec_ids[j] == hp.pad_id);
            if (!is_pad) ++run;
            const int32_t idx = (pos_mode == NLLB_POS_ABSOLUTE) ? (j + 1) : run;
            p[j] = is_pad ? hp.pad_id : idx + hp.pad_id;
        }
    }

    // Encoder output as a graph input: ne = [D, Ts], matching the row-major
    // [Ts, D] buffer nllb_encode returns.
    ggml_tensor * enc_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Ts);
    ggml_set_name(enc_t, "enc_out");
    std::memcpy(enc_t->data, enc.output.data(), (size_t) Ts * D * sizeof(float));

    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_ids);
    cur = ggml_scale(ctx, cur, hp.embd_scale);
    cur = ggml_add(ctx, cur, ggml_get_rows(ctx, model.pos_embd, inp_pos));

    std::vector<ggml_tensor *> hs;
    hs.reserve(L + 1);

    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < L; ++il) {
        const nllb_dec_layer & l = model.dec[il];
        hs.push_back(cur);   // HF decoder_hidden_states[il] == input to block il

        // ---- causal self-attention ----
        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.attn_norm_w), l.attn_norm_b);

        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, h), l.bq);
        Q = ggml_scale(ctx, Q, attn_scale);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, h), l.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, h), l.bv);

        Q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, Q, HD, H, Td), 0, 2, 1, 3));  // (HD, Td, H)
        K = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, K, HD, H, Td), 0, 2, 1, 3));  // (HD, Td, H)
        V = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, V, HD, H, Td), 1, 2, 0, 3));  // (Td, HD, H)

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                // (Td_k, Td_q, H)
        // The one structural difference from the encoder. n_past is 0 because
        // this is the un-cached path; with a KV cache it becomes the number of
        // already-decoded tokens.
        kq = ggml_diag_mask_inf(ctx, kq, 0);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);              // (HD, Td, H)

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // (HD, H, Td)
        cur = ggml_reshape_2d(ctx, kqv, D, Td);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, cur), l.bo);
        cur = ggml_add(ctx, cur, residual);

        // ---- cross-attention against the frozen encoder output ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.cross_norm_w), l.cross_norm_b);

        // Q from the decoder (Td rows), K and V from the encoder (Ts rows).
        // Mixing these up produces a graph that builds, runs, and is wrong.
        ggml_tensor * cQ = ggml_add(ctx, ggml_mul_mat(ctx, l.cwq, h), l.cbq);
        cQ = ggml_scale(ctx, cQ, attn_scale);
        ggml_tensor * cK = ggml_add(ctx, ggml_mul_mat(ctx, l.cwk, enc_t), l.cbk);
        ggml_tensor * cV = ggml_add(ctx, ggml_mul_mat(ctx, l.cwv, enc_t), l.cbv);

        cQ = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cQ, HD, H, Td), 0, 2, 1, 3));  // (HD, Td, H)
        cK = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cK, HD, H, Ts), 0, 2, 1, 3));  // (HD, Ts, H)
        cV = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cV, HD, H, Ts), 1, 2, 0, 3));  // (Ts, HD, H)

        ggml_tensor * ckq = ggml_mul_mat(ctx, cK, cQ);              // (Ts, Td, H)
        // No causal mask here: every target position may see the whole source.
        ckq = ggml_soft_max_ext(ctx, ckq, nullptr, 1.0f, 0.0f);
        ggml_tensor * ckqv = ggml_mul_mat(ctx, cV, ckq);            // (HD, Td, H)

        ckqv = ggml_cont(ctx, ggml_permute(ctx, ckqv, 0, 2, 1, 3)); // (HD, H, Td)
        cur = ggml_reshape_2d(ctx, ckqv, D, Td);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.cwo, cur), l.cbo);
        cur = ggml_add(ctx, cur, residual);

        // ---- feed forward ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.ffn_norm_w), l.ffn_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_up, h), l.b_up);
        h = ggml_relu(ctx, h);                                     // NLLB is relu
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_down, h), l.b_down);
        cur = ggml_add(ctx, h, residual);
    }

    cur = ggml_norm(ctx, cur, hp.eps);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, model.dec_out_norm_w), model.dec_out_norm_b);
    ggml_set_name(cur, "dec_out");
    hs.push_back(cur);   // HF decoder_hidden_states[n_dec_layer]

    // lm_head, tied to token_embd unless the checkpoint untied it. ne of the
    // embedding is [D, vocab], so mul_mat against [D, Td] gives [vocab, Td] -
    // row-major logits[t][v], no transpose anywhere.
    //
    // NLLB_LOGITS_LAST projects only the final row. Greedy decoding reads
    // nothing else, and on the 600M this is a [256206, 1024] x [1024, Td]
    // matmul - the dominant cost of the whole graph at Td > 1.
    ggml_tensor * logits = nullptr;
    if (want_logits) {
        ggml_tensor * lm_in = cur;
        if (logits_mode == NLLB_LOGITS_LAST && Td > 1) {
            lm_in = ggml_cont(ctx, ggml_view_2d(ctx, cur, D, 1, cur->nb[1],
                                                (size_t)(Td - 1) * cur->nb[1]));
        }
        logits = ggml_mul_mat(ctx, model.lm_head(), lm_in);
        ggml_set_name(logits, "logits");
        ggml_build_forward_expand(gf, logits);
    }

    ggml_build_forward_expand(gf, cur);
    if (want_hidden_states) {
        for (ggml_tensor * t : hs) ggml_build_forward_expand(gf, t);
    }

    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        err = "ggml_graph_compute failed (decoder), status " + std::to_string((int) st);
        ggml_free(ctx);
        return false;
    }

    out.n_tokens = Td;
    out.d_model  = D;
    out.n_vocab  = (int32_t) model.lm_head()->ne[1];
    out.output.resize((size_t) Td * D);
    std::memcpy(out.output.data(), cur->data, out.output.size() * sizeof(float));

    out.logits.clear();
    out.n_logit_rows = 0;
    if (want_logits) {
        out.n_logit_rows = logit_rows;
        out.logits.resize((size_t) logit_rows * out.n_vocab);
        std::memcpy(out.logits.data(), logits->data, out.logits.size() * sizeof(float));
    }

    out.hidden_states.clear();
    if (want_hidden_states) {
        out.hidden_states.resize(hs.size());
        for (size_t i = 0; i < hs.size(); ++i) {
            out.hidden_states[i].resize((size_t) Td * D);
            std::memcpy(out.hidden_states[i].data(), hs[i]->data,
                        out.hidden_states[i].size() * sizeof(float));
        }
    }

    ggml_free(ctx);
    return true;
}

// --------------------------------------------------------------------------
// KV cache
//
// Storage is in the shape attention consumes, not the shape the projections
// produce. The un-cached path spends a permute + ggml_cont on K and V every
// block; doing that here would copy the entire history once per block per step
// and give back most of what the cache is for. So the write does the permute
// once, into the slot for the new token, and the read is a plain view.
//
//   k_self [HD, n_ctx, H]   token t at row t of each head slice
//   v_self [n_ctx, HD, H]   transposed, because mul_mat wants (n_kv, HD, H)
//
// Both halves live in a ggml_context the cache owns. The per-step graph gets a
// context of its own and refers to these tensors across contexts, exactly as it
// already does for the model weights.
// --------------------------------------------------------------------------

nllb_kv_cache::~nllb_kv_cache() {
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
}

void nllb_kv_cache_reset(nllb_kv_cache & cache) {
    cache.n_past = 0;
}

bool nllb_kv_cache_init(const nllb_model & model,
                        const nllb_encode_result & enc,
                        int n_ctx_cache,
                        int n_threads,
                        nllb_kv_cache & cache,
                        std::string & err,
                        int kv_type) {
    const nllb_hparams & hp = model.hp;
    const int D  = hp.d_model;
    const int H  = hp.n_dec_head;
    const int HD = hp.dec_head_dim();
    const int L  = hp.n_dec_layer;
    const int Ts = enc.n_tokens;

    if (!model.has_decoder) { err = "GGUF has no decoder (converted with --encoder-only?)"; return false; }
    if (n_ctx_cache <= 0)   { err = "kv cache capacity must be > 0"; return false; }
    if (Ts <= 0)            { err = "empty encoder output"; return false; }
    if (enc.d_model != D)   { err = "encoder output d_model mismatch"; return false; }
    if ((int) enc.output.size() != Ts * D) { err = "encoder output size mismatch"; return false; }
    if (hp.n_ctx > 0 && n_ctx_cache > hp.n_ctx) {
        err = "kv cache capacity " + std::to_string(n_ctx_cache) +
              " exceeds the model's n_ctx " + std::to_string(hp.n_ctx);
        return false;
    }
    if (kv_type != NLLB_KV_F32 && kv_type != NLLB_KV_F16) {
        err = "unknown kv cache type " + std::to_string(kv_type);
        return false;
    }
    // The only place the type is turned into a ggml type. Everything below is
    // written in terms of `kvt` and ggml_type_size(kvt), so adding a third
    // option here is the whole change - and the arena formulas further down
    // stay in units of f32 ACTIVATIONS, which is still correct because the
    // cache is not in the graph arena.
    const ggml_type kvt = kv_type == NLLB_KV_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
    cache.k_self.clear();  cache.v_self.clear();
    cache.k_cross.clear(); cache.v_cross.clear();

    // Four tensors per layer. The self half is sized by capacity, the cross
    // half by the source length - which is why a new sentence needs a new init
    // and not a reset.
    const size_t cache_elems = (size_t) L * 2 * ((size_t) n_ctx_cache * D + (size_t) Ts * D);
    ggml_init_params cp = {};
    cp.mem_size   = cache_elems * ggml_type_size(kvt)
                  + (size_t) ggml_tensor_overhead() * (4 * L + 16)
                  + 1024u * 1024;
    cp.mem_buffer = nullptr;
    cp.no_alloc   = false;
    cache.ctx = ggml_init(cp);
    if (!cache.ctx) { err = "ggml_init failed (kv cache)"; return false; }

    cache.kv_type = kv_type;
    cache.n_ctx   = n_ctx_cache;
    cache.n_past  = 0;
    cache.n_src   = Ts;
    cache.d_model = D;
    cache.n_head  = H;
    cache.n_layer = L;

    cache.k_self.resize(L);  cache.v_self.resize(L);
    cache.k_cross.resize(L); cache.v_cross.resize(L);
    for (int il = 0; il < L; ++il) {
        cache.k_self[il]  = ggml_new_tensor_3d(cache.ctx, kvt, HD, n_ctx_cache, H);
        cache.v_self[il]  = ggml_new_tensor_3d(cache.ctx, kvt, n_ctx_cache, HD, H);
        cache.k_cross[il] = ggml_new_tensor_3d(cache.ctx, kvt, HD, Ts, H);
        cache.v_cross[il] = ggml_new_tensor_3d(cache.ctx, kvt, Ts, HD, H);
        if (!cache.k_self[il] || !cache.v_self[il] ||
            !cache.k_cross[il] || !cache.v_cross[il]) {
            err = "kv cache allocation failed at layer " + std::to_string(il);
            ggml_free(cache.ctx); cache.ctx = nullptr;
            return false;
        }
    }

    // ---- fill the cross-attention half: one graph, once per source ----
    size_t mem = ((size_t) L * 8 * Ts * D + (size_t) 8 * Ts * D) * sizeof(float);
    mem = mem * 2 + 32u * 1024 * 1024 + (size_t) ggml_tensor_overhead() * 1024
        + ggml_graph_overhead();

    ggml_init_params ip = {};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed (cross-attention cache build)";
                ggml_free(cache.ctx); cache.ctx = nullptr; return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * enc_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Ts);
    ggml_set_name(enc_t, "enc_out");
    std::memcpy(enc_t->data, enc.output.data(), (size_t) Ts * D * sizeof(float));

    for (int il = 0; il < L; ++il) {
        const nllb_dec_layer & l = model.dec[il];

        // Identical arithmetic to the un-cached path's cK/cV, including the
        // bias and the absence of the attention scale (that goes on Q). If this
        // ever stops matching, tests/test_cache.cpp reports it as a
        // cross-attention divergence at layer il rather than as wrong ids.
        ggml_tensor * cK = ggml_add(ctx, ggml_mul_mat(ctx, l.cwk, enc_t), l.cbk);
        ggml_tensor * cV = ggml_add(ctx, ggml_mul_mat(ctx, l.cwv, enc_t), l.cbv);

        cK = ggml_permute(ctx, ggml_reshape_3d(ctx, cK, HD, H, Ts), 0, 2, 1, 3); // (HD, Ts, H)
        cV = ggml_permute(ctx, ggml_reshape_3d(ctx, cV, HD, H, Ts), 1, 2, 0, 3); // (Ts, HD, H)

        ggml_build_forward_expand(gf, ggml_cpy(ctx, cK, cache.k_cross[il]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cV, cache.v_cross[il]));
    }

    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    ggml_free(ctx);
    if (st != GGML_STATUS_SUCCESS) {
        err = "ggml_graph_compute failed (cross-attention cache), status " +
              std::to_string((int) st);
        ggml_free(cache.ctx); cache.ctx = nullptr;
        return false;
    }
    return true;
}

bool nllb_decode_cached(const nllb_model & model,
                        nllb_kv_cache & cache,
                        const std::vector<int32_t> & new_ids,
                        int n_threads,
                        bool want_hidden_states,
                        int logits_mode,
                        nllb_decode_result & out,
                        std::string & err,
                        nllb_step_timing * timing) {
    const clk::time_point t_enter = clk::now();
    if (timing) *timing = nllb_step_timing();

    const nllb_hparams & hp = model.hp;
    const int Tn = (int) new_ids.size();
    const int D  = hp.d_model;
    const int H  = hp.n_dec_head;
    const int HD = hp.dec_head_dim();
    const int L  = hp.n_dec_layer;
    const int Ts = cache.n_src;

    if (!model.has_decoder) { err = "GGUF has no decoder (converted with --encoder-only?)"; return false; }
    if (!cache.ctx)         { err = "kv cache is not initialised (call nllb_kv_cache_init)"; return false; }
    if (Tn <= 0)            { err = "empty decoder input"; return false; }
    if (cache.n_layer != L || cache.d_model != D || cache.n_head != H) {
        err = "kv cache was built for a different model shape";
        return false;
    }
    const int n_past = cache.n_past;
    const int n_kv   = n_past + Tn;
    if (n_kv > cache.n_ctx) {
        err = "kv cache is full: " + std::to_string(n_past) + " + " +
              std::to_string(Tn) + " > capacity " + std::to_string(cache.n_ctx);
        return false;
    }

    const bool want_logits = logits_mode != NLLB_LOGITS_NONE;
    const int  logit_rows  = logits_mode == NLLB_LOGITS_LAST ? 1 : Tn;
    const int  vocab_rows  = (int) model.lm_head()->ne[1];

    // No Ts*D term per layer: the cross-attention projections are already in
    // the cache, which is the largest single saving here at short target
    // lengths. What is left scales with Tn, not with the prefix.
    size_t per_layer = (size_t) 48 * Tn * D
                     + (size_t) 12 * Tn * n_kv * H
                     + (size_t) 12 * Tn * Ts  * H
                     + (size_t)  6 * Tn * hp.n_dec_ff;
    size_t mem = ((size_t) L * per_layer + (size_t) 64 * Tn * D) * sizeof(float);
    if (want_logits) mem += (size_t) 3 * logit_rows * vocab_rows * sizeof(float);
    mem = mem * 2 + 64u * 1024 * 1024 + (size_t) ggml_tensor_overhead() * 8192
        + ggml_graph_overhead();

    const clk::time_point t_alloc0 = clk::now();
    ggml_init_params ip = {};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    // Reuse the arena across steps when the cache is carrying one, so the pages
    // are faulted in once rather than once per token. `mem_size` is the
    // buffer's real size rather than this step's estimate; ggml bump-allocates
    // within it either way. ggml_free will not free a buffer it did not
    // allocate (ggml.c: mem_buffer_owned = params.mem_buffer ? false : true),
    // so this allocation survives the loop and dies with the cache.
    //
    // `new char[]` and not vector/assign/memset: default-initialised, so the
    // pages stay untouched and the first step faults in only the ~4 MB it
    // actually uses. Pre-zeroing all 80 MB is a 360 ms bill against a 4 ms/step
    // saving - see nllb_kv_cache::scratch in nllb.h.
    //
    // THE HEADROOM IS NOT DECORATION. `mem` is not constant across steps: the
    // per-layer term carries 12 * Tn * n_kv * H and n_kv grows by one every
    // step, so a buffer grown to exactly `mem` is short again on the very next
    // one. With a zeroing allocator that mistake cost 50 ms/step and 20694
    // minor faults/step - worse than not reusing at all, and it looked like
    // reuse being a bad idea rather than like this line being wrong. Growing by
    // a quarter turns thousands of steps into one allocation.
    if (cache.reuse_scratch) {
        if (cache.scratch_size < mem) {
            cache.scratch_size = mem + mem / 4;
            cache.scratch.reset(new char[cache.scratch_size]);
        }
        ip.mem_size   = cache.scratch_size;
        ip.mem_buffer = cache.scratch.get();
    }
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed (cached decoder)"; return false; }
    const clk::time_point t_alloc1 = clk::now();
    if (timing) {
        timing->alloc = ms_between(t_alloc0, t_alloc1);
        timing->arena = ip.mem_size;
        timing->scratch_reused = cache.reuse_scratch;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tn);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tn);
    ggml_set_name(inp_ids, "dec_ids");
    ggml_set_name(inp_pos, "dec_pos");
    std::memcpy(inp_ids->data, new_ids.data(), Tn * sizeof(int32_t));

    // NLLB_POS_ABSOLUTE, and only that. HF's cached rule is
    // (cumsum(mask) + n_past) * mask + pad_id with the cumsum restarting each
    // step, which for a token at absolute index j collapses to j + 1 + pad_id -
    // or pad_id if the token IS pad. The un-cached CUMSUM rule cannot be
    // evaluated from n_past alone (it depends on how many pads the prefix
    // contained), and reproducing it here would mean matching HF's *other*
    // path, so it is not offered. See nllb_pos_mode in nllb.h.
    {
        int32_t * p = (int32_t *) inp_pos->data;
        for (int i = 0; i < Tn; ++i) {
            const int32_t j = n_past + i;
            p[i] = (new_ids[i] == hp.pad_id) ? hp.pad_id : j + 1 + hp.pad_id;
        }
    }

    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_ids);
    cur = ggml_scale(ctx, cur, hp.embd_scale);
    cur = ggml_add(ctx, cur, ggml_get_rows(ctx, model.pos_embd, inp_pos));

    std::vector<ggml_tensor *> hs;
    hs.reserve(L + 1);

    const float attn_scale = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < L; ++il) {
        const nllb_dec_layer & l = model.dec[il];
        hs.push_back(cur);

        // ---- causal self-attention over cache + new tokens ----
        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.attn_norm_w), l.attn_norm_b);

        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, h), l.bq);
        Q = ggml_scale(ctx, Q, attn_scale);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, h), l.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, h), l.bv);

        Q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, Q, HD, H, Tn), 0, 2, 1, 3));   // (HD, Tn, H)

        ggml_tensor * kc = cache.k_self[il];
        ggml_tensor * vc = cache.v_self[il];

        // Write the new rows in the layout the read expects. ggml_cpy handles
        // the strided destination, which is what buys the read below being a
        // view rather than a copy of the whole history.
        ggml_tensor * Kp = ggml_permute(ctx,
                ggml_reshape_3d(ctx, K, HD, H, Tn), 0, 2, 1, 3);    // (HD, Tn, H)
        ggml_tensor * Vp = ggml_permute(ctx,
                ggml_reshape_3d(ctx, V, HD, H, Tn), 1, 2, 0, 3);    // (Tn, HD, H)

        ggml_tensor * k_dst = ggml_view_3d(ctx, kc, HD, Tn, H,
                kc->nb[1], kc->nb[2], (size_t) n_past * kc->nb[1]);
        ggml_tensor * v_dst = ggml_view_3d(ctx, vc, Tn, HD, H,
                vc->nb[1], vc->nb[2], (size_t) n_past * vc->nb[0]);

        // ORDER MATTERS, and not in the way it first looks. These copies have
        // no data dependency on the reads below, so only graph order makes the
        // write happen first: ggml_graph_compute walks gf->nodes in insertion
        // order, and ggml_build_forward_expand appends depth-first.
        //
        // What actually protects this graph is that NOTHING downstream is
        // expanded during the layer loop - `cur` and the logits enter the graph
        // once, at the end - so the copies are the only nodes inserted here and
        // land ahead of every read. Measured, not assumed: moving these two
        // calls to the bottom of the block changes nothing (the reads are still
        // expanded later), while expanding the actual kq node before them
        // breaks 9 of 9 cases at layer 1. See claude/step5-cpp-kv-cache.md.
        //
        // The rule to keep, therefore: do not expand anything that reads the
        // cache before these two lines. A per-layer debug hook is exactly the
        // innocent-looking change that would.
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kp, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vp, v_dst));

        ggml_tensor * Kv = ggml_view_3d(ctx, kc, HD, n_kv, H,
                kc->nb[1], kc->nb[2], 0);                           // (HD, n_kv, H)
        ggml_tensor * Vv = ggml_view_3d(ctx, vc, n_kv, HD, H,
                vc->nb[1], vc->nb[2], 0);                           // (n_kv, HD, H)

        ggml_tensor * kq = ggml_mul_mat(ctx, Kv, Q);                // (n_kv, Tn, H)
        // n_past stops being 0 here. Query row j is absolute position
        // n_past + j and may see keys 0..n_past + j; ggml_diag_mask_inf masks
        // exactly i > n_past + j.
        //
        // At Tn == 1 the correct offset masks nothing, and the error this hides
        // is one-sided: an offset that is too LARGE still masks nothing and is
        // invisible to single-token generation (n_past + 1 passes every case of
        // the step comparison and fails the multi-token ones), while an offset
        // that is too small masks a real key and is caught immediately. That
        // asymmetry is why tests/test_cache.cpp feeds the same prefix in chunks
        // of more than one token as well as one at a time - single-token
        // stepping alone cannot test this line.
        kq = ggml_diag_mask_inf(ctx, kq, n_past);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, Vv, kq);              // (HD, Tn, H)

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));   // (HD, H, Tn)
        cur = ggml_reshape_2d(ctx, kqv, D, Tn);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, cur), l.bo);
        cur = ggml_add(ctx, cur, residual);

        // ---- cross-attention: K and V come straight out of the cache ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.cross_norm_w), l.cross_norm_b);

        ggml_tensor * cQ = ggml_add(ctx, ggml_mul_mat(ctx, l.cwq, h), l.cbq);
        cQ = ggml_scale(ctx, cQ, attn_scale);
        cQ = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, cQ, HD, H, Tn), 0, 2, 1, 3));  // (HD, Tn, H)

        ggml_tensor * ckq = ggml_mul_mat(ctx, cache.k_cross[il], cQ);  // (Ts, Tn, H)
        // Still unmasked: every target position may see the whole source.
        ckq = ggml_soft_max_ext(ctx, ckq, nullptr, 1.0f, 0.0f);
        ggml_tensor * ckqv = ggml_mul_mat(ctx, cache.v_cross[il], ckq); // (HD, Tn, H)

        ckqv = ggml_cont(ctx, ggml_permute(ctx, ckqv, 0, 2, 1, 3)); // (HD, H, Tn)
        cur = ggml_reshape_2d(ctx, ckqv, D, Tn);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, l.cwo, cur), l.cbo);
        cur = ggml_add(ctx, cur, residual);

        // ---- feed forward ----
        residual = cur;
        h = ggml_norm(ctx, cur, hp.eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, l.ffn_norm_w), l.ffn_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_up, h), l.b_up);
        h = ggml_relu(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, l.w_down, h), l.b_down);
        cur = ggml_add(ctx, h, residual);
    }

    cur = ggml_norm(ctx, cur, hp.eps);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, model.dec_out_norm_w), model.dec_out_norm_b);
    ggml_set_name(cur, "dec_out");
    hs.push_back(cur);

    ggml_tensor * logits = nullptr;
    if (want_logits) {
        ggml_tensor * lm_in = cur;
        if (logits_mode == NLLB_LOGITS_LAST && Tn > 1) {
            lm_in = ggml_cont(ctx, ggml_view_2d(ctx, cur, D, 1, cur->nb[1],
                                                (size_t)(Tn - 1) * cur->nb[1]));
        }
        logits = ggml_mul_mat(ctx, model.lm_head(), lm_in);
        ggml_set_name(logits, "logits");
        ggml_build_forward_expand(gf, logits);
    }

    ggml_build_forward_expand(gf, cur);
    if (want_hidden_states) {
        for (ggml_tensor * t : hs) ggml_build_forward_expand(gf, t);
    }

    const clk::time_point t_build1 = clk::now();
    if (timing) {
        timing->build   = ms_between(t_alloc1, t_build1);
        timing->n_nodes = ggml_graph_n_nodes(gf);
    }

    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        err = "ggml_graph_compute failed (cached decoder), status " + std::to_string((int) st);
        ggml_free(ctx);
        return false;
    }
    const clk::time_point t_compute1 = clk::now();
    if (timing) timing->compute = ms_between(t_build1, t_compute1);

    out.n_tokens = Tn;
    out.d_model  = D;
    out.n_vocab  = (int32_t) model.lm_head()->ne[1];
    out.output.resize((size_t) Tn * D);
    std::memcpy(out.output.data(), cur->data, out.output.size() * sizeof(float));

    out.logits.clear();
    out.n_logit_rows = 0;
    if (want_logits) {
        out.n_logit_rows = logit_rows;
        out.logits.resize((size_t) logit_rows * out.n_vocab);
        std::memcpy(out.logits.data(), logits->data, out.logits.size() * sizeof(float));
    }

    out.hidden_states.clear();
    if (want_hidden_states) {
        out.hidden_states.resize(hs.size());
        for (size_t i = 0; i < hs.size(); ++i) {
            out.hidden_states[i].resize((size_t) Tn * D);
            std::memcpy(out.hidden_states[i].data(), hs[i]->data,
                        out.hidden_states[i].size() * sizeof(float));
        }
    }

    const clk::time_point t_read1 = clk::now();
    if (timing) timing->readback = ms_between(t_compute1, t_read1);

    ggml_free(ctx);

    if (timing) {
        timing->release = ms_between(t_read1, clk::now());
        timing->total   = ms_between(t_enter, clk::now());
    }

    // Only now, after the graph really ran: a caller that retries after a
    // failure must not find the cache half-advanced.
    cache.n_past = n_kv;
    return true;
}

// --------------------------------------------------------------------------
// greedy generation
//
// One encoder pass, then one decoder pass per emitted token over the whole
// prefix. No KV cache: step t rebuilds and recomputes everything steps
// 0..t-1 already computed, which is O(n^2) in the target length and exactly
// what the cached path will be diffed against.
//
// Parity with HF generate(num_beams=1, do_sample=False):
//   - the sequence starts at [decoder_start_token_id] and that token counts as
//     position 0, so max_new_tokens caps the result at 1 + max_new_tokens;
//   - forced_bos_token_id replaces the argmax at the FIRST generated position
//     (HF does it by masking every other logit to -inf; same outcome, and the
//     unforced argmax is kept in the trace so a divergence is still readable);
//   - eos terminates and is included in the output;
//   - ties go to the lower token id, which is what torch.argmax does;
//   - positions follow NLLB_POS_ABSOLUTE, because generate() decodes with a
//     cache and HF's cached position rule is not the same as its un-cached one
//     once a pad_id appears in the prefix. See nllb_pos_mode in nllb.h.
// --------------------------------------------------------------------------

bool nllb_generate(const nllb_model & model,
                   const std::vector<int32_t> & src_ids,
                   const nllb_generate_params & params,
                   int n_threads,
                   nllb_generate_result & out,
                   std::string & err) {
    const nllb_hparams & hp = model.hp;

    if (!model.has_decoder) {
        err = "GGUF has no decoder (converted with --encoder-only?)";
        return false;
    }
    if (src_ids.empty())            { err = "empty source"; return false; }
    if (params.max_new_tokens <= 0) { err = "max_new_tokens must be > 0"; return false; }

    // Same restriction as nllb_decode, checked up front so the caller is told
    // before an encoder pass is spent on it.
    for (int32_t id : src_ids) {
        if (id == hp.pad_id) {
            err = "padded source sequences are not supported yet "
                  "(cross-attention is unmasked); see nllb_decode";
            return false;
        }
    }

    const int32_t eos   = params.eos_id       >= 0 ? params.eos_id       : hp.eos_id;
    const int32_t start = params.dec_start_id >= 0 ? params.dec_start_id : hp.dec_start_id;

    // These arrive from a manifest file and from CLI flags, and go straight
    // into ggml_get_rows as indices. Out of range, ggml aborts the process on
    // an assert; a CI run deserves a message instead of a stack trace.
    const int32_t n_vocab = (int32_t) model.lm_head()->ne[1];
    const int32_t embd_rows = (int32_t) model.tok_embd->ne[1];
    const int32_t id_limit = n_vocab < embd_rows ? n_vocab : embd_rows;
    struct { const char * what; int32_t id; } checked[] = {
        { "decoder_start_token_id", start },
        { "eos_id",                 eos   },
        { "forced_bos_id",          params.forced_bos_id },
    };
    for (const auto & c : checked) {
        if (c.id < -1 || c.id >= id_limit) {
            err = std::string(c.what) + " = " + std::to_string(c.id) +
                  " is out of range [0, " + std::to_string(id_limit) + ")";
            return false;
        }
    }
    if (start < 0) { err = "decoder_start_token_id must be >= 0"; return false; }
    if (eos   < 0) { err = "eos_id must be >= 0"; return false; }

    out = nllb_generate_result();
    out.n_src = (int32_t) src_ids.size();
    out.stop  = NLLB_STOP_MAX_NEW;
    out.used_cache = params.use_cache;

    nllb_encode_result enc;
    if (!nllb_encode(model, src_ids, n_threads, false, enc, err)) return false;

    out.ids.reserve((size_t) params.max_new_tokens + 1);
    out.ids.push_back(start);

    // The cached path needs its capacity up front: decoder_start plus every
    // token that could be emitted. Clamped to the model's n_ctx, so running out
    // of cache and running out of context are the same event with the same
    // stop reason rather than two different-looking failures.
    nllb_kv_cache cache;
    if (params.use_cache) {
        int cap = params.max_new_tokens + 1;
        if (hp.n_ctx > 0 && cap > hp.n_ctx) cap = hp.n_ctx;
        if (!nllb_kv_cache_init(model, enc, cap, n_threads, cache, err,
                                params.kv_type)) return false;
    }

    for (int step = 0; step < params.max_new_tokens; ++step) {
        // nllb_decode accepts Td == n_ctx, so the prefix may still be decoded
        // when it has exactly n_ctx tokens; only a longer one must be refused.
        if (hp.n_ctx > 0 && (int) out.ids.size() > hp.n_ctx) {
            out.stop = NLLB_STOP_CONTEXT;
            break;
        }

        nllb_decode_result dec;
        if (params.use_cache) {
            // Step 0 feeds [decoder_start]; every later step feeds only the
            // token just emitted, with the rest of the prefix already in the
            // cache. cache.n_past is therefore the absolute index of that
            // token, which is exactly what the position rule needs.
            const std::vector<int32_t> feed(1, out.ids.back());
            if ((int) cache.n_past != (int) out.ids.size() - 1) {
                err = "kv cache out of step with the generated prefix: n_past=" +
                      std::to_string(cache.n_past) + ", expected " +
                      std::to_string((int) out.ids.size() - 1);
                return false;
            }
            if (!nllb_decode_cached(model, cache, feed, n_threads,
                                    /*want_hidden_states=*/false,
                                    NLLB_LOGITS_LAST, dec, err)) {
                return false;
            }
        } else {
            // NLLB_POS_ABSOLUTE, not the teacher-forced default: the goldens
            // come from generate(), which decodes incrementally. The two rules
            // differ only if the model emits pad_id, and then this is the one
            // HF used - and the one the cached branch above has no choice about.
            if (!nllb_decode(model, enc, out.ids, src_ids, n_threads,
                             /*want_hidden_states=*/false, NLLB_LOGITS_LAST,
                             dec, err, NLLB_POS_ABSOLUTE)) {
                return false;
            }
        }
        if (dec.n_logit_rows != 1 || (int) dec.logits.size() != dec.n_vocab) {
            err = "decoder returned unexpected logits shape";
            return false;
        }

        // argmax plus runner-up, in one pass. Strict > keeps the first index on
        // a tie, matching torch.argmax.
        const float * p = dec.logits.data();
        const int n = dec.n_vocab;
        int best = 0, second = -1;
        for (int i = 1; i < n; ++i) {
            if (p[i] > p[best]) { second = best; best = i; }
            else if (second < 0 || p[i] > p[second]) second = i;
        }

        nllb_generate_step s;
        s.argmax = best;
        s.logit  = p[best];
        s.margin = second >= 0 ? p[best] - p[second] : 0.0f;
        s.forced = (step == 0 && params.forced_bos_id >= 0);
        s.token  = s.forced ? params.forced_bos_id : best;

        out.ids.push_back(s.token);
        out.steps.push_back(s);

        if (s.token == eos) {
            out.stop    = NLLB_STOP_EOS;
            out.hit_eos = true;
            break;
        }
    }

    return true;
}
