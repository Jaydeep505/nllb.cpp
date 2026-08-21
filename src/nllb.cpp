#include "nllb.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

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

ggml_tensor * need(ggml_context * ctx, const std::string & name, std::string & err) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t && err.empty()) err = "missing tensor: " + name;
    return t;
}

} // namespace

nllb_model::~nllb_model() {
    if (gguf)  { gguf_free(gguf);       gguf  = nullptr; }
    if (ctx_w) { ggml_free(ctx_w);      ctx_w = nullptr; }
}

// --------------------------------------------------------------------------
// load
// --------------------------------------------------------------------------

bool nllb_model_load(const std::string & path, nllb_model & model, std::string & err) {
    ggml_context * ctx_data = nullptr;
    gguf_init_params gp = {};
    gp.no_alloc = false;          // let gguf allocate + read the tensor data
    gp.ctx      = &ctx_data;

    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { err = "gguf_init_from_file failed: " + path; return false; }

    model.gguf  = g;
    model.ctx_w = ctx_data;

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
    {
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.padding_token_id");
        if (id >= 0) hp.pad_id = (int32_t) gguf_get_val_u32(g, id);
    }
    if (hp.d_model % hp.n_head != 0) {
        err = "d_model not divisible by n_head";
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
