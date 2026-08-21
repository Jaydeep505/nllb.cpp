// nllb.cpp - NLLB-200 encoder on ggml
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

struct nllb_hparams {
    int32_t d_model     = 0;
    int32_t n_head      = 0;
    int32_t n_layer     = 0;
    int32_t n_ff        = 0;
    int32_t n_vocab     = 0;
    int32_t n_ctx       = 0;
    int32_t pos_offset  = 2;   // HF SinusoidalPositionalEmbedding.offset
    int32_t pad_id      = 1;
    float   eps         = 1e-5f;
    float   embd_scale  = 1.0f; // sqrt(d_model)
    int32_t head_dim() const { return d_model / n_head; }
};

struct nllb_enc_layer {
    ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    ggml_tensor * wq = nullptr, * bq = nullptr;
    ggml_tensor * wk = nullptr, * bk = nullptr;
    ggml_tensor * wv = nullptr, * bv = nullptr;
    ggml_tensor * wo = nullptr, * bo = nullptr;
    ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
    ggml_tensor * w_up = nullptr, * b_up = nullptr;
    ggml_tensor * w_down = nullptr, * b_down = nullptr;
};

struct nllb_model {
    nllb_hparams hp;

    ggml_tensor * tok_embd   = nullptr;
    ggml_tensor * pos_embd   = nullptr;   // baked sinusoids, [n_ctx + 2, d_model]
    ggml_tensor * out_norm_w = nullptr;
    ggml_tensor * out_norm_b = nullptr;
    std::vector<nllb_enc_layer> enc;

    ggml_context * ctx_w = nullptr;
    gguf_context * gguf  = nullptr;

    ~nllb_model();
};

// hidden_states mirrors HF exactly: n_layer + 1 entries, where entry i (i <
// n_layer) is the INPUT to block i, and entry n_layer is the final output
// after the encoder's closing layer_norm. See tools/dump_reference.py.
struct nllb_encode_result {
    int32_t n_tokens = 0;
    int32_t d_model  = 0;
    std::vector<float>              output;         // [T, d_model] row-major
    std::vector<std::vector<float>> hidden_states;  // n_layer + 1 of [T, d_model]
};

bool nllb_model_load(const std::string & path, nllb_model & model, std::string & err);

bool nllb_encode(const nllb_model & model,
                 const std::vector<int32_t> & ids,
                 int n_threads,
                 bool want_hidden_states,
                 nllb_encode_result & out,
                 std::string & err);
