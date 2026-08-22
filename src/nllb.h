// nllb.cpp - NLLB-200 encoder + decoder on ggml
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

struct nllb_hparams {
    int32_t d_model     = 0;
    int32_t n_head      = 0;   // encoder
    int32_t n_layer     = 0;   // encoder
    int32_t n_ff        = 0;   // encoder
    int32_t n_vocab     = 0;
    int32_t n_ctx       = 0;
    int32_t pos_offset  = 2;   // HF SinusoidalPositionalEmbedding.offset
    int32_t pad_id      = 1;
    int32_t bos_id      = 0;
    int32_t eos_id      = 2;
    // NLLB starts the decoder with eos, then forces the target language code.
    int32_t dec_start_id = 2;
    float   eps         = 1e-5f;
    float   embd_scale  = 1.0f; // sqrt(d_model)

    int32_t n_dec_head  = 0;
    int32_t n_dec_layer = 0;
    int32_t n_dec_ff    = 0;

    int32_t head_dim()     const { return d_model / n_head; }
    int32_t dec_head_dim() const { return d_model / n_dec_head; }
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

// Decoder block: self-attention (causal), then cross-attention against the
// frozen encoder output, then FFN. Pre-layernorm throughout, like the encoder.
struct nllb_dec_layer {
    ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    ggml_tensor * wq = nullptr, * bq = nullptr;
    ggml_tensor * wk = nullptr, * bk = nullptr;
    ggml_tensor * wv = nullptr, * bv = nullptr;
    ggml_tensor * wo = nullptr, * bo = nullptr;

    ggml_tensor * cross_norm_w = nullptr, * cross_norm_b = nullptr;
    ggml_tensor * cwq = nullptr, * cbq = nullptr;
    ggml_tensor * cwk = nullptr, * cbk = nullptr;
    ggml_tensor * cwv = nullptr, * cbv = nullptr;
    ggml_tensor * cwo = nullptr, * cbo = nullptr;

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

    bool has_decoder = false;
    ggml_tensor * dec_out_norm_w = nullptr;
    ggml_tensor * dec_out_norm_b = nullptr;
    // lm_head. nullptr means it is tied to tok_embd, which is the NLLB default;
    // the converter only writes output.weight when the checkpoint untied them.
    ggml_tensor * output_w = nullptr;
    std::vector<nllb_dec_layer> dec;

    ggml_context * ctx_w = nullptr;
    gguf_context * gguf  = nullptr;

    ggml_tensor * lm_head() const { return output_w ? output_w : tok_embd; }

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

// Mirrors HF's decoder_hidden_states with the same indexing as the encoder:
// n_dec_layer + 1 entries, entry i (i < n_dec_layer) is the INPUT to block i,
// entry n_dec_layer is the output after dec.output_norm. Feeding that last
// entry through the tied lm_head reproduces model.logits exactly.
struct nllb_decode_result {
    int32_t n_tokens = 0;
    int32_t d_model  = 0;
    int32_t n_vocab  = 0;
    std::vector<float>              output;         // [Td, d_model] row-major
    std::vector<float>              logits;         // [Td, n_vocab], if requested
    std::vector<std::vector<float>> hidden_states;  // n_dec_layer + 1 of [Td, d_model]
};

// Teacher-forced decoder pass: the whole target sequence at once, causally
// masked, no KV cache. This is deliberately the un-cached form - it is what
// HF's golden dump does, so a mismatch here is a graph bug and nothing else.
// The incremental cached path is a separate, separately-verifiable change.
//
// `enc` must be the result of nllb_encode on the source sequence. Cross-
// attention is unmasked, exactly as the encoder's self-attention is: every
// golden case is a single unpadded sequence. Passing source ids containing
// pad_id is rejected rather than silently mis-computed - see src/nllb.cpp.
bool nllb_decode(const nllb_model & model,
                 const nllb_encode_result & enc,
                 const std::vector<int32_t> & dec_ids,
                 const std::vector<int32_t> & src_ids,  // may be empty; pad check only
                 int n_threads,
                 bool want_hidden_states,
                 bool want_logits,
                 nllb_decode_result & out,
                 std::string & err);
