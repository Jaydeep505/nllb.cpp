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
    int32_t n_logit_rows = 0;                       // Td, or 1 for NLLB_LOGITS_LAST
    std::vector<float>              output;         // [Td, d_model] row-major
    std::vector<float>              logits;         // [n_logit_rows, n_vocab]
    std::vector<std::vector<float>> hidden_states;  // n_dec_layer + 1 of [Td, d_model]
};

// How much of the lm_head to actually run. Greedy decoding only ever reads the
// last row, and on the 600M that row is 1/Td of a [Td, 256206] matmul - the
// single most expensive op in the graph. The values are chosen so the old
// `bool want_logits` call sites keep their meaning: false -> NONE, true -> ALL.
enum nllb_logits_mode {
    NLLB_LOGITS_NONE = 0,
    NLLB_LOGITS_ALL  = 1,
    NLLB_LOGITS_LAST = 2,
};

// How decoder position ids are derived from dec_ids. These two agree on every
// sequence with no pad_id in it, and disagree the moment one appears - which
// is not a corner case, because HF itself uses one rule in a full forward pass
// and the other inside generate().
//
//   HF, no cache (what the teacher-forced goldens are dumped from):
//       (cumsum(mask) + 0) * mask + pad_id          -> NLLB_POS_CUMSUM
//   HF, with cache (what generate() runs, one token at a time):
//       (cumsum(mask) + n_past) * mask + pad_id
//     with the cumsum restarting each step, so for a single new token this is
//       index + 1 + pad_id, or pad_id if the token IS pad -> NLLB_POS_ABSOLUTE
//
// The difference: CUMSUM makes a pad *shift every later position down by one*,
// ABSOLUTE does not. So a decoder that emits pad_id mid-sequence gets different
// positions from HF depending on whether HF was caching. Teacher-forced
// verification must use CUMSUM (that is what its goldens contain) and greedy
// generation must use ABSOLUTE (that is what generate() produced). Picking one
// globally would make one of the two gates unsatisfiable on any case where the
// model emits pad.
enum nllb_pos_mode {
    NLLB_POS_CUMSUM   = 0,   // HF forward pass with use_cache=False
    NLLB_POS_ABSOLUTE = 1,   // HF generate() / any incremental path
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
                 int logits_mode,            // nllb_logits_mode; bool still works
                 nllb_decode_result & out,
                 std::string & err,
                 int pos_mode = NLLB_POS_CUMSUM);

// --------------------------------------------------------------------------
// greedy generation
//
// Mirrors HF generate(num_beams=1, do_sample=False) exactly, and deliberately
// nothing more: argmax over the full vocabulary, forced_bos_token_id at the
// first generated position, stop on eos or max_new_tokens. No sampling, no
// repetition penalty, no length penalty - anything extra would be untestable
// against ref/<name>.gen.bin, which is the only golden that exists.
//
// There is no KV cache yet: each step re-decodes the whole prefix from
// scratch, which is O(n^2) and correct. That is the point of doing it in this
// order - the cached path is then diffed against this one rather than against
// torch, so a cache bug cannot hide behind a graph bug.
// --------------------------------------------------------------------------

struct nllb_generate_params {
    // HF counts max_new_tokens from the decoder_start token, so the returned
    // sequence is at most 1 + max_new_tokens long.
    int32_t max_new_tokens = 24;
    // Target language code, e.g. the id of "deu_Latn". -1 disables forcing.
    // Without it NLLB happily produces fluent output in the wrong language.
    int32_t forced_bos_id  = -1;
    int32_t eos_id         = -1;   // -1 -> model's
    int32_t dec_start_id   = -1;   // -1 -> model's (eos for NLLB, not bos)
};

// Per-step trace. `margin` is top1 - top2 over the vocabulary *before* forcing:
// it is what separates "the loop diverged because the graph is wrong" (margin
// of whole logits) from "the argmax flipped on a coin toss" (margin ~1e-5, i.e.
// float noise). The decoder gate has ULPs for this; generation has margins.
struct nllb_generate_step {
    int32_t token   = 0;      // the token actually emitted
    int32_t argmax  = 0;      // unforced argmax, == token unless forced
    float   logit   = 0.0f;   // logit of `argmax`
    float   margin  = 0.0f;   // logit(top1) - logit(top2)
    bool    forced  = false;  // true only at the forced_bos position
};

// Why the loop stopped. Distinguishing MAX_NEW from CONTEXT matters: the first
// is the caller's own limit, the second is a truncation the caller never asked
// for and would otherwise have to infer from a length.
enum nllb_stop_reason {
    NLLB_STOP_EOS     = 0,
    NLLB_STOP_MAX_NEW = 1,
    NLLB_STOP_CONTEXT = 2,   // ran out of n_ctx before eos
};

struct nllb_generate_result {
    // Includes the decoder_start token at [0] and the eos at the end, exactly
    // like HF's generate() output, so it can be memcmp'd against gen.bin.
    std::vector<int32_t>            ids;
    std::vector<nllb_generate_step> steps;   // one per emitted token
    int32_t stop    = NLLB_STOP_MAX_NEW;
    bool    hit_eos = false;                 // stop == NLLB_STOP_EOS
    int32_t n_src   = 0;
};

bool nllb_generate(const nllb_model & model,
                   const std::vector<int32_t> & src_ids,
                   const nllb_generate_params & params,
                   int n_threads,
                   nllb_generate_result & out,
                   std::string & err);
