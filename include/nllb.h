// nllb.cpp - NLLB-200 encoder + decoder on ggml
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;

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

    ggml_context * ctx_w = nullptr;   // tensor STRUCTS only; the bytes live in weights_buf
    gguf_context * gguf  = nullptr;

    // ---- execution (step 13: ggml-backend) ------------------------------
    //
    // `backend` is where the graphs run and where the weights and the KV cache
    // live: Metal when it was asked for and exists, otherwise the CPU. The CPU
    // backend is ALWAYS created as well and is always the LAST backend in the
    // scheduler, which is what makes it the fallback: an op the primary
    // backend does not implement is run on the CPU by ggml_backend_sched,
    // with the copies in and out inserted for it. With the CPU as primary the
    // two are the same object and the scheduler has one backend.
    //
    // On a device backend a ggml_tensor's ->data is a handle, not a pointer.
    // Nothing in src/ or tests/ may dereference it for a graph tensor; inputs
    // go in through ggml_backend_tensor_set and results come out through
    // ggml_backend_tensor_get. The CPU backend would not notice a violation -
    // its buffers are host memory - which is why that rule is written here.
    ggml_backend * backend     = nullptr;   // primary
    ggml_backend * backend_cpu = nullptr;   // fallback, and where n_threads applies
    std::string    backend_name;            // ggml's name for the primary, e.g. "CPU", "MTL0"
    bool           on_cpu      = true;      // backend == backend_cpu
    ggml_backend_buffer * weights_buf = nullptr;
    bool           weights_host_ptr   = false;   // the buffer wraps the mapping, zero-copy

    // The compute allocator for the one-shot graphs (encoder, un-cached
    // decoder, cross-attention cache build). ggml_gallocr underneath: it sizes
    // the buffer from the graph's actual tensor lifetimes and grows it when a
    // larger graph arrives, so the hand-written arena formulas of steps 2-6 are
    // gone. It is mutable because it is a workspace, not model state - and it
    // makes one nllb_model unsafe to use from two threads at once, which the
    // bump arenas were not. Use one model per thread.
    mutable ggml_backend_sched * sched = nullptr;
    // Metadata memory for those graphs' contexts (tensor structs + the graph,
    // never data), kept so a call does not malloc ~3 MB of it each time.
    mutable std::vector<unsigned char> graph_meta;

    // When the weights are mmap'ed, ctx_w owns only the ggml_tensor STRUCTS
    // and every ->data points into this read-only mapping of the file. When
    // they are read, this is null and ctx_w owns the bytes as well.
    void * mapping      = nullptr;
    size_t mapping_size = 0;
    bool   mapped       = false;   // what actually happened, not what was asked

    ggml_tensor * lm_head() const { return output_w ? output_w : tok_embd; }

    ~nllb_model();
};

// hidden_states mirrors HF exactly: n_layer + 1 entries, where entry i (i <
// n_layer) is the INPUT to block i, and entry n_layer is the final output
// after the encoder's closing layer_norm. See scripts/dump_reference.py.
struct nllb_encode_result {
    int32_t n_tokens = 0;
    int32_t d_model  = 0;
    std::vector<float>              output;         // [T, d_model] row-major
    std::vector<std::vector<float>> hidden_states;  // n_layer + 1 of [T, d_model]
};

// use_mmap maps the file read-only and points every weight tensor into the
// mapping, instead of copying the tensor blob onto the heap. The weights are
// never written, so the two paths are BIT-IDENTICAL by construction and that
// is the assertion tests/test-mmap.cpp makes - it is an allocation change, in
// the same family as the graph arena reuse in step 6, not a numerical one.
//
// What it buys is not FLOPs: it removes a memcpy of the whole tensor blob
// (~700 ms of the f32 file's 1100 ms load; the rest is the tokenizer), makes
// a second process share one copy rather than two, and turns dirty anonymous
// pages into clean file-backed ones the kernel can reclaim for free.
//
// It falls back to reading, with mapped=false, when the platform has no mmap
// or the file's alignment does not permit it. Nothing above this call has to
// care which happened.
//
// `backend` picks where the graphs run (step 13):
//   "cpu"          ggml's CPU backend. The configuration every claim in
//                  docs/dev/handoff.md was measured on, and bit-identical to the
//                  pre-backend CPU path (nllb-dump-states --diff).
//   "metal", "gpu" the first GPU device in this build; FAILS if there is none.
//   "auto"         the first GPU device if there is one, else the CPU.
//   a device name  as ggml names it, e.g. "MTL0", "CUDA0", "CPU".
//   ""             $NLLB_BACKEND if set, else "cpu".
// The library default is the CPU for the same reason use_cache defaults to
// false: a default that silently moved every gate onto different kernels
// would make every gate stop testing what its name says. nllb-cli defaults to
// "auto" - it is the product, and the product should use the GPU.
bool nllb_model_load(const std::string & path, nllb_model & model, std::string & err,
                     bool use_mmap = true, const std::string & backend = "");

// Every device this build of ggml can run on, one "name - description (type)"
// per entry. For --list-backends.
std::vector<std::string> nllb_list_backends();

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
// KV cache
//
// The un-cached decoder above recomputes the whole prefix every step. Almost
// all of that is provably redundant: block il's K and V for token t depend on
// token t alone once the prefix below it is fixed, and causality guarantees it
// is fixed. Cross-attention is stronger still - its K and V come from the
// encoder output, which does not depend on the decoder step at all, so they are
// computed once per source sentence rather than once per token.
//
// WHAT THIS IS DIFFED AGAINST. Not torch. The un-cached path is already proven
// exact against HF generate() on the real 600M (step 4), so the cache is
// verified against *it* - tests/test-cache.cpp. That keeps a cache bug from
// hiding behind a graph bug and vice versa, and it means the gate needs no
// goldens, no python and no download.
//
// THE POSITION RULE IS NOT NEGOTIABLE HERE. A cached step feeds one token with
// past_key_values_length = n_past, and HF's rule puts n_past *inside* the mask
// while restarting the cumsum, so it reduces to (absolute index + 1 + pad_id).
// That is NLLB_POS_ABSOLUTE, and it is what nllb_generate already uses. The
// teacher-forced NLLB_POS_CUMSUM rule cannot be reproduced incrementally - the
// running count is a property of the whole prefix - and asking for it is a
// symptom of diffing against the wrong reference, so it is rejected rather than
// approximated. See trap 12 in docs/dev/handoff.md.
//
// Layout: everything is stored in the shape attention consumes, so a step does
// no permuting and no copying of the history at all.
//
//   k_self [HD, n_ctx, H]    v_self [n_ctx, HD, H]     grows one slot per token
//   k_cross[HD, n_src, H]    v_cross[n_src, HD, H]     built once, then frozen
// --------------------------------------------------------------------------

// Storage type for the cached K and V. The WEIGHTS' precision is a property of
// the GGUF; this is a property of the run, so it is a parameter rather than a
// build option.
//
// f16 halves the cache and halves the bytes attention reads per step. It is NOT
// free and it is not the same kind of change as f16 weights: K and V are
// ACTIVATIONS on a pre-layernorm residual stream that reaches ~4200 by block 4
// and is never renormalised, so rounding them costs headroom that step 7
// measured at only 1.50x to begin with.
//
// AND IT CHANGES WHAT tests/test-cache.cpp CAN ASSERT, which is the part to
// read before turning it on. `full = exact` - one cached call over the whole
// prefix, bit-identical to the un-cached pass - is the project's strongest
// cache claim, and it holds because both sides do the same arithmetic on the
// same bytes. At NLLB_KV_F16 the cached side rounds K and V and the un-cached
// side does not, so the two CANNOT be bit-identical and asking for it would be
// trap 19 in a new costume. The gate therefore keeps `full = exact` as a
// required assertion at NLLB_KV_F32 and, under --kv-f16, runs BOTH and reports
// the f16 difference next to the f32 one. Red at f32 is a cache bug; red only
// at f16 is f16. See trap 37.
enum nllb_kv_type {
    NLLB_KV_F32 = 0,
    NLLB_KV_F16 = 1,
};

struct nllb_kv_cache {
    ggml_context * ctx = nullptr;          // owns the four tensor sets below (structs only)
    ggml_backend_buffer * buf = nullptr;   // the K/V bytes, in the model's primary backend

    // NLLB_KV_F32 or NLLB_KV_F16. Set by nllb_kv_cache_init and read by
    // nllb_decode_cached only through the tensors themselves, which carry
    // their own type - this field exists for reporting, not for dispatch.
    int32_t kv_type = NLLB_KV_F32;

    int32_t n_ctx   = 0;   // capacity in decoder tokens
    int32_t n_past  = 0;   // tokens currently stored in the self-attention half
    int32_t n_src   = 0;   // Ts the cross-attention half was built from (PADDED Ts if batched)

    // Rows decoded in lockstep. 1 for nllb_kv_cache_init, B for
    // nllb_kv_cache_init_batch. Every tensor below then carries a trailing
    // batch dimension ne[3] = n_batch, and nllb_decode_cached refuses a cache
    // with n_batch != 1 (use nllb_decode_cached_batch).
    int32_t n_batch = 1;
    // Real (unpadded) source length of each row; the cross-attention mask is
    // built from this. It lives on the cache, not in a per-step argument, because
    // it is a property of the encoder pass the cross half was built from - a
    // caller who could pass a different one every step could unmask a pad.
    std::vector<int32_t> src_len;
    int32_t d_model = 0;
    int32_t n_head  = 0;
    int32_t n_layer = 0;

    std::vector<ggml_tensor *> k_self;    // per layer, [HD, n_ctx, H]
    std::vector<ggml_tensor *> v_self;    // per layer, [n_ctx, HD, H]  (transposed)
    std::vector<ggml_tensor *> k_cross;   // per layer, [HD, n_src, H]
    std::vector<ggml_tensor *> v_cross;   // per layer, [n_src, HD, H]  (transposed)

    // The per-step compute allocator, kept across steps. It replaces step 6's
    // bump arena (`scratch`), and what it holds is the same kind of thing: NOT
    // cache state - no K or V, and nllb_kv_cache_reset does not touch it - but
    // the cache is the object that already lives exactly as long as the decode
    // loop, which is the lifetime this workspace wants.
    //
    // It is a ggml_backend_sched, so underneath it is ggml_gallocr, which is
    // the point of step 13's memory argument: the buffer is sized from the
    // step graph's real tensor lifetimes, reusing memory between tensors that
    // are never live at once, instead of from a formula with a 64 MB constant
    // in it. It grows when n_kv grows past what it was sized for, so trap 18's
    // "exact fit reallocates every step" does not arise - gallocr only
    // reallocates when the graph genuinely needs more.
    //
    // reuse_scratch = false builds a FRESH scheduler every step and frees it
    // after. That is the [arena] pass in tests/test-cache.cpp, and it matters
    // more under gallocr than it did under the bump arena: gallocr reuses
    // memory WITHIN a graph as well as across steps, so an op reading a tensor
    // it never wrote now sees some other tensor's values even on step one.
    ggml_backend_sched * sched = nullptr;
    std::vector<unsigned char> graph_meta;   // the step graph's metadata, reused too
    bool   reuse_scratch = true;
    const nllb_model * model = nullptr;   // the backends the step scheduler is built on

    nllb_kv_cache() = default;
    ~nllb_kv_cache();
    // Owns a ggml_context; copying it would double-free. Moving is not needed.
    nllb_kv_cache(const nllb_kv_cache &)             = delete;
    nllb_kv_cache & operator=(const nllb_kv_cache &) = delete;
};

// Allocates the cache and runs ONE graph that fills the cross-attention half
// from `enc`. After this the encoder result is no longer needed by the decode
// steps - which is the whole point, and also why `enc` is not held by pointer.
//
// n_ctx_cache is the longest decoder sequence this cache will hold, counting
// the decoder_start token. Asking for more than the model's n_ctx is refused.
//
// kv_type defaults to NLLB_KV_F32, which is the configuration every claim in
// docs/dev/handoff.md was measured on. Passing NLLB_KV_F16 halves the cache and
// costs headroom - see the note on nllb_kv_type above.
bool nllb_kv_cache_init(const nllb_model & model,
                        const nllb_encode_result & enc,
                        int n_ctx_cache,
                        int n_threads,
                        nllb_kv_cache & cache,
                        std::string & err,
                        int kv_type = NLLB_KV_F32);

// Rewind the self-attention half without touching the cross-attention half.
// Valid only for the same source sentence; use a fresh init for a new one.
void nllb_kv_cache_reset(nllb_kv_cache & cache);

// Per-call phase breakdown, in milliseconds. Optional: pass nullptr and nothing
// is measured. It exists because step 5 closed with an *inference* about where
// the cached step's 38 ms/token goes - "most likely the per-step ggml_init" -
// derived from weight sizes rather than from a clock, and acting on arithmetic
// like that is how an optimisation gets aimed at the wrong line.
//
// Read `alloc` and `compute` together. A freshly allocated compute buffer is
// untouched memory, so its page-fault cost is NOT in `alloc` - it lands inside
// `compute`, mixed in with the arithmetic (trap 17). `scratch_reused` says
// which of the two the numbers describe: false means a fresh scheduler per
// step, so those faults are still hiding in `compute`; true means the buffer
// was warm. The difference between two runs is the page-fault bill.
//
// Step 13 changed what two fields mean, not their names: `alloc` is now
// ggml_backend_sched_alloc_graph (graph split + gallocr placement, which can
// grow the buffer), and `arena` is the scheduler's real compute buffer size,
// summed over backends - a number the allocator computed, not a formula.
struct nllb_step_timing {
    double alloc    = 0.0;   // ggml_backend_sched_alloc_graph
    double build    = 0.0;   // graph construction, no arithmetic
    double compute  = 0.0;   // ggml_backend_sched_graph_compute (synchronised)
    double readback = 0.0;   // ggml_backend_tensor_get of outputs, logits, states
    double release  = 0.0;   // sched reset + ggml_free of the graph metadata
    double total    = 0.0;
    size_t arena    = 0;     // compute buffer bytes, all backends
    int    n_nodes  = 0;     // graph size, for the per-node cost of building
    bool   scratch_reused = false;
};

// One incremental step. `new_ids` are the tokens NOT yet in the cache - one per
// step during generation, or the whole starting prefix on the first call.
// Positions are NLLB_POS_ABSOLUTE, derived from cache.n_past; the causal mask
// is offset by n_past so the new rows see the whole history and nothing beyond
// themselves.
//
// out.output and out.hidden_states cover the NEW tokens only, [Tn, d_model].
// hidden_states keeps the same indexing as nllb_decode (n_dec_layer + 1
// entries, entry i = input to block i), which is what lets tests/test-cache.cpp
// bisect a divergence to a layer instead of just reporting that ids differ.
//
// On success cache.n_past advances by Tn. On failure the cache is left
// untouched and must be considered undefined only if err mentions the graph.
bool nllb_decode_cached(const nllb_model & model,
                        nllb_kv_cache & cache,
                        const std::vector<int32_t> & new_ids,
                        int n_threads,
                        bool want_hidden_states,
                        int logits_mode,            // nllb_logits_mode
                        nllb_decode_result & out,
                        std::string & err,
                        nllb_step_timing * timing = nullptr);

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
    // Off by default, deliberately. The un-cached path is the one proven exact
    // against HF, so it stays the reference the cache is diffed against; a
    // default that silently switched the reference out from under the step-4
    // gate would make that gate stop testing what its name says.
    bool    use_cache      = false;
    // Ignored unless use_cache. NLLB_KV_F32 is the default for the same reason
    // use_cache is false: it is the configuration the goldens were measured
    // against, and a default that quietly changed the arithmetic would make
    // every gate stop testing what its name says.
    int32_t kv_type        = NLLB_KV_F32;

    // Optional per-token hook, for a UI that streams the translation and can
    // cancel it (examples/nllb.swiftui). Called once per EMITTED token -
    // including the forced target code at step 0 and the final eos - AFTER the
    // token has been appended to out.ids. Return false to stop: generation
    // ends with NLLB_STOP_ABORTED and nllb_generate still returns true, with
    // the prefix produced so far. nullptr (the default) changes nothing, so
    // no gate sees a different code path unless it asks for one.
    bool  (*on_token)(int32_t token, void * user) = nullptr;
    void *  on_token_user  = nullptr;
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
    NLLB_STOP_ABORTED = 3,   // params.on_token returned false
};

struct nllb_generate_result {
    // Includes the decoder_start token at [0] and the eos at the end, exactly
    // like HF's generate() output, so it can be memcmp'd against gen.bin.
    std::vector<int32_t>            ids;
    std::vector<nllb_generate_step> steps;   // one per emitted token
    int32_t stop    = NLLB_STOP_MAX_NEW;
    bool    hit_eos = false;                 // stop == NLLB_STOP_EOS
    int32_t n_src   = 0;
    bool    used_cache = false;              // which path actually ran
};

bool nllb_generate(const nllb_model & model,
                   const std::vector<int32_t> & src_ids,
                   const nllb_generate_params & params,
                   int n_threads,
                   nllb_generate_result & out,
                   std::string & err);

// --------------------------------------------------------------------------
// batching (step 11, rebuilt on the step-13 backend)
//
// B sources through ONE encoder pass and ONE decoder step per token. The
// decoder step at B = 1 reads ~0.9 GB of weights to produce one token; at B it
// reads the same bytes for B tokens, so ms/token falls until arithmetic, not
// bandwidth, is the limit. Step 11 measured 2.67x at B = 8 on the shape fixture.
//
// These are NEW functions next to the single-sequence ones, not a rewrite of
// them: nllb_encode / nllb_kv_cache_init / nllb_decode_cached / nllb_generate
// are unchanged, so every gate that passed on Metal and CUDA
// still tests the code it passed on. tests/test-batch.cpp diffs these against
// those.
//
// Layout. Sources are RIGHT-padded to the longest in the batch (Ts). Right,
// not left: the encoder position rule is cumsum(mask)*mask + pad_id, so a pad
// shifts every LATER position; pads at the end have nothing after them. Real
// tokens keep the positions they would have alone (traps 2 and 12).
//
//   activations   [D, Ts*B] (encoder), [D, B] (decoder step) - so every weight
//                 matmul is ONE GEMM over all rows, which is the whole point
//   attention     [., ., H, B]: the batch is ggml's ne[3], so each row's
//                 attention is computed in its own slice
//   k_self  [HD, n_ctx, H, B]   v_self  [n_ctx, HD, H, B]
//   k_cross [HD, Ts,    H, B]   v_cross [Ts,    HD, H, B]
//
// Two masks are new, and they are the correctness-critical part:
//   encoder self-attention  [Ts, Ts, 1, B]  -inf on pad KEYS, broadcast over
//                           query rows (masking a pad's query row too would make
//                           a row of all -inf, and softmax of that is NaN)
//   cross-attention         [Ts, 1, 1, B]   -inf on pad source columns, every step
// The pad rows' encoder outputs are NOT zeroed: zeroing would make a missing
// mask almost-right instead of loudly wrong.
//
// A padded row cannot be bit-identical to its unpadded single call: padding
// raises the softmax length and regroups the same real additions (trap 42,
// which is trap 16 again). An UNPADDED row - and every row of an equal-length
// batch - is bit-identical on the CPU, and tests/test-batch.cpp asserts that.
// --------------------------------------------------------------------------

struct nllb_encode_batch_result {
    int32_t n_rows   = 0;
    int32_t n_tokens = 0;              // Ts, the PADDED length
    int32_t d_model  = 0;
    std::vector<int32_t> src_len;      // real length per row
    std::vector<float>   output;       // [B, Ts, D] row-major: row b starts at b*Ts*D
};

// pad_to > longest raises Ts (the [pad1] control: padding with no batch).
// filler is the token written into pad positions (default pad_id). Nothing in
// the graph may read it - positions and both masks come from the LENGTH - which
// is what the [mask] pass in tests/test-batch.cpp asserts by changing it.
// Sources may not contain pad_id.
bool nllb_encode_batch(const nllb_model & model,
                       const std::vector<std::vector<int32_t>> & srcs,
                       int n_threads,
                       nllb_encode_batch_result & out,
                       std::string & err,
                       int pad_to = 0,
                       int32_t filler = -1);

// Allocates a batched cache (n_batch = B) and fills its cross-attention half
// from `enc` in one graph. n_ctx_cache as for nllb_kv_cache_init.
bool nllb_kv_cache_init_batch(const nllb_model & model,
                              const nllb_encode_batch_result & enc,
                              int n_ctx_cache,
                              int n_threads,
                              nllb_kv_cache & cache,
                              std::string & err,
                              int kv_type = NLLB_KV_F32);

// One lockstep decoder step: ONE new token per row (new_ids.size() == B). All
// rows share n_past - the decoder side is never padded, because every row
// starts at decoder_start and grows by exactly one token per step. A finished
// row is fed pad_id (HF's batched generate() does the same); under
// NLLB_POS_ABSOLUTE a pad takes position pad_id and shifts nothing.
//
// out.n_tokens = B, out.output = [B, D], out.logits = [B, n_vocab] with
// n_logit_rows = B, and hidden_states (if asked) n_dec_layer + 1 of [B, D].
bool nllb_decode_cached_batch(const nllb_model & model,
                              nllb_kv_cache & cache,
                              const std::vector<int32_t> & new_ids,
                              int n_threads,
                              bool want_hidden_states,
                              nllb_decode_result & out,
                              std::string & err,
                              nllb_step_timing * timing = nullptr);

struct nllb_generate_batch_result {
    std::vector<nllb_generate_result> seq;   // one per input row, in input order
    int32_t steps       = 0;                 // decoder steps run
    int32_t n_src       = 0;                 // padded source length
    // Decoder token-slots computed (steps * B) and the ones that produced a kept
    // token. The difference is what one long row costs the others: finished
    // rows are not dropped from the batch (not implemented, deliberately - see
    // docs/dev/step11-batching.md).
    int64_t token_slots = 0;
    int64_t tokens_kept = 0;
};

// Greedy generation for B sources at once, always on the cached path.
// forced_bos_ids: one target-language id per row (rows may mix languages), or
// empty to use params.forced_bos_id for every row. params.use_cache is ignored
// (batching exists only on the cached path); params.on_token is not called -
// use on_row_token, which also says which row, and returning false stops THAT
// row with NLLB_STOP_ABORTED.
bool nllb_generate_batch(const nllb_model & model,
                         const std::vector<std::vector<int32_t>> & srcs,
                         const std::vector<int32_t> & forced_bos_ids,
                         const nllb_generate_params & params,
                         int n_threads,
                         nllb_generate_batch_result & out,
                         std::string & err,
                         bool (*on_row_token)(int row, int32_t token, void * user) = nullptr,
                         void * on_row_token_user = nullptr);

// Scheduling (step 11's three rules): sort by length, budget by PADDED tokens
// (group size * longest member) as well as by count, and return INPUT INDICES
// so the caller restores its own order. A single source over max_tokens still
// gets a group of its own: the budget is a hint, not a refusal.
// max_tokens <= 0 means no token budget.
std::vector<std::vector<int32_t>> nllb_plan_batches(const std::vector<int32_t> & lengths,
                                                    int max_seqs,
                                                    int max_tokens);
