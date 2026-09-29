/* nllb-c.h - a plain-C facade over nllb.h + nllb-vocab.h.
 *
 * nllb.h is C++ (std::vector, std::string), which Swift, Kotlin/JNI and
 * anything else with a C FFI cannot import. This header is the whole surface
 * an app needs - load a GGUF, list its languages, translate text - in the
 * shape whisper.h has: an opaque context, value-type params with a defaults
 * function, and C strings in and out. It is what build-xcframework.sh puts in
 * the framework's module map, so from Swift it is simply `import nllb`.
 *
 * THREADING. One nllb_ctx must be used from one thread at a time - the
 * model's scheduler is a mutable workspace (see nllb_model::sched). The ONE
 * exception is nllb_ctx_request_abort(), which may be called from any thread
 * while nllb_translate() is running on another; that is how a UI cancels a
 * translation that the user has already typed past.
 *
 * NOTHING HERE CHANGES THE ARITHMETIC. nllb_translate() is nllb-cli's
 * translate_one() - nllb_vocab_encode, nllb_generate with the cache on,
 * nllb_vocab_decode(skip_special) - applied line by line. The only addition is
 * the per-token hook, which reads the ids and never touches a tensor.
 */
#ifndef NLLB_C_H
#define NLLB_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nllb_ctx nllb_ctx;

struct nllb_ctx_params {
    /* "auto" (GPU if present, else CPU), "metal", "cpu", or a ggml device
     * name such as "MTL0". NULL means "auto". A named GPU backend that is not
     * present FAILS rather than falling back - that is the library contract. */
    const char * backend;
    /* Map the GGUF instead of reading it. On iOS this is the difference
     * between ~0.25 GB and ~2.7 GB of phys_footprint (trap 36), and jetsam
     * kills by footprint. Leave it on. */
    int use_mmap;
    /* CPU threads. On Metal they only run the ops Metal lacks. <= 0 means 4. */
    int n_threads;
};

struct nllb_ctx_params nllb_ctx_default_params(void);

/* Returns NULL on failure and writes a message into err (if err_len > 0). */
nllb_ctx * nllb_ctx_init(const char * model_path, struct nllb_ctx_params params,
                         char * err, size_t err_len);
void       nllb_ctx_free(nllb_ctx * ctx);

/* ggml's name for the device the graphs run on: "MTL0", "CPU", ... */
const char * nllb_ctx_backend_name(const nllb_ctx * ctx);
/* 1 when the weights are mmap'ed AND wrapped zero-copy by the backend. */
int          nllb_ctx_weights_zero_copy(const nllb_ctx * ctx);

/* The FLORES-200 codes in this GGUF ("eng_Latn", ...), in id order. 202 for
 * every stock NLLB-200 checkpoint. The returned pointers live as long as ctx. */
int          nllb_ctx_n_langs(const nllb_ctx * ctx);
const char * nllb_ctx_lang(const nllb_ctx * ctx, int i);
int          nllb_ctx_has_lang(const nllb_ctx * ctx, const char * code);

/* Called after every generated token with the WHOLE translation so far - all
 * finished lines joined by '\n', then the current line's partial text. That
 * is what a UI wants to display; it never has to stitch pieces together.
 * Return 0 to cancel (same effect as nllb_ctx_request_abort). */
typedef int (*nllb_partial_callback)(const char * text_so_far, void * user_data);

struct nllb_translate_params {
    const char * src_lang;        /* FLORES-200 code, e.g. "eng_Latn" */
    const char * tgt_lang;        /* FLORES-200 code, e.g. "deu_Latn" */
    int          max_new_tokens;  /* per LINE; <= 0 means 256 */
    nllb_partial_callback on_partial;  /* may be NULL */
    void *       user_data;
};

struct nllb_translate_params nllb_translate_default_params(void);

enum nllb_translate_status {
    NLLB_TRANSLATE_OK      =  0,
    NLLB_TRANSLATE_ABORTED =  1,   /* *out_text holds what was produced */
    NLLB_TRANSLATE_ERROR   = -1,   /* *out_text is NULL, err says why   */
};

/* Translates `text` one LINE at a time: NLLB is a sentence-level model, so a
 * paragraph per line is what it was trained on, and the output keeps the
 * input's line structure (blank lines stay blank). Each line gets its own
 * encoder pass and its own fresh KV cache, exactly as nllb-cli's stdin mode.
 *
 * On OK or ABORTED, *out_text is a malloc'd UTF-8 string the caller releases
 * with nllb_string_free(). out_n_tokens (may be NULL) receives the number of
 * tokens generated across all lines, for a tokens/s readout. */
int  nllb_translate(nllb_ctx * ctx, const char * text,
                    struct nllb_translate_params params,
                    char ** out_text, int * out_n_tokens,
                    char * err, size_t err_len);

/* Thread-safe. Makes a running nllb_translate() return ABORTED at the next
 * token boundary (one decoder step, ~20 ms on an A-series GPU). The flag is
 * cleared when the next nllb_translate() starts. */
void nllb_ctx_request_abort(nllb_ctx * ctx);

void nllb_string_free(char * s);

#ifdef __cplusplus
}
#endif

#endif /* NLLB_C_H */
