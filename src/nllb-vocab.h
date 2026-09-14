// nllb-vocab.h - NLLB's tokenizer, from the vocabulary embedded in the GGUF.
//
// This is the last Python dependency in the encode path. It is deliberately
// independent of nllb.h: it needs no tensors, so it loads from a vocab-only
// GGUF (tests/make_vocab_gguf.py) as happily as from the full 2.46 GB file,
// and the gate that checks it never has to touch the weights.
//
// WHAT THIS IMPLEMENTS, and why it is not what you would guess:
//
//   * NLLB's tokenizer is **BPE**, not Unigram. `sentencepiece.bpe.model`
//     means what its name says - trainer_spec.model_type == BPE - and
//     tokenizer.json is a BPE model with 514098 merges. There is no Viterbi
//     here and the `tokenizer.ggml.scores` array is not read.
//
//   * The goldens come from HuggingFace's FAST tokenizer, so this matches
//     HuggingFace's implementation, not sentencepiece's. Those differ, in two
//     ways that both change the ids. See nllb-vocab.cpp.
//
//   * The pipeline is: Precompiled(charsmap) -> Replace(" {2,}", " ")
//     -> Metaspace(U+2581, add_prefix_space) -> BPE. The charsmap comes out of
//     the GGUF; a file converted before step 8 does not have it and is
//     REFUSED rather than silently normalised differently.
//
// The full-sentence form NLLB's encoder wants is
//     [src_lang_code] token ... token [eos]
// which is nllb_vocab_encode(). nllb_vocab_encode_pieces() is the same thing
// without the two special ids, which is what a diff against
// `tokenizer.encode(text, add_special_tokens=False)` compares.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

struct nllb_vocab {
    std::vector<std::string>                     tokens;
    std::unordered_map<std::string, int32_t>     ids;     // piece -> id

    // BPE merges, keyed by (left id << 32 | right id). `rank` is the position
    // in HuggingFace's merge list; lower wins. Normally DERIVED from the
    // vocabulary order at load time; read verbatim from
    // `tokenizer.nllb.merges` when the file carries it, which is what a
    // trimmed or extended vocabulary needs. Either way `merges_hash` checks
    // the result.
    struct merge { uint32_t rank; int32_t id; };
    std::unordered_map<uint64_t, merge>          merges;
    bool                                         merges_from_file = false;

    // HuggingFace's AddedVocabulary: 207 strings - the four control tokens,
    // the 202 FLORES codes and <mask> - that are cut out of the input BEFORE
    // anything else looks at it, so a sentence containing the literal text
    // "eng_Latn" or "</s>" yields that id rather than being tokenized as
    // letters. Each surviving chunk is then normalized and pre-tokenized on
    // its own, which is why "x eng_Latn y" ends with a bare U+2581 between
    // "x" and the code.
    struct added_token {
        std::string text;
        int32_t     id      = -1;
        bool        lstrip  = false;   // absorb whitespace to the left
        bool        rstrip  = false;
        bool        normalized = false; // matched AFTER normalization
    };
    std::vector<added_token> added;      // normalized == false
    std::vector<added_token> added_norm; // normalized == true  (just <mask>)

    // SentencePiece's precompiled normalizer: a darts-clone double array over
    // raw bytes whose values are offsets into a blob of NUL-terminated
    // replacement strings.
    std::vector<uint32_t>                        trie;
    std::string                                  norm_blob;

    // UAX#29 grapheme cluster tables: `gcb` as (lo, hi, prop) triples, `ext`
    // as (lo, hi) pairs, both sorted. Always populated - from the GGUF when it
    // carries them, otherwise from the compiled-in src/nllb-grapheme-data.h -
    // so the normalizer has one code path rather than a branch. `from_file`
    // says which, because it is the difference between the ids being a
    // function of this file and a function of this file plus this binary.
    std::vector<uint32_t> gcb;
    std::vector<uint32_t> ext;
    bool                  tables_from_file = false;
    std::string           ucd_version;
    uint64_t              tables_hash = 0;

    int32_t bos_id = 0;
    int32_t pad_id = 1;
    int32_t eos_id = 2;
    int32_t unk_id = 3;

    // provenance, for the gate's banner and for bug reports
    uint64_t    merges_hash = 0;
    uint32_t    n_merges    = 0;
    std::string merges_hash_source;
    std::string vocab_source;

    int32_t id_of(const std::string & piece) const {
        auto it = ids.find(piece);
        return it == ids.end() ? -1 : it->second;
    }
};

// Load from a GGUF path. Reads metadata only - it never allocates tensor data,
// so pointing this at the full checkpoint costs no more than pointing it at a
// vocab-only file.
bool nllb_vocab_load(const std::string & path, nllb_vocab & v, std::string & err);

// Load from a gguf_context somebody else already opened (nllb_model holds one).
bool nllb_vocab_load_from_gguf(const gguf_context * g, nllb_vocab & v,
                               std::string & err);

// --- the pipeline, exposed one stage at a time so a mismatch can be bisected
// to a stage instead of guessed at ------------------------------------------

// Precompiled(charsmap) then Replace(" {2,}" -> " ").
std::string nllb_vocab_normalize(const nllb_vocab & v, const std::string & text);

// Metaspace: ' ' -> U+2581, prepend U+2581 unless already there, split before
// every U+2581 (SplitDelimiterBehavior::MergedWithNext).
std::vector<std::string> nllb_vocab_metaspace(const std::string & normalized);

// BPE over one Metaspace word. Unknown characters become unk, and RUNS of them
// fuse into a single unk (HF's fuse_unk).
std::vector<int32_t> nllb_vocab_bpe(const nllb_vocab & v, const std::string & word);

// normalize -> metaspace -> bpe. No special tokens.
std::vector<int32_t> nllb_vocab_encode_pieces(const nllb_vocab & v,
                                              const std::string & text);

// [src_lang] + pieces + [eos]. Returns false and sets err if src_lang is not a
// FLORES code in this vocabulary - a typo here would otherwise translate from
// whatever token happened to be at that id.
bool nllb_vocab_encode(const nllb_vocab & v, const std::string & text,
                       const std::string & src_lang,
                       std::vector<int32_t> & out, std::string & err);

// Ids -> text. `skip_special` drops the control tokens and the language codes.
std::string nllb_vocab_decode(const nllb_vocab & v,
                              const std::vector<int32_t> & ids,
                              bool skip_special = true);

// A FLORES-200 code ("eng_Latn") -> its id, or -1. This is a plain vocabulary
// lookup: the codes are real entries, added by the converter.
int32_t nllb_vocab_lang_id(const nllb_vocab & v, const std::string & code);

// Every FLORES code in this vocabulary, in id order. For `--list-langs`.
std::vector<std::string> nllb_vocab_langs(const nllb_vocab & v);
