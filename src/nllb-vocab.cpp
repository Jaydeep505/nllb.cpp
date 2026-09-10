// nllb-vocab.cpp - NLLB's BPE tokenizer, reading the GGUF's embedded vocab.
//
// ==========================================================================
// The two things that are NOT what you would guess, both measured
// ==========================================================================
//
// 1. The normalizer takes the SHORTEST trie match, not the longest, and a
//    matched grapheme is consumed WHOLE.
//
//    sentencepiece's own Normalizer does a longest match. HuggingFace's
//    `spm_precompiled`, which is what produced every golden in this repo,
//    calls common_prefix_search() and takes `results[0]` - the shortest. It
//    then advances past the entire grapheme cluster regardless of how much of
//    it the match covered. Consequences you can check by hand:
//
//        "A" U+0302 U+0300  ->  "A-circumflex"   (not "A-circumflex-grave";
//                                the U+0300 is DROPPED, because [41 302] is a
//                                shorter match than [41 302 300])
//        CR LF              ->  one space        (not two; the match was on
//                                the CR alone and the LF went with it)
//
//    Their own source says "Yes, this seems broken / No, I don't know why
//    Google did this." It is not our place to fix it: the goldens are
//    downstream of it.
//
// 2. The unit of iteration is the UAX#29 GRAPHEME CLUSTER, and only clusters
//    under 6 bytes take the whole-cluster path.
//
//    That 6 is a byte count on a UTF-8 cluster, so it selects almost exactly
//    "a base character plus one or two combining marks" - which is where the
//    charsmap's 2726 sub-6-byte multi-codepoint entries live. Everything
//    longer degrades to a per-codepoint walk, which is boundary-independent;
//    so the cluster boundaries only ever matter in the short regime. They do
//    matter there, which is why src/nllb-grapheme-data.h exists.
//
// The merge list is not in the GGUF. It is exactly reproducible from the
// vocabulary ORDER - see derive_merges() - and `tokenizer.nllb.merges_hash`
// turns that from an assumption into a load-time assertion.
// ==========================================================================

#include "nllb-vocab.h"
#include "nllb-grapheme-data.h"

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <string_view>

namespace {

// --------------------------------------------------------------------------
// UTF-8
// --------------------------------------------------------------------------

// Length of the UTF-8 sequence starting at `b`. Returns 1 for a continuation
// or invalid lead byte, so a malformed input walks forward one byte at a time
// instead of looping. nllb_vocab_encode() rejects invalid UTF-8 up front; the
// lower stages stay total on purpose, because a tokenizer that throws in the
// middle of a sentence is worse than one that produces unk.
inline size_t utf8_len(unsigned char b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

inline uint32_t utf8_cp(const char * s, size_t len) {
    const unsigned char * u = (const unsigned char *) s;
    switch (len) {
        case 1: return u[0];
        case 2: return ((u[0] & 0x1Fu) << 6)  |  (u[1] & 0x3Fu);
        case 3: return ((u[0] & 0x0Fu) << 12) | ((u[1] & 0x3Fu) << 6)  | (u[2] & 0x3Fu);
        case 4: return ((u[0] & 0x07u) << 18) | ((u[1] & 0x3Fu) << 12) |
                       ((u[2] & 0x3Fu) << 6)  |  (u[3] & 0x3Fu);
        default: return 0xFFFD;
    }
}

bool utf8_valid(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char b = (unsigned char) s[i];
        size_t n;
        uint32_t cp;
        if      (b < 0x80)          { n = 1; cp = b; }
        else if ((b & 0xE0) == 0xC0) { n = 2; cp = b & 0x1Fu; }
        else if ((b & 0xF0) == 0xE0) { n = 3; cp = b & 0x0Fu; }
        else if ((b & 0xF8) == 0xF0) { n = 4; cp = b & 0x07u; }
        else return false;
        if (i + n > s.size()) return false;
        for (size_t k = 1; k < n; k++) {
            const unsigned char c = (unsigned char) s[i + k];
            if ((c & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (c & 0x3Fu);
        }
        if (n == 2 && cp < 0x80)    return false;   // overlong
        if (n == 3 && cp < 0x800)   return false;
        if (n == 4 && cp < 0x10000) return false;
        if (cp > 0x10FFFF)          return false;
        if (cp >= 0xD800 && cp < 0xE000) return false;  // surrogate
        i += n;
    }
    return true;
}

// --------------------------------------------------------------------------
// UAX#29 grapheme cluster boundaries
// --------------------------------------------------------------------------

// Both tables are flat and sorted: `gcb` in (lo, hi, prop) triples, `ext` in
// (lo, hi) pairs. Whether they came from the GGUF or from the compiled-in
// header is settled at load time and invisible here.
uint8_t gcb_of(const std::vector<uint32_t> & t, uint32_t cp) {
    // Hangul syllables are computed, not tabulated: 798 ranges vs two lines.
    if (cp >= 0xAC00 && cp <= 0xD7A3) {
        return ((cp - 0xAC00) % 28 == 0) ? NLLB_GCB_LV : NLLB_GCB_LVT;
    }
    int lo = 0, hi = (int) (t.size() / 3) - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if      (cp < t[3 * mid])     hi = mid - 1;
        else if (cp > t[3 * mid + 1]) lo = mid + 1;
        else return (uint8_t) t[3 * mid + 2];
    }
    return NLLB_GCB_OTHER;
}

bool extpict_of(const std::vector<uint32_t> & t, uint32_t cp) {
    int lo = 0, hi = (int) (t.size() / 2) - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if      (cp < t[2 * mid])     hi = mid - 1;
        else if (cp > t[2 * mid + 1]) lo = mid + 1;
        else return true;
    }
    return false;
}

// Byte offset just past the grapheme cluster starting at `pos`.
size_t next_grapheme(const nllb_vocab & v, const std::string & s, size_t pos) {
    auto gcb     = [&](uint32_t cp) { return gcb_of(v.gcb, cp); };
    auto extpict = [&](uint32_t cp) { return extpict_of(v.ext, cp); };
    if (pos >= s.size()) return pos;

    size_t i  = pos;
    size_t n  = utf8_len((unsigned char) s[i]);
    if (i + n > s.size()) n = s.size() - i;
    uint32_t prev  = utf8_cp(s.data() + i, n);
    uint8_t  pprop = gcb(prev);
    i += n;

    // GB12/GB13 need the parity of the regional-indicator run ending at prev;
    // GB11 needs to know whether that run is `ExtPict Extend* ZWJ`.
    int  ri_run = (pprop == NLLB_GCB_RI) ? 1 : 0;
    bool emoji  = extpict(prev);

    while (i < s.size()) {
        size_t m = utf8_len((unsigned char) s[i]);
        if (i + m > s.size()) m = s.size() - i;
        const uint32_t cur   = utf8_cp(s.data() + i, m);
        const uint8_t  cprop = gcb(cur);

        bool brk;
        if      (pprop == NLLB_GCB_CR && cprop == NLLB_GCB_LF)            brk = false; // GB3
        else if (pprop == NLLB_GCB_CONTROL || pprop == NLLB_GCB_CR ||
                 pprop == NLLB_GCB_LF)                                    brk = true;  // GB4
        else if (cprop == NLLB_GCB_CONTROL || cprop == NLLB_GCB_CR ||
                 cprop == NLLB_GCB_LF)                                    brk = true;  // GB5
        else if (pprop == NLLB_GCB_L &&
                 (cprop == NLLB_GCB_L  || cprop == NLLB_GCB_V ||
                  cprop == NLLB_GCB_LV || cprop == NLLB_GCB_LVT))         brk = false; // GB6
        else if ((pprop == NLLB_GCB_LV || pprop == NLLB_GCB_V) &&
                 (cprop == NLLB_GCB_V  || cprop == NLLB_GCB_T))           brk = false; // GB7
        else if ((pprop == NLLB_GCB_LVT || pprop == NLLB_GCB_T) &&
                 cprop == NLLB_GCB_T)                                     brk = false; // GB8
        else if (cprop == NLLB_GCB_EXTEND || cprop == NLLB_GCB_ZWJ)       brk = false; // GB9
        else if (cprop == NLLB_GCB_SPACINGMARK)                           brk = false; // GB9a
        else if (pprop == NLLB_GCB_PREPEND)                               brk = false; // GB9b
        else if (pprop == NLLB_GCB_ZWJ && emoji && extpict(cur))          brk = false; // GB11
        else if (pprop == NLLB_GCB_RI && cprop == NLLB_GCB_RI &&
                 (ri_run % 2) == 1)                                       brk = false; // GB12/13
        else                                                              brk = true;  // GB999

        if (brk) break;

        ri_run = (cprop == NLLB_GCB_RI) ? ri_run + 1 : 0;
        if      (extpict(cur))                emoji = true;
        else if (cprop == NLLB_GCB_EXTEND ||
                 cprop == NLLB_GCB_ZWJ)       /* keep emoji as it was */ ;
        else                                  emoji = false;

        pprop = cprop;
        i += m;
    }
    return i;
}

// --------------------------------------------------------------------------
// darts-clone double array, exactly as spm_precompiled reads it
// --------------------------------------------------------------------------

inline bool da_has_leaf(uint32_t u) { return ((u >> 8) & 1u) == 1u; }
inline uint32_t da_value(uint32_t u) { return u & 0x7FFFFFFFu; }
inline uint32_t da_label(uint32_t u) { return u & (0x80000000u | 0xFFu); }
inline uint32_t da_offset(uint32_t u) { return (u >> 10) << ((u & 0x200u) >> 6); }

// common_prefix_search(key)[0]: the SHORTEST match. See the header comment.
bool trie_first(const std::vector<uint32_t> & arr,
                const char * key, size_t len, uint32_t & value) {
    if (arr.empty()) return false;
    uint32_t node = 0;
    uint32_t unit = arr[node];
    node ^= da_offset(unit);
    for (size_t i = 0; i < len; i++) {
        const uint32_t c = (unsigned char) key[i];
        if (c == 0) break;
        node ^= c;
        if (node >= arr.size()) return false;
        unit = arr[node];
        if (da_label(unit) != c) return false;
        node ^= da_offset(unit);
        if (da_has_leaf(unit)) {
            if (node >= arr.size()) return false;
            value = da_value(arr[node]);
            return true;
        }
    }
    return false;
}

// The trie value is a byte offset into a blob of NUL-terminated strings.
bool trie_transform(const nllb_vocab & v, const char * key, size_t len,
                    const char *& out, size_t & out_len) {
    uint32_t off;
    if (!trie_first(v.trie, key, len, off)) return false;
    if (off >= v.norm_blob.size()) return false;
    const size_t end = v.norm_blob.find('\0', off);
    out     = v.norm_blob.data() + off;
    out_len = (end == std::string::npos ? v.norm_blob.size() : end) - off;
    return true;
}

// --------------------------------------------------------------------------
// merges, derived from the vocabulary order
// --------------------------------------------------------------------------

inline uint64_t pair_key(int32_t a, int32_t b) {
    return ((uint64_t) (uint32_t) a << 32) | (uint32_t) b;
}

inline void fnv1a(uint64_t & h, const char * p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char) p[i];
        h *= 0x100000001B3ull;
    }
}

bool is_flores_code(const std::string & t) {
    // Every one of NLLB's 202 codes is [a-z]{3}_[A-Z][a-z]{3}, and no other
    // entry in the 256204-token vocabulary matches it. Checked, not assumed.
    if (t.size() != 8 || t[3] != '_') return false;
    for (int i = 0; i < 3; i++) if (t[i] < 'a' || t[i] > 'z') return false;
    if (t[4] < 'A' || t[4] > 'Z') return false;
    for (int i = 5; i < 8; i++) if (t[i] < 'a' || t[i] > 'z') return false;
    return true;
}

// --------------------------------------------------------------------------
// AddedVocabulary
// --------------------------------------------------------------------------

// Rust's char::is_whitespace. After the Precompiled normalizer every one of
// these except the ASCII set has already become U+0020, so in practice only
// the first line can fire - but lstrip runs on RAW text for a non-normalized
// added token, and none of NLLB's have lstrip, so the rest is belt and braces.
bool is_ws(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

size_t ws_extend_left(const std::string & s, size_t pos) {
    while (pos > 0) {
        // step back over one UTF-8 character
        size_t st = pos - 1;
        while (st > 0 && ((unsigned char) s[st] & 0xC0) == 0x80) st--;
        if (!is_ws(utf8_cp(s.data() + st, pos - st))) break;
        pos = st;
    }
    return pos;
}

size_t ws_extend_right(const std::string & s, size_t pos) {
    while (pos < s.size()) {
        size_t n = utf8_len((unsigned char) s[pos]);
        if (pos + n > s.size()) break;
        if (!is_ws(utf8_cp(s.data() + pos, n))) break;
        pos += n;
    }
    return pos;
}

// One chunk of the input: either a resolved added token, or raw text.
struct chunk { int32_t id; size_t begin, end; };

// Leftmost-longest scan for any of `set`, honouring lstrip/rstrip. This is
// aho-corasick's LeftmostLongest with HuggingFace's post-filters; the
// single_word filter is not implemented because the converter refuses any
// added token that sets it, so the case cannot arise.
std::vector<chunk> split_added(const std::string & s,
                               const std::vector<nllb_vocab::added_token> & set) {
    std::vector<chunk> out;
    if (set.empty()) { out.push_back({-1, 0, s.size()}); return out; }

    size_t plain = 0;                       // start of the pending text run
    size_t i = 0;
    while (i < s.size()) {
        const nllb_vocab::added_token * best = nullptr;
        for (const auto & t : set) {
            if (t.text.size() > s.size() - i) continue;
            if (std::memcmp(s.data() + i, t.text.data(), t.text.size()) != 0) continue;
            if (!best || t.text.size() > best->text.size()) best = &t;
        }
        if (!best) { i++; continue; }

        size_t b = i, e = i + best->text.size();
        if (best->lstrip) b = ws_extend_left(s, b);
        if (best->rstrip) e = ws_extend_right(s, e);
        if (b > plain) out.push_back({-1, plain, b});
        else if (b < plain) b = plain;      // lstrip cannot eat a previous token
        out.push_back({best->id, b, e});
        plain = e;
        i = e;
    }
    if (plain < s.size()) out.push_back({-1, plain, s.size()});
    return out;
}

} // namespace

// --------------------------------------------------------------------------
// loading
// --------------------------------------------------------------------------

bool nllb_vocab_load_from_gguf(const gguf_context * g, nllb_vocab & v,
                               std::string & err) {
    const int64_t k_tok = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (k_tok < 0) { err = "no tokenizer.ggml.tokens in this GGUF"; return false; }

    const size_t n = gguf_get_arr_n(g, k_tok);
    v.tokens.resize(n);
    v.ids.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        v.tokens[i] = gguf_get_arr_str(g, k_tok, i);
        v.ids.emplace(v.tokens[i], (int32_t) i);
    }

    auto kv_i32 = [&](const char * key, int32_t & dst) {
        const int64_t id = gguf_find_key(g, key);
        if (id >= 0) dst = (int32_t) gguf_get_val_u32(g, id);
    };
    kv_i32("tokenizer.ggml.bos_token_id", v.bos_id);
    kv_i32("tokenizer.ggml.eos_token_id", v.eos_id);
    kv_i32("tokenizer.ggml.padding_token_id", v.pad_id);
    kv_i32("tokenizer.ggml.unknown_token_id", v.unk_id);

    {
        const int64_t id = gguf_find_key(g, "tokenizer.nllb.vocab_source");
        if (id >= 0) v.vocab_source = gguf_get_val_str(g, id);
        const int64_t id2 = gguf_find_key(g, "tokenizer.nllb.merges_hash_source");
        if (id2 >= 0) v.merges_hash_source = gguf_get_val_str(g, id2);
    }

    // ---- grapheme tables --------------------------------------------------
    // The file's copy wins; the compiled-in header is the fallback for a GGUF
    // written before step 8b, or by some other tool. Either way v.gcb / v.ext
    // end up holding the same shape, so nothing downstream has to care.
    {
        const int64_t k_g = gguf_find_key(g, "tokenizer.nllb.gcb_ranges");
        const int64_t k_e = gguf_find_key(g, "tokenizer.nllb.extpict_ranges");
        if (k_g >= 0 && k_e >= 0) {
            const size_t ng = gguf_get_arr_n(g, k_g);
            const size_t ne = gguf_get_arr_n(g, k_e);
            if (ng == 0 || ng % 3 != 0 || ne % 2 != 0) {
                err = "tokenizer.nllb.gcb_ranges / extpict_ranges have the "
                      "wrong shape (want (lo,hi,prop) triples and (lo,hi) pairs)";
                return false;
            }
            const int32_t * pg = (const int32_t *) gguf_get_arr_data(g, k_g);
            const int32_t * pe = (const int32_t *) gguf_get_arr_data(g, k_e);
            v.gcb.assign(pg, pg + ng);
            v.ext.assign(pe, pe + ne);
            v.tables_from_file = true;
            const int64_t k_u = gguf_find_key(g, "tokenizer.nllb.ucd_version");
            if (k_u >= 0) v.ucd_version = gguf_get_val_str(g, k_u);
        } else {
            for (int i = 0; i < NLLB_GCB_TABLE_N; i++) {
                v.gcb.push_back(NLLB_GCB_TABLE[i].lo);
                v.gcb.push_back(NLLB_GCB_TABLE[i].hi);
                v.gcb.push_back(NLLB_GCB_TABLE[i].prop);
            }
            for (int i = 0; i < NLLB_EXTPICT_TABLE_N; i++) {
                v.ext.push_back(NLLB_EXTPICT_TABLE[i].lo);
                v.ext.push_back(NLLB_EXTPICT_TABLE[i].hi);
            }
            v.tables_from_file = false;
            v.ucd_version = "compiled-in";
        }
        // Sortedness is what the binary searches assume; a table that is not
        // sorted returns NLLB_GCB_OTHER for real properties and the failure
        // looks like a normalizer bug rather than like a bad file.
        for (size_t i = 3; i < v.gcb.size(); i += 3) {
            if (v.gcb[i] <= v.gcb[i - 3]) {
                err = "tokenizer.nllb.gcb_ranges is not sorted by codepoint";
                return false;
            }
        }
        uint64_t h = 0xCBF29CE484222325ull;
        for (uint32_t x : v.gcb) fnv1a(h, (const char *) &x, 4);
        for (uint32_t x : v.ext) fnv1a(h, (const char *) &x, 4);
        v.tables_hash = h;
    }

    // ---- added tokens -----------------------------------------------------
    {
        const int64_t k_at = gguf_find_key(g, "tokenizer.nllb.added_tokens");
        const int64_t k_af = gguf_find_key(g, "tokenizer.nllb.added_flags");
        if (k_at < 0 || k_af < 0) {
            err = "no tokenizer.nllb.added_tokens in this GGUF. This file was "
                  "converted before step 8; re-run tools/convert_nllb_to_gguf.py. "
                  "Without them a sentence containing the literal text "
                  "'eng_Latn' or '</s>' tokenizes as letters instead of as that "
                  "token, which is a silent divergence from the reference.";
            return false;
        }
        const size_t na = gguf_get_arr_n(g, k_at);
        if (gguf_get_arr_n(g, k_af) != na) {
            err = "tokenizer.nllb.added_flags has a different length to "
                  "tokenizer.nllb.added_tokens";
            return false;
        }
        const int32_t * flags = (const int32_t *) gguf_get_arr_data(g, k_af);
        for (size_t i = 0; i < na; i++) {
            nllb_vocab::added_token t;
            t.text = gguf_get_arr_str(g, k_at, i);
            t.id   = v.id_of(t.text);
            if (t.id < 0) {
                err = "added token '" + t.text + "' is not in the vocabulary";
                return false;
            }
            const uint32_t f = (uint32_t) flags[i];
            if (f & 1u) {   // single_word
                err = "added token '" + t.text + "' sets single_word, which "
                      "this tokenizer does not implement. The converter is "
                      "supposed to have refused this file.";
                return false;
            }
            t.lstrip     = (f & 2u) != 0;
            t.rstrip     = (f & 4u) != 0;
            t.normalized = (f & 8u) != 0;
            (t.normalized ? v.added_norm : v.added).push_back(t);
        }
    }

    // ---- the normalizer ---------------------------------------------------
    const int64_t k_cm = gguf_find_key(g, "tokenizer.ggml.precompiled_charsmap");
    if (k_cm < 0) {
        err = "no tokenizer.ggml.precompiled_charsmap in this GGUF. This file "
              "was converted before step 8; re-run tools/convert_nllb_to_gguf.py. "
              "Refusing rather than tokenizing without SentencePiece's "
              "normalizer, which would silently differ from the goldens on any "
              "input that is not plain ASCII.";
        return false;
    }
    if (gguf_get_arr_type(g, k_cm) != GGUF_TYPE_UINT8) {
        err = "tokenizer.ggml.precompiled_charsmap is not a uint8 array";
        return false;
    }
    {
        const size_t nb = gguf_get_arr_n(g, k_cm);
        const uint8_t * p = (const uint8_t *) gguf_get_arr_data(g, k_cm);
        if (nb < 8) { err = "precompiled_charsmap is too short"; return false; }
        uint32_t trie_bytes = 0;
        std::memcpy(&trie_bytes, p, 4);
        if (trie_bytes % 4 != 0 || (size_t) trie_bytes + 4 > nb) {
            err = "precompiled_charsmap trie size is out of range";
            return false;
        }
        v.trie.resize(trie_bytes / 4);
        std::memcpy(v.trie.data(), p + 4, trie_bytes);
        v.norm_blob.assign((const char *) p + 4 + trie_bytes,
                           nb - 4 - (size_t) trie_bytes);
    }

    // ---- merges, derived then checked -------------------------------------
    // transformers' SentencePieceExtractor.extract(vocab_scores=None): for
    // each piece in ID ORDER, every split whose two halves are both in the
    // vocabulary, those splits ordered by (left id, right id). Splits are at
    // CHARACTER boundaries - the Python iterates a str, not bytes - which is
    // why this walks UTF-8 rather than every offset.
    // A view-keyed index, so the ~1.3M lookups below allocate nothing. Keys
    // point into v.tokens, which is filled and never touched again.
    std::unordered_map<std::string_view, int32_t> view_ids;
    view_ids.reserve(n * 2);
    for (size_t i = 0; i < n; i++) view_ids.emplace(v.tokens[i], (int32_t) i);
    auto view_id = [&](std::string_view s) -> int32_t {
        auto it = view_ids.find(s);
        return it == view_ids.end() ? -1 : it->second;
    };

    uint64_t hash = 0xCBF29CE484222325ull;
    uint32_t rank = 0;
    v.merges.reserve(n * 3);            // ~2 merges per piece on the real vocab

    // A vocabulary that was trimmed or extended after training - which is what
    // a language-specific fine-tune usually does - reindexes the pieces, and
    // the merge order HuggingFace kept is then no longer a function of that
    // order. The converter notices and writes the list; if it is here it wins,
    // and merges_hash below checks the transfer instead of the derivation.
    const int64_t k_ml = gguf_find_key(g, "tokenizer.nllb.merges");
    if (k_ml >= 0) {
        if (gguf_get_arr_type(g, k_ml) != GGUF_TYPE_STRING) {
            err = "tokenizer.nllb.merges is not a string array";
            return false;
        }
        const size_t nm = gguf_get_arr_n(g, k_ml);
        v.merges.reserve(nm);
        for (size_t i = 0; i < nm; i++) {
            const std::string m = gguf_get_arr_str(g, k_ml, i);

            // "left right". A piece can in principle contain a space, so take
            // the split that resolves rather than assuming the first one does.
            int32_t l = -1, r = -1;
            for (size_t sp = m.find(' '); sp != std::string::npos;
                 sp = m.find(' ', sp + 1)) {
                const int32_t a = view_id(std::string_view(m).substr(0, sp));
                if (a < 0) continue;
                const int32_t b = view_id(std::string_view(m).substr(sp + 1));
                if (b < 0) continue;
                l = a; r = b;
                break;
            }
            if (l < 0) {
                err = "merge " + std::to_string(i) + " ('" + m + "') has a half "
                      "that is not in this vocabulary";
                return false;
            }
            const int32_t merged = view_id(v.tokens[l] + v.tokens[r]);
            if (merged < 0) {
                err = "merge " + std::to_string(i) + " ('" + m + "') produces a "
                      "piece that is not in this vocabulary";
                return false;
            }
            if (rank) { hash ^= 0x0A; hash *= 0x100000001B3ull; }
            fnv1a(hash, m.data(), m.size());

            // Last one wins, matching BpeBuilder::build()'s HashMap::insert.
            v.merges[pair_key(l, r)] = nllb_vocab::merge{rank, merged};
            rank++;
        }
        v.merges_from_file = true;
    } else {
        std::vector<std::pair<int32_t, int32_t>> local;
        for (size_t id = 0; id < n; id++) {
            const std::string & piece = v.tokens[id];
            const std::string_view pv(piece);
            local.clear();
            for (size_t i = 1; i < piece.size(); i++) {
                if (((unsigned char) piece[i] & 0xC0) == 0x80) continue; // mid-char
                const int32_t l = view_id(pv.substr(0, i));
                if (l < 0) continue;
                const int32_t r = view_id(pv.substr(i));
                if (r < 0) continue;
                local.emplace_back(l, r);
            }
            std::sort(local.begin(), local.end());
            for (const auto & lr : local) {
                if (rank) { hash ^= 0x0A; hash *= 0x100000001B3ull; }
                const std::string & ls = v.tokens[lr.first];
                const std::string & rs = v.tokens[lr.second];
                fnv1a(hash, ls.data(), ls.size());
                const char sp = ' ';
                fnv1a(hash, &sp, 1);
                fnv1a(hash, rs.data(), rs.size());

                // Last one wins, matching BpeBuilder::build()'s HashMap::insert.
                v.merges[pair_key(lr.first, lr.second)] =
                    nllb_vocab::merge{rank, (int32_t) id};
                rank++;
            }
        }
    }
    v.n_merges = rank;

    const int64_t k_h = gguf_find_key(g, "tokenizer.nllb.merges_hash");
    if (k_h < 0) {
        err = "no tokenizer.nllb.merges_hash in this GGUF. This file was "
              "converted before step 8; re-run tools/convert_nllb_to_gguf.py.";
        return false;
    }
    v.merges_hash = gguf_get_val_u64(g, k_h);
    if (v.merges_hash != hash) {
        char buf[400];
        snprintf(buf, sizeof(buf),
                 "merge %s mismatch: %u merges hashing to 0x%016llx, GGUF says "
                 "0x%016llx (%s). %s Refusing to load.",
                 v.merges_from_file ? "list" : "derivation",
                 rank, (unsigned long long) hash,
                 (unsigned long long) v.merges_hash,
                 v.merges_hash_source.empty() ? "unknown source"
                                              : v.merges_hash_source.c_str(),
                 v.merges_from_file
                     ? "tokenizer.nllb.merges does not hash to what the "
                       "converter wrote, so this file is damaged."
                     : "The vocabulary order in this file does not reproduce "
                       "the merge list HuggingFace serialised, so BPE here "
                       "would not match the reference. Re-run "
                       "tools/convert_nllb_to_gguf.py, which embeds the list "
                       "when the derivation does not fit the vocabulary.");
        err = buf;
        return false;
    }
    const int64_t k_nm = gguf_find_key(g, "tokenizer.nllb.n_merges");
    if (k_nm >= 0 && gguf_get_val_u32(g, k_nm) != rank) {
        err = "merge count mismatch against tokenizer.nllb.n_merges";
        return false;
    }
    return true;
}

bool nllb_vocab_load(const std::string & path, nllb_vocab & v, std::string & err) {
    gguf_init_params gp = {};
    gp.no_alloc = true;             // metadata only; never touch the weights
    gp.ctx      = nullptr;
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { err = "gguf_init_from_file failed: " + path; return false; }
    const bool ok = nllb_vocab_load_from_gguf(g, v, err);
    gguf_free(g);
    return ok;
}

// --------------------------------------------------------------------------
// the pipeline
// --------------------------------------------------------------------------

std::string nllb_vocab_normalize(const nllb_vocab & v, const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);

    size_t i = 0;
    while (i < text.size()) {
        const size_t end = next_grapheme(v, text, i);
        const size_t len = end - i;

        const char * rep = nullptr;
        size_t rep_len = 0;
        if (len < 6 && trie_transform(v, text.data() + i, len, rep, rep_len)) {
            out.append(rep, rep_len);      // the WHOLE cluster is consumed
        } else {
            size_t j = i;
            while (j < end) {
                size_t cl = utf8_len((unsigned char) text[j]);
                if (j + cl > end) cl = end - j;
                if (trie_transform(v, text.data() + j, cl, rep, rep_len)) {
                    out.append(rep, rep_len);
                } else {
                    out.append(text, j, cl);
                }
                j += cl;
            }
        }
        i = end;
    }

    // Replace(Regex(" {2,}"), " ") - the second half of HF's normalizer stack.
    std::string collapsed;
    collapsed.reserve(out.size());
    for (size_t k = 0; k < out.size(); ) {
        if (out[k] == ' ') {
            size_t run = 0;
            while (k + run < out.size() && out[k + run] == ' ') run++;
            collapsed.push_back(' ');
            k += run;
        } else {
            collapsed.push_back(out[k]);
            k++;
        }
    }
    return collapsed;
}

std::vector<std::string> nllb_vocab_metaspace(const std::string & normalized) {
    static const char * MS = "\xe2\x96\x81";     // U+2581 LOWER ONE EIGHTH BLOCK

    std::vector<std::string> words;
    if (normalized.empty()) return words;        // HF drops the empty split

    std::string s;
    s.reserve(normalized.size() + 3);
    for (char c : normalized) {
        if (c == ' ') s += MS;
        else          s.push_back(c);
    }
    if (s.compare(0, 3, MS) != 0) s = std::string(MS) + s;

    // split on U+2581 with SplitDelimiterBehavior::MergedWithNext
    size_t i = 0;
    std::string cur;
    while (i < s.size()) {
        if (i + 3 <= s.size() && std::memcmp(s.data() + i, MS, 3) == 0) {
            if (!cur.empty()) words.push_back(cur);
            cur.assign(MS);
            i += 3;
        } else {
            size_t cl = utf8_len((unsigned char) s[i]);
            if (i + cl > s.size()) cl = s.size() - i;
            cur.append(s, i, cl);
            i += cl;
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

std::vector<int32_t> nllb_vocab_bpe(const nllb_vocab & v, const std::string & word) {
    struct sym { int32_t id; int prev, next; bool alive; };
    std::vector<sym> syms;
    syms.reserve(word.size());

    for (size_t i = 0; i < word.size(); ) {
        size_t cl = utf8_len((unsigned char) word[i]);
        if (i + cl > word.size()) cl = word.size() - i;
        const int32_t id = v.id_of(word.substr(i, cl));
        if (id < 0) {
            // fuse_unk: a RUN of unknown characters is one unk, not several.
            if (!syms.empty() && syms.back().id == v.unk_id) { i += cl; continue; }
            syms.push_back({v.unk_id, (int) syms.size() - 1, (int) syms.size() + 1, true});
        } else {
            syms.push_back({id, (int) syms.size() - 1, (int) syms.size() + 1, true});
        }
        i += cl;
    }
    if (syms.empty()) return {};
    syms.back().next = -1;

    // Lowest rank wins; ties go to the leftmost position. That is HF's
    // BinaryHeap<Merge> ordering (rank reversed, then pos reversed) and it is
    // also what a plain left-to-right scan does, so the two are equivalent.
    struct cand {
        uint32_t rank; int pos;
        bool operator<(const cand & o) const {              // min-heap
            return rank != o.rank ? rank > o.rank : pos > o.pos;
        }
    };
    std::priority_queue<cand> q;
    auto push_pair = [&](int a) {
        if (a < 0) return;
        const int b = syms[a].next;
        if (b < 0) return;
        auto it = v.merges.find(pair_key(syms[a].id, syms[b].id));
        if (it != v.merges.end()) q.push({it->second.rank, a});
    };
    for (int a = 0; a + 1 < (int) syms.size(); a++) push_pair(a);

    while (!q.empty()) {
        const cand c = q.top();
        q.pop();
        const int a = c.pos;
        if (!syms[a].alive) continue;
        const int b = syms[a].next;
        if (b < 0 || !syms[b].alive) continue;
        auto it = v.merges.find(pair_key(syms[a].id, syms[b].id));
        if (it == v.merges.end() || it->second.rank != c.rank) continue; // stale

        syms[a].id   = it->second.id;
        syms[b].alive = false;
        syms[a].next = syms[b].next;
        if (syms[a].next >= 0) syms[syms[a].next].prev = a;

        push_pair(syms[a].prev);
        push_pair(a);
    }

    std::vector<int32_t> out;
    for (int i = 0; i >= 0 && i < (int) syms.size(); i = syms[i].next) {
        if (syms[i].alive) out.push_back(syms[i].id);
    }
    return out;
}

std::vector<int32_t> nllb_vocab_encode_pieces(const nllb_vocab & v,
                                              const std::string & text) {
    std::vector<int32_t> out;

    // Stage 1: cut out added tokens that are matched on the RAW text - the
    // control tokens and the 202 language codes. Stage 2 does the same on the
    // NORMALIZED text for the ones flagged normalized (only <mask>). Each
    // surviving run is normalized and pre-tokenized ON ITS OWN, so
    // add_prefix_space fires once per run, not once per sentence.
    for (const chunk & c1 : split_added(text, v.added)) {
        if (c1.id >= 0) { out.push_back(c1.id); continue; }
        const std::string norm =
            nllb_vocab_normalize(v, text.substr(c1.begin, c1.end - c1.begin));
        for (const chunk & c2 : split_added(norm, v.added_norm)) {
            if (c2.id >= 0) { out.push_back(c2.id); continue; }
            for (const std::string & w :
                 nllb_vocab_metaspace(norm.substr(c2.begin, c2.end - c2.begin))) {
                const std::vector<int32_t> ids = nllb_vocab_bpe(v, w);
                out.insert(out.end(), ids.begin(), ids.end());
            }
        }
    }
    return out;
}

bool nllb_vocab_encode(const nllb_vocab & v, const std::string & text,
                       const std::string & src_lang,
                       std::vector<int32_t> & out, std::string & err) {
    if (!utf8_valid(text)) {
        err = "input is not valid UTF-8";
        return false;
    }
    const int32_t lang = nllb_vocab_lang_id(v, src_lang);
    if (lang < 0) {
        err = "unknown source language code '" + src_lang +
              "'. Expected a FLORES-200 code such as eng_Latn.";
        return false;
    }
    out.clear();
    out.push_back(lang);
    const std::vector<int32_t> ids = nllb_vocab_encode_pieces(v, text);
    out.insert(out.end(), ids.begin(), ids.end());
    out.push_back(v.eos_id);
    return true;
}

std::string nllb_vocab_decode(const nllb_vocab & v,
                              const std::vector<int32_t> & ids,
                              bool skip_special) {
    std::string joined;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t) v.tokens.size()) continue;
        const std::string & t = v.tokens[id];
        if (skip_special) {
            if (id == v.bos_id || id == v.eos_id ||
                id == v.pad_id || id == v.unk_id) continue;
            if (is_flores_code(t) || t == "<mask>") continue;
        }
        joined += t;
    }

    // Metaspace decode: U+2581 -> ' ', then drop one leading space.
    std::string out;
    out.reserve(joined.size());
    for (size_t i = 0; i < joined.size(); ) {
        if (i + 3 <= joined.size() &&
            std::memcmp(joined.data() + i, "\xe2\x96\x81", 3) == 0) {
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(joined[i]);
            i++;
        }
    }
    if (!out.empty() && out[0] == ' ') out.erase(0, 1);
    return out;
}

int32_t nllb_vocab_lang_id(const nllb_vocab & v, const std::string & code) {
    if (!is_flores_code(code)) return -1;
    return v.id_of(code);
}

std::vector<std::string> nllb_vocab_langs(const nllb_vocab & v) {
    std::vector<std::string> out;
    for (const std::string & t : v.tokens) if (is_flores_code(t)) out.push_back(t);
    return out;
}
