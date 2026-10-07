// tests/test-tokenizer.cpp - the tokenizer gate.
//
// Five passes, because they have different blind spots:
//
//   [vocab]   the vocabulary loaded at all, the merge derivation matched its
//             hash, and the 202 FLORES codes are where they should be. This is
//             the pass that would have caught the fairseq offset being wrong.
//   [tables]  where the UAX#29 grapheme tables came from - the GGUF, or this
//             binary. The only input with two possible sources, so the only
//             pass that has to say which, and to price any disagreement in
//             IDS rather than in table rows.
//   [golden]  ref/manifest.json - the same nine prompts the encoder goldens
//             use, compared WITH the [src_lang] ... [eos] wrapping. Exact ids.
//             This is the gate the handoff asked for.
//   [corpus]  ref/tokenizer_corpus.txt - ~240k cases from
//             scripts/dump_tokenizer_reference.py, comparing pieces (no
//             wrapping) AND the decode. Nine sentences cannot cover a
//             224711-entry normalizer trie; this can.
//   [sweep]   ref/tokenizer_sweep.txt - one hash over the ids of every one of
//             the 1112064 Unicode codepoints. Catches the case the corpus
//             structurally cannot: the C++ matching something the reference
//             leaves alone.
//
// There is no tolerance anywhere in this file. Ids are right or wrong. What
// replaces the tolerance is the bisect: on a mismatch it prints the normalized
// string, the Metaspace words and the per-word ids, so a normalizer bug and a
// BPE bug do not look alike.
//
// A pass whose input file is missing is a FAILURE, not a skip - deleting the
// goldens must not turn the gate green (step 4 shipped exactly that bug). Say
// --no-golden / --no-corpus / --no-sweep to leave one out on purpose.

#include "nllb-vocab.h"
#include "nllb-grapheme-data.h"

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// just enough JSON for ref/manifest.json
// --------------------------------------------------------------------------
namespace mini {

struct value;
using object = std::vector<std::pair<std::string, value>>;
using array  = std::vector<value>;

struct value {
    enum kind { NUL, BOOL, NUM, STR, ARR, OBJ } k = NUL;
    bool        b = false;
    double      n = 0;
    std::string s;
    std::vector<value>  arr;
    std::vector<std::pair<std::string, value>> obj;

    const value * find(const char * key) const {
        for (const auto & kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct parser {
    const std::string & in;
    size_t i = 0;
    bool   ok = true;
    std::string err;

    explicit parser(const std::string & s) : in(s) {}

    void ws() { while (i < in.size() && (unsigned char) in[i] <= ' ') i++; }
    bool eat(char c) { ws(); if (i < in.size() && in[i] == c) { i++; return true; } return false; }
    void fail(const char * m) { if (ok) { ok = false; err = m; } }

    void utf8(uint32_t cp, std::string & out) {
        if (cp < 0x80) out.push_back((char) cp);
        else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }

    uint32_t hex4() {
        uint32_t v = 0;
        for (int k = 0; k < 4 && i < in.size(); k++, i++) {
            const char c = in[i];
            v <<= 4;
            if      (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else { fail("bad \\u escape"); return 0; }
        }
        return v;
    }

    std::string str() {
        std::string out;
        if (!eat('"')) { fail("expected string"); return out; }
        while (i < in.size()) {
            const char c = in[i++];
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            if (i >= in.size()) break;
            const char e = in[i++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < in.size() &&
                        in[i] == '\\' && in[i + 1] == 'u') {
                        i += 2;
                        const uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(cp, out);
                    break;
                }
                default: fail("bad escape"); return out;
            }
        }
        fail("unterminated string");
        return out;
    }

    value parse() {
        value v;
        ws();
        if (i >= in.size()) { fail("eof"); return v; }
        const char c = in[i];
        if (c == '{') {
            i++; v.k = value::OBJ;
            ws();
            if (eat('}')) return v;
            while (ok) {
                std::string key = str();
                if (!eat(':')) { fail("expected :"); break; }
                v.obj.emplace_back(key, parse());
                if (eat(',')) continue;
                if (eat('}')) break;
                fail("expected , or }"); break;
            }
        } else if (c == '[') {
            i++; v.k = value::ARR;
            ws();
            if (eat(']')) return v;
            while (ok) {
                v.arr.push_back(parse());
                if (eat(',')) continue;
                if (eat(']')) break;
                fail("expected , or ]"); break;
            }
        } else if (c == '"') {
            v.k = value::STR; v.s = str();
        } else if (!strncmp(in.c_str() + i, "true", 4))  { i += 4; v.k = value::BOOL; v.b = true; }
        else if (!strncmp(in.c_str() + i, "false", 5))   { i += 5; v.k = value::BOOL; v.b = false; }
        else if (!strncmp(in.c_str() + i, "null", 4))    { i += 4; v.k = value::NUL; }
        else {
            char * end = nullptr;
            v.n = strtod(in.c_str() + i, &end);
            if (end == in.c_str() + i) { fail("bad number"); return v; }
            i = (size_t) (end - in.c_str());
            v.k = value::NUM;
        }
        return v;
    }
};

} // namespace mini

// --------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static bool b64_decode(const std::string & in, std::string & out) {
    static int8_t T[256];
    static bool init = false;
    if (!init) {
        memset(T, -1, sizeof(T));
        const char * A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char) A[i]] = (int8_t) i;
        init = true;
    }
    out.clear();
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\r' || c == '\n') continue;
        const int8_t d = T[(unsigned char) c];
        if (d < 0) return false;
        acc = (acc << 6) | (uint32_t) d;
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((char) ((acc >> bits) & 0xFF)); }
    }
    return true;
}

// Printable form of an arbitrary byte string, for failure reports.
static std::string esc(const std::string & s, size_t limit = 220) {
    std::string out;
    for (size_t i = 0; i < s.size() && out.size() < limit; i++) {
        const unsigned char c = (unsigned char) s[i];
        if (c >= 0x20 && c < 0x7F && c != '\\') out.push_back((char) c);
        else {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            out += buf;
        }
    }
    if (out.size() >= limit) out += "...";
    return out;
}

static std::string ids_str(const std::vector<int32_t> & v, size_t limit = 24) {
    std::string s = "[";
    for (size_t i = 0; i < v.size() && i < limit; i++) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    if (v.size() > limit) s += ", ...";
    return s + "]";
}

static std::string pieces_str(const nllb_vocab & v, const std::vector<int32_t> & ids,
                              size_t limit = 24) {
    std::string s;
    for (size_t i = 0; i < ids.size() && i < limit; i++) {
        if (i) s += " ";
        s += (ids[i] >= 0 && ids[i] < (int32_t) v.tokens.size())
             ? v.tokens[ids[i]] : "<?>";
    }
    if (ids.size() > limit) s += " ...";
    return s;
}

// The bisect. Which stage is wrong is not obvious from a list of ids.
static void report(const nllb_vocab & v, const std::string & text,
                   const std::vector<int32_t> & got,
                   const std::vector<int32_t> & exp) {
    size_t d = 0;
    while (d < got.size() && d < exp.size() && got[d] == exp[d]) d++;
    printf("      text       %s\n", esc(text).c_str());
    printf("      normalized %s\n", esc(nllb_vocab_normalize(v, text)).c_str());
    const std::vector<std::string> ws =
        nllb_vocab_metaspace(nllb_vocab_normalize(v, text));
    std::string joined;
    for (size_t i = 0; i < ws.size() && i < 12; i++) {
        if (i) joined += " | ";
        joined += ws[i];
    }
    printf("      words      %s\n", esc(joined).c_str());
    printf("      expected   %s\n", ids_str(exp).c_str());
    printf("      got        %s\n", ids_str(got).c_str());
    printf("      exp pieces %s\n", esc(pieces_str(v, exp)).c_str());
    printf("      got pieces %s\n", esc(pieces_str(v, got)).c_str());
    printf("      first difference at index %zu\n", d);
}

struct opts {
    std::string gguf, ref = "ref";
    std::string corpus, manifest, sweep, one_text;
    bool have_text = false;
    bool golden = true, do_corpus = true, do_sweep = true;
    bool verbose = false;
    int  max_report = 5;
};

// --------------------------------------------------------------------------
// passes
// --------------------------------------------------------------------------

static int pass_vocab(const nllb_vocab & v, const opts & o) {
    printf("[vocab]\n");
    printf("  %zu tokens, %u merges (hash %#018" PRIx64 ", %s)\n",
           v.tokens.size(), v.n_merges, v.merges_hash,
           v.merges_hash_source.empty() ? "source unrecorded"
                                        : v.merges_hash_source.c_str());
    printf("  source %s; bos=%d pad=%d eos=%d unk=%d\n",
           v.vocab_source.empty() ? "?" : v.vocab_source.c_str(),
           v.bos_id, v.pad_id, v.eos_id, v.unk_id);

    int bad = 0;
    const std::vector<std::string> langs = nllb_vocab_langs(v);
    printf("  %zu FLORES codes; eng_Latn=%d deu_Latn=%d\n", langs.size(),
           nllb_vocab_lang_id(v, "eng_Latn"), nllb_vocab_lang_id(v, "deu_Latn"));
    if (langs.empty()) { printf("  FAIL: no FLORES language codes\n"); bad++; }
    if (nllb_vocab_lang_id(v, "eng_Latn") < 0) { printf("  FAIL: eng_Latn missing\n"); bad++; }

    // A vocabulary whose ids are shifted is the classic fairseq-offset bug and
    // it is invisible unless something pins an absolute id.
    struct { const char * piece; int32_t id; } anchors[] = {
        {"<s>", 0}, {"<pad>", 1}, {"</s>", 2}, {"<unk>", 3},
    };
    for (const auto & a : anchors) {
        const int32_t id = v.id_of(a.piece);
        if (id != a.id) {
            printf("  FAIL: %s is id %d, expected %d\n", a.piece, id, a.id);
            bad++;
        }
    }
    // An unknown code must be refused, not silently resolved to some token.
    if (nllb_vocab_lang_id(v, "zzz_Zzzz") >= 0) {
        printf("  FAIL: a nonexistent language code resolved\n"); bad++;
    }
    if (o.verbose) {
        printf("  first codes:");
        for (size_t i = 0; i < langs.size() && i < 6; i++)
            printf(" %s", langs[i].c_str());
        printf(" ...\n");
    }
    printf("  %s\n\n", bad ? "FAIL" : "ok");
    return bad;
}

// The tables are the one input that can come from either the file or the
// binary, so this is the one pass whose job is to say WHICH - and, when the
// two differ, to price the difference in ids rather than in table rows.
static int pass_tables(const nllb_vocab & v, const opts & o) {
    printf("[tables]  UAX#29 grapheme boundaries\n");
    printf("  %zu GCB ranges + %zu pictographic, hash %#018" PRIx64 "\n",
           v.gcb.size() / 3, v.ext.size() / 2, v.tables_hash);
    printf("  source: %s (UCD %s)\n",
           v.tables_from_file ? "THE GGUF" : "the compiled-in header",
           v.ucd_version.empty() ? "unrecorded" : v.ucd_version.c_str());

    if (!v.tables_from_file) {
        printf("  This file predates the baked-in tables, so its ids depend on\n"
               "  which Unicode release THIS BINARY was built against. That is\n"
               "  not wrong, it is just not self-contained - reconvert to pin it.\n");
        printf("  ok\n\n");
        return 0;
    }

    // Same numbers on both sides is the common case and needs no id run.
    nllb_vocab builtin;
    for (int i = 0; i < NLLB_GCB_TABLE_N; i++) {
        builtin.gcb.push_back(NLLB_GCB_TABLE[i].lo);
        builtin.gcb.push_back(NLLB_GCB_TABLE[i].hi);
        builtin.gcb.push_back(NLLB_GCB_TABLE[i].prop);
    }
    for (int i = 0; i < NLLB_EXTPICT_TABLE_N; i++) {
        builtin.ext.push_back(NLLB_EXTPICT_TABLE[i].lo);
        builtin.ext.push_back(NLLB_EXTPICT_TABLE[i].hi);
    }
    if (builtin.gcb == v.gcb && builtin.ext == v.ext) {
        printf("  identical to the compiled-in tables, row for row - this "
               "binary and\n  this file agree on every cluster boundary\n");
        printf("  ok\n\n");
        return 0;
    }

    // They differ. That is legal - a file pinned to an older UCD, or a binary
    // rebuilt on a newer one - and the only question that matters is whether
    // any id moves. Run the corpus under the compiled-in tables and diff.
    printf("  DIFFER from the compiled-in tables (%zu vs %d GCB ranges).\n",
           v.gcb.size() / 3, NLLB_GCB_TABLE_N);
    printf("  Legal, but it means this binary alone would tokenize differently.\n"
           "  Re-running the corpus under the compiled-in tables to price it:\n");

    nllb_vocab alt = v;              // same vocab, other tables
    alt.gcb = builtin.gcb;
    alt.ext = builtin.ext;

    std::ifstream f(o.corpus);
    if (!f) {
        printf("  FAIL: cannot read %s, so the difference cannot be priced\n\n",
               o.corpus.c_str());
        return 1;
    }
    size_t n = 0, moved = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t t1 = line.find('\t');
        if (t1 == std::string::npos) continue;
        std::string text;
        if (!b64_decode(line.substr(0, t1), text)) continue;
        n++;
        if (nllb_vocab_encode_pieces(v, text) !=
            nllb_vocab_encode_pieces(alt, text)) {
            if (moved < (size_t) o.max_report) {
                printf("    case %zu moves: %s\n", n, esc(text, 80).c_str());
            }
            moved++;
        }
    }
    printf("  %zu cases, %zu tokenize differently under the two tables\n", n, moved);
    printf("  %s\n\n", moved ? "FAIL" : "ok (the difference is inert here)");
    return moved ? 1 : 0;
}

static int pass_golden(const nllb_vocab & v, const opts & o) {
    printf("[golden]  ref/manifest.json, with the language code and eos\n");
    std::string raw;
    if (!read_file(o.manifest, raw)) {
        printf("  FAIL: cannot read %s\n", o.manifest.c_str());
        printf("        run scripts/dump_reference.py first, or pass --no-golden\n\n");
        return 1;
    }
    mini::parser p(raw);
    mini::value root = p.parse();
    if (!p.ok) { printf("  FAIL: %s at byte %zu\n\n", p.err.c_str(), p.i); return 1; }

    const mini::value * lang = root.find("src_lang");
    const mini::value * cases = root.find("cases");
    if (!lang || lang->k != mini::value::STR) {
        printf("  FAIL: manifest has no src_lang string. These goldens were "
               "dumped with --random-ids, so there is no text to tokenize.\n\n");
        return 1;
    }
    if (!cases || cases->k != mini::value::ARR || cases->arr.empty()) {
        printf("  FAIL: manifest has no cases\n\n");
        return 1;
    }

    int bad = 0, shown = 0;
    for (const mini::value & c : cases->arr) {
        const mini::value * nm = c.find("name");
        const mini::value * tx = c.find("text");
        const mini::value * id = c.find("ids");
        if (!nm || !tx || !id || id->k != mini::value::ARR) {
            printf("  FAIL: malformed case\n"); bad++; continue;
        }
        std::vector<int32_t> exp;
        for (const mini::value & x : id->arr) exp.push_back((int32_t) x.n);

        std::vector<int32_t> got;
        std::string err;
        if (!nllb_vocab_encode(v, tx->s, lang->s, got, err)) {
            printf("  %-16s FAIL: %s\n", nm->s.c_str(), err.c_str());
            bad++; continue;
        }
        const bool ok = got == exp;
        if (!ok) bad++;
        if (ok) {
            if (o.verbose)
                printf("  %-16s T=%-3zu ok\n", nm->s.c_str(), got.size());
        } else {
            printf("  %-16s T=%zu got, %zu expected  MISMATCH\n",
                   nm->s.c_str(), got.size(), exp.size());
            if (shown++ < o.max_report) report(v, tx->s, got, exp);
        }
    }
    printf("  %zu cases, %d failed  %s\n\n", cases->arr.size(), bad,
           bad ? "FAIL" : "ok");
    return bad;
}

static int pass_corpus(const nllb_vocab & v, const opts & o) {
    printf("[corpus]  %s, pieces only, plus the decode\n", o.corpus.c_str());
    std::ifstream f(o.corpus);
    if (!f) {
        printf("  FAIL: cannot read it. Run "
               "scripts/dump_tokenizer_reference.py, or pass --no-corpus\n\n");
        return 1;
    }
    size_t n = 0, bad_ids = 0, bad_dec = 0, shown = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t t1 = line.find('\t');
        const size_t t2 = line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) {
            printf("  FAIL: malformed line %zu\n", n + 1);
            return 1;
        }
        std::string text, dec;
        if (!b64_decode(line.substr(0, t1), text) ||
            !b64_decode(line.substr(t2 + 1), dec)) {
            printf("  FAIL: bad base64 on line %zu\n", n + 1);
            return 1;
        }
        std::vector<int32_t> exp;
        {
            const std::string s = line.substr(t1 + 1, t2 - t1 - 1);
            size_t i = 0;
            while (i < s.size()) {
                size_t j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) exp.push_back((int32_t) strtol(s.substr(i, j - i).c_str(), nullptr, 10));
                i = j + 1;
            }
        }
        n++;

        const std::vector<int32_t> got = nllb_vocab_encode_pieces(v, text);
        if (got != exp) {
            bad_ids++;
            if (shown++ < (size_t) o.max_report) {
                printf("  case %zu  MISMATCH\n", n);
                report(v, text, got, exp);
            }
            continue;   // the decode of wrong ids says nothing
        }
        const std::string got_dec = nllb_vocab_decode(v, got, true);
        if (got_dec != dec) {
            bad_dec++;
            if (shown++ < (size_t) o.max_report) {
                printf("  case %zu  DECODE MISMATCH\n", n);
                printf("      text     %s\n", esc(text).c_str());
                printf("      expected %s\n", esc(dec).c_str());
                printf("      got      %s\n", esc(got_dec).c_str());
            }
        }
    }
    printf("  %zu cases, %zu id mismatches, %zu decode mismatches  %s\n\n",
           n, bad_ids, bad_dec, (bad_ids || bad_dec) ? "FAIL" : "ok");
    if (n == 0) { printf("  FAIL: corpus was empty\n\n"); return 1; }
    return (int) (bad_ids + bad_dec);
}

static int pass_sweep(const nllb_vocab & v, const opts & o) {
    printf("[sweep]   every Unicode codepoint, reduced to one hash\n");
    std::string raw;
    if (!read_file(o.sweep, raw)) {
        printf("  FAIL: cannot read %s (--no-sweep to leave it out)\n\n",
               o.sweep.c_str());
        return 1;
    }
    unsigned long long exp_n = 0, exp_h = 0;
    if (sscanf(raw.c_str(), "n=%llu hash=%llx", &exp_n, &exp_h) != 2) {
        printf("  FAIL: cannot parse %s\n\n", o.sweep.c_str());
        return 1;
    }

    uint64_t h = 0xCBF29CE484222325ull;
    uint64_t n = 0;
    std::string buf;
    for (uint32_t cp = 1; cp < 0x110000; cp++) {
        if (cp >= 0xD800 && cp < 0xE000) continue;
        buf.clear();
        if (cp < 0x80) buf.push_back((char) cp);
        else if (cp < 0x800) {
            buf.push_back((char) (0xC0 | (cp >> 6)));
            buf.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            buf.push_back((char) (0xE0 | (cp >> 12)));
            buf.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            buf.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            buf.push_back((char) (0xF0 | (cp >> 18)));
            buf.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            buf.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            buf.push_back((char) (0x80 | (cp & 0x3F)));
        }
        const std::vector<int32_t> ids = nllb_vocab_encode_pieces(v, buf);
        char line[256];
        int len = snprintf(line, sizeof(line), "%u:", cp);
        for (size_t i = 0; i < ids.size() && len < (int) sizeof(line) - 16; i++) {
            len += snprintf(line + len, sizeof(line) - (size_t) len,
                            "%s%d", i ? "," : "", ids[i]);
        }
        len += snprintf(line + len, sizeof(line) - (size_t) len, "\n");
        for (int i = 0; i < len; i++) {
            h ^= (unsigned char) line[i];
            h *= 0x100000001B3ull;
        }
        n++;
    }
    const bool ok = (n == exp_n) && (h == exp_h);
    printf("  n=%" PRIu64 " (expected %llu), hash %#018" PRIx64
           " (expected %#018llx)\n", n, exp_n, h, exp_h);
    if (!ok) {
        printf("  FAIL: at least one codepoint tokenizes differently. The "
               "corpus pass cannot find this one - rerun with --text on a "
               "suspect character.\n");
    }
    printf("  %s\n\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------

int main(int argc, char ** argv) {
    opts o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if      (a == "--corpus"   && i + 1 < argc) o.corpus   = argv[++i];
        else if (a == "--manifest" && i + 1 < argc) o.manifest = argv[++i];
        else if (a == "--sweep"    && i + 1 < argc) o.sweep    = argv[++i];
        else if (a == "--text"     && i + 1 < argc) { o.one_text = argv[++i]; o.have_text = true; }
        else if (a == "--max-report" && i + 1 < argc) o.max_report = atoi(argv[++i]);
        else if (a == "--no-golden") o.golden    = false;
        else if (a == "--no-corpus") o.do_corpus = false;
        else if (a == "--no-sweep")  o.do_sweep  = false;
        else if (a == "-v")          o.verbose   = true;
        else if (a == "-h" || a == "--help") {
            printf("usage: %s <model.gguf> [ref_dir] [options]\n"
                   "  --text STR      tokenize one string and print every stage\n"
                   "  --manifest F    default <ref_dir>/manifest.json\n"
                   "  --corpus F      default <ref_dir>/tokenizer_corpus.txt\n"
                   "  --sweep F       default <ref_dir>/tokenizer_sweep.txt\n"
                   "  --no-golden|--no-corpus|--no-sweep\n"
                   "  --max-report N  mismatches to print in full (default 5)\n"
                   "  -v              per-case lines on the passes that pass\n",
                   argv[0]);
            return 0;
        }
        else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
        else pos.push_back(a);
    }
    if (pos.empty()) {
        fprintf(stderr, "usage: %s <model.gguf> [ref_dir] [options]\n", argv[0]);
        return 2;
    }
    o.gguf = pos[0];
    if (pos.size() > 1) o.ref = pos[1];
    if (o.manifest.empty()) o.manifest = o.ref + "/manifest.json";
    if (o.corpus.empty())   o.corpus   = o.ref + "/tokenizer_corpus.txt";
    if (o.sweep.empty())    o.sweep    = o.ref + "/tokenizer_sweep.txt";

    nllb_vocab v;
    std::string err;
    if (!nllb_vocab_load(o.gguf, v, err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    if (o.have_text) {
        const std::string norm = nllb_vocab_normalize(v, o.one_text);
        printf("text       %s\n", esc(o.one_text, 4096).c_str());
        printf("normalized %s\n", esc(norm, 4096).c_str());
        const std::vector<std::string> ws = nllb_vocab_metaspace(norm);
        for (const std::string & w : ws) {
            const std::vector<int32_t> ids = nllb_vocab_bpe(v, w);
            printf("  word %-24s %s\n", esc(w, 40).c_str(), ids_str(ids).c_str());
        }
        std::vector<int32_t> full;
        if (nllb_vocab_encode(v, o.one_text, "eng_Latn", full, err))
            printf("eng_Latn   %s\n           %s\n", ids_str(full, 64).c_str(),
                   esc(pieces_str(v, full, 64), 4096).c_str());
        else
            printf("encode failed: %s\n", err.c_str());
        printf("decoded    %s\n",
               esc(nllb_vocab_decode(v, full, true), 4096).c_str());
        return 0;
    }

    int bad = pass_vocab(v, o);
    bad += pass_tables(v, o);
    if (o.golden)    bad += pass_golden(v, o);   else printf("[golden]  SKIPPED on purpose\n\n");
    if (o.do_corpus) bad += pass_corpus(v, o);   else printf("[corpus]  SKIPPED on purpose\n\n");
    if (o.do_sweep)  bad += pass_sweep(v, o);    else printf("[sweep]   SKIPPED on purpose\n\n");

    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
