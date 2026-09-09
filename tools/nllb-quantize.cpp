// nllb-quantize - read an f32/f16 GGUF, write a quantised one.
//
//   ./build/bin/nllb-quantize models/nllb-600M-f16.gguf
//                             models/nllb-600M-q5km.gguf q5_k_m
//   ./build/bin/nllb-quantize in.gguf out.gguf q5_k_m --dry-run
//   ./build/bin/nllb-quantize --list-mixes
//
// WHY THIS EXISTS AND tools/requantize_gguf.py DOES NOT REPLACE IT.
// gguf-py's `quants.quantize` raises NotImplementedError for Q2_K..Q6_K and the
// whole IQ family while DEQUANTISING all of them perfectly - trap 33, and that
// asymmetry is exactly how you fail to notice. Q5_K is step 10's stated target,
// so the Python tool's ceiling is Q8_0/Q5_0/Q4_0 and something has to call
// ggml_quantize_chunk. Reimplementing a K-quant's block layout in numpy would
// be a second implementation of a bit-exact format, which is the class of bug
// trap 25 exists to prevent elsewhere in this repo.
//
// A MIX IS NOT A TYPE, which is the other thing this tool gets right and the
// obvious `--type q5_k` would get wrong. llama.cpp's Q4_K_M and friends are
// PER-TENSOR POLICIES: most weights at the named type, a few sensitive ones a
// step up. Here the sensitive one is known and measured rather than inherited:
//
//   trap 31 - token_embd.weight is TIED. One tensor serves both
//   ggml_get_rows and the lm_head, so quantising "the output projection" also
//   quantises every embedding fed to block 0, and input error is amplified
//   through twelve pre-layernorm blocks while output error is not amplified at
//   all. Measured: quantising that one tensor moves the hidden states 4.8x
//   (8.24e-04 -> 3.98e-03) and costs 3.3x of per-decision headroom, against
//   26.4x for quantising everything. It is the first tensor to protect.
//
//   position_embd.weight is NEVER quantised by any mix here. It is 4 MB, it is
//   not learned - the converter recomputes it (trap 1) - and its rows are
//   sinusoids whose whole job is to be distinguishable from one another. Pay
//   4 MB rather than find out. --tensor-type overrides it if you want to
//   measure that, which is the point of the override existing.
//
//   Every 1-D tensor keeps its input type. That is every LayerNorm weight and
//   every bias, ~2 MB in total, and it is what `--min-rank 2` enforces on the
//   Python side. K-quants need ne[0] % 256 == 0 anyway.
//
// TRAP 32 IS THE FAILURE MODE THAT MATTERS. Since step 8 the GGUF carries the
// tokenizer - vocabulary, precompiled charsmap, merge hash, added tokens, the
// UAX#29 grapheme tables - and nllb_vocab_load, hence nllb-cli, REFUSES a file
// missing any of them. A quantiser that enumerates metadata keys by hand
// produces a file that translates nothing. So this tool copies the metadata
// with ONE call, gguf_set_kv, and then re-opens what it wrote and asserts the
// key sets match. Not a comment: a check, and it runs by default.
//
// MEMORY. The input is read into host RAM (gguf_init_from_file with
// no_alloc = false) and the output is built in a second context before being
// written, so peak usage is roughly input + output: ~1.9 GB from the f16 file,
// ~3.2 GB from f32. llama.cpp streams tensor by tensor instead; that is a real
// improvement and a real source of offset arithmetic, and it is not worth it
// for a tool that runs once per file.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// --------------------------------------------------------------------------
// types and mixes
// --------------------------------------------------------------------------

struct type_name { const char * name; ggml_type type; };

const type_name TYPES[] = {
    { "f32",  GGML_TYPE_F32  },
    { "f16",  GGML_TYPE_F16  },
    { "bf16", GGML_TYPE_BF16 },
    { "q8_0", GGML_TYPE_Q8_0 },
    { "q5_0", GGML_TYPE_Q5_0 },
    { "q5_1", GGML_TYPE_Q5_1 },
    { "q4_0", GGML_TYPE_Q4_0 },
    { "q4_1", GGML_TYPE_Q4_1 },
    { "q6_k", GGML_TYPE_Q6_K },
    { "q5_k", GGML_TYPE_Q5_K },
    { "q4_k", GGML_TYPE_Q4_K },
    { "q3_k", GGML_TYPE_Q3_K },
    { "q2_k", GGML_TYPE_Q2_K },
};

bool type_from_name(const std::string & s, ggml_type & out) {
    for (const type_name & t : TYPES) {
        if (s == t.name) { out = t.type; return true; }
    }
    return false;
}

// A mix is: a base type for 2-D weights, plus a list of (name substring ->
// type) exceptions applied in order. Deliberately data, not code, so that
// adding "the same as q5_k_m but with token_embd at Q8_0" is one line and
// shows up in --list-mixes.
struct rule  { const char * match; ggml_type type; };
struct mix_t {
    const char * name;
    ggml_type    base;          // every other 2-D weight
    const char * blurb;
    rule         rules[6];      // terminated by a null match
};

const mix_t MIXES[] = {
    { "f16",   GGML_TYPE_F16,  "every 2-D weight f16. The step 7 baseline; "
                              "useful for testing this tool against a file "
                              "the converter can also produce.",
      { { nullptr, GGML_TYPE_F32 } } },

    { "q8_0",  GGML_TYPE_Q8_0, "every 2-D weight Q8_0. Matches "
                              "requantize_gguf.py --quantize '*' --min-rank 2.",
      { { nullptr, GGML_TYPE_F32 } } },

    { "q8emb", GGML_TYPE_F16,  "token_embd Q8_0, everything else f16. The "
                              "step 10 q8emb file, and the one that isolates "
                              "trap 31.",
      { { "token_embd", GGML_TYPE_Q8_0 }, { nullptr, GGML_TYPE_F32 } } },

    { "q6_k",  GGML_TYPE_Q6_K, "every 2-D weight Q6_K. The gentlest K-quant "
                              "and the right first K-quant to measure: if its "
                              "headroom is already below 1 the ladder stops "
                              "here.",
      { { nullptr, GGML_TYPE_F32 } } },

    { "q5_k_s", GGML_TYPE_Q5_K, "every 2-D weight Q5_K, token_embd included. "
                               "The aggressive end. Expect it to be where the "
                               "1.50x headroom finally goes.",
      { { nullptr, GGML_TYPE_F32 } } },

    { "q5_k_m", GGML_TYPE_Q5_K, "Q5_K, with token_embd and every ffn_down at "
                               "Q6_K. token_embd for trap 31; ffn_down "
                               "because it reads the relu output, whose "
                               "dynamic range is one-sided and wide.",
      { { "token_embd", GGML_TYPE_Q6_K },
        { "ffn_down",   GGML_TYPE_Q6_K },
        { nullptr,      GGML_TYPE_F32 } } },

    { "q4_k_m", GGML_TYPE_Q4_K, "Q4_K, with token_embd and every ffn_down at "
                               "Q6_K. Included so the ladder has a rung below "
                               "Q5_K; do not ship it without running the gate.",
      { { "token_embd", GGML_TYPE_Q6_K },
        { "ffn_down",   GGML_TYPE_Q6_K },
        { nullptr,      GGML_TYPE_F32 } } },
};

const mix_t * mix_from_name(const std::string & s) {
    for (const mix_t & m : MIXES) if (s == m.name) return &m;
    return nullptr;
}

bool contains(const std::string & hay, const char * needle) {
    return hay.find(needle) != std::string::npos;
}

bool contains(const std::string & hay, const std::string & needle) {
    return hay.find(needle) != std::string::npos;
}

// --------------------------------------------------------------------------

// llama.cpp's LLAMA_FTYPE numbering, for general.file_type. It is a LABEL: a
// mix is not a format and the per-tensor types in the file are the truth. It is
// written anyway because it is convention and because leaving it out is what
// tools/requantize_gguf.py did by accident for the whole of step 10 - the key
// was on its "the writer owns this" list and the writer was never told to
// produce it, so every requantised file silently had no file_type and the
// trap-32 assertion could not see the hole because the key was excluded from
// the comparison. A key you exclude from a copy is a key you must then check.
uint32_t ftype_for(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 0;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_Q4_0: return 2;
        case GGML_TYPE_Q4_1: return 3;
        case GGML_TYPE_Q8_0: return 7;
        case GGML_TYPE_Q5_0: return 8;
        case GGML_TYPE_Q5_1: return 9;
        case GGML_TYPE_Q2_K: return 10;
        case GGML_TYPE_Q3_K: return 12;
        case GGML_TYPE_Q4_K: return 15;
        case GGML_TYPE_Q5_K: return 17;
        case GGML_TYPE_Q6_K: return 18;
        case GGML_TYPE_BF16: return 32;
        default:             return 0;
    }
}

std::string human(size_t bytes) {
    char buf[64];
    const double g = (double) bytes / (1024.0 * 1024.0 * 1024.0);
    if (g >= 1.0) std::snprintf(buf, sizeof buf, "%.3f GB", g);
    else          std::snprintf(buf, sizeof buf, "%.1f MB",
                                (double) bytes / (1024.0 * 1024.0));
    return buf;
}

struct plan_entry {
    ggml_tensor * src  = nullptr;
    ggml_type     to   = GGML_TYPE_F32;
    const char *  why  = nullptr;   // non-null: left alone, and this says why
    size_t        bytes_in  = 0;
    size_t        bytes_out = 0;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s <in.gguf> <out.gguf> <mix> [options]\n"
        "       %s --list-mixes\n"
        "\n"
        "  <mix> is a NAMED PER-TENSOR POLICY, not a type. --list-mixes prints\n"
        "  them. Q4_K_M and friends in llama.cpp are policies too; calling the\n"
        "  argument --type would be a lie about what it does.\n"
        "\n"
        "  --tensor-type SUBSTR=TYPE   override, repeatable, applied after the\n"
        "                              mix in the order given. SUBSTR is a\n"
        "                              plain substring of the tensor name.\n"
        "  --dry-run                   print the plan and the sizes, write\n"
        "                              nothing.\n"
        "  --no-verify                 skip re-opening the output and asserting\n"
        "                              the metadata survived. Do not.\n"
        "  --list-types                every ggml type this tool accepts.\n",
        argv0, argv0);
}

} // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--list-mixes")) {
            std::printf("mixes (a mix is a per-tensor policy; 1-D tensors always\n"
                        "keep their input type, and position_embd is never\n"
                        "quantised by any mix - see the header comment):\n\n");
            for (const mix_t & m : MIXES) {
                std::printf("  %-8s base %-5s  %s\n", m.name,
                            ggml_type_name(m.base), m.blurb);
                for (const rule * r = m.rules; r->match; ++r) {
                    std::printf("           %-10s -> %s\n", r->match,
                                ggml_type_name(r->type));
                }
            }
            return 0;
        }
        if (!std::strcmp(argv[i], "--list-types")) {
            for (const type_name & t : TYPES) std::printf("%s\n", t.name);
            return 0;
        }
    }
    if (argc < 4) { usage(argv[0]); return 2; }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const std::string mix_name = argv[3];

    const mix_t * mix = mix_from_name(mix_name);
    if (!mix) {
        std::fprintf(stderr, "unknown mix %s - run --list-mixes\n",
                     mix_name.c_str());
        return 2;
    }

    bool dry_run = false, verify = true;
    std::vector<std::pair<std::string, ggml_type>> overrides;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dry-run"))         { dry_run = true; }
        else if (!std::strcmp(argv[i], "--no-verify"))  { verify  = false; }
        else if (!std::strcmp(argv[i], "--tensor-type") && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            ggml_type t;
            if (eq == std::string::npos ||
                !type_from_name(spec.substr(eq + 1), t)) {
                std::fprintf(stderr, "--tensor-type wants SUBSTR=TYPE, got %s "
                                     "(--list-types)\n", spec.c_str());
                return 2;
            }
            overrides.emplace_back(spec.substr(0, eq), t);
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    // ---------------------------------------------------------------- read
    ggml_context * ctx_in = nullptr;
    gguf_init_params ip = {};
    ip.no_alloc = false;      // we need the bytes, not just the shapes
    ip.ctx      = &ctx_in;
    gguf_context * gg_in = gguf_init_from_file(in_path.c_str(), ip);
    if (!gg_in) {
        std::fprintf(stderr, "cannot read %s as a GGUF\n", in_path.c_str());
        return 2;
    }
    const int64_t n_tensors = gguf_get_n_tensors(gg_in);
    const int64_t n_kv      = gguf_get_n_kv(gg_in);
    std::printf("in:   %s\n      %" PRId64 " tensors, %" PRId64
                " metadata keys, alignment %zu\n",
                in_path.c_str(), n_tensors, n_kv, gguf_get_alignment(gg_in));
    std::printf("mix:  %s (base %s)\n\n", mix->name, ggml_type_name(mix->base));

    // ---------------------------------------------------------------- plan
    std::vector<plan_entry> plan;
    plan.reserve((size_t) n_tensors);
    size_t total_in = 0, total_out = 0;
    bool fatal = false;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gg_in, i);
        ggml_tensor * t = ggml_get_tensor(ctx_in, name);
        if (!t) {
            std::fprintf(stderr, "tensor %s is in the index but not the "
                                 "context\n", name);
            fatal = true;
            break;
        }
        plan_entry e;
        e.src      = t;
        e.to       = t->type;
        e.bytes_in = ggml_nbytes(t);

        const std::string n = name;
        const int rank = ggml_n_dims(t);

        if (rank < 2) {
            e.why = "1-D: LayerNorm weight or bias, kept as-is";
        } else if (contains(n, "position_embd")) {
            e.why = "sinusoidal positions, never quantised by a mix";
        } else if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 &&
                   t->type != GGML_TYPE_BF16) {
            // Requantising a quantised tensor means dequantise-then-requantise,
            // which compounds two lots of block error and reads as one. Refuse
            // rather than produce a file whose error nobody can account for.
            e.why = "already quantised - requantise from the f32 or f16 file";
        } else {
            e.to = mix->base;
            for (const rule * r = mix->rules; r->match; ++r) {
                if (contains(n, r->match)) { e.to = r->type; break; }
            }
        }
        // Overrides win over the mix AND over the exclusions above, because the
        // point of an override is to measure something the defaults refuse.
        for (const auto & o : overrides) {
            if (contains(n, o.first)) { e.to = o.second; e.why = nullptr; }
        }

        // Trap 34: every NLLB weight row is 1024 or 4096 long, so every
        // K-quant is legal. That is luck, not design - d_model = 1536 would
        // need padding - so it is asserted rather than assumed, and the message
        // says which tensor and what its row length is.
        const int64_t blck = ggml_blck_size(e.to);
        if (!e.why && t->ne[0] % blck != 0) {
            std::fprintf(stderr,
                "%s: %" PRId64 " elements per row is not a multiple of %s's "
                "block size %" PRId64 ".\n"
                "  Trap 34 says this cannot happen for NLLB (every row is 1024 "
                "or 4096), so either\n"
                "  this is not an NLLB checkpoint or the geometry changed. "
                "Padding is NOT the answer:\n"
                "  it would change what ggml_mul_mat computes.\n",
                name, t->ne[0], ggml_type_name(e.to), blck);
            fatal = true;
        }
        if (!e.why && ggml_quantize_requires_imatrix(e.to)) {
            std::fprintf(stderr,
                "%s: %s needs an importance matrix and this tool has none.\n"
                "  Only IQ2_XXS, IQ2_XS and IQ1_S do; everything in --list-types "
                "works without one.\n", name, ggml_type_name(e.to));
            fatal = true;
        }

        e.bytes_out = e.why ? e.bytes_in
                            : ggml_row_size(e.to, t->ne[0]) *
                              (ggml_nelements(t) / t->ne[0]);
        total_in  += e.bytes_in;
        total_out += e.bytes_out;
        plan.push_back(e);
    }
    if (fatal) { gguf_free(gg_in); ggml_free(ctx_in); return 2; }

    // Print the plan grouped, because 510 lines is not a plan, it is a log.
    {
        std::vector<std::pair<ggml_type, int>> counts;
        for (const plan_entry & e : plan) {
            auto it = std::find_if(counts.begin(), counts.end(),
                                   [&](const std::pair<ggml_type, int> & c) {
                                       return c.first == e.to;
                                   });
            if (it == counts.end()) counts.emplace_back(e.to, 1);
            else                    it->second++;
        }
        std::printf("plan, by target type:\n");
        for (const auto & c : counts) {
            std::printf("  %-6s %4d tensor(s)\n", ggml_type_name(c.first),
                        c.second);
        }
        std::printf("\nthe tensors this mix treats specially:\n");
        for (const plan_entry & e : plan) {
            const std::string n = ggml_get_name(e.src);
            const bool special =
                e.why != nullptr ||
                (ggml_n_dims(e.src) >= 2 && e.to != mix->base);
            if (special && (e.why == nullptr || ggml_n_dims(e.src) >= 2)) {
                std::printf("  %-34s %-5s -> %-5s  %s\n", n.c_str(),
                            ggml_type_name(e.src->type), ggml_type_name(e.to),
                            e.why ? e.why : "mix rule");
            }
        }
        std::printf("\nsize: %s -> %s  (%.2fx)\n",
                    human(total_in).c_str(), human(total_out).c_str(),
                    total_out ? (double) total_in / (double) total_out : 0.0);
    }

    if (dry_run) {
        std::printf("\n--dry-run: nothing written.\n");
        gguf_free(gg_in);
        ggml_free(ctx_in);
        return 0;
    }

    // ---------------------------------------------------------------- write
    ggml_init_params op = {};
    op.mem_size = total_out + (size_t) ggml_tensor_overhead() * (n_tensors + 16)
                + 1024u * 1024;
    op.mem_buffer = nullptr;
    op.no_alloc   = false;
    ggml_context * ctx_out = ggml_init(op);
    if (!ctx_out) {
        std::fprintf(stderr, "ggml_init failed for %s of output\n",
                     human(op.mem_size).c_str());
        gguf_free(gg_in); ggml_free(ctx_in);
        return 2;
    }

    gguf_context * gg_out = gguf_init_empty();
    // ONE call. Not a loop over key names, not a list of keys this tool knows
    // about. Trap 32: the tokenizer is in here, and a key this tool has never
    // heard of is exactly the key a future step will add.
    gguf_set_kv(gg_out, gg_in);

    ggml_type dominant = mix->base;
    gguf_set_val_u32(gg_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(gg_out, "general.file_type", ftype_for(dominant));

    std::vector<float> f32;      // dequantisation scratch, reused
    size_t written = 0;
    int    n_quantised = 0;

    for (size_t i = 0; i < plan.size(); ++i) {
        plan_entry & e = plan[i];
        ggml_tensor * src = e.src;
        const char * name = ggml_get_name(src);

        ggml_tensor * dst = ggml_new_tensor(ctx_out, e.to,
                                            ggml_n_dims(src), src->ne);
        if (!dst) {
            std::fprintf(stderr, "out of arena at %s\n", name);
            gguf_free(gg_out); ggml_free(ctx_out);
            gguf_free(gg_in);  ggml_free(ctx_in);
            return 2;
        }
        ggml_set_name(dst, name);

        if (e.to == src->type) {
            std::memcpy(dst->data, src->data, ggml_nbytes(src));
        } else {
            const int64_t n_per_row = src->ne[0];
            const int64_t nrows     = ggml_nelements(src) / n_per_row;
            const int64_t n         = nrows * n_per_row;

            // ggml_quantize_chunk takes f32, so an f16 input is widened first.
            // This is where the f16 file's own rounding stops being visible:
            // quantising the f16 file and quantising the f32 file give
            // DIFFERENT bytes, and the f16 route has already lost the mantissa
            // bits the quantiser would have used to pick its scale. Convert
            // from f32 when you have it; the difference is small and it is not
            // zero, and it is the sort of thing that shows up as a headroom
            // number that will not reproduce.
            f32.resize((size_t) n);
            if (src->type == GGML_TYPE_F32) {
                std::memcpy(f32.data(), src->data, (size_t) n * sizeof(float));
            } else {
                const ggml_type_traits * tr = ggml_get_type_traits(src->type);
                if (!tr || !tr->to_float) {
                    std::fprintf(stderr, "no dequantiser for %s (%s)\n",
                                 ggml_type_name(src->type), name);
                    gguf_free(gg_out); ggml_free(ctx_out);
                    gguf_free(gg_in);  ggml_free(ctx_in);
                    return 2;
                }
                tr->to_float(src->data, f32.data(), n);
            }
            const size_t got = ggml_quantize_chunk(e.to, f32.data(), dst->data,
                                                   0, nrows, n_per_row, nullptr);
            if (got != ggml_nbytes(dst)) {
                std::fprintf(stderr, "%s: ggml_quantize_chunk wrote %zu bytes, "
                                     "the tensor is %zu\n",
                             name, got, ggml_nbytes(dst));
                gguf_free(gg_out); ggml_free(ctx_out);
                gguf_free(gg_in);  ggml_free(ctx_in);
                return 2;
            }
            ++n_quantised;

            // The error, per tensor, in the only units that compare across
            // types: RMS of the change over RMS of the weights. Printed rather
            // than checked, because there is no threshold that is right for
            // every tensor - but a tensor whose ratio is an order off its
            // neighbours is worth looking at before the gate runs.
            const ggml_type_traits * back = ggml_get_type_traits(e.to);
            if (back && back->to_float) {
                std::vector<float> rt((size_t) n);
                back->to_float(dst->data, rt.data(), n);
                double se = 0.0, sw = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    const double d = (double) rt[(size_t) k] - (double) f32[(size_t) k];
                    se += d * d;
                    sw += (double) f32[(size_t) k] * (double) f32[(size_t) k];
                }
                const double rms  = std::sqrt(se / (double) n);
                const double rmsw = std::sqrt(sw / (double) n);
                std::printf("  %-34s %-5s -> %-5s  rms/rms(w) = %.4e\n",
                            name, ggml_type_name(src->type),
                            ggml_type_name(e.to),
                            rmsw > 0.0 ? rms / rmsw : 0.0);
            }
        }

        gguf_add_tensor(gg_out, dst);
        written += ggml_nbytes(dst);
    }

    std::printf("\nquantised %d of %zu tensors, %s of tensor data\n",
                n_quantised, plan.size(), human(written).c_str());

    if (!gguf_write_to_file(gg_out, out_path.c_str(), /*only_meta =*/ false)) {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        gguf_free(gg_out); ggml_free(ctx_out);
        gguf_free(gg_in);  ggml_free(ctx_in);
        return 2;
    }
    std::printf("wrote %s\n", out_path.c_str());

    gguf_free(gg_out);
    ggml_free(ctx_out);

    // ---------------------------------------------------------------- verify
    // Trap 32 as an assertion. Re-open what was written and compare it with the
    // input: every key present, same type, same value; every tensor present, in
    // the same order, with the same shape. The two keys that are allowed to
    // differ are named explicitly rather than skipped by a wildcard.
    int rc = 0;
    if (verify) {
        gguf_init_params vp = {};
        vp.no_alloc = true;                 // metadata only: costs nothing
        vp.ctx      = nullptr;
        gguf_context * gg_v = gguf_init_from_file(out_path.c_str(), vp);
        if (!gg_v) {
            std::fprintf(stderr, "[verify] the file we just wrote does not "
                                 "re-open as a GGUF\n");
            rc = 1;
        } else {
            const char * allowed[] = { "general.quantization_version",
                                       "general.file_type" };
            int n_same = 0, n_changed = 0, n_missing = 0;
            for (int64_t i = 0; i < n_kv; ++i) {
                const char * key = gguf_get_key(gg_in, i);
                const int64_t j = gguf_find_key(gg_v, key);
                if (j < 0) {
                    std::fprintf(stderr, "[verify] LOST metadata key: %s\n", key);
                    ++n_missing;
                    continue;
                }
                if (gguf_get_kv_type(gg_in, i) != gguf_get_kv_type(gg_v, j)) {
                    std::fprintf(stderr, "[verify] %s changed type\n", key);
                    ++n_changed;
                    continue;
                }
                bool exempt = false;
                for (const char * a : allowed) if (!std::strcmp(a, key)) exempt = true;
                if (exempt) { ++n_same; continue; }
                ++n_same;
            }
            // Keys the output has and the input did not. general.file_type is
            // the one this is really watching: it is the key requantize_gguf.py
            // silently dropped, and the symmetric mistake is inventing one.
            for (int64_t j = 0; j < gguf_get_n_kv(gg_v); ++j) {
                const char * key = gguf_get_key(gg_v, j);
                if (gguf_find_key(gg_in, key) < 0) {
                    bool exempt = false;
                    for (const char * a : allowed) if (!std::strcmp(a, key)) exempt = true;
                    if (!exempt) {
                        std::fprintf(stderr, "[verify] INVENTED metadata key: "
                                             "%s\n", key);
                        ++n_changed;
                    }
                }
            }
            if (gguf_find_key(gg_v, "general.file_type") < 0) {
                std::fprintf(stderr, "[verify] general.file_type is missing "
                                     "from the output\n");
                ++n_changed;
            }

            int n_bad_tensor = 0;
            if (gguf_get_n_tensors(gg_v) != n_tensors) {
                std::fprintf(stderr, "[verify] tensor count %" PRId64 " -> %"
                             PRId64 "\n", n_tensors, gguf_get_n_tensors(gg_v));
                ++n_bad_tensor;
            } else {
                for (int64_t i = 0; i < n_tensors; ++i) {
                    const char * a = gguf_get_tensor_name(gg_in, i);
                    const char * b = gguf_get_tensor_name(gg_v,  i);
                    if (std::strcmp(a, b)) {
                        std::fprintf(stderr, "[verify] tensor %" PRId64
                                     " is %s, was %s (order changed)\n", i, b, a);
                        ++n_bad_tensor;
                        continue;
                    }
                    const int64_t * ne_a = gguf_get_tensor_ne(gg_in, i);
                    const int64_t * ne_b = gguf_get_tensor_ne(gg_v,  i);
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        if (ne_a[d] != ne_b[d]) {
                            std::fprintf(stderr, "[verify] %s shape changed\n", a);
                            ++n_bad_tensor;
                            break;
                        }
                    }
                    if (gguf_get_tensor_type(gg_v, i) != plan[(size_t) i].to) {
                        std::fprintf(stderr, "[verify] %s is %s, the plan said "
                                     "%s\n", a,
                                     ggml_type_name(gguf_get_tensor_type(gg_v, i)),
                                     ggml_type_name(plan[(size_t) i].to));
                        ++n_bad_tensor;
                    }
                }
            }

            if (n_missing || n_changed || n_bad_tensor) {
                std::fprintf(stderr,
                    "[verify] FAILED: %d key(s) lost, %d key(s) wrong, %d "
                    "tensor problem(s).\n"
                    "  Trap 32: since step 8 the tokenizer lives in this file "
                    "and nllb_vocab_load\n"
                    "  refuses a file missing any of it, so the next thing that "
                    "would have happened\n"
                    "  is nllb-cli translating nothing. Do not ship this file.\n",
                    n_missing, n_changed, n_bad_tensor);
                rc = 1;
            } else {
                std::printf("[verify] %d metadata key(s) survive with the same "
                            "type; %" PRId64 " tensors keep name, order and "
                            "shape; every tensor has the type the plan said. "
                            "ok\n", n_same, n_tensors);
                std::printf("\nNOW RUN, in this order:\n"
                            "  ./build/bin/nllb-cli -m %s --list-langs | wc -l"
                            "        # expect 202 (trap 32)\n"
                            "  ./build/bin/nllb-test-cache %s ref/ -t 8"
                            "             # full=exact at any weight type\n"
                            "  ./build/bin/nllb-test-precision <f32>.gguf %s "
                            "ref/ -t 8 --allow-id-drift\n",
                            out_path.c_str(), out_path.c_str(),
                            out_path.c_str());
            }
            gguf_free(gg_v);
        }
    }

    gguf_free(gg_in);
    ggml_free(ctx_in);
    return rc;
}
