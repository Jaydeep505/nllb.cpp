// nllb-dump-states - everything the public API computes, written as raw bytes.
//
// WHY THIS EXISTS. Every other gate compares two things inside ONE build. This
// one is for comparing two BUILDS: run it from the old library and from the
// new one over the same GGUF and the same manifest, then `cmp` the outputs.
// Equal files mean the change moved no arithmetic anywhere the API can see -
// encoder states, teacher-forced decoder states and logits, both generation
// paths, and every cached step's states and full logits.
//
// Use it for anything that changes the execution model that is not meant to
// change arithmetic - e.g. a ggml submodule bump. It links only the public header, so the same source compiles
// against any version of libnllb that has these entry points.
//
//   nllb-dump-states <model.gguf> <ref_dir> <out.bin> [-t N] [--kv-f16] [--lite]
//                    [--no-mmap] [--backend B]
//
// --lite hashes every logits record instead of storing it. Use it on the real
// checkpoint: a full dump of the nine goldens is gigabytes of logits.
//
// Against an older library without backend support, compile with
// -DNLLB_PRE_BACKEND: its loader has no backend parameter.
//
// Output: a sequence of records, each  u32 tag_len | tag | u64 n_bytes | bytes.
// Tags name what the bytes are, so a diff tool can say WHERE two files part.
// `--diff a.bin b.bin` does exactly that and exits non-zero on any difference.
#include "nllb.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool read_i32(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    if (n <= 0 || n % 4) return false;
    f.seekg(0);
    out.resize((size_t) n / 4);
    return (bool) f.read((char *) out.data(), n);
}

struct writer {
    FILE * f = nullptr;
    size_t n_records = 0, n_bytes = 0;
    // --lite: logits are recorded as a 64-bit FNV-1a hash of their bytes
    // instead of the bytes. Still an exact test - one flipped bit changes the
    // hash - but a real-checkpoint dump drops from gigabytes (every row is
    // 256206 floats) to megabytes. Hidden states stay whole, so a difference
    // can still be located to a layer.
    bool lite = false;
    void put_logits(const std::string & tag, const std::vector<float> & v) {
        if (!lite) { put(tag, v); return; }
        uint64_t h = 1469598103934665603ull;
        const unsigned char * p = (const unsigned char *) v.data();
        for (size_t i = 0; i < v.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
        put(tag + "#fnv", &h, sizeof h);
    }
    void put(const std::string & tag, const void * data, uint64_t nb) {
        const uint32_t tl = (uint32_t) tag.size();
        std::fwrite(&tl, 4, 1, f);
        std::fwrite(tag.data(), 1, tl, f);
        std::fwrite(&nb, 8, 1, f);
        if (nb) std::fwrite(data, 1, nb, f);
        ++n_records; n_bytes += nb;
    }
    void put(const std::string & tag, const std::vector<float> & v) {
        put(tag, v.data(), v.size() * sizeof(float));
    }
    void put(const std::string & tag, const std::vector<int32_t> & v) {
        put(tag, v.data(), v.size() * sizeof(int32_t));
    }
};

struct record { std::string tag; std::vector<char> bytes; };

bool read_all(const char * path, std::vector<record> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    for (;;) {
        uint32_t tl = 0;
        if (std::fread(&tl, 4, 1, f) != 1) break;
        record r; r.tag.resize(tl);
        uint64_t nb = 0;
        if (std::fread(&r.tag[0], 1, tl, f) != tl || std::fread(&nb, 8, 1, f) != 1) {
            std::fclose(f); return false;
        }
        r.bytes.resize(nb);
        if (nb && std::fread(r.bytes.data(), 1, nb, f) != nb) { std::fclose(f); return false; }
        out.push_back(std::move(r));
    }
    std::fclose(f);
    return true;
}

// Compare two dumps record by record. Float records report how many floats
// differ and the worst relative difference, so "not identical" comes with a
// size rather than just a verdict.
int diff_main(const char * a, const char * b) {
    std::vector<record> ra, rb;
    if (!read_all(a, ra) || !read_all(b, rb)) {
        std::fprintf(stderr, "cannot read %s or %s\n", a, b);
        return 2;
    }
    if (ra.empty()) {
        std::fprintf(stderr, "%s is empty - a dump that recorded nothing proves nothing\n", a);
        return 2;
    }
    int n_diff = 0;
    size_t n = ra.size() < rb.size() ? ra.size() : rb.size();
    for (size_t i = 0; i < n; ++i) {
        const record & x = ra[i];
        const record & y = rb[i];
        if (x.tag != y.tag) {
            std::printf("record %zu: tag %s vs %s - the two runs did different things\n",
                        i, x.tag.c_str(), y.tag.c_str());
            return 1;
        }
        if (x.bytes == y.bytes) continue;
        ++n_diff;
        if (x.bytes.size() != y.bytes.size()) {
            std::printf("  %-40s size %zu vs %zu\n", x.tag.c_str(), x.bytes.size(), y.bytes.size());
            continue;
        }
        const bool opaque = x.tag.find("ids") != std::string::npos ||
                            x.tag.find('#') != std::string::npos;
        if (opaque) {
            std::printf("  %-40s differs\n", x.tag.c_str());
            continue;
        }
        const float * p = (const float *) x.bytes.data();
        const float * q = (const float *) y.bytes.data();
        const size_t nf = x.bytes.size() / 4;
        size_t nd = 0; double worst = 0.0, mag = 0.0;
        for (size_t k = 0; k < nf; ++k) {
            const double m = std::fabs((double) p[k]);
            if (m > mag) mag = m;
        }
        for (size_t k = 0; k < nf; ++k) {
            if (std::memcmp(&p[k], &q[k], 4) == 0) continue;
            ++nd;
            const double d = std::fabs((double) p[k] - (double) q[k]) / (mag > 0 ? mag : 1.0);
            if (d > worst) worst = d;
        }
        std::printf("  %-40s %zu of %zu floats differ, worst rel %.3e\n",
                    x.tag.c_str(), nd, nf, worst);
    }
    if (ra.size() != rb.size()) {
        std::printf("record count %zu vs %zu\n", ra.size(), rb.size());
        return 1;
    }
    size_t total = 0;
    for (const record & r : ra) total += r.bytes.size();
    if (n_diff == 0) {
        std::printf("IDENTICAL - %zu records, %zu bytes, 0 differ\n", ra.size(), total);
        return 0;
    }
    std::printf("DIFFERENT - %d of %zu records differ\n", n_diff, ra.size());
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--diff") == 0) return diff_main(argv[2], argv[3]);
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <ref_dir> <out.bin> [-t N] [--kv-f16] [--lite] [--no-mmap] [--backend B]\n"
            "       %s --diff a.bin b.bin\n", argv[0], argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';
    const char * out_path = argv[3];
    int n_threads = 4;
    int kv_type = NLLB_KV_F32;
    bool use_mmap = true;
    bool lite = false;
    std::string backend;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-t") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--kv-f16")) kv_type = NLLB_KV_F16;
        else if (!std::strcmp(argv[i], "--no-mmap")) use_mmap = false;
        else if (!std::strcmp(argv[i], "--lite"))    lite = true;
        else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    nllb_model model;
    std::string err;
#ifdef NLLB_PRE_BACKEND
    // Compiled against an older libnllb, whose loader has no backend
    // parameter: -DNLLB_PRE_BACKEND. --no-mmap is honoured, --backend is not.
    if (!backend.empty()) { std::fprintf(stderr, "--backend needs a build with backend support\n"); return 2; }
    if (!nllb_model_load(model_path, model, err, use_mmap)) {
#else
    if (!nllb_model_load(model_path, model, err, use_mmap, backend)) {
#endif
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    std::ifstream mf(ref_dir + "manifest_gen.txt");
    if (!mf) { std::fprintf(stderr, "no manifest_gen.txt in %s\n", ref_dir.c_str()); return 2; }

    writer w;
    w.lite = lite;
    w.f = std::fopen(out_path, "wb");
    if (!w.f) { std::fprintf(stderr, "cannot write %s\n", out_path); return 2; }

    std::string line;
    int n_cases = 0;
    while (std::getline(mf, line)) {
        std::istringstream ss(line);
        std::string name;
        int Tg = 0, bos = -1, max_new = 0, start = -1;
        if (!(ss >> name >> Tg >> bos >> max_new >> start)) continue;
        std::vector<int32_t> src;
        if (!read_i32(ref_dir + name + ".ids.bin", src)) {
            std::fprintf(stderr, "%s: missing .ids.bin - refusing to dump a partial file\n",
                         name.c_str());
            std::fclose(w.f);
            return 1;
        }
        ++n_cases;

        // encoder, with every hidden state
        nllb_encode_result enc;
        if (!nllb_encode(model, src, n_threads, true, enc, err)) goto fail;
        w.put(name + "/enc/out", enc.output);
        for (size_t i = 0; i < enc.hidden_states.size(); ++i)
            w.put(name + "/enc/hs" + std::to_string(i), enc.hidden_states[i]);

        {
            // both generation paths, with the per-step trace
            nllb_generate_params gp;
            gp.max_new_tokens = max_new;
            gp.forced_bos_id  = bos;
            gp.dec_start_id   = start;
            gp.kv_type        = kv_type;
            nllb_generate_result g[2];
            for (int c = 0; c < 2; ++c) {
                gp.use_cache = c == 1;
                if (!nllb_generate(model, src, gp, n_threads, g[c], err)) goto fail;
                const std::string p = name + (c ? "/gen_cached" : "/gen_uncached");
                w.put(p + "/ids", g[c].ids);
                std::vector<float> tr;
                for (const auto & s : g[c].steps) { tr.push_back(s.logit); tr.push_back(s.margin); }
                w.put(p + "/logit_margin", tr);
            }

            // teacher-forced decoder over the generated prefix: all states and
            // ALL logits rows, both position rules
            const std::vector<int32_t> & tgt = g[0].ids;
            for (int pm = 0; pm < 2; ++pm) {
                nllb_decode_result dec;
                if (!nllb_decode(model, enc, tgt, src, n_threads, true, NLLB_LOGITS_ALL,
                                 dec, err, pm)) goto fail;
                const std::string p = name + (pm ? "/dec_abs" : "/dec_cumsum");
                w.put(p + "/out", dec.output);
                w.put_logits(p + "/logits", dec.logits);
                for (size_t i = 0; i < dec.hidden_states.size(); ++i)
                    w.put(p + "/hs" + std::to_string(i), dec.hidden_states[i]);
            }

            // cached: one token at a time, then the whole prefix in chunks of 3,
            // every step's states and full logits
            for (int chunk : {1, 3}) {
                nllb_kv_cache cache;
                if (!nllb_kv_cache_init(model, enc, (int) tgt.size(), n_threads, cache, err,
                                        kv_type)) goto fail;
                for (size_t pos = 0; pos < tgt.size(); pos += chunk) {
                    const size_t end = pos + chunk < tgt.size() ? pos + chunk : tgt.size();
                    std::vector<int32_t> feed(tgt.begin() + pos, tgt.begin() + end);
                    nllb_decode_result dec;
                    if (!nllb_decode_cached(model, cache, feed, n_threads, true,
                                            NLLB_LOGITS_ALL, dec, err)) goto fail;
                    const std::string p = name + "/cached_c" + std::to_string(chunk) +
                                          "_at" + std::to_string(pos);
                    w.put_logits(p + "/logits", dec.logits);
                    for (size_t i = 0; i < dec.hidden_states.size(); ++i)
                        w.put(p + "/hs" + std::to_string(i), dec.hidden_states[i]);
                }
            }
        }
        std::fprintf(stderr, "  %-20s ok\n", name.c_str());
        continue;
    fail:
        std::fprintf(stderr, "%s: %s\n", name.c_str(), err.c_str());
        std::fclose(w.f);
        return 1;
    }
    std::fclose(w.f);
    if (n_cases == 0) { std::fprintf(stderr, "no cases in the manifest\n"); return 1; }
    std::fprintf(stderr, "wrote %s: %d cases, %zu records, %zu bytes\n",
                 out_path, n_cases, w.n_records, w.n_bytes);
    return 0;
}
