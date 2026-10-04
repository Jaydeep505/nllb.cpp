// nllb-server: an HTTP front end for nllb.cpp, built for batching.
//
//   POST /translate   {"text": "Hello!" | ["a", "b"], "src": "eng_Latn", "tgt": "deu_Latn"}
//   GET  /languages   every FLORES-200 code in the GGUF
//   GET  /health      readiness + queue/batch statistics
//   GET  /            a small test page
//
// How it differs from whisper.cpp's server
// ----------------------------------------
// whisper-server holds one mutex around one context, so concurrent requests
// run one at a time inside the HTTP threads. Here the HTTP threads never touch
// the model. They split each request into LINES, push the lines onto one queue
// and wait. One worker thread owns the model (nllb.h: one nllb_model must not be
// used from two threads at once): it takes up to -b queued lines, waiting at
// most --batch-wait-ms for more, translates them as one group, and hands each
// line its result.
//
// Today a group is translated row by row with nllb_generate, because the
// library has no batched generate in this tree. The queue, the grouping and the
// HTTP side are already shaped for it: engine::translate() receives the whole
// group, and that function is the only thing that changes when
// nllb_generate_batch lands. Rows may mix language pairs: the source code is
// inside each row's ids and the target code is a per-row forced token.
//
// Every call into nllb.cpp is in the "engine" section and nowhere else.
// Build with -DNLLB_SERVER_MOCK to replace the model with a stand-in that needs
// no weights; that is how the HTTP and queueing logic is tested.

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using json  = nlohmann::ordered_json;
using clk   = std::chrono::steady_clock;
using ms_d  = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

struct server_params {
    std::string hostname        = "127.0.0.1";
    int32_t     port            = 8080;
    int32_t     http_threads    = 16;    // concurrent connections being served
    int32_t     read_timeout    = 60;
    int32_t     write_timeout   = 60;
    int32_t     request_timeout = 120;   // seconds a request may wait for its result

    int32_t     batch_size      = 8;     // max lines per group handed to the engine
    int32_t     batch_wait_ms   = 5;     // how long the worker waits to fill a batch
    int32_t     max_queue       = 1024;  // lines waiting; beyond this -> 503
    int32_t     max_lines       = 64;    // lines per request
    int32_t     max_chars       = 2000;  // characters per line
};

struct model_params {
    std::string model;                   // path to the .gguf
    std::string backend        = "auto"; // as nllb_model_load: auto|cpu|metal|gpu|<device name>; "" = $NLLB_BACKEND
    int32_t     n_threads      = std::min(8, (int32_t) std::max(1u, std::thread::hardware_concurrency()));
    int32_t     max_new_tokens = 256;
    bool        use_mmap       = true;
    bool        verbose        = false;  // -v: log each line's src/out ids to stderr, like nllb-cli -v
    bool        cancel_hook    = true;   // on_token hook that stops abandoned lines; --no-cancel-hook
};

static void print_usage(const char * argv0, const server_params & sp, const model_params & mp) {
    fprintf(stderr, "\nusage: %s -m MODEL.gguf [options]\n\n", argv0);
    fprintf(stderr, "model:\n");
    fprintf(stderr, "  -m,  --model PATH          GGUF file (weights + tokenizer)\n");
    fprintf(stderr, "  -t,  --threads N           CPU threads             [%d]\n", mp.n_threads);
    fprintf(stderr, "  -n,  --max-new-tokens N    per line cap            [%d]\n", mp.max_new_tokens);
    fprintf(stderr, "       --backend NAME        auto|cpu|metal|gpu|DEV  [%s]\n", mp.backend.c_str());
    fprintf(stderr, "       --no-mmap             read weights instead of mapping them\n");
    fprintf(stderr, "  -v,  --verbose             log src/out ids per line (stderr)\n");
    fprintf(stderr, "       --no-cancel-hook      do not set on_token (debugging)\n");
    fprintf(stderr, "server:\n");
    fprintf(stderr, "       --host HOST           bind address            [%s]\n", sp.hostname.c_str());
    fprintf(stderr, "       --port N              port                    [%d]\n", sp.port);
    fprintf(stderr, "       --http-threads N      connection threads      [%d]\n", sp.http_threads);
    fprintf(stderr, "       --request-timeout S   max wait per request    [%d]\n", sp.request_timeout);
    fprintf(stderr, "batching:\n");
    fprintf(stderr, "  -b,  --batch-size N        lines per group (1 = no grouping) [%d]\n", sp.batch_size);
    fprintf(stderr, "       --batch-wait-ms N     wait to fill a batch             [%d]\n", sp.batch_wait_ms);
    fprintf(stderr, "       --max-queue N         queued lines before 503          [%d]\n", sp.max_queue);
    fprintf(stderr, "       --max-lines N         lines per request                [%d]\n", sp.max_lines);
    fprintf(stderr, "       --max-chars N         characters per line              [%d]\n\n", sp.max_chars);
}

static bool parse_args(int argc, char ** argv, server_params & sp, model_params & mp) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-h"  || a == "--help")           { print_usage(argv[0], sp, mp); exit(0); }
        else if (a == "-m"  || a == "--model")          { mp.model          = next(); }
        else if (a == "-t"  || a == "--threads")        { mp.n_threads      = std::stoi(next()); }
        else if (a == "-n"  || a == "--max-new-tokens") { mp.max_new_tokens = std::stoi(next()); }
        else if (              a == "--backend")        { mp.backend        = next(); }
        else if (              a == "--no-mmap")        { mp.use_mmap       = false; }
        else if (a == "-v"  || a == "--verbose")        { mp.verbose        = true; }
        else if (              a == "--no-cancel-hook") { mp.cancel_hook    = false; }
        else if (              a == "--host")           { sp.hostname       = next(); }
        else if (              a == "--port")           { sp.port           = std::stoi(next()); }
        else if (              a == "--http-threads")   { sp.http_threads   = std::stoi(next()); }
        else if (              a == "--request-timeout"){ sp.request_timeout= std::stoi(next()); }
        else if (a == "-b"  || a == "--batch-size")     { sp.batch_size     = std::stoi(next()); }
        else if (              a == "--batch-wait-ms")  { sp.batch_wait_ms  = std::stoi(next()); }
        else if (              a == "--max-queue")      { sp.max_queue      = std::stoi(next()); }
        else if (              a == "--max-lines")      { sp.max_lines      = std::stoi(next()); }
        else if (              a == "--max-chars")      { sp.max_chars      = std::stoi(next()); }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); return false; }
    }
    if (mp.model.empty()) { fprintf(stderr, "error: -m MODEL.gguf is required\n"); return false; }
    if (sp.batch_size < 1 || sp.max_queue < 1 || sp.max_lines < 1 || mp.n_threads < 1) {
        fprintf(stderr, "error: sizes and thread counts must be >= 1\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// engine: the ONLY code that calls nllb.cpp
// ---------------------------------------------------------------------------

struct row_in {
    std::string text;
    std::string src;   // FLORES code, already validated against the vocabulary
    std::string tgt;
    // Set by the HTTP thread when its client has given up (timeout). The engine
    // polls it once per token through nllb_generate_params::on_token, so an
    // abandoned line stops generating instead of running to eos for nobody.
    const std::atomic<bool> * abandoned = nullptr;
};

struct row_out {
    bool        ok = false;
    std::string text;
    std::string error;
    int32_t     n_tokens = 0;   // generated tokens, excluding the decoder start token
    std::string stop;           // "eos" | "max_new" | "ctx" | "aborted"
};

#ifdef NLLB_SERVER_MOCK
// ---- stand-in: no weights. Cost model = fixed per-call overhead + small per-row
// cost, the shape a batched step has (weights read once per step, not per row).
struct engine {
    std::vector<std::string> codes = {"eng_Latn", "deu_Latn", "fra_Latn", "hin_Deva", "guj_Gujr", "jpn_Jpan"};
    bool load(const model_params &) { return true; }
    std::vector<std::string> languages() const { return codes; }
    std::string backend_name() const { return "MOCK"; }
    bool batched() const { return true; }
    void translate(const std::vector<row_in> & in, std::vector<row_out> & out, const model_params &) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40 + 2 * (int) in.size()));
        out.assign(in.size(), {});
        for (size_t i = 0; i < in.size(); i++) {
            out[i].ok       = true;
            out[i].text     = "[" + in[i].src + "->" + in[i].tgt + "] " + in[i].text;
            out[i].n_tokens = (int32_t) in[i].text.size() / 4 + 1;
            out[i].stop     = "eos";
        }
    }
};
#else
// ---- the real thing: the same calls as nllb-cli's translate_one().
#include "nllb.h"
#include "nllb-vocab.h"

struct engine {
    // nllb.h: the scheduler inside nllb_model makes one model unsafe to use from
    // two threads at once. Only the batcher's worker thread calls translate().
    nllb_model model;
    nllb_vocab vocab;

    bool load(const model_params & mp) {
        std::string err;
        if (!nllb_model_load(mp.model, model, err, mp.use_mmap, mp.backend)) {
            fprintf(stderr, "error: model load failed: %s\n", err.c_str());
            return false;
        }
        if (!model.has_decoder) {
            fprintf(stderr, "error: this GGUF has no decoder (converted with --encoder-only?)\n");
            return false;
        }
        // Reuse the gguf_context the model already holds, as nllb-cli does.
        if (!nllb_vocab_load_from_gguf(model.gguf, vocab, err)) {
            fprintf(stderr, "error: tokenizer load failed: %s\n"
                            "  a GGUF converted before step 8 has no tokenizer; reconvert it.\n", err.c_str());
            return false;
        }
        return true;
    }

    std::vector<std::string> languages() const { return nllb_vocab_langs(vocab); }

    std::string backend_name() const {
        return model.backend_name + (model.weights_host_ptr ? " (zero-copy)" : "");
    }

    // false until nllb_generate_batch exists; main() prints it so nobody
    // benchmarks -b 8 and wonders why it equals -b 1.
    bool batched() const { return false; }

    static const char * stop_str(int32_t s) {
        switch (s) {
            case NLLB_STOP_EOS:     return "eos";
            case NLLB_STOP_MAX_NEW: return "max_new";
            case NLLB_STOP_CONTEXT: return "ctx";
            case NLLB_STOP_ABORTED: return "aborted";
        }
        return "?";
    }

    static bool keep_going(int32_t /*token*/, void * user) {
        const auto * abandoned = static_cast<const std::atomic<bool> *>(user);
        return !(abandoned && abandoned->load(std::memory_order_relaxed));
    }

    // One group from the queue. Today each row runs through nllb_generate on its
    // own; the group is still useful (one wake-up, one stats entry, and the place
    // the batch call goes). THIS LOOP IS THE BATCHING HOOK: when the step-11
    // branch lands, tokenize every row first, then hand all of them to
    // nllb_generate_batch with a per-row forced_bos vector - rows may mix
    // src/tgt pairs, because src is inside each row's ids and tgt is per row.
    void translate(const std::vector<row_in> & in, std::vector<row_out> & out, const model_params & mp) {
        out.assign(in.size(), {});
        for (size_t i = 0; i < in.size(); i++) {
            const row_in & r = in[i];
            row_out      & o = out[i];
            std::string err;

            std::vector<int32_t> src_ids;
            if (!nllb_vocab_encode(vocab, r.text, r.src, src_ids, err)) {
                o.error = "tokenize: " + err;
                continue;
            }
            // Validated by the HTTP thread already; re-checked because -1 here
            // means "force nothing" and would translate into an arbitrary language.
            const int32_t tgt_id = nllb_vocab_lang_id(vocab, r.tgt);
            if (tgt_id < 0) { o.error = "unknown target language " + r.tgt; continue; }

            nllb_generate_params gp;
            gp.max_new_tokens = mp.max_new_tokens;
            gp.forced_bos_id  = tgt_id;
            gp.use_cache      = true;          // the library default is off; the product's is on
            if (mp.cancel_hook) {
                gp.on_token      = &engine::keep_going;
                gp.on_token_user = const_cast<std::atomic<bool> *>(r.abandoned);
            }

            nllb_generate_result gen;
            if (!nllb_generate(model, src_ids, gp, mp.n_threads, gen, err)) {
                o.error = "generate: " + err;
                continue;
            }
            // skip_special drops decoder start, forced target code and eos.
            o.ok       = true;
            o.text     = nllb_vocab_decode(vocab, gen.ids, /*skip_special=*/true);
            o.n_tokens = gen.ids.empty() ? 0 : (int32_t) gen.ids.size() - 1;
            o.stop     = stop_str(gen.stop);

            if (mp.verbose) {
                // Same "src ids (N): ..." / "out ids (N): ..." format as nllb-cli -v,
                // so the two logs can be compared with grep | sort | uniq -c.
                static uint64_t seq = 0;
                std::string line = "[" + std::to_string(seq++) + "] src ids (" + std::to_string(src_ids.size()) + "):";
                for (int32_t id : src_ids) line += " " + std::to_string(id);
                line += "\n[" + std::to_string(seq - 1) + "] out ids (" + std::to_string(gen.ids.size()) + "):";
                for (int32_t id : gen.ids) line += " " + std::to_string(id);
                line += std::string("\n[") + std::to_string(seq - 1) + "] stop: " + o.stop + "\n";
                fputs(line.c_str(), stderr);
            }
        }
    }
};
#endif

// ---------------------------------------------------------------------------
// the batcher: many HTTP threads in, one model thread out
// ---------------------------------------------------------------------------

struct work_item {
    row_in                 in;
    std::promise<row_out>  done;
    std::atomic<bool>      abandoned{false};   // the client timed out; skip it
    clk::time_point        queued_at = clk::now();
};
using work_ptr = std::shared_ptr<work_item>;

struct batcher {
    std::mutex              m;
    std::condition_variable cv;
    std::deque<work_ptr>    q;
    bool                    stopping = false;

    // statistics, read by /health
    std::atomic<uint64_t> n_batches{0}, n_rows{0}, n_tokens{0};
    std::atomic<double>   busy_ms{0.0};

    // All-or-nothing: a request's lines are queued together or not at all,
    // so a 503 never leaves half a request running.
    bool submit(const std::vector<work_ptr> & items, size_t max_queue) {
        {
            std::lock_guard<std::mutex> lk(m);
            if (stopping || q.size() + items.size() > max_queue) return false;
            for (auto & it : items) q.push_back(it);
        }
        cv.notify_one();
        return true;
    }

    size_t depth() { std::lock_guard<std::mutex> lk(m); return q.size(); }

    void stop() {
        { std::lock_guard<std::mutex> lk(m); stopping = true; }
        cv.notify_all();
    }

    void run(engine & eng, const model_params & mp, const server_params & sp) {
        const size_t B = (size_t) sp.batch_size;
        for (;;) {
            std::vector<work_ptr> batch;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stopping || !q.empty(); });
                if (stopping && q.empty()) break;

                // Give other requests a moment to arrive. Under load the queue is
                // already full and this returns at once; when idle a lone request
                // pays at most batch_wait_ms.
                if (B > 1 && q.size() < B && sp.batch_wait_ms > 0) {
                    cv.wait_for(lk, std::chrono::milliseconds(sp.batch_wait_ms),
                                [&] { return stopping || q.size() >= B; });
                }
                while (!q.empty() && batch.size() < B) {
                    work_ptr w = q.front(); q.pop_front();
                    if (w->abandoned.load()) continue;     // nobody is waiting for it
                    batch.push_back(std::move(w));
                }
            }
            if (batch.empty()) continue;

            std::vector<row_in> in;
            in.reserve(batch.size());
            for (auto & w : batch) in.push_back(w->in);

            std::vector<row_out> out;
            const auto t0 = clk::now();
            try {
                eng.translate(in, out, mp);
            } catch (const std::exception & e) {
                out.assign(in.size(), {});
                for (auto & o : out) o.error = std::string("internal error: ") + e.what();
            }
            const double dt = ms_d(clk::now() - t0).count();

            n_batches++;
            n_rows += batch.size();
            busy_ms = busy_ms.load() + dt;
            for (size_t i = 0; i < batch.size(); i++) {
                if (i < out.size()) { n_tokens += out[i].n_tokens; batch[i]->done.set_value(std::move(out[i])); }
                else                { row_out e; e.error = "missing result"; batch[i]->done.set_value(e); }
            }
        }
        // Anything still queued at shutdown is answered, not dropped.
        std::lock_guard<std::mutex> lk(m);
        for (auto & w : q) { row_out e; e.error = "server shutting down"; w->done.set_value(e); }
        q.clear();
    }
};

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

static void send_json(httplib::Response & res, int status, const json & j) {
    res.status = status;
    res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
}

static void send_error(httplib::Response & res, int status, const std::string & msg) {
    send_json(res, status, json{{"error", msg}});
}

// "a\nb\n\nc" -> ["a","b","","c"]. The model is sentence-level, and the CLI treats
// one line as one call; the server does the same and rejoins with '\n'.
static std::vector<std::string> split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        size_t nl = s.find('\n', start);
        std::string line = s.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(std::move(line));
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    return out;
}

static bool is_blank(const std::string & s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return c == ' ' || c == '\t'; });
}

static size_t utf8_len(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

static const char * k_index_html = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nllb-server</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}
textarea{width:100%;box-sizing:border-box;min-height:7rem;font:inherit;padding:.5rem}
input{font:inherit;width:9rem;padding:.25rem}button{font:inherit;padding:.35rem 1rem}
pre{background:#f4f4f4;padding:.75rem;white-space:pre-wrap;min-height:3rem}
code{background:#f4f4f4;padding:0 .25rem}
</style></head><body>
<h2>nllb-server</h2>
<p><input id="src" value="eng_Latn"> &rarr; <input id="tgt" value="deu_Latn">
<button onclick="go()">Translate</button> <span id="t"></span></p>
<textarea id="text">Hello, how are you today?
The weather is nice.</textarea>
<pre id="out"></pre>
<p>One sentence per line. API: <code>POST /translate</code>, <code>GET /languages</code>, <code>GET /health</code>.</p>
<script>
async function go(){
  const t0=performance.now(), out=document.getElementById('out');
  out.textContent='...';
  const r=await fetch('translate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:document.getElementById('text').value,
      src:document.getElementById('src').value,tgt:document.getElementById('tgt').value})});
  const j=await r.json();
  out.textContent=j.translation??j.error;
  document.getElementById('t').textContent=(performance.now()-t0).toFixed(0)+' ms';
}
</script></body></html>)HTML";

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static std::function<void()> g_shutdown;
static void on_signal(int) { if (g_shutdown) g_shutdown(); }

int main(int argc, char ** argv) {
    server_params sp;
    model_params  mp;
    if (!parse_args(argc, argv, sp, mp)) { print_usage(argv[0], sp, mp); return 1; }

    engine eng;
    const auto t_load = clk::now();
    if (!eng.load(mp)) return 1;
    fprintf(stderr, "model:   %s (backend %s) loaded in %.0f ms\n",
            mp.model.c_str(), eng.backend_name().c_str(), ms_d(clk::now() - t_load).count());

    // The language list is fixed for the life of the model. Copying it here means
    // the HTTP threads validate requests without ever calling into the library.
    const std::vector<std::string>        lang_list = eng.languages();
    const std::unordered_set<std::string> langs(lang_list.begin(), lang_list.end());

    batcher bat;
    std::thread worker([&] { bat.run(eng, mp, sp); });

    httplib::Server svr;
    svr.new_task_queue = [&] { return new httplib::ThreadPool((size_t) sp.http_threads); };
    svr.set_read_timeout(sp.read_timeout);
    svr.set_write_timeout(sp.write_timeout);
    svr.set_payload_max_length(1 << 20);   // 1 MiB of JSON is far beyond max_lines * max_chars
    svr.set_default_headers({
        {"Server",                       "nllb.cpp"},
        {"Access-Control-Allow-Origin",  "*"},
        {"Access-Control-Allow-Headers", "content-type"},
    });

    svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(k_index_html, "text/html; charset=utf-8");
    });

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        const uint64_t nb = bat.n_batches, nr = bat.n_rows;
        send_json(res, 200, json{
            {"status",          "ok"},
            {"backend",         eng.backend_name()},
            {"queue",           bat.depth()},
            {"batches",         nb},
            {"lines",           nr},
            {"tokens",          (uint64_t) bat.n_tokens},
            {"avg_batch_size",  nb ? double(nr) / double(nb) : 0.0},
            {"busy_ms",         bat.busy_ms.load()},
        });
    });

    svr.Get("/languages", [&](const httplib::Request &, httplib::Response & res) {
        send_json(res, 200, json{{"languages", lang_list}});
    });

    svr.Options("/translate", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    svr.Post("/translate", [&](const httplib::Request & req, httplib::Response & res) {
        const auto t0 = clk::now();

        // ---- parse and validate. Every bad input is a 400 that says what was wrong;
        //      nothing is ever translated "as best we can".
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception &) { return send_error(res, 400, "body is not valid JSON"); }
        if (!body.is_object()) return send_error(res, 400, "body must be a JSON object");

        auto get_code = [&](const char * key, std::string & dst) -> bool {
            if (!body.contains(key) || !body[key].is_string()) {
                send_error(res, 400, std::string("'") + key + "' is required, e.g. \"eng_Latn\"");
                return false;
            }
            dst = body[key].get<std::string>();
            if (!langs.count(dst)) {
                send_error(res, 400, "unknown language '" + dst + "' in '" + key + "' (see GET /languages)");
                return false;
            }
            return true;
        };
        std::string src, tgt;
        if (!get_code("src", src) || !get_code("tgt", tgt)) return;

        // "text" is a string (lines split on '\n', rejoined in the reply) or an array of strings.
        if (!body.contains("text")) return send_error(res, 400, "'text' is required");
        const json & jt = body["text"];
        const bool  as_array = jt.is_array();
        std::vector<std::string> lines;
        if (jt.is_string()) {
            lines = split_lines(jt.get<std::string>());
        } else if (as_array) {
            for (const auto & e : jt) {
                if (!e.is_string()) return send_error(res, 400, "'text' array must contain only strings");
                const std::string & s = e.get_ref<const std::string &>();
                if (s.find('\n') != std::string::npos) {
                    return send_error(res, 400, "array items must not contain newlines; send one sentence per item");
                }
                lines.push_back(s);
            }
        } else {
            return send_error(res, 400, "'text' must be a string or an array of strings");
        }
        if ((int) lines.size() > sp.max_lines) {
            return send_error(res, 413, "too many lines: " + std::to_string(lines.size()) +
                                        " (max " + std::to_string(sp.max_lines) + ")");
        }
        for (const auto & l : lines) {
            if ((int) utf8_len(l) > sp.max_chars) {
                return send_error(res, 413, "line longer than " + std::to_string(sp.max_chars) + " characters");
            }
        }

        // ---- queue the non-blank lines; blank lines stay blank without a model call.
        std::vector<work_ptr> items(lines.size());
        std::vector<work_ptr> to_submit;
        for (size_t i = 0; i < lines.size(); i++) {
            if (is_blank(lines[i])) continue;
            items[i] = std::make_shared<work_item>();
            items[i]->in = row_in{lines[i], src, tgt, &items[i]->abandoned};
            to_submit.push_back(items[i]);
        }
        if (!to_submit.empty() && !bat.submit(to_submit, (size_t) sp.max_queue)) {
            res.set_header("Retry-After", "1");
            return send_error(res, 503, "server busy: queue full");
        }

        // ---- wait for every line, against one deadline for the whole request.
        const auto deadline = t0 + std::chrono::seconds(sp.request_timeout);
        std::vector<std::string> outs(lines.size());
        int32_t n_tokens = 0;
        for (size_t i = 0; i < lines.size(); i++) {
            if (!items[i]) continue;
            auto fut = items[i]->done.get_future();
            if (fut.wait_until(deadline) != std::future_status::ready) {
                for (auto & w : to_submit) w->abandoned = true;   // let the worker skip what is left
                return send_error(res, 504, "translation timed out");
            }
            row_out r = fut.get();
            if (!r.ok) return send_error(res, 500, "line " + std::to_string(i) + ": " + r.error);
            outs[i]   = std::move(r.text);
            n_tokens += r.n_tokens;
        }

        json reply;
        if (as_array) {
            reply["translation"] = outs;
        } else {
            std::string joined;
            for (size_t i = 0; i < outs.size(); i++) { if (i) joined += '\n'; joined += outs[i]; }
            reply["translation"] = joined;
        }
        reply["src"]      = src;
        reply["tgt"]      = tgt;
        reply["tokens"]   = n_tokens;
        reply["time_ms"]  = ms_d(clk::now() - t0).count();
        send_json(res, 200, reply);
    });

    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string msg = "internal error";
        try { std::rethrow_exception(ep); } catch (const std::exception & e) { msg += std::string(": ") + e.what(); } catch (...) {}
        send_error(res, 500, msg);
    });

    svr.set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.body.empty()) send_error(res, res.status, "not found");
    });

    if (!svr.bind_to_port(sp.hostname, sp.port)) {
        fprintf(stderr, "error: could not bind %s:%d\n", sp.hostname.c_str(), sp.port);
        bat.stop(); worker.join();
        return 1;
    }

    g_shutdown = [&] { svr.stop(); };
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    fprintf(stderr, "batch:   up to %d lines per group, %d ms wait%s\n", sp.batch_size, sp.batch_wait_ms,
            eng.batched() ? "" : " (rows run one at a time: no batched generate in this build)");
    fprintf(stderr, "listen:  http://%s:%d\n", sp.hostname.c_str(), sp.port);

    svr.listen_after_bind();   // returns after svr.stop()

    fprintf(stderr, "shutting down\n");
    bat.stop();
    worker.join();
    return 0;
}
