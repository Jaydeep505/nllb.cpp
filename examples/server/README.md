# nllb.cpp/examples/server

An HTTP server for nllb.cpp, shaped for batching: lines from concurrent requests
are queued and handed to one model thread in groups.

**Current state:** this tree has no batched generate, so a group runs row by row
through `nllb_generate`, and `-b` does not change throughput yet. When
`nllb_generate_batch` lands, only `engine::translate()` in `server.cpp` changes.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target nllb-server   # NLLB_BUILD_SERVER=ON by default
./build/bin/nllb-server -m models/nllb-600M-f16.gguf --host 0.0.0.0 --port 8080 -b 8
```

Open `http://localhost:8080/` for a test page.

## API

### `POST /translate`

```bash
curl localhost:8080/translate -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you?\nThe weather is nice.", "src": "eng_Latn", "tgt": "deu_Latn"}'
```

```json
{"translation": "Hallo, wie geht es dir?\nDas Wetter ist schön.",
 "src": "eng_Latn", "tgt": "deu_Latn", "tokens": 17, "time_ms": 212.4}
```

* `text`: a string or an array of strings. A string is split on `\n`, each line
  is translated on its own, and the result is rejoined with `\n`. Blank lines stay
  blank and cost nothing. An array returns an array in the same order.
* `src`, `tgt`: FLORES-200 codes (`GET /languages`). An unknown code is a 400 that
  names the field. It is never ignored or guessed.

NLLB is a sentence-level model. A paragraph on one line often stops after the
first sentence (same as HF `generate()`), so send one sentence per line.

| status | when |
|---|---|
| 400 | bad JSON, missing field, unknown language code, newline inside an array item |
| 413 | more than `--max-lines` lines or a line over `--max-chars` |
| 503 | queue full (`Retry-After: 1`) |
| 504 | not finished within `--request-timeout` |

### `GET /languages`, `GET /health`

`/health` also reports `queue`, `batches`, `lines` and `avg_batch_size`. If
`avg_batch_size` stays near 1 under load, raise `--batch-wait-ms`.

## How batching works

```
HTTP threads (--http-threads)          one model thread
  request -> split into lines  ─┐
  request -> split into lines  ─┼─> queue ─> take up to -b lines (wait ≤ --batch-wait-ms)
  request -> split into lines  ─┘
        ▲                                    engine::translate(group)
        │                                      today: nllb_generate per row
        │                                      later: nllb_generate_batch
        └──────── per-line promise ◄──────── decode, hand each line its result
```

* Only the worker thread calls the model, so there is no mutex around it
  (`nllb.h`: one `nllb_model` must not be used from two threads at once).
* Rows can mix language pairs: the source code is in each row's ids and the target
  code is a per-row forced token.
* A request's lines are queued all-or-nothing, so a 503 never leaves half a request running.
* When a request times out, its queued lines are skipped, and a line already
  generating stops at the next token (`nllb_generate_params::on_token`).
* `-b 1` turns batching off and gives whisper-server's one-at-a-time behaviour.

Mock run (no weights, `-DNLLB_SERVER_MOCK`, a fake batched engine with 40 ms fixed
cost per call, 64 concurrent one-line requests): `-b 1` 3149 ms, `-b 8` 922 ms.
That run checks the queue and the routing only. It is the shape batching will
give, not a measurement of nllb.cpp.

## Options

```
-m MODEL  -t THREADS  -n MAX_NEW_TOKENS  --backend auto|cpu|metal|gpu|<device>  --no-mmap
--host  --port  --http-threads  --request-timeout
-b/--batch-size  --batch-wait-ms  --max-queue  --max-lines  --max-chars
```

## Testing without weights

```bash
g++ -std=c++17 -O2 -DNLLB_SERVER_MOCK server.cpp -o nllb-server-mock -lpthread
./nllb-server-mock -m x --port 8080
```

## Model licence

The NLLB-200 weights are CC-BY-NC 4.0. A public server running them is
non-commercial use only.

## Docker (CPU)

```bash
docker build -t nllb-server .          # Dockerfile at the repo root
docker run -p 8080:8080 -v $PWD/models:/models nllb-server
```
