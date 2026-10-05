# nllb.cpp/examples/server

An HTTP server for nllb.cpp with **dynamic batching**: lines from concurrent
requests are queued, and the model thread translates up to `--batch` of them in
one `nllb_generate_batch` call - one encoder pass and one decoder step per token
for all of them. A step reads ~0.9 GB of weights whether it produces 1 token or 8,
so that is where the throughput is. Measured on an M1 Air, real 600M f16, decoder
step: CPU 17.1 -> 3.4 ms/token at 16 rows (5.1x), Metal 16.8 -> 1.9 ms/token (9.0x).
`--batch` defaults to **16**. Measure your machine with
`nllb-test-batch MODEL --bench 1,2,4,8,16 [--backend metal]`.
`--batch` is a MAXIMUM: a group is whatever is queued when the model thread is
free, so under light load most groups are 1-3 lines.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target nllb-server   # NLLB_BUILD_SERVER=ON by default
./build/bin/nllb-server -m models/nllb-600M-f16.gguf --host 0.0.0.0 --port 8080
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
  request -> split into lines  ─┼─> queue ─> take up to --batch lines (wait ≤ --batch-wait-ms)
  request -> split into lines  ─┘
        ▲                                    engine::translate(group)
        │                                      1 line : nllb_generate (nllb-cli's path)
        │                                      more   : planned by length, then
        │                                               nllb_generate_batch
        └──────── per-line promise ◄──────── decode, hand each line its result
```

* Only the worker thread calls the model, so there is no mutex around it
  (`nllb.h`: one `nllb_model` must not be used from two threads at once).
* Rows can mix language pairs: the source code is in each row's ids and the target
  code is a per-row forced token.
* A request's lines are queued all-or-nothing, so a 503 never leaves half a request running.
* When a request times out, its queued lines are skipped, and a line already
  generating stops at the next token (`nllb_generate_params::on_token`).
* `--batch 1` turns batching off and gives whisper-server's one-at-a-time behaviour.
* `-b` is rejected on purpose: in `nllb-cli` it means `--backend`.

Correctness: a batched line is byte-identical to the same line translated alone
on the CPU when the batch rows have equal length, and within float noise when
they are padded (`nllb-test-batch`; the product-level check is
`python tests/verify_translation.py ./build/bin/nllb-cli MODEL ref/ --batch 4`).
`-v` logs each line's ids with `(single)` or `(batch of N)`.

## Options

```
-m MODEL  -t THREADS  -n MAX_NEW_TOKENS  --backend auto|cpu|metal|gpu|<device>  --no-mmap
--host  --port  --http-threads  --request-timeout
--batch  --batch-tokens  --batch-wait-ms  --max-queue  --max-lines  --max-chars
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
