# nllb-server tests

pytest suite for `examples/server`. Three files:

| file | runs against | checks |
|---|---|---|
| `test_api.py` | mock **and** real | endpoints, JSON shapes, every 400/413 message, blank/CRLF lines, CORS, SIGTERM, port already in use |
| `test_mock.py` | `nllb-server-mock` | order of replies, batching under concurrency, mixed language pairs, 503 all-or-nothing, 504 timeout |
| `test_real.py` | `nllb-server` + a real GGUF | output is byte-identical to `nllb-cli`, single line, batched and concurrent |

```bash
pip install -r tests/server/requirements.txt

# mock: no model needed (what server.yml runs)
cmake -B build -DNLLB_BUILD_SERVER_MOCK=ON && cmake --build build -j --target nllb-server-mock
pytest tests/server

# real model (what real-model.yml and ci/run.sh run)
cmake --build build -j --target nllb-server nllb-cli
NLLB_SERVER_BIN=build/bin/nllb-server NLLB_SERVER_MODEL=models/nllb-600M-f16.gguf \
    pytest tests/server -m "not mock"
```
