---
license: cc-by-nc-4.0
base_model: facebook/nllb-200-distilled-600M
pipeline_tag: translation
library_name: ggml
tags:
  - gguf
  - nllb
  - nllb.cpp
  - translation
  - ggml
---

# NLLB-200 distilled 600M — GGUF for nllb.cpp

`facebook/nllb-200-distilled-600M` converted for
[nllb.cpp](https://github.com/Jaydeep505/nllb.cpp): a C++ / ggml inference
engine in the spirit of whisper.cpp. One file holds the weights **and** the
tokenizer, so nothing else is needed at runtime — no Python, no
`tokenizer.json`, no HF snapshot.

| file | size | use it for | sha256 |
|---|---|---|---|
| `nllb-600M-f16.gguf` | @F16_SIZE@ | **the default.** Greedy output identical to HF `generate()` on every golden | `@F16_SHA@` |
| `nllb-600M-q8emb.gguf` | @Q8E_SIZE@ | smaller and ~1.33x faster on CPU; `token_embd` at Q8_0, rest f16. corpus chrF 96.57 vs HF, 8/9 goldens byte-identical | `@Q8E_SHA@` |
| `nllb-600M-f32.gguf` | @F32_SIZE@ | the reference precision: running the verification gates, and the input to `nllb-quantize` | `@F32_SHA@` |

More aggressive quantisations (whole-model Q8_0, Q5_K_M) were built and
**rejected on measured translation quality** — see `QUANTIZE.md` in the repo.
They are not published on purpose.

## Use

```bash
git clone --recursive https://github.com/Jaydeep505/nllb.cpp && cd nllb.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./models/download-nllb-model.sh f16          # verifies SHA256SUMS
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --src eng_Latn --tgt deu_Latn "Hello, world!"
```

Or with the Hugging Face CLI:

```bash
hf download @REPO@ nllb-600M-f16.gguf --local-dir models
```

`nllb-cli -m <file> --list-langs` prints the 202 FLORES-200 language codes.

## Provenance

| | |
|---|---|
| source checkpoint | `facebook/nllb-200-distilled-600M` @ `@SOURCE_REV@` |
| converted with | nllb.cpp @ `@NLLB_REV@` (`tools/convert_nllb_to_gguf.py`, `nllb-quantize q8emb`) |
| gated | @GATED@ |
| built | @DATE@ |

These files require an nllb.cpp build from step 8 or later (they carry the
precompiled charsmap, merge hash and UAX#29 tables). An older binary ignores
those keys; a newer one refuses a file that lacks them, by name.

## License

The weights are Meta's NLLB-200, released under
**CC-BY-NC 4.0 — non-commercial use only**. Converting them to GGUF does not
change that. This repository redistributes them unmodified in value (f32), at
reduced precision (f16) or partially quantised (q8emb), with attribution to:

> NLLB Team et al., *No Language Left Behind: Scaling Human-Centered Machine
> Translation*, 2022. arXiv:2207.04672

The nllb.cpp code is licensed separately; see its repository.
