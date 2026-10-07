# scripts/

Developer and maintainer tools. **You don't need anything in this folder to
use nllb.cpp.** To get a model, see [`models/README.md`](../models/README.md).

These scripts generate the reference files the tests compare against, and
publish the models to Hugging Face. CI runs several of them, so keep them in
the repo.

## Reference files for the tests

These run the original Hugging Face model and save its outputs into `ref/`,
so the C++ tests can check they produce the same results. They need torch and
transformers:

```bash
pip install -r scripts/requirements-verify.txt
```

| script | what it writes |
|---|---|
| `dump_reference.py` | encoder hidden states |
| `dump_decoder_reference.py` | decoder hidden states and logits |
| `dump_tokenizer_reference.py` | tokenizer test corpus with the HF token ids |
| `dump_translation_reference.py` | full HF translations, for the CLI tests |

## Reference implementations

| script | what it is |
|---|---|
| `ref_encoder.py` | the NLLB encoder in plain numpy, reading only the GGUF file |
| `ref_decoder.py` | the same for the decoder |

## Code generation

| script | what it does |
|---|---|
| `gen_grapheme_tables.py` | regenerates `src/nllb-grapheme-data.h` (Unicode grapheme tables used by the tokenizer) |

## Quality gates

| script | what it does |
|---|---|
| `step10-measure.sh` | runs the precision and quantization checks on f32/f16 GGUF files |

## Publishing (maintainer only)

| file | what it does |
|---|---|
| `publish-models.sh` | builds the f16, q8emb and f32 GGUFs, checks them, writes `SHA256SUMS` and uploads them to Hugging Face |
| `model-card.md` | template for the Hugging Face model card, filled in by `publish-models.sh` |
