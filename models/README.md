## nllb.cpp models

nllb.cpp runs a single GGUF file that holds both the weights and the
tokenizer of [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
(or any M2M100-shaped fine-tune of it). GGUF files go in this folder and are
git-ignored; only the scripts and this README are tracked.

There are three ways to get a model:

1. Use the `download-nllb-model.sh` script (recommended)
2. Manually download a pre-converted file from Hugging Face
3. Convert your own (fine-tuned) checkpoint with `convert-nllb-to-gguf.py`, then quantize it

### Available models

| name    | file                   | size    | notes |
|---------|------------------------|---------|-------|
| `f16`   | `nllb-600M-f16.gguf`   | 1.24 GB | **Default.** Same output as the original Hugging Face model. |
| `q8emb` | `nllb-600M-q8emb.gguf` | 0.99 GB | `token_embd` at Q8_0, rest f16. ~1.33x faster on an M1 CPU, chrF 96.57 vs HF. Good for phones. |
| `f32`   | `nllb-600M-f32.gguf`   | 2.46 GB | Reference precision. Use it as the input to `nllb-quantize`. |

| file                   | SHA256 |
|------------------------|--------|
| `nllb-600M-f16.gguf`   | `e8e202c28ae5f43a7905e4790f930e16807e14eeb294a844400e64be57606f9a` |
| `nllb-600M-q8emb.gguf` | `aebad3ca0507c8a882659e10cda4861ed246068ac871c91e784d97cd1b17a73e` |
| `nllb-600M-f32.gguf`   | `150f7eab66d6a7e469930a4ab66835277a0635868f6cc4d19b62723d6fe6aa5e` |

## 1. Download with the script

```text
$ ./models/download-nllb-model.sh f16
downloading nllb-600M-f16.gguf from https://huggingface.co/JaydeepGupta/nllb-200-distilled-600M-GGUF/resolve/main
verifying sha256...

saved models/nllb-600M-f16.gguf

try it:
  ./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --src eng_Latn --tgt deu_Latn "Hello, world!"
```

```bash
./models/download-nllb-model.sh            # f16, the default
./models/download-nllb-model.sh q8emb      # smaller and faster on CPU
./models/download-nllb-model.sh f16 ~/nllb # save into another directory
./models/download-nllb-model.sh --list     # show all models
```

On Windows use `.\models\download-nllb-model.ps1` with the same arguments.

The script needs only `curl` (or `wget`) and `sha256sum`/`shasum`. It checks
every file against the repo's `SHA256SUMS` before naming it `*.gguf`, and an
interrupted download resumes when you run it again.

Pin a revision for anything reproducible (CI, apps, papers):

```bash
NLLB_HF_REV=<commit-sha> ./models/download-nllb-model.sh f16
```

Other overrides: `NLLB_HF_REPO` (a fork or mirror), `NLLB_BASE_URL` (any
server with the same layout) and `HF_TOKEN` (private or gated repos).

## 2. Manually download pre-converted models

GGUF models are available from:

- https://huggingface.co/JaydeepGupta/nllb-200-distilled-600M-GGUF/tree/main

Download a file from the browser into `models/`, or use the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download JaydeepGupta/nllb-200-distilled-600M-GGUF nllb-600M-f16.gguf --local-dir models
```

Unlike the script, this does not check the file. To check it yourself,
compare the output with the SHA256 table above:

```bash
sha256sum models/nllb-600M-f16.gguf        # Linux
shasum -a 256 models/nllb-600M-f16.gguf    # macOS
```

## 3. Convert your own (fine-tuned) model with `convert-nllb-to-gguf.py`

Use this for your own fine-tune of NLLB-200, or to build every file from the
original checkpoint yourself. The converter reads everything from
`config.json`, so any M2M100-shaped checkpoint converts at whatever depth and
width it has.

### 3.1 Install the converter dependencies

```bash
pip install -r models/requirements-convert.txt   # numpy, gguf, sentencepiece, regex - no torch needed
```

`regex` must be installed. Without it the converter writes a file with no
grapheme tables and only prints a one-line warning.

### 3.2 Put your checkpoint in a folder

The folder is a normal Hugging Face snapshot: the output of
`model.save_pretrained(...)` + `tokenizer.save_pretrained(...)`, or a
`hf download` of a repo.

```text
models/my-nllb-finetune/
├── config.json
├── model.safetensors            # or sharded *.safetensors + index, or pytorch_model.bin
├── tokenizer.json               # recommended: lets the converter verify the merges
├── sentencepiece.bpe.model
├── special_tokens_map.json
└── tokenizer_config.json
```

```bash
# your own fine-tune on the Hub
hf download <your-user>/<your-nllb-finetune> --local-dir models/my-nllb-finetune

# or the original base model
hf download facebook/nllb-200-distilled-600M --local-dir models/nllb-600M
```

`model.safetensors` is preferred. A `pytorch_model.bin` checkpoint also
works, but needs `pip install torch` to read it.

### 3.3 Convert to GGUF

Convert to **f32** first. It is the reference file and the input you quantize
from:

```bash
python models/convert-nllb-to-gguf.py models/my-nllb-finetune \
    --outfile models/my-nllb-f32.gguf --outtype f32 --name my-nllb-finetune
```

Optionally also write an **f16** file (half the size, the recommended
everyday file):

```bash
python models/convert-nllb-to-gguf.py models/my-nllb-finetune \
    --outfile models/my-nllb-f16.gguf --outtype f16 --name my-nllb-finetune
```

Useful flags:

| flag | what it does |
|---|---|
| `--outtype f32\|f16\|bf16` | type for all weights (norms, biases and position embeddings always stay f32) |
| `--embed-merges` | store the BPE merge list in the file (~6 MB, ~0.4 s faster load). Done automatically if your fine-tune trimmed or reordered the vocabulary |
| `--name` | sets `general.name` in the GGUF |

### 3.4 Quantize

Build the quantizer, then always quantize **from the f32 file**, never from
f16:

```bash
cmake -B build
cmake --build build --config Release --target nllb-quantize

./build/bin/nllb-quantize --list-mixes        # show available mixes
./build/bin/nllb-quantize models/my-nllb-f32.gguf models/my-nllb-q8emb.gguf q8emb
```

The last argument is a **mix**, a per-tensor policy, not a single type:

| mix | what it does |
|---|---|
| `q8emb`  | `token_embd` Q8_0, everything else f16. **Recommended**: same recipe as the published `q8emb` |
| `q8_0`   | every 2-D weight Q8_0 |
| `q6_k`   | every 2-D weight Q6_K |
| `q5_k_s` | every 2-D weight Q5_K, `token_embd` included |
| `q5_k_m` | Q5_K, with `token_embd` and every `ffn_down` at Q6_K |
| `q4_k_m` | Q4_K, with `token_embd` and every `ffn_down` at Q6_K |

LayerNorms, biases and `position_embd` are never quantized by any mix.

If you're not sure which one to pick, use `q8emb`. It is the only mix that
kept translation quality close to f16 on the base model. The lower-bit mixes
are smaller, but whole-model `q8_0` and `q5_k_m` lost too much quality to be
published.

The Python converter can also write the linear quants (Q8_0/Q5_0/Q4_0)
directly from a checkpoint, without the f32 file in between. K-quants
(`q6_k`, `q5_k`, `q4_k`, ...) need `nllb-quantize`.

```bash
# q8emb straight from the checkpoint
python models/convert-nllb-to-gguf.py models/my-nllb-finetune \
    --outfile models/my-nllb-q8emb.gguf --outtype f16 \
    --quantize token_embd.weight --type q8_0
```

### 3.5 Run it

```bash
./build/bin/nllb-cli -m models/my-nllb-q8emb.gguf --src eng_Latn --tgt deu_Latn "Hello, world!"
```

## License

The NLLB-200 weights are Meta's and licensed **CC-BY-NC 4.0, for
non-commercial use only**. Converting or quantizing them, or fine-tuning
them, does not change that.

See [`../docs/models.md`](../docs/models.md) and
[`../docs/quantize.md`](../docs/quantize.md) for details.
