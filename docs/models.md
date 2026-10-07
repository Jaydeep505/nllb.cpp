# Getting a model

nllb.cpp runs one GGUF file that holds both the weights and the tokenizer.
There are two ways to get one:

1. **Download a pre-converted file** (recommended). No Python needed.
2. **Convert it yourself** from a Hugging Face checkpoint. Use this for
   fine-tunes, or if you want to check every byte yourself.

## 1. Download

```bash
./models/download-nllb-model.sh            # f16, the default
./models/download-nllb-model.sh q8emb      # smaller and faster on CPU
./models/download-nllb-model.sh --list
```
```bash
NLLB_HF_REV=e8d8e2a441a677f1aea388a02b9f713ffac81c18 ./models/download-nllb-model.sh f16
```
```powershell
.\models\download-nllb-model.ps1           # Windows; same arguments
```

| name | file | size | when to use it |
|---|---|---|---|
| `f16` | `nllb-600M-f16.gguf` | 1.24 GB | **The default.** Greedy output matches HF `generate()` byte for byte on every golden. The only file with per-decision headroom above 1 (`docs/precision.md`). Use this one on Metal: matmuls there run in half precision anyway (trap 48). |
| `q8emb` | `nllb-600M-q8emb.gguf` | 0.99 GB | When memory or CPU speed matters more than exactness: 1.33x faster per step on an M1 CPU, corpus chrF 96.57 against HF, 8 of 9 goldens byte-identical (`docs/quantize.md`). Good for phones. |
| `f32` | `nllb-600M-f32.gguf` | 2.46 GB | Running the verification gates, and as the input to `nllb-quantize`. Not for everyday use. |

Whole-model Q8_0 and Q5_K_M were built and **rejected on measured
translation quality**, so they are not offered. If you want them anyway,
build them with `nllb-quantize` (section 2) and run the gates on them first.

The script:

- downloads `SHA256SUMS` first, then the model into `<file>.part`, checks
  its hash, and only then renames it to `<file>`. A cut-off or corrupted
  download never ends up with a `.gguf` name;
- **resumes**: run it again after an interruption and it picks up from the
  `.part` file;
- does nothing if the file is already there and its hash matches.

Files are served from the Hugging Face repo
[`JaydeepGupta/nllb-200-distilled-600M-GGUF`](https://huggingface.co/JaydeepGupta/nllb-200-distilled-600M-GGUF).
You can override where they come from:

| variable | default | |
|---|---|---|
| `NLLB_HF_REPO` | `JaydeepGupta/nllb-200-distilled-600M-GGUF` | a fork or mirror on HF |
| `NLLB_HF_REV` | `main` | a branch, tag or commit sha. **Pin a sha for anything reproducible** (CI, papers, apps): a new upload to `main` changes the bytes you get. |
| `NLLB_BASE_URL` | `https://huggingface.co/$REPO/resolve/$REV` | any HTTP server laid out the same way (`SHA256SUMS` next to the files) |
| `HF_TOKEN` | — | sent as a bearer token, for private or gated repos |

With the Hugging Face CLI instead (`pip install -U huggingface_hub`):

```bash
hf download JaydeepGupta/nllb-200-distilled-600M-GGUF nllb-600M-f16.gguf --local-dir models
```

That works too, but it does not check `SHA256SUMS`. Hugging Face checks its
own hashes, so this matters mostly for mirrors.

Then check it:

```bash
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --list-langs | wc -l     # 202
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --src eng_Latn --tgt deu_Latn "Hello, world!"
```

### On a phone

Download on your computer, then copy the file over: `adb push` for Android,
and for iOS either the app bundle or the Files app
(`examples/nllb.swiftui/README.md`). The iOS example picks `f16` first, then
`q8emb`.

### License

The weights are Meta's NLLB-200 and are licensed **CC-BY-NC 4.0, for
non-commercial use only**. Converting them to GGUF does not change that, and
neither does nllb.cpp's own license. If you ship them in a product, that
restriction comes with them.

## 2. Convert it yourself

```bash
pip install -r models/requirements-convert.txt        # numpy, gguf, sentencepiece, regex. No torch.
hf download facebook/nllb-200-distilled-600M --local-dir models/nllb-600M

python models/convert-nllb-to-gguf.py models/nllb-600M --outfile models/nllb-600M-f32.gguf --outtype f32
python models/convert-nllb-to-gguf.py models/nllb-600M --outfile models/nllb-600M-f16.gguf --outtype f16
./build/bin/nllb-quantize models/nllb-600M-f32.gguf models/nllb-600M-q8emb.gguf q8emb
```

`regex` must be importable. Without it, the converter writes a file with no
grapheme tables and only says so in one line of output. Always quantise from
f32, never from f16 (`docs/quantize.md`).

## For maintainers: publishing

`scripts/publish-models.sh` builds the three files above from a snapshot. It
then checks that each one lists 202 languages and translates a sentence,
runs `tests/verify_translation.py` (when `--ref` is given), writes
`SHA256SUMS`, and fills in the model card from `scripts/model-card.md` with
the source checkpoint's revision and nllb.cpp's commit. With `--upload` it
also pushes the files.

```bash
hf auth login
./scripts/publish-models.sh models/nllb-600M --ref ref/              # builds and checks into dist/
./scripts/publish-models.sh models/nllb-600M --ref ref/ --upload     # and publishes
```

**Republish after any change to the GGUF format.** Old files will either be
refused by name (missing keys) or will silently not use new keys (a stale
binary ignores them; see the handoff on step 8). After uploading, pin the new
commit sha wherever reproducibility matters.
