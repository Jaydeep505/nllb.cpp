# nllb-quantize

Quantizes an nllb.cpp GGUF model to make it smaller and faster.

Always quantize **from the f32 file**, never from f16. To get an f32 file,
download it with `./models/download-nllb-model.sh f32`, or convert your own
checkpoint (see [`models/README.md`](../../models/README.md)).

## Build

```bash
cmake -B build
cmake --build build --config Release --target nllb-quantize
```

## Usage

```bash
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

## Quantizing with the Python converter

The Python converter can also write the linear quants (Q8_0/Q5_0/Q4_0)
directly from a checkpoint, without the f32 file in between. K-quants
(`q6_k`, `q5_k`, `q4_k`, ...) need `nllb-quantize`.

```bash
# q8emb straight from the checkpoint
python models/convert-nllb-to-gguf.py models/my-nllb-finetune \
    --outfile models/my-nllb-q8emb.gguf --outtype f16 \
    --quantize token_embd.weight --type q8_0
```

## Run it

```bash
./build/bin/nllb-cli -m models/my-nllb-q8emb.gguf --src eng_Latn --tgt deu_Latn "Hello, world!"
```
