# nllb.cpp — step 1: NLLB → GGUF

Converts `facebook/nllb-200-distilled-600M` to GGUF and proves the result
reproduces HuggingFace's encoder to float tolerance. Nothing here needs a GPU
and the converter itself needs neither torch nor transformers.

```
tools/convert_nllb_to_gguf.py   HF snapshot  -> .gguf         (numpy + safetensors + gguf)
tools/ref_encoder.py            .gguf        -> hidden states (numpy only; the C++ spec)
tools/dump_reference.py         HF model     -> ref/*.bin     (golden files for C++ tests)
tests/make_tiny_model.py        random tiny NLLB, no download
tests/verify_encoder.py         HF vs GGUF diff — the milestone-1 gate
```

## Run it

```bash
pip install numpy safetensors gguf sentencepiece      # converter only
pip install torch transformers                        # verification only

huggingface-cli download facebook/nllb-200-distilled-600M \
    --local-dir models/nllb-600M

python tools/convert_nllb_to_gguf.py models/nllb-600M \
    --outfile models/nllb-600M-f32.gguf --outtype f32

python tests/verify_encoder.py models/nllb-600M models/nllb-600M-f32.gguf
```

Expect `PASS` with `max|d|` around `1e-6`. If it prints `MISMATCH`, stop —
do not start on ggml until this is green.

Fast loop with no download at all:

```bash
python tests/make_tiny_model.py models/tiny-nllb
python tools/convert_nllb_to_gguf.py models/tiny-nllb --outfile models/tiny.gguf
python tests/verify_encoder.py models/tiny-nllb models/tiny.gguf
```

Measured on the tiny model: **f32 → `8.3e-07`**, **f16 → `8.9e-04`**. Three
orders of magnitude. Convert f32 for the milestone; f16 error is large enough
to hide a real bug in the same digits.

## What NLLB actually is

`M2M100ForConditionalGeneration` — a pre-layernorm encoder-decoder. 600M-distilled:

| | |
|---|---|
| d_model | 1024 |
| encoder / decoder layers | 12 / 12 |
| attention heads | 16 (head_dim 64) |
| FFN dim | 4096 |
| activation | **relu**, not gelu |
| vocab | 256206 |
| max positions | 1024 |
| pad / bos / eos | 1 / 0 / 2 |
| LayerNorm eps | 1e-5 (torch default) |
| lm_head | tied to `model.shared.weight` |

Encoder block, in order:

```
r = x;  x = LN(x, attn_norm);  x = self_attn(x);  x = r + x
r = x;  x = LN(x, ffn_norm);   x = fc2(relu(fc1(x)));  x = r + x
```

and one final `LN(x, enc.output_norm)` after the last block. Attention has bias
on all four projections.

## The five things that will bite you

**1. Position embeddings are sinusoidal and are NOT in the checkpoint.**
HF holds them in a non-persistent buffer, so they never hit disk. The converter
recomputes the table and bakes it in as `position_embd.weight`. Do not generate
sinusoids in C++ until the encoder is green — the formula has three separate
places to get it subtly wrong:

- the denominator is `log(10000) / (half_dim − 1)`, not `/ half_dim`
- layout is `[sin(all freqs) | cos(all freqs)]` **concatenated**, not the
  interleaved layout in the original Transformer paper and most blog posts
- row `padding_idx` (= 1) is zeroed

**2. Positions are offset by 2.** HF computes
`cumsum(mask) * mask + padding_idx` with `padding_idx = 1`, against a table
built with `offset = 2`. For an unpadded sequence, token *j* reads table row
**j + 2**. Rows 0 and 1 are never used. Off-by-one here gives plausible-looking
garbage that survives layer 1 and diverges slowly — the worst failure mode.

**3. Embeddings are scaled by `sqrt(d_model)` = 32, positions are not.**
`x = embed[ids] * 32.0 + pos_table[j + 2]`.

**4. Do not transpose linear weights.** torch stores `(out, in)`; gguf reverses
shape on write, so ggml sees `ne = [in, out]`, which is exactly what
`ggml_mul_mat` expects. Writing them "corrected" is the single most common
conversion bug.

**5. Attention scaling is applied to Q, not to the logits.**
`q = q_proj(x) * head_dim**-0.5`. Algebraically identical, but if you dump
intermediate tensors to bisect a mismatch, matching HF's placement means your
numbers line up at every step instead of only at the end.

## GGUF layout produced

Arch string `nllb`. Metadata beyond the usual: `nllb.encoder.block_count`,
`nllb.encoder.attention.head_count`, `nllb.encoder.feed_forward_length`
(and `nllb.decoder.*`), plus `nllb.embedding_scale`, `nllb.position_offset`,
`nllb.pre_layernorm`, `nllb.activation`, `nllb.attention.qkv_bias`.

Tensors:

```
token_embd.weight                 [vocab, d_model]
position_embd.weight              [max_pos + 2, d_model]   baked sinusoids, always f32
enc.blk.{i}.attn_{q,k,v,out}.{weight,bias}
enc.blk.{i}.attn_norm.{weight,bias}
enc.blk.{i}.ffn_{up,down}.{weight,bias}          up = fc1, down = fc2
enc.blk.{i}.ffn_norm.{weight,bias}               = HF final_layer_norm
enc.output_norm.{weight,bias}
dec.blk.{i}.*                                    + cross_attn_{q,k,v,out,norm}
dec.output_norm.{weight,bias}
output.weight                                    only if lm_head is untied
```

`--encoder-only` drops the decoder (52 tensors instead of 106) if you want a
smaller file while working the milestone.

In f16 mode all biases, all norms, and the position table stay f32 — they are
tiny and they are exactly where precision loss compounds.

## Tokenizer

Read from `tokenizer.json` when present, which already has NLLB's fairseq `+1`
id offset and the 200+ FLORES language codes resolved — so the offset
arithmetic is never guessed. Falls back to `sentencepiece.bpe.model` plus the
documented offset rule. Scores come from the spm model, keyed by piece text.

`vocab_size` (256206) exceeds the real token count; the embedding matrix is
padded. The converter warns and continues — that warning is expected on NLLB.

Encoder verification doesn't need the tokenizer at all (it feeds raw ids), so
if the vocab path gives you trouble, `--no-tokenizer` unblocks the milestone.

**The tokenizer is now implemented in C++ — see `TOKENIZER.md`.** Two things
that document exists to correct, because both were wrong here for seven steps:
NLLB's tokenizer is **BPE**, not Unigram (`sentencepiece.bpe.model` means what
its name says), and the reference to match is HuggingFace's *fast* tokenizer,
not sentencepiece — they normalize differently, and the difference changes ids.
A GGUF converted before that step lacks the precompiled charsmap and is refused
at load rather than tokenizing ASCII correctly and everything else silently
wrong.

## Next

`tools/dump_reference.py --per-layer` writes golden `.bin` files plus every
intermediate layer's output. When the C++ encoder's final output is wrong, diff
layer by layer — it tells you *where* it broke rather than *that* it broke.
`tools/ref_encoder.py` is the line-by-line spec to port; each numpy line maps
to one ggml op.
