# Step 3: the C++ decoder on ggml

```
tools/dump_decoder_reference.py  HF teacher-forced decoder -> ref/*.dec*.bin
tools/ref_decoder.py             .gguf -> decoder states + logits (numpy; the C++ spec)
tests/verify_decoder.py          HF vs numpy-from-GGUF   (the python gate)
tests/test_decoder.cpp           HF goldens vs ggml, with per-layer bisect
tests/corrupt_gguf.py            break one tensor on purpose, to test the tests
src/nllb.{h,cpp}                 nllb_decode + decoder tensor loading
```

## Complete on the real 600M

`facebook/nllb-200-distilled-600M`, 9 cases, 0 failed, `argmax=same` on every
one:

```
model: d_model=1024 dec_layers=12 dec_heads=16 dec_ffn=4096 vocab=256206
       eps=1e-05 embd_scale=32 dec_start=2 pad=1 eos=2 lm_head=tied
tolerances are RELATIVE: states 5.0e-05, logits 1.0e-04

  00_hello_world  Td=6   rel=8.400e-07  logit=1.634e-06  argmax=same  ok
  01_quick_brown  Td=16  rel=9.216e-07  logit=1.097e-06  argmax=same  ok
  02_empty        Td=2   rel=5.171e-07  logit=1.045e-06  argmax=same  ok
  03_single_char  Td=3   rel=4.569e-07  logit=7.782e-07  argmax=same  ok
  04_long         Td=37  rel=7.430e-07  logit=1.060e-06  argmax=same  ok
  05_german       Td=21  rel=1.114e-06  logit=6.650e-07  argmax=same  ok
  06_japanese     Td=13  rel=1.443e-06  logit=8.079e-07  argmax=same  ok
  07_arabic       Td=19  rel=1.494e-06  logit=9.061e-07  argmax=same  ok
  08_repeated     Td=10  rel=5.142e-07  logit=9.045e-07  argmax=same  ok

worst rel = 1.494e-06 (07_arabic), worst logit rel = 1.634e-06
```

That is a factor of 33 inside the state tolerance and 60 inside the logit one.
Re-running it after step 4 changed `nllb_decode`'s signature (`logits_mode`,
`pos_mode`) reproduced these digits **exactly**, which is the cheapest possible
proof that the refactor was numerically a no-op.

Same ladder that made the encoder trustworthy, one rung up:

| comparison | worst |
|---|---|
| HF torch vs numpy-from-GGUF, synthetic 3-layer / 2-dec | 9.5e-07 abs |
| HF torch vs ggml C++, synthetic 3-layer / 2-dec | 1.6e-06 abs |
| HF torch vs numpy-from-GGUF, synthetic 12-layer / 16-head / untied | 7.6e-06 abs |
| HF torch vs ggml C++, synthetic 12-layer / 16-head / untied | 2.0e-06 abs |
| HF torch vs numpy-from-GGUF, **real 600M** | 3.3e-06 rel |
| **HF torch vs ggml C++, real 600M** | **1.5e-06 rel** |

Tolerances are relative here and absolute on the synthetic rows for a reason:
see the note in `claude/handoff.md` about the pre-layernorm residual stream
reaching ~4200 by block 4, where one float32 ULP is ~5e-4 and an absolute
threshold becomes unsatisfiable rather than strict.

## Teacher forced, no KV cache - on purpose

The whole target sequence goes through in one causally-masked pass. That is
what HF's dump does, so any mismatch here is a graph bug and nothing else.
Building the incremental cached path at the same time would mean the first
failing diff could be either, and you would have no way to tell which.

The cache is the *next* change, and it is separately verifiable: run cached and
un-cached over the same ids and diff them against each other, with HF out of
the loop entirely.

## Run

```bash
source .venv/bin/activate

# 1. python gate - if this is not green, do not start on ggml
python tests/verify_decoder.py models/nllb-600M models/nllb-600M-f32.gguf

# 2. goldens. The encoder dumper must run first: the decoder dumper reads
#    ref/<name>.ids.bin back so the two cannot drift apart.
python tools/dump_reference.py         models/nllb-600M --out ref/ --lang eng_Latn --per-layer
python tools/dump_decoder_reference.py models/nllb-600M --out ref/ \
    --tgt-lang deu_Latn --per-layer --greedy

# 3. the C++ diff
cmake --build build -j
./build/bin/nllb-test-decoder models/nllb-600M-f32.gguf ref/ -t 8
```

`--greedy` also writes `ref/<name>.gen.bin` (ids from `generate(num_beams=1)`)
and `ref/manifest_gen.txt`. Both are consumed by `bin/nllb-test-generate` in
step 4, and `--greedy` is no longer optional if you intend to run that gate.
Run it here anyway while the model is already loaded - it costs nothing.

Each case runs the *real* pipeline - `nllb_encode` on the source ids, then
`nllb_decode` against that output - rather than feeding the decoder a golden
encoder state it would never see at runtime. The decoder therefore inherits the
encoder's ~1e-6, which is the same composition the translator will do.

Tolerances: `2e-4` on hidden states, `2e-3` on logits. Logits are a projection
of a d_model vector onto a 256k-row matrix, so they carry more accumulated
error than the states they come from. The test also checks **argmax agreement**
separately: a hidden-state diff inside tolerance that still flips the argmax is
a real failure, and it is the only property greedy decoding actually depends on.

## Decoder block

```
r = x;  x = LN(x, attn_norm);        x = self_attn(x, causal);    x = r + x
r = x;  x = LN(x, cross_attn_norm);  x = cross_attn(x, enc_out);  x = r + x
r = x;  x = LN(x, ffn_norm);         x = fc2(relu(fc1(x)));       x = r + x
```

then `LN(x, dec.output_norm)`, then the tied lm_head. Everything the encoder
does - pre-layernorm, relu, bias on all four projections, scale on Q not on the
logits, sinusoid lookup at `j + 2`, embedding scale 32 - carries over unchanged.

Three things that are new, and all three are silent when wrong:

1. **Self-attention is causal, cross-attention is not.** `ggml_diag_mask_inf(kq, 0)`
   before the softmax on self-attention only. `n_past` is 0 because this is the
   un-cached path; with a cache it becomes the number of already-decoded tokens.
2. **Cross-attention K and V come from the encoder output, not from `x`.** Q has
   `Td` rows, K and V have `Ts`. Wiring `x` into K/V builds and runs and is wrong.
   K and V here do not depend on the decoder step at all, which is exactly why
   they are the first thing worth caching.
3. **The lm_head is tied to `token_embd.weight` and has no bias.** The converter
   only writes `output.weight` when the checkpoint actually untied them, so a
   missing `output.weight` is the expected case, not a load failure.

`decoder_start_token_id` is **eos (2)**, not bos. The converter now writes it to
the GGUF as `nllb.decoder_start_token_id` so the runtime never guesses; the C++
fallback is eos rather than the more obvious-looking bos. Guessing bos here
produces fluent output in the wrong language - a bug that survives a test suite
happily.

## Testing the tests

A green suite only proves the suite did not fire. `tests/corrupt_gguf.py` breaks
one tensor on purpose so you can watch it fire on the right line:

```bash
cp models/tiny.gguf /tmp/bad.gguf
python tests/corrupt_gguf.py /tmp/bad.gguf --tensor dec.blk.1.cross_attn_k.weight --mode transpose
./build/bin/nllb-test-decoder /tmp/bad.gguf ref/
```

```
  00_hello_world  Td=3  max|d|=3.132e-02 ... FAIL
      dec_layer_00  max|d|=1.192e-07  ok   (embeddings)
      dec_layer_01  max|d|=2.384e-07  ok
      dec_layer_02  max|d|=3.132e-02  FAIL   (after dec.output_norm)
```

Four corruptions, four correct attributions:

| corruption | first red line | reads as |
|---|---|---|
| `dec.blk.1.cross_attn_k.weight --mode transpose` | `dec_layer_02` | block 1 |
| `dec.blk.0.attn_q.weight --mode transpose` | `dec_layer_01` | block 0 |
| `position_embd.weight --mode shift` | `dec_layer_00` | embeddings / position offset |
| `dec.output_norm.weight --mode scale` | last entry only | the closing norm, not a block |

And one that is not a tensor at all: deleting the `ggml_diag_mask_inf` line
fails **8 of 9** cases at `dec_layer_01`. The ninth is `02_empty`, where `Td=1` -
a 1x1 softmax is 1.0 whatever the mask says, so causality is vacuous. That case
also survives a corrupted `attn_q`, for the same reason. Worth knowing before
you conclude from a single short case that something works.

Indexing, same trap as the encoder, verified against the installed transformers
rather than assumed - `dec_layer_NN` fed through the tied lm_head reproduces
`model.logits` exactly:

```
dec_layer_00        embeddings (scale + positions) = INPUT to block 0
dec_layer_i         input to block i = output of block i-1
dec_layer_NN        output of the last block AFTER dec.output_norm
```

## Found on the way: untied embeddings

`tie_word_embeddings=False` unties **four** parameters in transformers, not one:
`model.shared`, `encoder.embed_tokens`, `decoder.embed_tokens` and `lm_head` all
become independent. The GGUF schema stores a single `token_embd.weight`, so a
checkpoint whose two towers disagree cannot be converted faithfully - and
nothing downstream could detect it, because the shapes match and the numbers
look plausible. The converter now compares them and refuses.

NLLB itself always ties the two towers, so this cannot fire on the real model.
It fired on a synthetic one within minutes of building it, which is the point.

## Still not proven

- **KV cache.** Nothing incremental exists yet.
- **Padded batches.** Cross-attention is unmasked, exactly as the encoder's
  self-attention is. `nllb_decode` *rejects* source ids containing `pad_id`
  rather than returning plausible numbers; lifting that means a mask tensor
  into `ggml_soft_max_ext` and goldens that actually exercise padding. Note
  this is about the *source*: a decoder that emits `pad_id` is handled, via
  `nllb_pos_mode` - see `GENERATE.md`.
- **The tokenizer.** Still HF's. The GGUF's embedded vocab remains unread.
- f16, performance, Metal.

## Next

1. ~~**Greedy loop**~~ - done, see `GENERATE.md`. One finding from it belongs
   here: that gate compares token ids exactly, but it is blind to any error
   that preserves the argmax ordering - a 1.01x scale on `dec.output_norm`
   fails all 9 cases *here* and none of them there. It does not replace this
   gate; run both.
2. **KV cache**, verified against the un-cached path rather than against HF.
3. **Tokenizer in C++** against the embedded vocab - the last Python dependency
   and the remaining silent-bug reservoir.
4. **Then** optimise: `ggml_gallocr`, f16, Metal.
