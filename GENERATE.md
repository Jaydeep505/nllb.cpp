# Step 4: greedy generation

```
src/nllb.{h,cpp}                 nllb_generate, NLLB_LOGITS_LAST, nllb_pos_mode
tests/test_generate.cpp          HF generate() goldens vs ggml, exact id diff
tools/dump_decoder_reference.py  --greedy now also writes ref/manifest_gen.txt
tests/corrupt_gguf.py            break one tensor on purpose, to test the tests
```

The loop is the only new computation: encode once, then run the proven
teacher-forced decoder over the growing prefix, take the argmax of the last
row, append, repeat. Everything under it was already green in step 3.

## Complete on the real 600M

`facebook/nllb-200-distilled-600M`, eng_Latn -> deu_Latn, 9 cases, **exact id
match on every one**:

```
model: d_model=1024 dec_layers=12 dec_heads=16 vocab=256206
       dec_start=2 eos=2 lm_head=tied
greedy: exact id comparison, no tolerance. tie floor 1.0e-03

  00_hello_world  Tg=7   got=7   eos  ok        05_german    Tg=25  got=25  cap  ok
  01_quick_brown  Tg=19  got=19  eos  ok        06_japanese  Tg=13  got=13  eos  ok
  02_empty        Tg=4   got=4   eos  ok        07_arabic    Tg=17  got=17  eos  ok
  03_single_char  Tg=5   got=5   eos  ok        08_repeated  Tg=16  got=16  eos  ok
  04_long         Tg=25  got=25  cap  ok

9 of 9 manifest cases ran, 0 failed.
smallest logit margin over all steps: 1.822e-02 (05_german step 23)
```

Seven cases ran to eos; `04_long` and `05_german` hit the 24-token cap, where
HF's golden is truncated at the same point, so the comparison still holds.
`--max-new-tokens 64` gets full translations if you want them.

`--check-modes` on the real vocabulary: `max|d| = 0.000e+00`, **0 of 256206
floats differ**, `argmax=same`, all 9 cases. `NLLB_LOGITS_LAST` is bit-exact
against the full projection at production scale, not just on a toy vocab.

The decoder gate re-run afterwards is digit-for-digit identical to its step-3
result (worst rel 1.494e-06, `07_arabic`), so `logits_mode` and `pos_mode`
changed nothing on the path that was already proven.

## Parity, and nothing beyond it

`nllb_generate` implements exactly what `generate(num_beams=1, do_sample=False)`
does and deliberately nothing else:

- the sequence starts at `decoder_start_token_id` (eos, 2 - not bos), and that
  token counts as position 0, so the result is at most `1 + max_new_tokens`
  long;
- `forced_bos_token_id` replaces the argmax at the **first generated position**.
  HF implements it by masking every other logit to `-inf`; this does the same
  thing directly, and keeps the unforced argmax in the step trace so a
  divergence stays readable;
- eos terminates and is included in the output;
- ties go to the lower token id, which is what `torch.argmax` does;
- positions follow the **cached** rule - see below, it is not the same as the
  teacher-forced one.

No sampling, no beams, no repetition penalty, no length penalty. Every one of
those would be untestable against `ref/<name>.gen.bin`, which is the only
golden that exists. `dump_decoder_reference.py --greedy` warns if the
checkpoint's `generation_config` turns on a processor this loop does not
implement (`bad_words_ids`, `no_repeat_ngram_size`, `min_new_tokens`,
`sequence_bias`, `remove_invalid_values`, ...), because that would otherwise
surface as a mysterious tail divergence rather than as a missing feature. It
also asserts what the C++ loop assumes about HF's own output: the sequence
starts at `decoder_start`, `forced_bos` lands at position 1, and the length
never exceeds `1 + max_new_tokens`.

`forced_bos_token_id` and `max_new_tokens` are *inputs* to generation, so they
come from `ref/manifest_gen.txt` rather than being read off the golden. Taking
`gen.bin[1]` as the forced token would have made the test pass by construction
on exactly the mistake it exists to catch - and getting the target language
code wrong is the mistake that produces fluent output in the wrong language.
The manifest records the **effective** forced token, not the one passed on the
command line: omitting the argument falls back to `generation_config`, and
mBART-style checkpoints set it there, so a dumper that recorded its own
argument would tell the C++ loop not to force something HF did force.

## The position rule is not the same in both HF paths

This is the one place where "the cached and un-cached paths are the same
computation up to rounding" is false, and it is worth stating precisely because
the KV cache lands next and will be diffed against this code.

HF's `create_position_ids_from_input_ids` is

```python
(cumsum(mask) + past_key_values_length) * mask + padding_idx
```

In a forward pass with `use_cache=False` the whole sequence goes in at once and
`past = 0`, so **a pad shifts every later position down by one**. Inside
`generate()` the same function is called with one token and `past = j`, and the
cumsum restarts each step, so a non-pad token at index *j* always gets
`j + 1 + padding_idx` no matter how many pads preceded it.

The two agree on any sequence with no pad in it, which is every teacher-forced
golden - and disagree the moment the model emits `pad_id`. So:

| path | rule | used by |
|---|---|---|
| `NLLB_POS_CUMSUM` | `cumsum(mask) * mask + pad` | `nllb_decode` default, teacher-forced gate |
| `NLLB_POS_ABSOLUTE` | `(j + 1) * mask + pad` | `nllb_generate`, and the coming KV cache |

Picking one globally would make one of the two gates unsatisfiable on any case
where the model emits pad. `ref-pad/` exists to hold that case: goldens dumped
with `--forced-bos 1` (pad itself), where **7 of 9 cases fail under the
un-cached rule and all 9 pass under the cached one**. Without that fixture the
whole suite is blind to the difference - all three original fixtures pass
either way.

## No KV cache, on purpose - again

Every step re-decodes the whole prefix from scratch. That is O(n²) and correct,
and it is the same reasoning as step 3: the cached path is next, and it will be
diffed against *this* path rather than against torch, so a cache bug cannot
hide behind a graph bug.

The one concession to cost is `NLLB_LOGITS_LAST`. Greedy reads exactly one row
of logits, and on the 600M the lm_head is a `[256206, 1024] x [1024, Td]`
matmul - the dominant op in the graph. Projecting only the final row cuts it by
a factor of Td. `--check-modes` proves that row is bit-identical to the last
row of `NLLB_LOGITS_ALL` (measured: `max|d| = 0`, 0 differing floats out of
256206 on the real model and out of 2000 on the synthetic ones, over
Td = 1..24), which is what lets the decoder gate's numbers say
anything about generation at all. A differing argmax there fails the run;
differing floats with the same argmax only warn, since a new backend can
legitimately do that.

## Run

```bash
# goldens: encoder dumper first, then the decoder dumper with --greedy
python tools/dump_reference.py         models/nllb-600M --out ref/ --lang eng_Latn --per-layer
python tools/dump_decoder_reference.py models/nllb-600M --out ref/ \
    --tgt-lang deu_Latn --per-layer --greedy

cmake --build build -j
./build/bin/nllb-test-decoder  models/nllb-600M-f32.gguf ref/ -t 8
./build/bin/nllb-test-generate models/nllb-600M-f32.gguf ref/ -t 8

# occasionally, and after any backend change:
./build/bin/nllb-test-generate models/nllb-600M-f32.gguf ref/ -t 8 --check-modes
```

Do not paste a trailing `# comment` after one of these lines in zsh:
interactive comments are off by default, so the `#` and what follows become
arguments, a `?` in them glob-fails, and the whole line is skipped without
running. A skipped gate looks exactly like a gate that had nothing to say.

`ref/<name>.dec.json` carries `greedy_text` next to the ids, so a passing run
can be read as German rather than as integers.

## This gate is exact, and that is a different thing from strict

Ids are right or wrong; there is no tolerance to set. But that makes the gate
*narrower*, not stronger, and it is worth being precise about how:

Greedy only depends on which logit is largest. Any error that preserves the
ordering is invisible here. Corrupting the deep synthetic model (12+12 layers,
16 heads, untied lm_head, 9 cases) makes the point:

| corruption | generate gate | decoder gate |
|---|---|---|
| `dec.blk.0.attn_q --transpose` | 7/9 diverge | 8/9 FAIL |
| `dec.blk.11.attn_q --transpose` | 6/9 | 8/9 |
| `dec.blk.5.cross_attn_k --transpose` | 4/9 | 9/9 |
| `position_embd --shift` | 8/9 | 9/9 |
| `ggml_diag_mask_inf` deleted | 8/9 | 8/9 |
| `dec.blk.3.ffn_up --scale 1.01` | **0/9** | 9/9 |
| `dec.output_norm --scale 1.01` | **0/9** | 9/9 |
| `token_embd --scale 1.01` | **0/9** | 9/9 |

The three zeros are not a bug in the gate. A near-uniform rescaling of the
residual stream rescales the logits and leaves the argmax where it was - the
decoder gate reports those cases as `argmax=same` while failing on magnitude,
which is precisely the property greedy decoding rides on. **The generate gate
does not subsume the decoder gate. Run both.**

The two also have *different* blind spots, which is the argument for keeping
both rather than picking the stricter one. Deleting the causal mask fails 8 of
9 in each - but not the same 8. The decoder gate misses `02_empty`, where
`Td = 1` makes a 1×1 softmax equal to 1.0 whatever the mask says; the generate
gate misses `08_repeated`, whose golden output is one token repeated 24 times,
degenerate enough that the corruption never flips the top slot. Between them
the coverage is complete.

## Margins: the generation analogue of the ULP report

The decoder gate answers "is this diff real or is it float32 noise?" with ULPs.
Generation answers the same question with the logit margin.

Every step records `top1 - top2`. On a mismatch the gate re-decodes the shared
prefix and prices the disagreement directly - `logit(ours)` vs `logit(HF's)` -
and classifies it:

- **DECISIVE** (gap ≥ `--tie-floor`, default 1e-3): the graph disagrees with
  HF. Go and run the decoder gate.
- **NEAR TIE** (gap < tie floor): rounding decided this fork, not the graph.
  Note that this describes *this fork*, not the model - a genuinely corrupted
  model can still diverge first at a coin toss, as `cross_attn_k --transpose`
  does in one of its four failures.

Three mismatch shapes get their own message because they are not numeric at
all: a divergence at **position 0** means `manifest_gen.txt` and the GGUF
disagree about `decoder_start_token_id`; a divergence at the **forced
position** means `forced_bos_id` is wrong (nothing about the graph is under
test yet); and identical prefixes with **different lengths** mean a
stopping-rule difference - eos handling, or how `max_new_tokens` is counted.
All three were confirmed by deliberately passing the wrong values.

A passing run still prints the smallest margin it saw. That number is the
headroom before a rounding change forks the output, and it is the number that
decides whether f16 can be adopted:

| model | smallest margin | vs f16's 8.9e-04 |
|---|---|---|
| deep synthetic 12+12 | 4.2e-04 | **below** it |
| **real 600M** | **1.8e-02** | 20x above it |

The synthetic model was the alarming case: at 4.2e-04 f16 would produce
different text while every float gate stayed green. The real checkpoint is not
that narrow - no greedy decision came within 20x of flipping - which is an
argument for trying f16, not proof. It is 9 cases, one language pair, <=24
tokens each; longer outputs are more chances to land on a tie.

Under `--check-modes` the trace is itself re-derived from an independent full
logits row and compared, so the number the gate reports is checked rather than
merely printed.

## A skip is a failure

A missing or stale `.gen.bin` used to print to stderr and drop the case. With
eight of nine goldens deleted the run said `1 cases, 0 failed. PASS` and exited
0 - so a partial regeneration or a renamed prompt would silently shrink the
gate to nothing. Skips are now counted, reported against the manifest line
count (`9 of 9 manifest cases ran`), and exit non-zero.

## Verified

| fixture | model | covers | result |
|---|---|---|---|
| **`ref`** | **real 600M, eng->deu, 24 tokens** | **the whole pipeline** | **exact, 9/9** |
| `ref-tiny` | tiny 3+2 / 4 heads / tied, forced_bos 7 | the loop, the cap | exact |
| `ref-eos` | same model, no forced_bos | the eos stop (Tg=2) | exact |
| `ref-deep` | deep 12+12 / 16 heads / untied, forced_bos 123 | depth, untied lm_head, varied output | exact |
| `ref-pad` | deep model, forced_bos = pad (1) | the position rule | exact |

Exact means the emitted id sequence equals `generate()`'s byte for byte,
including the leading `decoder_start_token_id` and the trailing eos.

Note what each fixture does *not* prove. `ref-tiny`'s golden is `[2] + 24x7`
and `ref-eos`'s is `[2, 2]`: both are reproduced byte for byte by a loop that
forces `forced_bos` at *every* step, so neither actually pins the
forced-position rule. Only `ref-deep`, whose output varies
(`123, 769, 1137, 1851, ...`), does. Their value is the two stopping rules, not
the forcing.

Rebuilding all four from nothing, no download involved:

```bash
python tests/make_tiny_model.py models/tiny-nllb
python tests/make_tiny_model.py models/deep-nllb --layers 12 --heads 16 \
    --d-model 256 --ffn 512 --vocab 2000 --untied
python tools/convert_nllb_to_gguf.py models/tiny-nllb --outfile models/tiny-f32.gguf --outtype f32
python tools/convert_nllb_to_gguf.py models/deep-nllb --outfile models/deep-f32.gguf --outtype f32

for d in tiny deep; do
  python tools/dump_reference.py models/$d-nllb --out ref-$d/ --per-layer --random-ids
done
python tools/dump_decoder_reference.py models/tiny-nllb --out ref-tiny/ --per-layer --random-ids --greedy --forced-bos 7
python tools/dump_decoder_reference.py models/deep-nllb --out ref-deep/ --per-layer --random-ids --greedy --forced-bos 123
# ref-eos and ref-pad reuse another fixture's source ids:
mkdir -p ref-eos ref-pad
cp ref-tiny/*.ids.bin ref-tiny/manifest.* ref-eos/
cp ref-deep/*.ids.bin ref-deep/manifest.* ref-pad/
python tools/dump_decoder_reference.py models/tiny-nllb --out ref-eos/ --random-ids --greedy
python tools/dump_decoder_reference.py models/deep-nllb --out ref-pad/ --random-ids --greedy --forced-bos 1
```

A random model wants to emit eos immediately - the decoder starts on the eos
embedding and the tied lm_head scores that same row highest - so
`--forced-bos` was added to the dumper to get a synthetic run past step 0 at
all. With `--random-ids` there is no tokenizer and therefore no target language
code, so without it there is no loop to test.

## Still not proven

- **KV cache.** Still nothing incremental. It should implement
  `NLLB_POS_ABSOLUTE`, and be diffed against this path.
- **Beam search.** No goldens dumped for it yet.
- **Long outputs and other language pairs.** Everything measured stops at 24
  new tokens, eng->deu only. The margin statistic is only as good as the
  number of decisions it has seen, and 9 short sentences is not many.
- **Padded batches.** `nllb_generate` rejects source ids containing `pad_id`
  up front, before spending an encoder pass on them. Note this is about the
  *source*; the decoder emitting pad is handled, and `ref-pad` covers it.
- **The tokenizer.** Still HF's; the GGUF's embedded vocab remains unread.
- f16, performance, Metal. See the margin note above before reaching for f16.
