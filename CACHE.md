# Step 5: the KV cache

```
src/nllb.{h,cpp}         nllb_kv_cache, nllb_kv_cache_init/reset, nllb_decode_cached,
                         nllb_generate_params::use_cache
tests/test_cache.cpp     cached vs un-cached, four passes, no goldens and no torch
tests/test_generate.cpp  + --use-cache, to run the HF comparison through the cache
CMakeLists.txt           + bin/nllb-test-cache
.github/workflows/       cache gate in both ci.yml and real-model.yml
```

Project copy of this document: `claude/step5-cpp-kv-cache.md`.

## Status: COMPLETE on the real 600M

`facebook/nllb-200-distilled-600M`, eng_Latn → deu_Latn, 9 cases, goldens dumped
with `--max-new-tokens 64`:

```
9 of 9 manifest cases ran, 0 failed.
worst incremental state rel = 4.128e-07 (04_long), worst logit rel = 1.939e-07
control: un-cached decoding of GROWING prefixes moves the same states by 4.608e-07
one cached call over the whole prefix is BIT-IDENTICAL to the un-cached pass -
  0 differing floats
generation wall clock over all cases: un-cached 21824 ms, cached 7287 ms (2.99x),
  185 tokens
PASS
```

`nllb-test-generate --use-cache` on the same goldens: **exact id match, 9/9**,
same lengths and same stop reasons as the un-cached run, and the smallest logit
margin is `1.822e-02` at `05_german` step 23 — identical to the un-cached
step-4 figure to four significant figures. Turning the cache on does not move
the number the f16 decision rests on.

The four synthetic fixtures are green as well, on both the cache gate and
`--use-cache`:

| fixture | model | cache gate | generate `--use-cache` |
|---|---|---|---|
| **`ref/`** | **real 600M, eng→deu, 64 tokens** | **PASS** | **PASS** |
| `ref-tiny` | tiny 3+2 / 4 heads / tied | PASS | PASS |
| `ref-eos`  | same model, Tg=2 | PASS | PASS |
| `ref-deep` | 12+12 / 16 heads / untied | PASS | PASS |
| `ref-pad`  | deep model, emits pad at position 1 | PASS | PASS |

## Correctness: three numbers, in order of how much they say

### 1. `full = exact`, 0 differing floats, all nine cases

The whole prefix fed to the cache in **one** call, at the same sequence length
as the un-cached reference. No addition can regroup, so the only remaining
difference is the cache itself — and there is none, across 13 layers at the real
model's ~4200 residual magnitudes. The layout, the strided writes and the views
are exact rather than close. This is the assertion; the tolerances below are
backstops.

### 2. The incremental drift does not accumulate with length

| case | T | `step` rel |
|---|---|---|
| `02_empty` | 4 | 1.45e-07 |
| `01_quick_brown` | 19 | 4.04e-07 |
| `05_german` | 39 | 3.09e-07 |
| `04_long` | **65** | **4.13e-07** |

Token 65 is no worse than token 19. Whatever rounding the incremental path
introduces is bounded, not compounding — which is the thing that would actually
matter for long-form translation.

### 3. The control came out *larger* than the cache

```
worst incremental state rel = 4.128e-07   (cached, stepping)
control (grow)              = 4.608e-07   (no cache anywhere)
```

`grow` decodes prefixes of length 1, 2, … T with the ordinary un-cached decoder
and keeps the last row of each. No cache is involved. It moves the same states
by the same amount — slightly more, in fact — so the incremental difference is
attributable to the sequence length changing the grouping of float additions,
and not to the cache. Per case the two straddle each other in both directions
with no consistent sign.

`worst logit rel = 1.939e-07` is four orders of magnitude below the 1.822e-02
margin, so nothing is remotely close to forking the output.

## Speed: the cached cost per token is flat

Real 600M, 8 threads, MacBook Air:

| case | T | un-cached | cached | ratio | cached ms/token |
|---|---|---|---|---|---|
| `06_japanese` | 13 | 809 ms | 513 ms | 1.6x | 39.5 |
| `08_repeated` | 16 | 1047 ms | 582 ms | 1.8x | 36.4 |
| `07_arabic` | 17 | 1201 ms | 643 ms | 1.9x | 37.8 |
| `01_quick_brown` | 19 | 2046 ms | 822 ms | 2.5x | 43.3 |
| `05_german` | 39 | 4239 ms | 1469 ms | 2.9x | 37.7 |
| `04_long` | **65** | 11244 ms | 2472 ms | **4.5x** | **38.0** |

Token 65 costs what token 13 costs. The un-cached column over the same cases
runs 62 → 65 → 71 → 108 → 109 → **173** ms/token, rising with length. That is
the whole O(n²) → O(n) claim, measured.

Checking the shape directly, `05_german` (39) against `04_long` (65), a length
ratio of 1.67:

| | growth | predicted |
|---|---|---|
| un-cached | **2.65x** | n² → 1.67² = 2.78 |
| cached | **1.68x** | n → 1.67 |

The un-cached side lands slightly under n² because part of its cost is genuinely
linear: the `lm_head` runs once per step in both paths.

**The speedup is not a constant and should never be quoted as one.** It is
roughly proportional to output length, because n²/n = n. The aggregate 2.99x is
an artefact of these nine cases averaging 20 tokens.

Cases below ~10 tokens are measurement noise, not signal: `03_single_char`
reports 0.9x — cached *slower* — and `00_hello_world` measured 4.6x in one run
and 2.1x in the next, same binary, same work. At those lengths the encoder pass,
the one-time cross-attention cache build and scheduler jitter dominate.

### Where the remaining time goes — a hypothesis, not a measurement

Fitting `total ≈ n·L + d·n²/2` to the two trustworthy un-cached cases (39 and 65
tokens) gives a fixed per-step cost `L ≈ 12 ms` and a per-token decode cost
`d ≈ 5 ms`. That 12 ms is about what streaming the weights should cost: the
`lm_head` alone is 256206 × 1024 × 4 = **1.05 GB read per step**, plus ~0.6 GB of
decoder weights.

So the cached path should be running near 16 ms/token and it runs at 38. The
~22 ms gap is most likely `nllb_decode_cached` allocating a fresh ggml context
every step — the sizing works out to roughly 72 MB, so that is a 72 MB malloc
plus several thousand page faults per token, freed and repeated.

If that is right, `ggml_gallocr` plus a reused graph would roughly halve the
cached time again, and it is a far bigger win than anything left in the cache.
**It is arithmetic from weight sizes, not a profile.** Put a timer around
`ggml_init` before acting on it.

## What is cached, and what that buys

| | recomputed per step before | now |
|---|---|---|
| self-attention K, V for the prefix | all of it | only the new token |
| cross-attention K, V | all of it, every block, every step | once per source sentence |
| cross-attention projections | `[D,D] x [D,Ts]` per block per step | once |

### Layout: stored the way attention consumes it

The un-cached path spends a `ggml_permute` + `ggml_cont` on K and V in every
block. Storing the cache in projection layout and repeating that per step would
copy the entire history once per block per step — O(n) memory traffic to save
O(n) arithmetic, which gives most of the cache back. So the permute happens on
the *write*, for the new token only, and the read is a plain view:

```
k_self  [HD, n_ctx, H]     token t at row t of each head slice
v_self  [n_ctx, HD, H]     transposed, because mul_mat wants (n_kv, HD, H)
k_cross [HD, n_src, H]
v_cross [n_src, HD, H]
```

`ggml_cpy` handles the strided destination. The transposed `v_self` is the same
trick llama.cpp uses, and it is the one piece here whose correctness is not
obvious by inspection — see the C4 corruption below, which single-token stepping
cannot detect at all.

## The position rule, which is the trap this step was warned about

`nllb_decode_cached` implements `NLLB_POS_ABSOLUTE` and refuses to offer
anything else. HF's cached rule is

```python
(cumsum(mask) + past_key_values_length) * mask + padding_idx
```

with `past` *inside* the mask and the cumsum restarting every step, so for one
token at absolute index *j* it collapses to `j + 1 + pad_id`, or `pad_id` if the
token is itself pad. The un-cached `NLLB_POS_CUMSUM` rule cannot be evaluated
from `n_past` alone — the running count is a property of the whole prefix — and
reproducing it would mean matching HF's *other* path.

Two corruptions make the shape of this trap concrete, and **they produce
identical output**:

| | what is wrong | ref-deep | ref-pad |
|---|---|---|---|
| C5 | cache uses the cumsum rule (the wrong belief) | PASS | FAIL, layer 0, rel 2.747e-01 |
| C6 | cache is correct; the gate's *reference* uses cumsum | PASS | FAIL, layer 0, rel 2.747e-01 |

Same case, same layer, same number to four significant figures. A broken cache
and a correctly-built cache diffed against the wrong reference are
indistinguishable from the gate's output, which is why `tests/test_cache.cpp`
passes `NLLB_POS_ABSOLUTE` explicitly with a comment saying why, rather than
taking `nllb_decode`'s default.

And both are invisible without `ref-pad`: no pad in the prefix, no disagreement.

## The gate: four passes, one of them a control

`tests/test_cache.cpp` compares two C++ paths, so it needs no reference dump, no
python and no download. It reads `manifest_gen.txt` only to reuse the same nine
sources and the same `forced_bos` / `max_new` settings.

| pass | what runs | held to |
|---|---|---|
| `gen` | `nllb_generate` twice, `use_cache` false then true | exact id equality |
| `full` | the whole prefix in ONE cached call | **bit-identity** |
| `step` | one token per cached call | relative tolerance |
| `chunk` | 3 tokens per cached call | relative tolerance |
| `grow` | **no cache at all** — un-cached decode of prefixes 1, 2, … Td | it *is* the control |

Each pass is compared per layer against one un-cached teacher-forced pass, so a
divergence names a layer instead of a token: layer 0 means the embeddings and
therefore the position rule, layer 0 clean with everything after it bad means
cross-attention, clean up to layer *k* means self-attention from block *k-1*.

### Bit-identity is required of `full` and of nothing else

`full` runs the same sequence length as the reference, so no arithmetic can
regroup and the only remaining difference is the cache. The incremental passes
run at a *different* length by construction: attention at position *t* sums over
`n_kv` terms and `n_kv` is the sequence length, so in a full pass the extra terms
carry softmax weight exactly 0 — mathematically nothing, arithmetically a
different grouping of the same additions.

An earlier draft of this file asserted bit-identity everywhere. That was wrong,
and adding `grow` is what proved it: the un-cached decoder does exactly the same
thing to itself when the length varies.

The practical form of the rule, which will come up again with f16 and Metal: an
incremental decoder cannot be bit-identical to a full pass and asking for it is
asking for the wrong thing. What it can be — and is — is bit-identical **at the
same sequence length**.

### The `differing floats` counts are restricted, and here is why

`grow`'s last iteration decodes the entire prefix, which is bit-for-bit the same
call as the reference. Its final row and all of its logits are equal *by
construction*. `step` and `chunk` carry their drift into that same final row, and
on the 600M the logits alone are 256206 floats — enough to make the incremental
passes look an order of magnitude worse than the control while their relative
errors are identical. The first version of this gate printed exactly that and
invited exactly that misreading:

```
differing floats: full=0 step=292429 chunk=292429 grow=21050     <- misleading
```

The `-v` line now counts only hidden-state rows 0..T-2, logits excluded — the
population that means the same thing for all four passes. `full` keeps its
complete count, because that one is an assertion rather than a comparison.

## Testing the tests

Seven deliberate corruptions. The column that matters is which pass caught each
one — three of them are invisible to single-token stepping, which is to say
invisible to how the cache is actually used at inference time.

| # | corruption | `gen` ids | `full` | `step` | `chunk` | first bad layer |
|---|---|---|---|---|---|---|
| C1 | mask offset `n_past + 1` (too large) | same | FAIL | **pass** | FAIL | 1 |
| C2 | cache write expanded after the block is built | same | pass | pass | pass | — (see below) |
| C2c | the real `kq` node ordered before the write | DIFF | FAIL | FAIL | FAIL | 1 |
| C3 | every layer reads layer 0's cross K/V | DIFF | FAIL | FAIL | FAIL | 2 |
| C4 | self V written in K's layout | same | FAIL | **pass** | FAIL | 1 |
| C5 | cumsum position rule in the cached path | same | FAIL | FAIL | FAIL | 0 (ref-pad only) |
| C6 | correct cache, reference decoded with cumsum | same | FAIL | FAIL | FAIL | 0 (ref-pad only) |
| C7 | mask offset `n_past - 1` (too small) | DIFF | **pass** | FAIL | FAIL | 1 |

Three findings in that table are worth carrying forward.

**C1 and C4 pass the `step` comparison.** A causal mask offset that is too
*large* still masks nothing when `Tn == 1`, and at `Tn == 1` the two candidate V
permutations produce the same bytes. Both corruptions therefore survive
single-token generation completely intact — ids identical, states identical to
4.5e-07 — and are caught only by feeding more than one token per call. The
`chunk` pass is not thoroughness, it is the only coverage those two lines have.

**C7 passes the `full` comparison,** because `full` runs at `n_past == 0` where
an offset that is too small clamps to correct. So the error is one-sided in both
directions and neither pass covers the other. Both are required.

**C2 did not fire, and that is a correction to a comment I wrote.** The first
draft of `nllb_decode_cached` claimed that moving the two
`ggml_build_forward_expand(cpy)` calls below the attention would silently read
the previous step's tail. It does not: nothing downstream is expanded during the
layer loop — `cur` and the logits enter the graph once, at the end — so the
copies are the only nodes inserted there and land ahead of every read wherever
in the block they sit. C2c is the shape that does break it: expanding the actual
`kq` node before the copies moves the read ahead of the write and fails 9 of 9 at
layer 1, rel 1.2e-01. The comment in the source now says that, and says which of
the two was measured.

### What this gate deliberately does not catch

A corrupted weight tensor leaves it **green**, because both sides of the
comparison read the same corrupted weight and it cancels. That is the division
of labour, not a hole: the decoder and generate gates own "are the weights and
the graph right", this one owns "does caching change the answer". Reading a
green cache gate as evidence about the model is the mistake it invites, and
`ci.yml` says so at the point where the other two gates are asserted to fire.

## `use_cache` is off by default, on purpose

`nllb_generate_params::use_cache` defaults to `false`. The un-cached path is the
one proven exact against HF on the real 600M, so it stays the reference the
cache is diffed against; a default that quietly switched paths would make
`nllb-test-generate` stop testing the thing its name and its step-4 result refer
to. `--use-cache` runs the same HF comparison through the cache, which closes
the loop directly instead of by composition, and CI runs both.

## Reproducing

```bash
source .venv/bin/activate

python tools/dump_reference.py         models/nllb-600M --out ref/ --lang eng_Latn --per-layer
python tools/dump_decoder_reference.py models/nllb-600M --out ref/ \
    --tgt-lang deu_Latn --per-layer --greedy --max-new-tokens 64

cmake --build build -j
./build/bin/nllb-test-generate models/nllb-600M-f32.gguf ref/ -t 8
./build/bin/nllb-test-generate models/nllb-600M-f32.gguf ref/ -t 8 --use-cache
./build/bin/nllb-test-cache    models/nllb-600M-f32.gguf ref/ -t 8 -v
```

`--max-new-tokens 64` rather than the default 24 because `04_long` and
`05_german` were both truncated at 24. At 64, `05_german` finishes on eos at 39
tokens and `04_long` still hits the cap at 65 — it wants more. Raising it added
54 greedy decisions (131 → 185 tokens) and **the smallest margin did not move**,
which is the strongest thing that has been said about the f16 headroom so far.

The synthetic fixtures are built with `max_position_embeddings = 64`, so do not
change the dumper's default to 64 without also pinning `--max-new-tokens 24` in
`ci.yml`: those models would generate 65 tokens, one past `n_ctx`, and the
`cap` stopping path would stop being exercised.

If a future run goes red, in order of likelihood:

1. **`full` is not exact.** The cache is wrong. Magnitude does not matter, only
   that the count is non-zero. Read the first-bad-layer line: 0 is the position
   rule, 0-clean-then-all-bad is cross-attention, clean-to-*k* is self-attention
   from block *k-1*.
2. **`step` well above `grow`.** The two should be within a small factor. If the
   cache is genuinely drifting more than the length alone accounts for, this is
   where it shows.
3. **Ids differ with `full` exact.** A near-tie. The generate gate's margin trace
   prices it.
4. **Only `chunk` fails.** The causal mask's `n_past` offset. Nothing else in the
   suite touches it.

## Still not proven

- **Beam search.** The cache is per-sequence; beams need one per beam plus a
  reorder step when the beams permute. It will also need real probabilities, so
  the "no softmax at the output head" shortcut ends there.
- **Reuse across sentences.** `nllb_kv_cache_reset` rewinds the self-attention
  half only, which is correct exactly when the source is unchanged. There is no
  guard that enforces it and no test that covers it — it exists for a future
  caller, and today every generation builds a fresh cache.
- **Other language pairs, and outputs past 65 tokens.** eng→deu only. `04_long`
  is still truncated.
- **Long contexts.** The cache is allocated at full capacity up front:
  `2 * L * n_ctx * D` floats, ~100 MB for the 600M at `n_ctx = 1024`. Generation
  sizes it to `max_new_tokens + 1`, so this does not bite yet.
- **Padded batches.** Unchanged from step 4 — source ids containing `pad_id` are
  still rejected up front, and cross-attention is still unmasked.
- f16, quantised K/V, Metal, `ggml_gallocr` — and see the per-step allocation
  hypothesis above, which says `ggml_gallocr` is now the biggest remaining win.
