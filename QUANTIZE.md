# Step 10, part 2 — the measurement, the new gate, `nllb-quantize`, an f16 KV cache

Repo copy of `claude/step10-progress.md`. Step 10's plan is
`claude/step10-quantisation-plan.md`.

**§0 IS THE ONE THAT MATTERS.** `./tools/step10_measure.sh` has now been run on
the real `facebook/nllb-200-distilled-600M` on the M1, which closes the single
outstanding action in the project. Read it before anything else here.

**§1 onward** cover §7.3 (the new assertion), §7.1/§7.2 (the C++ quantiser and
its mixes) and §7.5 (the f16 KV cache). Those were written and exercised in the
Claude sandbox against `tests/make_shape_model.py` — real 600M geometry, random
weights, no download — so unless a number says otherwise it is a claim about
**the code and the format**, not about the checkpoint. §0 is the exception and
is entirely real-model.

---

## 0. THE MEASUREMENT — RUN, on the real 600M, on the M1

`./tools/step10_measure.sh` has been run on `facebook/nllb-200-distilled-600M`,
8 threads, M1 Air. This closes the single outstanding action in the project. It
was run with the **pre-patch** script, so it carries no `[topk]`, no chrF and no
`--kv` line; everything below is the §7.0 ladder and the bench.

### The headroom ladder

| candidate | state rel | logit err | tightest headroom | unsafe | ids |
|---|---|---|---|---|---|
| f16 everywhere | 2.319e-03 | 3.697e-02 | **1.50x** @ 06_japanese pos 3 | 0/176 | exact 9/9 |
| Q8_0 `token_embd`, f16 rest | 6.567e-03 | 1.223e-01 | **0.45x** @ 06_japanese pos 3 | **3/176** | **8/9** |
| Q8_0 every 2-D weight | 2.997e-02 | 4.347e-01 | **0.19x** @ 05_german pos 8 | **13/176** | **8/9** |

**The f16 row reproduces step 7 to four significant figures** — 1.50x at
06_japanese pos 3, worst logit error 3.697e-02, 0 of 176 unsafe. The f32 file
was reconverted since then and the number did not move, which is the check that
this ladder's reference is the same object step 7 measured against.

### The prediction, scored

| mix | predicted | measured | verdict |
|---|---|---|---|
| Q8_0 `token_embd` | ~0.45x (cost 3.3x) | **0.45x** (cost **3.33x**) | **exact, to two significant figures** |
| Q8_0 everything | ~0.06x (cost 26.4x) | **0.19x** (cost **7.89x**) | **3.3x too pessimistic** |

So the shape fixture's ratio transferred **perfectly for the single-tensor mix
and overestimated the cost by 3.3x for the whole-model one**, and it erred in
the direction opposite to the one trap 29 warned about — the handoff said 3.3x
and 26.4x were *lower* bounds because trained weights have fatter tails than
`standard_normal`, and the whole-model number came in far under.

The pattern — exact at one tensor, 3.3x optimistic at 194 — points at
**accumulation, not at the tails**. On random iid weights the per-tensor errors
are independent and add in quadrature through 24 blocks; on a trained network
they are not independent, and some of the error a layer introduces is in
directions the next layer attenuates. That is a hypothesis with one data point
behind it, and the way to test it is the q6_k / q5_k_m ladder in §10: if the
fixture keeps over-predicting in proportion to the *number* of quantised
tensors, that is the mechanism.

### What actually broke, and it is one sentence

Both Q8 mixes emit different ids on **`05_german` and nothing else**.

```
q8emb  first difference at position 24 (step 23): 8688 vs 183051
       the reference's margin there was 1.822090e-02
       39 reference tokens -> 34 candidate tokens

q8all  first difference at position 11 (step 10): 566 vs 229190
       the reference's margin there was 3.449154e-02
       39 reference tokens -> 14 candidate tokens
```

**`1.822090e-02` is the project's known corpus minimum.** It has been quoted in
`claude/handoff.md` since step 4 as "the smallest logit margin over all steps:
1.822e-02 (05_german step 23)", it survived the re-dump from 131 to 185
decisions unchanged to four significant figures, and it is the exact decision
that breaks first under quantisation. The margin statistic identified the
failure three steps before it happened.

**The two failures are not the same kind of failure.** q8emb flips one near-tie
at the corpus minimum and produces a slightly shorter output. q8all diverges at
step 10 on a margin nearly twice as large and **collapses a 39-token
translation to 14** — it hits eos 25 tokens early. That is a truncation, not a
reordering, and no top-k or id-drift argument makes it acceptable.

Headroom below 1 by case: q8emb **2 of 9** (05_german 0.6x, 06_japanese 0.5x);
q8all **6 of 9** (00_hello 0.8x, 02_empty 0.6x, 04_long 0.3x, 05_german 0.2x,
06_japanese 0.5x, 08_repeated 0.3x).

### Trap 31, visible in the per-case output

At q8emb the worst layer is **layer 0** on seven of nine cases, at an identical
`3.49e-03` — layer 0 is the embeddings, and that flat number is the embedding
table's own quantisation error before any block has touched it. At f16 and at
q8all the worst layer is 12 on every case, because there the error accumulates
through the stack. Quantising the tied tensor puts error at the decoder
**input**, exactly as trap 31 says, and the gate's per-layer column shows it
without being asked.

### `full = exact` holds at Q8_0 on the real model

9 of 9, **0 differing floats**, arena reuse bit-identical, and the incremental
passes land on top of their control (`step` 1.337e-02 against `grow`
1.337e-02 on `04_long`). Trap 20's `max(tolerance, 4x the control)` rule doing
its job at a third precision. Four cases are bit-identical incrementally as
well.

### Speed — and the M1 lands where the bytes said it would

Reused-arena medians, 8 threads, `--src 24 --steps 64`:

| file | disk | cached step | lm_head | vs f16 | predicted from bytes |
|---|---|---|---|---|---|
| f16 | 1.242 GB | **18.52 ms** | 9.59 ms (52%) | 1.00x | 1.00x |
| q8emb | 0.996 GB | **13.94 ms** | 5.14 ms (37%) | **1.33x** | 1.36x |
| q8all | 0.662 GB | **10.59 ms** | 6.23 ms (59%) | **1.75x** | 1.88x |

**The lm_head alone hits 1.87x against 1.88x predicted** from 2 bytes → 1.0625.
That is the same result step 7 got for f32 → f16 (2.04x against a theoretical
2.04x): on this machine the step is bandwidth-bound and bytes removed convert
one-for-one into time removed. The sandbox's 1.21x / 1.47x were pessimistic
exactly as step 7 predicted they would be, for the same reason — a 2-core VM
that was never bandwidth-bound.

Two things in that table to be careful about:

* **q8all's lm_head reads 6.23 ms against q8emb's 5.14 ms for the identical
  tensor at the identical type.** 21% apart. The share is measured by
  differencing against `NLLB_LOGITS_NONE`, so it is not a direct timing, and in
  q8all everything around it got cheaper. Re-measure before quoting either
  number; do not explain it away.
* **The fixed costs are becoming visible.** Arena reuse is +0.9% of the step at
  f16 and **+5.1% at q8all**, because the step shrank and the 0.35 ms outside
  `compute` did not. `claude/step10-quantisation-plan.md` §4 predicted exactly
  this ("at ~6 ms/step that is 5%, not 0.9%"). At Q5_K-everything it will
  matter more, and it is the first argument for `ggml_gallocr` on CPU that is
  about speed rather than about memory.

### The verdict

**f16 stays the default.** 1.50x headroom, exact ids, 1.92x over f32 already.

**Neither Q8_0 mix is shippable as a default on this evidence.** q8emb is the
live candidate — 1.33x, 0.996 GB, and its only casualty is a coin toss at the
corpus minimum — but 0.45x headroom means "not provably safe", and whether it is
*acceptable* is a chrF question this run could not ask. q8all is out: a
39-token translation became 14 tokens.

**And the whole verdict rests on one sentence.** `05_german` is the only case
whose ids move, in both mixes, and it holds the corpus minimum margin. Nine
sentences, one language pair, 176 decisions. Widening that sample has been
filed under "cheap and worth doing alongside" since step 4; it is now the thing
the decision depends on.

---

## 1. The new assertion: `[topk]` in `nllb-test-precision`

Exact id equality is spent (trap 30). Its replacement is:

> **`[topk]`** — at every teacher-forced decision, the *reference's* argmax must
> still be inside the *candidate's* top k. k is small and fixed; default 4.

Ids are an **outcome** and reorder on a coin toss. The ranking is the arithmetic
itself. A quantised model is allowed to reorder a near-tie — that is what
"near-tie" means — and is not allowed to move the winner out of the top few,
because that is no longer rounding, it is a different model.

**Ranks use the project's tie rule** (trap 11): equal logits order by ascending
id, matching `torch.argmax` and `nllb_generate`'s strict `>`. Getting that wrong
would report rank 1 for a decision the candidate would in fact have made — a
gate lying in the direction that matters.

### `--allow-id-drift` is the switch, and it is not a default

Without it, **nothing changes**: exact ids remain the assertion and every f16
number in `claude/handoff.md` reproduces. With it, id drift is reported and
`[topk]` is what fails the run.

The flag is opt-in so that a green line always names the claim it is green
about. The PASS line says which:

```
PASS - exact ids AND [topk] at k=4.
PASS - [topk] held at k=4. NOT a claim of identical ids: 2 cases drifted.
```

### AND `[topk]` IS A BLUNT INSTRUMENT — measured, not assumed

This is the most important caveat the new gate carries, and it was found by
running `tests/corrupt_gguf.py --mode quantize` rather than by reasoning.

Round-tripping `token_embd.weight` through **Q4_0** on the shape fixture — a
relative weight perturbation of 8.1e-02, far past anything a shippable mix
would do:

| | f16 baseline | + token_embd through Q4_0 |
|---|---|---|
| tightest headroom | 5.73x | **0.26x** |
| decisions with headroom ≤ 1 | 0 of 96 | **1 of 96** |
| worst rank in `[topk]` | 0 | **0** |
| decisions whose argmax moved | 0 | **0** |
| ids | same | same |

**A perturbation that takes the headroom from 5.7x to 0.26x moves zero ranks.**
`[topk]` is a replacement for *exact id equality*, not a replacement for
`[margin]`. The gate now prints a loud paragraph saying exactly that whenever
`out-of-topk == 0` and `unsafe > 0`, and `--strict` turns an unsafe decision
into a failure.

The full ladder, one mid-stack tensor (`dec.blk.3.ffn_down.weight`)
round-tripped through progressively coarser types, f32 as the reference:

| candidate | injected max&#124;Δw&#124;/max&#124;w&#124; | tightest headroom | ranks moved |
|---|---|---|---|
| f16, untouched | — | 5.73x | 0 of 96 |
| f16 + ffn_down via Q8_0 | 3.8e-03 | 3.33x | 0 |
| f16 + ffn_down via Q5_0 | 3.7e-02 | 2.06x | 0 |
| f16 + ffn_down via Q4_0 | 7.7e-02 | 1.57x | 0 |
| f16 + **token_embd** via Q4_0 | 8.1e-02 | **0.26x** | 0 |

Two readings. The headroom audit degrades monotonically and is the sensitive
instrument. And the last row against the fourth is trap 31 in one line: the same
size of perturbation costs **6x more headroom** when it lands on the tied
embedding table than on a mid-stack FFN matrix.

### `--kv f32:f16`

The weights' precision is a property of the file; the KV cache's is a property
of the run. So the KV comparison passes **the same file twice** and differs only
in `nllb_kv_type`. It also switches which decoder produces the teacher-forced
logits — `nllb_decode_cached` on **both** sides, because the un-cached path has
no KV cache to be f16. With both sides at f32 KV the gate still uses the
un-cached `nllb_decode`, so every historical number reproduces.

---

## 2. chrF in `tests/verify_translation.py`

The product-level half of §7.3, behind the same `--allow-id-drift`.

**chrF, not BLEU**: character n-gram F-score degrades smoothly when one token
changes, and it needs no tokenizer of its own — which matters when the target
script is Devanagari or Gujarati and a word tokenizer would be a second thing to
be wrong.

**Implemented in-file, with no dependency**, because running with no
dependencies is this gate's existing property. When `sacrebleu` happens to be
importable it is used as a **cross-check** and the delta is printed and gated at
0.01. Measured here:

```
corpus  mine=92.300186  sacrebleu=92.300186  delta=0.00000000
per-segment mismatches: 0 of 6      (incl. an empty pair and a Devanagari pair)
```

**The cross-check earned its keep immediately.** The first implementation
averaged precision and recall over all six orders. sacrebleu uses **effective
order** — an order in which either side has no n-grams does not contribute and
is not counted in the divisor — and the two disagreed by **50 points** on a
corpus of short sentences, because a three-character segment has no 4-, 5- or
6-grams at all. Written from the paper, that bug ships.

Consequence to know: two *empty* strings have effective order zero and score
**0.0**, not 100. `02_empty` is one of the nine goldens, so the metric keeps
sacrebleu's answer and the **gate** handles it — a byte-identical segment is
counted as identical and is exempt from the per-segment floor, because "close
but wrong" is not a thing it can be.

### Two floors, and why one is not enough

`--min-chrf 95.0` (corpus) and `--min-seg-chrf 75.0` (worst non-identical
segment). Measured, not argued: under the `lead-space` mutation this gate
reports **corpus chrF 100.00 with a worst segment of 16.67**. Summed statistics
are dominated by long segments, so a destroyed short one vanishes into the
corpus figure.

Calibration:

| | chrF |
|---|---|
| identical | 100.00 |
| one word changed in a long sentence | 96.45 |
| `</s>` left in the string | 91.17 |
| language code left in the string | 83.87 |
| **leading space kept** | **100.00** |
| a clause dropped | 39.11 |
| the wrong language entirely | 0.00 |

**chrF cannot see whitespace at all** — it strips it, which is what lets it work
on Japanese. A stray leading space scores 100.00 and `[golden]` is blind to it.
`[wiring]` catches it structurally, which is why that pass exists.

### The mutation matrix now has two columns

`tests/fake_cli.py --matrix` runs every breakage under **both** assertions:

```
mutation                rc  strict: passes that fired         rc  --allow-id-drift
(none - honest stand-in)   0  -                                  0  -
swap                     1  wiring                             1  wiring
ignore-tgt               1  wiring                             1  wiring
keep-eos                 1  golden, stdin, wiring              1  golden, wiring
keep-lang                1  golden, stdin, wiring              1  golden, wiring
lead-space               1  golden, stdin, wiring              1  golden, wiring
stdout-noise             1  wiring                             1  wiring
drop-empty               1  stdin                              1  stdin
no-ids                   1  wiring                             1  wiring
shrug                    1  errors                             1  errors
join-argv                1  errors                             1  errors

PASS - 10 mutations, all caught by the strict gate
```

All ten survive the loosening. The strict column is unchanged from step 9, which
is the check that the gate did not lose sensitivity while gaining a mode.

---

## 3. `tools/nllb-quantize.cpp` — and the project's first K-quant

A CMake target linking ggml directly (it never builds a graph). It reads an
f32/f16 GGUF, applies a **named mix**, and writes a quantised one via
`ggml_quantize_chunk`.

```
./build/bin/nllb-quantize --list-mixes
./build/bin/nllb-quantize in.gguf out.gguf q5_k_m [--dry-run]
./build/bin/nllb-quantize in.gguf out.gguf q5_k_m --tensor-type ffn_up=q4_K
```

**A mix is a per-tensor policy, not a type** — `--type q5_k` would be a lie
about what the argument does. The mixes are data, not code, so they print
themselves:

| mix | base | exceptions |
|---|---|---|
| `f16` | F16 | — |
| `q8_0` | Q8_0 | — |
| `q8emb` | F16 | `token_embd` → Q8_0 |
| `q6_k` | Q6_K | — |
| `q5_k_s` | Q5_K | — |
| `q5_k_m` | Q5_K | `token_embd`, `*ffn_down*` → Q6_K |
| `q4_k_m` | Q4_K | `token_embd`, `*ffn_down*` → Q6_K |

`token_embd` is protected because of trap 31 and because §1's ladder measured
the cost of not protecting it. `*ffn_down*` because it reads the relu output,
whose dynamic range is one-sided and wide. **`position_embd` is never quantised
by any mix**: 4 MB, not learned (the converter recomputes it, trap 1), and its
rows exist to be distinguishable from one another. Every 1-D tensor keeps its
input type. `--tensor-type` overrides all of that, which is the point of having
it.

**Trap 32 is handled with one call and then asserted.** Metadata is copied with
`gguf_set_kv(out, in)` — never a hand-written list of keys, because the key a
future step adds is exactly the key a hand-written list omits — and the tool
then **re-opens what it wrote** and checks every key survived with the same
type, every tensor kept its name, order and shape, and every tensor has the type
the plan said. On by default; `--no-verify` exists and should not be used.

Refusals rather than guesses: a row length that is not a multiple of the target
block size (trap 34 says this cannot happen for NLLB, so the message says either
this is not NLLB or the geometry changed, and that padding is not the answer); a
type needing an importance matrix; and an **already-quantised** source tensor,
because dequantise-then-requantise compounds two lots of block error and reads
as one.

**Measured, shape fixture, f16 in:**

```
in:   models/shape-600M-f16.gguf     510 tensors, 23 metadata keys
mix:  q5_k_m (base q5_K)
plan: q5_K 168, q6_K 25, f32 316, f16 1 (position_embd)
size: 1.148 GB -> 452.5 MB  (2.60x)   [tensor data; on disk 1.233 GB -> 474.5 MB]
[verify] 23 metadata key(s) survive with the same type; 510 tensors keep name,
         order and shape; every tensor has the type the plan said.  ok
```

and then, on that Q5_K file:

```
./build/bin/nllb-test-cache models/shape-600M-q5km.gguf ref-shape/ -t 2
  full = exact, 0 differing floats
  arena reuse bit-identical, 0 differing floats
  worst incremental state rel = 1.092e-02, control = 1.100e-02
  PASS
```

**`full = exact` holds at a K-quant.** That is trap 16's assertion at a
precision nothing in this project had ever produced, and the control being
*larger* than the incremental number is trap 20's rule doing its job for the
third precision in a row.

### A bug found in `tools/requantize_gguf.py` on the way

**Every file it has ever written is missing `general.file_type`.** The key is on
its `WRITER_OWNED` list — excluded from the copy loop because the writer is
supposed to produce it — and the writer was never told to. The trap-32 assertion
could not see it, because it checks `set(reader.fields) - SKIP_KEYS` and
`FILE_TYPE` is *inside* `SKIP_KEYS`.

Nothing in `src/nllb.cpp` reads the key, so no file ever broke, which is exactly
why it survived. Fixed, and the lesson is the one trap 32 already states and the
code did not obey: **a key you exclude from a copy because something else owns
it is a key you must then check that something else actually wrote.**
`nllb-quantize`'s verify pass checks for it explicitly.

---

## 4. The f16 KV cache — a memory change, not a speed one

`nllb_kv_cache_init` takes a `kv_type` (`NLLB_KV_F32` default, `NLLB_KV_F16`),
`nllb_generate_params` carries it, and `nllb-cli` gained `--kv-f16`,
`nllb-bench` `--kv-f16`, `nllb-test-cache` `--kv-f16`, `nllb-test-precision`
`--kv f32:f16`. The library change is small: one `ggml_type`, threaded through
four `ggml_new_tensor_3d` calls and one arena size. `ggml_cpy` already converts
f32 → f16 into a strided destination and `ggml_mul_mat` already takes an f16
`src0`, so the graph needed nothing — the same precision-agnosticism step 7
found for the weights.

### It changes what `test_cache.cpp` can assert, and that needed care

`full = exact` holds because both sides do the same arithmetic on the same
bytes. At f16 KV the cached side rounds K and V and the un-cached side does not,
so the two **cannot** be bit-identical, and demanding it would be trap 19 in a
new costume.

So `--kv-f16` **adds a fifth pass** rather than changing the four:

* `[full]` still runs at f32 KV and is still held to bit-identity. **Red there
  is a cache bug.**
* `[kv16]` is the identical single cached call with K/V at f16, **measured and
  never gated**. `[full]` is its control in the exact sense trap 20 means: it
  contains no f16, so anything `[kv16]` has that `[full]` does not is the f16 KV
  cache and nothing else.

```
shape_00_len6   [kv16] same single cached call, K/V at f16: rel=2.342e-04  (full=exact)
shape_01_len12  [kv16] ...                                  rel=2.222e-04  (full=exact)
shape_02_len24  [kv16] ...                                  rel=2.501e-04  (full=exact)
shape_03_len40  [kv16] ...                                  rel=2.700e-04  (full=exact)
```

Priced in the units that decide the output, `--kv f32:f16` on the same file:

```
worst state divergence = 2.700e-04     worst logit error = 7.160e-04
tightest headroom      = 195.16x       96 decisions, 0 unsafe, all rank 0
```

195x on **this fixture**, whose f16-weights headroom is 5.73x on the same four
cases — so f16 KV is roughly **34x cheaper in headroom than f16 weights here**.
That ratio is the transferable part; against the real model's 1.50x it suggests
f16 KV is comfortably affordable, and it is a suggestion until the real
checkpoint says so.

### AND IT IS NOT FASTER. Do not turn it on for speed.

Sandbox, 2 threads, `--src 24 --steps 32`, reused arena, medians:

| | ms/step | lm_head |
|---|---|---|
| f16 weights, **KV f32** | **29.87** | 18.31 ms |
| f16 weights, **KV f16** | **30.85** | 17.88 ms |

Slower, by about 3%, which is inside this box's noise but is certainly not a
win. The arithmetic says why and it is not a sandbox artefact: at 32 decoder
tokens the whole self-attention cache is
`12 × 2 × 32 × 1024 × 4 B ≈ 3 MB` against **~930 MB** of weights read per step,
and the lm_head alone is 54–65% of that. Halving 3 MB cannot move a 930 MB step,
and the f32 → f16 conversion of K and V is new work on the critical path. Even
at 256 tokens the cache is ~25 MB — under 3%.

**What it is for is memory.** Step 10 measured 99 MB of dirty pages with the
weights mmap'ed, nearly all of it the ~101 MiB graph arena; the KV cache is the
next allocation down, and f16 halves it. That is the case for the flag, and it
is why the flag is **off by default** — an arithmetic change that costs headroom
and buys no time should be asked for by name.

---

## 5. `tools/step10_measure.sh`, rewritten

Six sections instead of five, and the two Q8 runs **no longer end in `|| true`**.
They pass `--allow-id-drift`, which means their red is a failure again rather
than a shrug — there is a real assertion to make now, which there was not when
the script was written. The f16 run keeps exact id equality, because it held in
step 7 and a run asserting less than the truth is how a regression gets in.

New in the script: `3b` (`--kv f32:f16`), `4b` (`nllb-test-cache --kv-f16`), a
fourth `nllb-bench` line at KV f16, and section `6`, `verify_translation.py` at
f16 (byte equality) and at both Q8 mixes (chrF).

---

## 6. `tests/make_shape_ref.py` — new, and deliberately loud about its limits

The shape model never had a `ref/`, so `nllb-test-cache` and
`nllb-test-precision` could not run against it without the real checkpoint's
goldens. This writes `manifest_gen.txt` and the `.ids.bin` files and **nothing
else** — no `.gen.bin`, so `nllb-test-generate` refuses to run rather than
passing against a target it invented. That is the step-4 SKIP lesson applied in
advance: a fixture that silently satisfies a gate it cannot support is worse
than a missing one.

It supports the C++/C++ comparisons and gate plumbing. It supports **no headroom
number**, nothing the tokenizer or CLI touches (the shape model has no
vocabulary), and nothing about what the model says.

---

## 7. A fourth harness that lied toward green

`claude/handoff.md` catalogues three. Here is the fourth, and it happened while
measuring `[topk]`'s sensitivity.

`tests/corrupt_gguf.py --mode quantize` on `token_embd.weight` held the f32
copy, the packed blocks and the dequantised result at once — about 2.5 GB — and
was **OOM-killed**. The probe script piped it into `head -2`, and `set -e`
inspects only the last command of a pipeline, so the kill was invisible: the
gate then ran against an **uncorrupted file** and printed PASS, with numbers
identical to the f16 baseline to three significant figures.

What gave it away was that identity. The tell was not an error; it was a result
that was *too exactly* the control.

Fixed twice over: the round-trip now works in 4096-row blocks (block quantisers
work per row, so chunking changes no byte), and the probe checks the exit code.
**Every harness needs a check that the measured process actually did the work** —
a negative duration, a 2 ms translation, a matrix row where every mutation
blames the same pass, and now a corruption whose numbers match the baseline.

---

## 8. Traps to add

**37. An f16 KV cache changes what the cache gate can assert, and the answer is
to add a pass, not to loosen one.** `full = exact` holds because both sides do
the same arithmetic on the same bytes; at f16 KV the cached side rounds K and V
and the un-cached side does not, so bit-identity is unavailable *by
construction*. `nllb-test-cache --kv-f16` therefore keeps `[full]` at f32 KV as
a required assertion and adds `[kv16]` as a measured, never-gated quantity, with
`[full]` as its control. Red at `[full]` is a cache bug; a number at `[kv16]` is
f16. And note the thing the flag does **not** buy: at 32 decoder tokens the
whole cache is ~3 MB against ~930 MB of weights read per step, so f16 KV
measured 3% *slower* and is a memory change only.

**38. `[topk]` replaces exact id equality, NOT the margin audit.** Measured:
round-tripping `token_embd` through Q4_0 collapses the tightest headroom from
5.73x to 0.26x and puts a decision inside the error band while leaving all 96
ranks at 0 and every id unchanged. Top-k agreement is coarse by construction —
it asks whether the two models agree about which tokens are plausible, not
whether this decision could have gone the other way. Read `[topk]` and
`[margin]` together, and use `--strict` when the claim is that a decision could
not have flipped.

**39. A key excluded from a copy because "the writer owns it" is a key you must
check the writer wrote.** `tools/requantize_gguf.py` put `general.file_type` on
its `WRITER_OWNED` list and never called `add_file_type`, so every file it wrote
for the whole of step 10 lacked the key — and its own trap-32 assertion could
not see it, because the assertion compares `set(reader.fields) - SKIP_KEYS` and
the key is inside `SKIP_KEYS`. Nothing reads it in this project, which is why it
survived. An exclusion list is an assertion about someone else's behaviour.

**40. chrF cannot see whitespace, and a corpus figure hides a destroyed short
segment.** It strips whitespace by design — that is what lets one metric work on
German and Japanese — so a stray leading space scores 100.00 against the golden.
And summed statistics are dominated by long segments: measured, `lead-space`
gives corpus chrF 100.00 with a worst segment of 16.67. Hence two floors, and
hence `[wiring]`'s structural checks staying the thing that catches a
whitespace bug.

---

## 9. Calibration ladder — rows to add

| comparison | worst |
|---|---|
| **cached vs un-cached, same length, Q5_K_M (shape fixture)** | **bit-identical, 0 floats** |
| f16 KV vs f32 KV, same file, same length (shape fixture) | 2.700e-04 relative |
| f16 KV headroom cost vs f32 KV (shape fixture) | 195x headroom, 0 of 96 unsafe |
| f16-weights headroom, same 4 cases (shape fixture) | 5.73x |
| f16 + `dec.blk.3.ffn_down` via Q8_0 (shape fixture) | 3.33x, 0 ranks moved |
| f16 + `dec.blk.3.ffn_down` via Q4_0 (shape fixture) | 1.57x, 0 ranks moved |
| **f16 + `token_embd` via Q4_0 (shape fixture)** | **0.26x, 0 ranks moved** |
| **our chrF2 vs sacrebleu, corpus and per segment** | **byte-identical, delta 0.00000000** |

---

## 10. What to do next, in order

1. **Re-run `./tools/step10_measure.sh` with this patch applied.** §0's run
   used the pre-patch script, so it produced no `[topk]` histogram, no chrF and
   no `--kv` line — and chrF is the number that decides whether q8emb ships,
   because "34 tokens instead of 39 on one sentence" is not yet a quality
   judgement. The re-run costs one command and answers it.
2. **Build a real Q5_K file and put it through the same three gates.** §0 makes
   this the most interesting thing on the list rather than merely the next one:
   the damage is concentrated exactly where trap 31 said it would be, and
   `q5_k_m` puts `token_embd` at Q6_K, whose 16-weight sub-block scales are the
   direct answer to "one outlier sets the step size for 32 neighbours". Q6_K is
   FEWER bits than Q8_0 (6.5625 vs 8.5), so whether finer scale granularity
   beats coarser quantisation is a question to measure here, not to take on
   llama.cpp's authority.
   ```
   ./build/bin/nllb-quantize models/nllb-600M-f32.gguf \
       models/nllb-600M-q5km.gguf q5_k_m
   ./build/bin/nllb-cli -m models/nllb-600M-q5km.gguf --list-langs | wc -l
   ./build/bin/nllb-test-cache models/nllb-600M-q5km.gguf ref/ -t 8
   ./build/bin/nllb-test-precision models/nllb-600M-f32.gguf \
       models/nllb-600M-q5km.gguf ref/ -t 8 --allow-id-drift
   python tests/verify_translation.py ./build/bin/nllb-cli \
       models/nllb-600M-q5km.gguf ref/ -t 8 --allow-id-drift
   ```
   Quantise from the **f32** file, not the f16 one: the f16 route has already
   lost the mantissa bits the quantiser would have used to choose its scales.
3. **Re-run `tests/corrupt_gguf.py --mode quantize` on the real checkpoint**, on
   `token_embd` and on a mid-stack `ffn_down`, at q8_0 / q5_0 / q4_0. The
   sandbox ladder is a claim about the gate; on the real model it is a claim
   about the model, and the real model's activations are 4000x larger.
4. **Dump a second language pair. This has stopped being optional.** §0's entire
   quantisation verdict turns on `05_german`, the one case out of nine whose ids
   move and the holder of the corpus minimum margin. One `--tgt-lang hin_Deva`
   invocation of each dumper roughly doubles the decision pool and adds the
   first non-Latin target script. Until that exists, every headroom and chrF
   number in this document is a statement about nine English sentences
   translated into German.
