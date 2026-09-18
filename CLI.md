# Step 9 — the CLI, and the first text → text claim

`nllb.cpp` takes a sentence and prints a translation. No Python, no goldens at
runtime, one GGUF.

```bash
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf \
    --src eng_Latn --tgt deu_Latn "Hello, world!"
```

**Status: written, gated, and NOT YET RUN against the real checkpoint.** Every
line below about how the binary behaves on `facebook/nllb-200-distilled-600M`
is a prediction until `tests/verify_translation.py` has been run on a machine
that has the weights. The Claude sandbox cannot: 2.47 GB, `huggingface.co`
blocked. What *has* been run there is the build (clean under `-Wall -Wextra`),
every argument-error path that does not need a model, and the gate itself
against a stand-in — see "Testing the gate" below.

## Why this is not glue

Every gate in this repo stops at ids.

| gate | proves |
|---|---|
| `nllb-test-tokenizer` | text → ids |
| `nllb-test-decoder` | ids → states |
| `nllb-test-generate` | ids → ids, exact against HF `generate()` |
| `nllb-test-cache` | ids → ids, cached == un-cached |
| `nllb-test-precision` | ids → ids, f16 == f32 |

**Nothing has ever compared text to text.** The CLI is the first thing to cross
that seam, so composition of green gates is not evidence about it. The failure
modes it can have are exactly the ones nothing above can see:

1. **The target code at the wrong position.** It belongs at the first
   *generated* position (trap 10), not in the encoder input. A CLI that put it
   in the source would still emit a fluent sentence.
2. **`--src` and `--tgt` swapped, or one ignored.** Both are strings that index
   the same vocabulary; nothing about the types distinguishes them.
3. **Metaspace detokenization** — a lost or doubled leading space. `▁` → `' '`
   and then *one* leading space is dropped; off by one either way is invisible
   in ids.
4. **The eos, the decoder start token or the language code surviving into the
   printed string.** `skip_special_tokens` is a decision, not a default.

## What shipped

```
cli/nllb-cli.cpp                    the binary
tools/dump_translation_reference.py HF translations -> ref/translations.txt
tests/verify_translation.py         THE GATE. Runs the binary, diffs stdout.
tests/fake_cli.py                   a stand-in + 10 breakages, to test the gate
```

Scope is deliberately small — a positional text argument or stdin, `--src`,
`--tgt`, `--list-langs`, `-t`, `-n`, `--no-cache`, `-v`. **No batch mode.** Each
input line is an independent call with its own encoder pass and its own fresh
KV cache. Reusing a cache across sentences means `nllb_kv_cache_reset`, which
rewinds only the self-attention half and is correct exactly when the source is
unchanged — something nothing in the repo enforces or tests. Amortising the
model load across lines is free; amortising the cache is not, so it is not done.

Three smaller decisions worth knowing:

* **Only the translation goes to stdout.** Banners, ids, timings and warnings
  go to stderr, which is what lets the gate compare stdout byte for byte and
  lets a user pipe the output without filtering it. `-v` adds nothing to stdout.
* **The cache is ON by default here**, the opposite of `nllb_generate_params`.
  The library default stays off so the step-4 gate keeps testing the un-cached
  path it made its claim about; a user-facing binary wants the fast path, and
  step 5 proved the two are bit-identical at equal length.
* **`--list-langs` does not load the weights.** `nllb_vocab_load` opens the
  GGUF `no_alloc = true`, so it costs the same on the 2.46 GB checkpoint as on
  the 6.7 MB vocab-only file.

## The gate

`tools/dump_translation_reference.py` is the only part that needs torch. It
writes `ref/translations.txt`: a `# key<TAB>value` header, then one case per
line with the source, the translation and the generated ids JSON-encoded so a
tab or a newline in a prompt cannot break the format. The prompts are
**imported** from `tools/dump_reference.py` rather than copied — two lists that
must agree and are maintained separately eventually do not.

It also records two things about itself:

* **`pipeline(...)` vs `generate()` + `batch_decode(skip_special_tokens=True)`.**
  The goldens are the second, because that is the operation the CLI performs.
  The first is compared against it so a pipeline post-processing quirk cannot
  be attributed to `nllb.cpp`, and `pipeline_agrees` in the header says whether
  they ever differed.
* **`generation_config.num_beams`.** The open question from the handoff: if the
  checkpoint's default is above 1, then every "exact against `generate()`"
  result in this project is exact against a *non-default* configuration. The
  dumper forces `num_beams=1` and records what the default was, so the answer
  ends up written down instead of assumed. **Read that line on the first real
  run.**

`tests/verify_translation.py` needs no torch. Four passes, different blind
spots:

| pass | what it runs | what only it can see |
|---|---|---|
| `[golden]` | one process per case, text as an argument | the claim itself: typing a sentence prints the right translation |
| `[stdin]` | all cases in one process, one line each | the stdin path, and that lines stay in correspondence |
| `[wiring]` | `-v`, checking ids **by position**, then with the flags swapped | `--src`/`--tgt` swapped or ignored — and it works on a random model, because it asserts about structure, not about output quality |
| `[errors]` | a bad `--tgt`, a bad `--src`, a missing `--tgt`, an unquoted sentence | that bad input fails instead of translating something else |

A missing golden file is a **FAILURE**, not a skip — the step-4 lesson, where a
SKIP that was not a FAIL meant deleting eight of nine goldens still printed
PASS. `--no-golden` / `--no-stdin` / `--no-wiring` / `--no-errors` leave a pass
out on purpose.

**On a mismatch the gate bisects.** It re-runs the case with `-v`, parses the
`out ids` line and says which side moved:

* ids identical → the divergence is in **detokenization**, look at
  `nllb_vocab_decode`;
* ids differ at index 0 → the decoder start token (eos, not bos — trap 8);
* ids differ at index 1 → the **forced target code** (trap 10);
* ids differ later → an ids-level divergence, which `nllb-test-generate` owns.

That is why `-v`'s `src ids (N):` and `out ids (N):` lines have a stable
format. Changing them changes the bisect.

## Testing the gate

This gate can only run where the real weights are, which is exactly where a
broken gate is least likely to be noticed. So `tests/fake_cli.py` is an honest
stand-in for `nllb-cli` that looks the answers up, plus ten deliberate
breakages — each a mistake the real CLI could plausibly make. No weights, no
torch, about a second:

```bash
python tests/fake_cli.py --make-goldens ref-fake/
python tests/verify_translation.py tests/fake_cli.py MODEL ref-fake/   # PASS
python tests/fake_cli.py --matrix ref-fake/
```

```
mutation                rc  passes that fired
----------------------------------------------------------------------
(none - honest stand-in)   0  -
swap                     1  wiring
ignore-tgt               1  wiring
keep-eos                 1  golden, stdin, wiring
keep-lang                1  golden, stdin, wiring
lead-space               1  golden, stdin, wiring
stdout-noise             1  wiring
drop-empty               1  stdin
no-ids                   1  wiring
shrug                    1  errors
join-argv                1  errors

PASS - 10 mutations, all caught
```

Read the first two rows: **`swap` and `ignore-tgt` are caught only by
`[wiring]`.** Against a stand-in that looks its answers up, a wrong language
code at position 1 does not change the printed text, so `[golden]` sees
nothing. On the real model it would — but that is a fact about the model's
output, not about the gate, and `[wiring]` is the pass that catches it
structurally. `drop-empty` is caught only by `[stdin]`, and `stdout-noise`
only by `[wiring]`, which is the argument for all four existing.

Two bugs this harness found in its own gate, both worth naming because both
made the gate quietly weaker rather than loud:

* `[wiring]` checked "the printed text has no leading space" against
  `p.stdout.strip()` — which removes the very space it was looking for. The
  `lead-space` mutation was caught by `[golden]` and `[stdin]` and the check
  that existed specifically for it stayed green.
* The matrix attributed every FAIL to the last `[section]` header seen, and the
  summary line `FAIL - n check(s)` starts with FAIL, so every mutation looked
  like it had fired `[errors]`. Same shape as the bug in the first version of
  `tests/corrupt_tokenizer.py`: a harness that mis-reports *which* pass fired
  is how a load-bearing pass gets deleted as redundant.

## Running it for real

Two things left over from step 8b, both to do **before** the first CLI run:

```bash
# 1. the f16 checkpoint predates the grapheme tables. Reconvert, then confirm
#    the gate says `source: THE GGUF` rather than the compiled-in fallback.
python tools/convert_nllb_to_gguf.py models/nllb-600M \
    --outfile models/nllb-600M-f16.gguf --outtype f16
./build/bin/nllb-test-tokenizer models/nllb-600M-f16.gguf ref/ -v

# 2. settle the cross-UCD question: this should print nothing.
pip install regex && python tools/gen_grapheme_tables.py > src/nllb-grapheme-data.h
git diff --stat src/nllb-grapheme-data.h
```

Then:

```bash
cmake --build build -j

# the goldens (the only step that needs torch)
python tools/dump_translation_reference.py models/nllb-600M --out ref/ \
    --src-lang eng_Latn --tgt-lang deu_Latn --max-new-tokens 64

# THE GATE
python tests/verify_translation.py ./build/bin/nllb-cli \
    models/nllb-600M-f32.gguf ref/ -t 8

# and at f16, which is the file you would actually ship
python tests/verify_translation.py ./build/bin/nllb-cli \
    models/nllb-600M-f16.gguf ref/ -t 8
```

**Expect it to find something.** Four seams cross here for the first time and
none of them has ever been checked. Treat any difference as a finding, not a
rounding — and if the gate is green on the first run, be suspicious of the
gate before believing the code, which is what `tests/fake_cli.py --matrix` is
there to answer.

Note that `f32` and `f16` are two separate runs of the same gate, not one:
`nllb-test-precision` proves the two produce identical *ids*, and identical ids
through the same detokenizer produce identical text — so a difference between
those two runs would mean something is wrong upstream of the CLI, and the
precision gate is where to look.
