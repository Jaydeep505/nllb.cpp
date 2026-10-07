# Step 7: f16 weights

```
tests/test-precision.cpp   the f16 gate: exact ids + a per-decision margin audit
tests/test-decoder.cpp     says why it is the wrong instrument for a non-f32 GGUF
tests/test-cache.cpp       incremental passes now judged against the CONTROL, not
                           an absolute tolerance; the control is no longer gated
examples/bench/bench.cpp     byte accounting from ggml_nbytes, not dims x 4
tests/make_shape_model.py  --outtype f16
CMakeLists.txt             + bin/nllb-test-precision
```

Project copy of this document: `docs/dev/step7-cpp-f16.md`.

## Status: f16 runs, it is ~1.65x faster, and it needed no library changes at all

Step 6 measured the cached decode step and found **54–65% of it is one matmul** —
the `[256206, 1024]` output projection, reading 0.98 GiB of f32 per token, at a
memory bandwidth the CPU cannot exceed. The only lever on bandwidth-bound work
is moving fewer bytes.

`models/convert-nllb-to-gguf.py` already had an `--outtype f16` path, written in
step 1 and never once run. It works, and so does everything downstream:
`src/nllb.{h,cpp}` needed **zero changes**. ggml's `mul_mat` takes an f16 `src0`
against f32 activations, and `ggml_get_rows` on an f16 embedding table returns
f32, so the graph is precision-agnostic as written.

Real-600M *shape*, f32 vs f16, 2 threads, medians over 20 steps, three runs:

| | f32 | f16 | |
|---|---|---|---|
| whole cached step | 98.8 / 99.8 / 99.5 ms | 60.2 / 59.5 / 62.4 ms | **~1.65x** |
| the lm_head alone | 56.9 / 59.8 / 60.3 ms | 36.6 / 35.9 / 31.5 ms | ~1.7x |
| file on disk | 2.30 GiB | 1.15 GiB | exactly half |

**It is not the full 2x, and the reason is worth keeping.** Effective bandwidth
goes *down*: f32 moves 1.73 GiB in 99 ms (17.9 GB/s), f16 moves 0.87 GiB in
60 ms (14.8 GB/s). Halving the bytes moved the workload partway out of the
bandwidth-bound regime step 6 found it in, so the remaining time is
increasingly the parts that are not weight streaming — softmax, the norms, and
the activations, which are still f32. Expect the same shape from quantisation:
the first halving buys the most.

## The decoder gate is the wrong instrument, and now says so

Pointing `nllb-test-decoder` at an f16 GGUF fails all nine cases at
`7.978e-04`. That number is not a bug — it is almost exactly the `8.9e-04` the
handoff independently recorded as f16's error, measured a different way.

The gate's tolerances are the **float32 noise floor**: `rel ~1e-7` is
arithmetic, `rel ~1e-2` is a bug. f16 sits at ~8e-04 — three orders above the
floor and four below a real defect — so every case fails and the failure
carries no information.

Loosening the tolerance to 1e-03 would be worse than useless: it would then
pass a genuine graph error of exactly the size this gate exists to catch. So
the gate keeps its f32 tolerances, keeps failing, and **explains itself**:

```
!! this GGUF stores its weights as f16, and this gate's tolerances are the FLOAT32
   noise floor. Expect every case to FAIL at roughly 1e-03, which is what f16 costs
   and NOT evidence of a bug - a real defect lands at 1e-02 and flips the argmax.
   Do not loosen the tolerance to make this green: at 1e-03 it would also pass a
   genuine graph error. Verify reduced precision with the gate built for it, which
   asserts on ids rather than on distances:
     ./build/bin/nllb-test-precision <f32.gguf> models/deep-f16.gguf <ref-dir>
```

A red gate that means "this is f16" and reads as "this is broken" is a trap for
whoever runs it next. It costs six lines to remove.

## The new gate, and the argument it replaces

`tests/test-precision.cpp` diffs one model against itself at two precisions. No
torch, no goldens beyond the source ids — same trick as the cache gate.

```
./build/bin/nllb-test-precision <reference.gguf> <candidate.gguf> <ref-dir> [-t 8] [-v]
```

| pass | held to |
|---|---|
| `[gen]` | **exact id equality** — the only thing a user of a translator can observe |
| `[state]` | reported, **not gated** — f16 moves the numbers, that is what f16 is |
| `[margin]` | the per-decision audit below |

### The margin audit is the point

The project's case for f16 has been two aggregate numbers side by side: *"the
smallest greedy margin is 1.822e-02, f16's error is 8.9e-04, so there is 20x of
headroom."* That compares the **worst** margin against the **worst** error as if
they happened at the same decision. Nobody checked whether they did. It is an
argument, not a measurement.

This gate measures it per decision. Teacher-force the reference's own output
through both models with logits at every position; at each position the argmax
cannot change while, for **every** other token *i*:

```
(logit[best] - logit[i])  >  err[best] + err[i]
```

The winner can only fall by its own error and a rival can only rise by its own.
The gate reports the tightest such ratio over the row.

**Two drafts of this were wrong, and the difference is large.** The first used
the row's *maximum* error for both terms — an upper bound, but it overstates
the error at the two tokens actually competing, and it cried wolf: it reported
headroom of 1.1x and 1.4x on cases that are in fact at 10x and 13x. The
quantifier also matters: checking only the runner-up would miss a distant token
with a large error overtaking, which is why it ranges over every token.

### What it found

On the deep synthetic fixture, f16 reproduces the f32 ids exactly, 9 of 9 — and
the gate still refuses to call that evidence:

```
tightest headroom      = 0.40x at 01_quick_brown pos 15
      pos 15   margin 4.215e-04  row err 1.668e-03  headroom 0.40x  argmax agreed

!! 1 of 216 greedy decisions had headroom <= 1 - the argmax COULD have flipped.
  THIS FIXTURE CANNOT SUPPORT A CLAIM ABOUT f16, however green it looks.
```

That `4.215e-04` is the number `nllb-test-generate` independently reports as the
deep model's smallest margin, at the same case and the same step — two tools
agreeing from different directions. It is also the handoff's long-standing
"synthetic scare" (`deep synthetic 12+12 | 4.2e-04 | below it`), now localised
to a position rather than quoted as an aggregate.

**So: on this fixture the ids match by luck, and a suite that only checks ids
cannot tell.** That is the whole reason this gate exists. The claim about f16
has to rest on the real 600M, whose margin is 1.822e-02 — and that run has not
happened yet (see "Not proven").

## Found on the way: the cache gate printed its control and never used it

`tests/test-cache.cpp` has always said, in prose:

> Read the two together: the cache is only responsible for whatever the first
> number has that the second does not.

and then gated `step` and `chunk` against an **absolute** tolerance, using the
`grow` control for nothing but display. That works only while the absolute
number happens to suit the checkpoint. On an f16 model it stops working
completely: all three incremental passes fail at ~2.7e-04 against a 5e-05
tolerance — **the control included, at the identical number, with `full` still
exact**. The gate reported "the cache is broken" for a fact about f16.

Worse, the control was itself being *gated*. It contains no cache; there is
nothing it could be failing at. It is the yardstick.

Both fixed:

- the incremental passes are held to `max(tolerance, 4x the control)`, so the
  gate means the same thing at any weight precision;
- the control is measured and never gated;
- `full` keeps bit-identity, which needs no tolerance and **still holds at f16**
   — 0 differing floats, so the cache's layout, writes and reads are exact
  regardless of weight type.

The logit tolerance needed a different yardstick, for a reason the file already
documented elsewhere: `grow`'s last iteration decodes the whole prefix, which is
bit-for-bit the reference call, so its logits are equal *by construction* and
are zero. Useless as a control. The state control is used instead, with argmax
equality as the hard assertion.

### The gate did not lose sensitivity

All four step-6 corruptions still fire, with the identical pass-by-pass
signature, **at both precisions** — they land three to four orders above the
control, so a 4x factor cannot hide them:

| corruption | `gen` | `full` | `step` | `chunk` | `arena` |
|---|---|---|---|---|---|
| reads graph memory nobody wrote | DIFF | exact | FAIL | FAIL | FAIL |
| mask offset `n_past + 1` | same | FAIL | pass | FAIL | pass |
| self V in K's layout | same | FAIL | pass | FAIL | pass |
| the real `kq` before the write | DIFF | FAIL | FAIL | FAIL | FAIL |

## Also fixed: nllb-bench was reporting f32 byte counts for an f16 model

The header line computed `dims x 4`, so it printed `0.98 GiB read per step` for
a model reading half that. On a change whose entire mechanism is "move fewer
bytes", that is the one number that must not be wrong. It now uses
`ggml_nbytes` on the real tensors and names the type:

```
lm_head is 256206 x 1024 f16 = 0.49 GiB read per step; decoder weights are 0.38 GiB (f16)
```

## Reproducing

```bash
python models/convert-nllb-to-gguf.py models/nllb-600M \
    --outfile models/nllb-600M-f16.gguf --outtype f16

# the assertion: same ids, and every decision provably safe
./build/bin/nllb-test-precision models/nllb-600M-f32.gguf \
                                models/nllb-600M-f16.gguf ref/ -t 8

# ids against HF, through the cache, at f16
./build/bin/nllb-test-generate models/nllb-600M-f16.gguf ref/ -t 8 --use-cache
./build/bin/nllb-test-cache    models/nllb-600M-f16.gguf ref/ -t 8

# the speed claim
./build/bin/nllb-bench models/nllb-600M-f32.gguf -t 8 --src 24 --steps 64
./build/bin/nllb-bench models/nllb-600M-f16.gguf -t 8 --src 24 --steps 64
```

Do **not** run `nllb-test-decoder` against an f16 file expecting green. It will
fail, correctly, and tell you why.

## Verification

38 gate invocations across `ref-tiny`, `ref-eos`, `ref-deep` and `ref-pad`, at
both precisions — 0 failures. f32 keeps the full suite; f16 gets generation,
generation through the cache, the cache gate and the precision gate.

## Still not proven

- **f16 on the real 600M.** Everything here is synthetic. The prediction: ids
  identical to f32 9 of 9, and **zero** decisions inside the error band, because
  the real model's smallest margin is 1.822e-02 against an f16 logit error
  around 1e-03 — roughly 10x of headroom where the deep synthetic has 0.4x.
  Until that run happens, the f16 claim rests on a fixture the gate itself says
  cannot support it.
- **The f16 KV cache.** Only the *weights* are f16. `nllb_kv_cache_init` still
  allocates `GGML_TYPE_F32` for K and V, so the cache is unchanged and its
  bit-identity assertion is unaffected. Trap 16 says an f16 cache must still be
  held to bit-identity against an f16 full pass at equal length, and that is a
  separate change with a separate argument.
- **Quantisation.** f16 is the first halving; Q8_0 would quarter the lm_head and
  Q5_K more. The bandwidth number above predicts diminishing returns, which is
  worth measuring rather than assuming.
- **Whether f16 is the right default.** It is not wired to anything: the
  converter takes `--outtype`, and nothing chooses for the user.
