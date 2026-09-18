# Step 6: where the cached decode step's time actually goes

```
src/nllb.h              nllb_step_timing, nllb_kv_cache::scratch / reuse_scratch
src/nllb.cpp            per-phase timers in nllb_decode_cached, reused graph arena
tests/bench_cached.cpp  the profile: phase split, lm_head price, cold vs warm arena
tests/make_shape_model.py  a random GGUF with the real 600M's SHAPE, no torch
tests/test_cache.cpp    + the [arena] pass: reused arena vs fresh, bit-identical
CMakeLists.txt          + bin/nllb-bench, and -Wall -Wextra on our own targets
```

Project copy of this document: `claude/step6-cpp-graph-alloc.md`.

## Status: measured. `ggml_gallocr` is NOT worth writing, and here is the clock

Step 5 closed with a hypothesis dressed as a plan:

> the cached path *should* run near 16 ms/token and it runs at 38. The likely
> culprit is `nllb_decode_cached` allocating a fresh ~72 MB ggml context every
> step — a malloc plus several thousand page faults per token. […] Put a timer
> around `ggml_init` before acting on it.

The timer is now around `ggml_init`. **`ggml_init` is 0.02 ms — 0.02% of the
step.** The 22 ms gap it was supposed to explain is not there. Almost all of it
is one matmul.

`facebook/nllb-200-distilled-600M`, f32, **M2 Air, 8 threads**, medians over 64
steps — the machine and the checkpoint that actually matter:

| phase | ms/step | share |
|---|---|---|
| `ggml_init` (the arena) | **0.00** | 0.0% |
| graph build | 0.24 | 0.7% |
| **`ggml_graph_compute`** | **35.68** | **99.1%** |
| readback (memcpy out) | 0.04 | 0.1% |
| `ggml_free` | 0.05 | 0.1% |

and inside that 99%, from running the identical step with `NLLB_LOGITS_NONE`:

| | ms/step |
|---|---|
| whole step | 36.0 |
| step with the lm_head switched off | 16.4 |
| **the lm_head alone** | **19.6 (54%)** |

Across runs and both machines the lm_head is **54–65%** of the cached step. The
other ~40% is the twelve decoder blocks. Everything an allocator could possibly
touch — `ggml_init`, graph construction, readback, `ggml_free` — is **0.33
ms/step combined, 0.9%**, and it is remarkably stable.

So the answer to "where does the cached step's time go" is: **55% output
projection, 44% decoder blocks, 1% everything else.**

The development numbers, taken in a 2-core Linux sandbox on a shape-faithful
weight-random GGUF, said the same thing with different constants: `ggml_init`
0.02 ms, build 1.08, compute 88.60, non-compute 1.6–1.7 ms. Everything
transferred except the arena reuse's saving — see that section.

### The measured number does not depend on the thread count, and that is what makes it portable

The sandbox has 2 cores and the MacBook has 8, so a percentage measured here
does not transfer. An absolute overhead does, and this is the check:

| phase | 1 thread | 2 threads | |
|---|---|---|---|
| compute | 189.80 | 97.96 | **1.94x — scales** |
| graph build | 1.02 | 1.09 | flat |
| `ggml_init` | 0.02 | 0.02 | flat |
| readback | 0.33 | 0.35 | flat |
| `ggml_free` | 0.18 | 0.36 | flat |

Only `compute` moves. The prediction was that the ~1.7 ms would therefore hold
on the MacBook too. **It came out better than predicted: 0.33 ms of a 36 ms
token, 0.9%.** So the ceiling on step 5's entire proposed optimisation is under
one percent on the machine that matters. It cannot be 22 ms on any machine,
because it is not a function of how fast the machine is.

## What step 5 got right, and what it got wrong

Right: there *is* a per-step allocation cost, and it *is* invisible where you
would first look for it.

Wrong in two specific ways, both worth keeping:

1. **`ggml_init` does not touch the memory it returns.** With
   `mem_buffer = nullptr` it is a `malloc`, glibc serves ~80 MB from a fresh
   `mmap`, and the kernel does nothing until the pages are written. So the cost
   of a fresh arena is *not* in `ggml_init` — it lands inside
   `ggml_graph_compute`, mixed in with the arithmetic and indistinguishable
   from it. A timer around `ggml_init` alone would have reported 0.02 ms and
   been read as "no allocation cost at all", which is also wrong.
2. **"several thousand page faults per token" is 1040**, from `getrusage`. And
   the reason is the interesting part: the arena is *sized* at 81 MiB but a step
   *touches* about 4 MiB of it. The sizing formula is a worst-case bound with a
   64 MB constant in it; the graph never comes close.

That second point is what decides how to fix it, and it is the thing arithmetic
from weight sizes could not have told you.

## The change: keep the arena, do not pre-touch it

`nllb_kv_cache` now carries the graph arena across steps
(`scratch` / `reuse_scratch`, default on). `ggml_free` does not free a buffer it
did not allocate — `ggml.c` sets `mem_buffer_owned = params.mem_buffer ? false :
true` — so one allocation serves the whole decode loop.

| | ms/step (median) | minor faults/step |
|---|---|---|
| fresh arena every step | 94.06 | 1027 |
| **arena reused** | **90.88** | **72** |

3.4% and 93% of the page faults, *in the sandbox*. Interleaved rather than run
back to back: run in sequence the second loop inherits the first's allocator
state, and any thermal or scheduling drift over a few seconds lands entirely on
one of them.

**On an M2 Air the time saving disappears, and that is the honest headline.**
Same binary, same model, medians:

| | cold | reused |
|---|---|---|
| real 600M, 8 threads | 34.72 ms | 34.77 ms |
| shape model, 8 threads | 35.71 ms | 35.66 ms |

A dead heat both times. The mechanism the saving came from is first-touch page
faults, and macOS on Apple silicon uses 16 KiB pages: the fault count is 399/step
there against 1037 in a 4 KiB-page Linux VM, and Apple's are cheap. What reuse
still buys on that machine is 353 fewer faults per step, 0.17 ms off build and
free, and no per-token 80 MB malloc — all real, none of it a measurable speed-up.

So: **keep the change, do not sell it as a speed-up.** It is bit-identical, it
removes work, and on some platforms that work was worth something. The reason
this is stated so carefully is in "Found on the way" below — the first version of
this document claimed 2–4% on the strength of a summary line that was computing
from means.

**I got this wrong twice first, and both mistakes are instructive.**

**Mistake 1: growing the buffer to exactly what the step asks for.** `mem` is
not constant across steps — the per-layer term carries `12 * Tn * n_kv * H` and
`n_kv` grows by one every token — so a buffer grown to exactly `mem` is short
again on the very next step. It reallocated every single token. Measured: **50
ms/step and 20694 minor faults/step**, half as fast again as not reusing at all.
It reads like "reuse is a bad idea" and is actually one line being wrong. The
buffer now grows by a quarter.

**Mistake 2: pre-zeroing the arena.** The first version used
`std::vector<char>::assign`, which writes every byte — pre-touching all 81 MiB
so no faults land in `compute`. But a step touches ~4 MiB, so that pays for 20x
the pages the work needs: **~360 ms up front against a ~4 ms/step saving, which
does not break even until ~90 tokens** — longer than most translations, and a
straight loss on all nine golden cases. Handing over an *untouched*
`new char[]` buffer gets the same win with no up-front bill: the first step
faults in exactly the pages it uses and every later step finds them warm. That
is the difference between the 20694-faults row and the 56-faults row.

## The verdict on `ggml_gallocr`

After the arena reuse, what is left for a reused graph plus `ggml_gallocr` to
recover, measured on the M2 Air against the real checkpoint:

| | ms/step |
|---|---|
| graph build | 0.12 |
| readback (a memcpy `gallocr` does not touch) | 0.04 |
| `ggml_init` + `ggml_free` | 0.00 |

**0.16 ms/step, against a step of 36 ms.** Under half a percent. Even before the
arena reuse — i.e. what `gallocr` alone would have been worth against the step-5
code — the whole category is 0.34 ms, under one percent.

That is not worth the change. `ggml_gallocr` means a graph that lives across
steps, and trap 15 says the thing that breaks this graph is precisely a
reordering of the cache write against the cache reads — "reusing a graph across
steps is exactly the change that can move a cache read ahead of its write". The
gate would catch it (C2c below still fails 9 of 9 at layer 1), but paying that
risk for 0.3 ms is a bad trade, and the 1.6 ms it targets was measured rather
than inferred.

**The next real optimisation is the lm_head, because that is where 60% of the
step is.** It is a `[256206, 1024] x [1024, 1]` matvec reading 0.98 GiB of f32
per token. Two honest observations about it:

- It is not yet bandwidth-saturated *here*: 1 → 2 threads takes it from 122.2 to
  59.9 ms, a 2.04x, i.e. perfect scaling. At 2 threads that is 16.4 GB/s.
- On the MacBook it is closer to the wall: ~60% of a 38 ms token is ~23 ms for
  1.05 GiB, about 45 GB/s against an M2's ~100 GB/s.

So f16 (or a quantised lm_head) should nearly halve the dominant 60%, and is
worth more than everything in this document put together. That makes step 5's
"next steps" item 5 the real item 1. Note the constraint it inherits: trap 16
says an f16 cache must still be bit-identical to an f16 full pass at equal
length, and the 1.822e-02 greedy margin is what says f16's 8.9e-04 has room.

## The new gate pass: `[arena]`

Reusing the arena changes what a latent bug does. A fresh `mmap` reads as zero,
so an op that read graph memory it had not written would silently see `0.0`
today — and see **the previous step's values** once the arena is reused. That is
a real behavioural change and it needed its own assertion.

None of the four existing passes covers it. `full` feeds the whole prefix in one
call, so its arena is fresh no matter what the flag says. `step` and `chunk` are
held to a relative tolerance a small stale read could hide under.

So `[arena]` runs the identical stepping pass twice — once with reuse, once
without — and holds the two to **bit-identity**. Same graph, same inputs, and an
allocator has no business changing a float.

```
reusing the graph arena across steps is BIT-IDENTICAL to allocating a fresh one -
  0 differing floats. Nothing in the graph reads memory it did not write, which is
  what makes keeping the arena safe rather than merely faster.
```

### Testing the tests

Four corruptions on the deep synthetic fixture. C1, C4 and C2c reproduce step
5's recorded behaviour exactly, which is the check that the gate did not lose
sensitivity while gaining a pass:

| # | corruption | `gen` ids | `full` | `step` | `chunk` | `arena` |
|---|---|---|---|---|---|---|
| C8 | reads graph memory nobody wrote | DIFF | **exact** | FAIL | FAIL | **FAIL** |
| C1 | mask offset `n_past + 1` (trap 13) | same | FAIL | **pass** | FAIL | pass |
| C4 | self V in K's layout (trap 14) | same | FAIL | **pass** | FAIL | pass |
| C2c | the real `kq` expanded before the write (trap 15) | DIFF | FAIL | FAIL | FAIL | FAIL |

C8 is the corruption `[arena]` exists for, and note that **`full` stays exact on
it** — the single-call pass reads the same zeros the un-cached path does.

C1 and C4 do not trip `[arena]`, which is right: they are deterministic bugs,
unaffected by where the memory came from.

**Two notes on writing C8 and C2c, because both took a wrong first draft.**

`ggml_new_tensor` bump-allocates in graph order, so a tensor created *before*
the layer loop lands at the same offset every step and nothing ever writes
there — it keeps the zeros the first `mmap` handed back and reads identically
with and without reuse. The probe has to be allocated **late**, after the loop,
where its offset falls on bytes an earlier step really wrote. The first version
was allocated early and was invisible to every pass.

And adding a spare `ggml_mul_mat` that reads the cache before the write does
**nothing** — its result is unused and the `kq` that feeds the output is still a
later node. What actually moves the read ahead of the write is
`ggml_build_forward_expand` on the node the block really uses. That is the
corruption step 5 described, and the first draft here was not it.

## Found on the way: the build had no warnings turned on

The `[arena]` report shipped its first version with a `printf` missing its first
argument — a `%-20s` reading a `long` as a pointer. It is invisible on the
passing path and a **segfault on the failing one**, so a gate that had just
correctly diagnosed a corruption crashed instead of reporting it, and the
corruption read as MISSED.

`-Wformat` catches that, and it comes free with `-Wall`. `CMakeLists.txt` had
neither. It now sets `-Wall -Wextra` on this project's own targets and leaves
`vendor/ggml` to its own flags. A gate that crashes is worse than no gate,
because it fails in the direction of looking green.

## Found on the way: the summary line was computing from means

The first version of `nllb-bench` reported the cold-vs-reused arena difference
from the **means** of the per-step totals. On the M2 Air that printed

```
-> reusing the arena saves 2.09 ms/step of 39.48 (5.3%)
```

while the medians of the same two runs were 35.71 and 35.66 — a dead heat. The
5.3% was one or two outlying steps, and it went into the first draft of this
document as a real result.

A decode step has a long right tail that has nothing to do with the code: the
first touch of a 2.3 GB model file, and a fanless laptop throttling partway
through 64 steps that each stream a gigabyte. Both inflate a mean and leave the
typical step alone. Three fixes, all in `tests/bench_cached.cpp`:

- **medians are the headline**, means are a footnote;
- **the spread is printed** — median, mean, p90, max — with a marker when the
  mean runs more than 15% above the median, so the reader can see *why* the
  median is the number being quoted;
- **a noise floor.** A difference under 0.5% of the step now prints as "inside
  the noise floor ... a wash on time" rather than as a number with two decimal
  places that invites belief in all of them;
- **untimed warm-up steps** before each measured loop, so the timed ones are not
  paying to page in the model. Without it the first block's mean ran ~30% above
  its median for reasons entirely external to the thing being measured.

The general lesson is the same one this whole step is about: a number that is
easy to compute is not the same as a number that answers the question. Step 5
computed a per-step cost from weight sizes; this tool computed a difference from
means. Both were arithmetic, both were confidently wrong, and both were caught
only by measuring the thing directly.

## Reproducing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# a random GGUF with the real 600M's geometry - no torch, no download.
# vocab 256206 is the point: on a 2000-word fixture the lm_head is not the
# bottleneck and the profile answers a different question.
python tests/make_shape_model.py models/shape-600M-f32.gguf

./build/bin/nllb-bench models/shape-600M-f32.gguf -t 8 --src 24 --steps 64

# and on the real checkpoint, which is the number that counts:
./build/bin/nllb-bench models/nllb-600M-f32.gguf -t 8 --src 24 --steps 64
./build/bin/nllb-test-cache models/nllb-600M-f32.gguf ref/ -t 8 -v
```

`nllb-bench` prints the phase split, the lm_head's share, the cold-vs-warm arena
comparison and the un-cached curve for scale.

## Caveats — read these before quoting any absolute number

- **Two machines, and they disagree about one thing.** The development numbers
  are the Claude sandbox: 2 cores, 4 KiB pages, f32, random weights at the real
  600M's shape. The confirmation numbers are an M2 Air, 8 threads, 16 KiB pages,
  against the real checkpoint. The phase split, the lm_head's share and the
  thread-independence of the overhead transfer cleanly. **The arena reuse's time
  saving does not** — 3.4% in the sandbox, a wash on the Mac. Quote the Mac.
- **The weights are random.** That is fine for timing — a matmul does not care
  what the floats are — and fine for `nllb-test-cache`, which compares two C++
  paths so the weights cancel. It is not fine for anything that reads what comes
  out, and `make_shape_model.py` says so.
- **The correctness runs were on the four synthetic fixtures**, all green: 20 of
  20 gate invocations (decoder, generate, generate `--use-cache`, generate
  `--check-modes`, cache) across `ref-tiny`, `ref-eos`, `ref-deep`, `ref-pad`,
  plus the encoder gate. `full = exact`, 0 differing floats, everywhere.
  **The real 600M still has to be run locally.**

## Still not proven

- ~~**The real checkpoint.**~~ **Run, on an M2 Air with 8 threads against
  `facebook/nllb-200-distilled-600M`.** `ggml_init` **0.00 ms**, graph build
  **0.24 ms**, non-compute overhead **0.33 ms of a 36 ms token (0.9%)**, lm_head
  **54-59%**. Every prediction held or came out better than predicted; the one
  thing that did NOT transfer is the arena reuse's time saving (3.4% in the
  sandbox, a wash there — see the arena section). `nllb-test-cache` on the real
  goldens: 9 of 9, `full = exact`, the arena comparison 0 differing floats, and
  step 5's drift figures reproduced to four significant figures (4.128e-07
  against a 4.608e-07 control).
- **Any machine that is neither of those two.** The split should hold wherever
  the lm_head is 57% of the bytes read per token, which is a property of the
  checkpoint rather than the host. The arena reuse is the part that is
  platform-dependent, and the tool now says so rather than printing a number.
- **The arena's memory cost under beam search.** The scratch is ~101 MiB and
  hangs off the cache, so one cache per beam is one arena per beam. Beams could
  share a single workspace; nothing does that yet, and nothing needs to until
  beam search exists.
- **`nllb_kv_cache_reset` and the scratch.** Reset rewinds `n_past` and
  deliberately leaves the scratch alone, which is right — it holds no cache
  state. Untested, like the rest of reset.
- f16, quantised K/V, Metal — and `ggml_gallocr`, now with a measured reason not
  to bother rather than an assumption that it was the biggest win left.
