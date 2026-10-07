# Batching (step 11, rebuilt on the step-13 backend)

B sources through one encoder pass and **one decoder step per token for all of
them**. The cached step at B = 1 reads the whole decoder plus the 256206-row
lm_head (~0.9 GB at f16) to produce one token; at B it reads the same bytes for
B tokens. Nothing else on the roadmap divides the weight traffic.

The original step-11 code was lost (a `git stash` that was never promoted to a
branch). This is a rebuild from `docs/dev/step11-batching.md`, written against
today's `ggml_backend_sched` code rather than the pre-step-13 arena code the
stash was written for.

## Design decisions

**New functions, not a rewrite.** `nllb_encode_batch`, `nllb_kv_cache_init_batch`,
`nllb_decode_cached_batch`, `nllb_generate_batch` and `nllb_plan_batches` sit
next to the single-sequence functions, which are unchanged. Every gate that
passed on Metal and CUDA still tests the code it passed on, and
`nllb-dump-states --diff` between the build before and after this change reports
**IDENTICAL, 912 records, 0 differ** at f32 and at f16. The cost is two graph
builders to keep in step; `tests/test-batch.cpp` is what holds them together.

**Layout.** Activations carry every row (`[D, T*B]`), so each weight matmul is
one GEMM. Attention carries the batch as ggml's `ne[3]`, so each row is computed
in its own slice. The KV cache is `[HD, n_ctx, H, B]` / `[n_ctx, HD, H, B]`.

**Right padding.** The encoder position rule makes a pad shift every LATER
position; pads at the end have nothing after them (traps 2 and 12).

**Two new masks**, both uploaded as backend input tensors (no `->data`):

| mask | shape | what it hides |
|---|---|---|
| encoder self-attention | `[Ts, Ts, 1, B]` | pad KEYS, broadcast over queries. A pad QUERY row is not masked (an all `-inf` row softmaxes to NaN); its garbage is hidden by the next mask |
| cross-attention | `[Ts, 1, 1, B]` | each row's pad source columns, every step. Built from `nllb_kv_cache::src_len`, which lives on the cache so no caller can pass a different one |

Pad rows of the encoder output are **not** zeroed: that would make a missing
mask almost right instead of loudly wrong.

**Finished rows** are fed `pad_id` (HF parity) and their outputs discarded.
They are NOT dropped from the batch; `token_slots - tokens_kept` reports what
that costs.

**`enc_max_tokens` is gone.** It was step 11's workaround for the hand-written
arena formula (3401 MiB asked at B = 8, Ts = 96; `ggml_init` returning null at
Ts = 200). gallocr sizes from tensor lifetimes: B = 16, Ts = 24 on the 600M's
geometry needs **9.0 MiB** of encoder compute buffer.

## The gate: `nllb-test-batch`

Seeded random sources, no `ref/`, no goldens, no torch.

| pass | claim | result (shape fixtures, CPU) |
|---|---|---|
| `[plan]` | every input planned once; both budgets respected | ok |
| `[b1]` | batch of one = `nllb_generate`: ids, stop, every step's logit and margin | **bit-identical** |
| `[equal]` | equal-length rows vs single calls, every step's logits | **0 differing floats** (4, 8, 16 rows; f32, f16, KV f16; vocab 4000 and 256206) |
| `[pad1]` | one padded row, no batch: the control | measured, never gated |
| `[pad]` | unpadded row bit-identical; padded rows within an ABSOLUTE tolerance; ids exact | padded rows **equal to their control** to every printed digit |
| `[perm]` | the padded batch reversed | **bit-identical** |
| `[mask]` | a different filler token behind the masks | **bit-identical** |
| `[eos]` | an extra row stopping at step 0 changes nothing else | **bit-identical** |

`[pad]` is held to an absolute tolerance, never to "N x the control": `[pad1]`
builds a mask of its own, so a broken mask inflates both (trap 44).

## Does the gate fire? `tests/corrupt_batch.py`

Ten mutations of the batched code, each rebuilt with the project's own CMake.
**The table is identical at f32 and at f16:**

```
mutation                 plan  b1    equal pad   perm  mask  eos
no-cross-mask                              FAIL        FAIL
no-encoder-mask                            FAIL        FAIL
cross-mask-off-by-one                      FAIL        FAIL
encoder-mask-off-by-one                    FAIL        FAIL
mask-from-row-zero                         FAIL  FAIL
kv-write-batch-stride                FAIL  FAIL  FAIL  FAIL  FAIL
argmax-from-row-zero                       FAIL
never-finish                                                 FAIL
pos-ignores-pad                                                     (inert, expected)
feed-last-not-pad                                                   (inert, expected)
```

Every mask bug is caught twice, by `[mask]` (no threshold) and `[pad]`. The two
inert mutations are trap 43: a finished row cannot reach a running one. If
either ever fires, something has coupled the rows.

## Speed (decoder step only; `nllb-test-batch MODEL --bench 1,2,4,8,16`)

600M geometry (`make_shape_model.py`, f16), 2-core Linux VM, src 24 tokens:

| B | ms/step | ms/token | vs B=1 | compute buffer |
|---|---|---|---|---|
| 1 | 58.98 | 58.98 | 1.00x | 1.0 MiB |
| 2 | 79.76 | 39.88 | 1.48x | 2.0 MiB |
| 4 | 108.14 | 27.04 | 2.18x | 3.9 MiB |
| 8 | 176.47 | **22.06** | **2.67x** | 7.9 MiB |
| 16 | 329.36 | 20.58 | 2.87x | 15.8 MiB |

The same 2.67x at B = 8 the original step 11 measured. Read ms/token: the ratio
is bounded by the arithmetic floor and hides where it is. Re-run on the target.

## Using it

```bash
# CLI, stdin only, opt-in. Output order = input order; not streamed.
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --tgt deu_Latn --batch 8 < in.txt > out.txt

# server: batching is ON, up to 16 lines per decode (--batch N to change, 1 = off)
./build/bin/nllb-server -m models/nllb-600M-f16.gguf
```

## On the M1 Air, real 600M f16 (2026-10-05)

**The product claim holds.** `verify_translation.py ... --batch 4` on Metal:
9/9 byte-identical to the HF goldens, chrF 100.00, every pass green.

**CPU gate:** PASS. `[equal]` bit-identical on 4 real-geometry rows; padded rows
2.515e-03 against a control of 2.515e-03 (tol 5e-3); ids 4/4 exact.

**Metal gate:** `[perm]`, `[mask]`, `[eos]`, `[b1]` bit-identical. `[equal]`
and the unpadded row of `[pad]` differed from the single path by ~3e-7
relative (f32 reordering), ids exact. Cause, read off ggml's own pipeline log:

| columns | weight matmul kernel Metal picks |
|---|---|
| 1 | `kernel_mul_mv_f16_f32_4` (matrix x vector) |
| 2-5 | `kernel_mul_mv_ext_f16_f32_r1_N` (N columns per weight read) |
| 6-8 | plain `mul_mv` - no ext kernel, no mm: the SLOW band |
| > 8 | `kernel_mul_mm_f16_f32` (tiled; inputs rounded to half, trap 48) |

Different kernels sum in a different order. So on a GPU the gate now holds the
single-path comparisons to 1e-5 (rows <= 8) or the padded tolerance (rows > 8,
the half path) and prints `[MTL0 kernel selection, ...]`; `--strict` restores
bit-identity. The batched-vs-batched passes are unchanged on every backend.

**Speed** (`--bench 1,2,4,8,16`, 8 threads, src 24, decoder step):

| B | CPU ms/token | vs B=1 | Metal ms/token | vs B=1 |
|---|---|---|---|---|
| 1 | 17.09 | 1.00x | 16.82 | 1.00x |
| 2 | 9.23 | 1.85x | 9.56 | 1.76x |
| 4 | 5.38 | 3.18x | 4.66 | 3.61x |
| 8 | 3.88 | 4.41x | 4.28 | 3.93x |
| 16 | **3.38** | **5.06x** | **1.88** | **8.96x** |

Where it comes from: t(B) ~ 14.6 ms + 2.5 ms x B on the CPU. At B = 1 ~85% of a
step is streaming ~0.86 GB of weights (~50 GB/s, near the M1's ceiling); batching
shares that read across B rows. The GPU wins at 16 because by then the step is
arithmetic-bound and it has more arithmetic. Metal's B = 8 is the kernel gap in
the table above (34 ms/step, slower than B = 4).

**The server defaults to `--batch 16`.** A group is whatever is queued, so the
6-8 row band will still occur on Metal under some loads; mapping it exactly is
`--bench 4,5,6,7,8,9,12,16 --backend metal`.

## Still open

1. The 6-8 row Metal band: measure it, then decide whether padding such a group
   to 9 rows (into the mm kernel) is worth its half-precision inputs.
2. Mixed language pairs in one batch are supported (per-row forced token) and
   exercised by the server, but the gate uses one target id for every row.
3. Finished rows are not dropped from a running batch.
