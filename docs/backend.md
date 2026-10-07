# Step 13: ggml-backend — Metal first, the CPU as fallback

Until this step every graph ran through `ggml_graph_compute_with_ctx`, the CPU
executor, on a bump arena sized by hand-written formulas. Metal was found at
configure time, linked, and never used. This step moves the whole library onto
`ggml-backend`: weights and the KV cache live in backend buffers, every graph is
scheduled by `ggml_backend_sched`, every intermediate is placed by `gallocr`,
and data crosses the host boundary only through `ggml_backend_tensor_set` /
`_get`.

```bash
./build/bin/nllb-cli --list-backends
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --tgt deu_Latn "Hello, world!"            # auto: Metal if built with it
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --tgt deu_Latn --backend cpu "Hello!"     # the reference
NLLB_BACKEND=metal ./build/bin/nllb-test-cache models/nllb-600M-f16.gguf ref/ -t 8         # any gate, on Metal
```

**Status: built and verified in the Linux sandbox on the CPU backend only.
Nothing here has run on Metal.** The sandbox has no GPU. Every Metal claim below is either a
reading of the ggml-metal source or a prediction for the M1 to confirm, and is
labelled as such. The acceptance run is at the end of this file.

## The six host-memory lines, and what replaced them

| before (CPU-only) | after |
|---|---|
| `ggml_init(no_alloc = false)` per graph, sized by formula | `no_alloc = true` context for structs + graph; `ggml_backend_sched_alloc_graph` places the data |
| `ggml_graph_compute_with_ctx(ctx, gf, n_threads)` | `ggml_backend_sched_graph_compute`; `n_threads` goes to the CPU backend |
| `gguf_init_from_file(no_alloc = false)` / `->data = mmap + off` | always `no_alloc = true`; the mapping is wrapped in ONE backend buffer (`ggml_backend_dev_buffer_from_host_ptr`) and each tensor placed in it with `ggml_backend_tensor_alloc`; without mmap, `ggml_backend_alloc_ctx_tensors_from_buft` + chunked `ggml_backend_tensor_set` |
| `memcpy(inp->data, ...)` | `ggml_set_input` + `ggml_backend_tensor_set` after allocation |
| `memcpy(out, t->data, ...)` | `ggml_set_output` + `ggml_backend_tensor_get` |
| KV cache in a `ggml_context` with data | structs in a `no_alloc` context, bytes in a buffer on the primary backend |

Plus one change to the graph itself: the **cached** step's causal mask is now a
mask *tensor* passed to `ggml_soft_max_ext`, not `ggml_diag_mask_inf`, because
Metal does not implement `DIAG_MASK_INF` and the scheduler would bounce every
layer's `kq` to the CPU and back. The **un-cached** decoder deliberately keeps
`ggml_diag_mask_inf` — see "The mask" below.

## Choosing a backend

| request | meaning |
|---|---|
| `cpu` | ggml's CPU backend. Bit-identical to every pre-step-13 build. |
| `metal`, `gpu` | the first GPU device. **Fails** if there is none — it never silently falls back. |
| `auto` | the first GPU if the build has one, else the CPU. |
| a device name | as `--list-backends` prints it: `MTL0`, `CPU`, ... |
| empty | `$NLLB_BACKEND` if set, else `cpu`. |

**The library default is `cpu`; the CLI default is `auto`.** Same reasoning as
`use_cache` (library off, CLI on): the library default is the configuration
every gate was measured on, and a default that silently moved them onto other
kernels would make every gate stop testing what its name says. The product
should use the GPU it has.

`NLLB_BACKEND` is read by `nllb_model_load` whenever the caller passes no
backend, so **every existing gate runs on Metal with no new flags**:
`NLLB_BACKEND=metal ./build/bin/nllb-test-cache ...`.

**The CPU is always the scheduler's last backend**, which is what makes it the
fallback: an op the GPU backend does not implement is run on the CPU with the
copies in and out inserted automatically. `NLLB_ARENA_TRACE=1` prints the
number of splits per graph; `splits 1` on a GPU backend means nothing fell back.

BLAS/Accelerate (`ACCEL` devices) are deliberately not scheduled: the old path
never used them, and adding them would change which kernels the CPU reference
runs.

## Evidence (sandbox, x86, 2 cores, CPU backend)

### Bit-identity against the pre-port build

`nllb-dump-states` (new) writes everything the public API computes — encoder
states, teacher-forced decoder states and logits under both position rules,
both generation paths with their margin traces, and every cached step's states
and logits at chunk 1 and chunk 3 — and `--diff` compares two dumps. Built
against the pre-port library and the new one, same GGUF:

| fixture | configuration | result |
|---|---|---|
| small shape model, f32 | default | **IDENTICAL, 912 records, 9.16 MB, 0 differ** |
| small shape model, f16 | default | **IDENTICAL** |
| small shape model, f16 | `--kv-f16` | **IDENTICAL** |
| small shape model, f32 | `--no-mmap` (read path) | **IDENTICAL** |
| real-geometry shape model (vocab 256206), f16 | default | **IDENTICAL, 432 records, 79.5 MB** |

So on the CPU the port changed **no float anywhere the API can see**, including
the mask swap, which was checked in the kernel before it was measured:
`ggml_compute_forward_soft_max_f32` computes `wp = kq*scale + slope*mask` with
`scale = slope = 1`, and adding exactly `0.0f` is an exact no-op.

The instrument was itself checked: dumping with the mask offset one too large
gives `DIFFERENT - 228 of 912 records differ`, and none of the differing records
is a single-token cached step — trap 13 reproduced by a tool that had never
heard of it.

### Memory: what gallocr actually needs vs what the formulas asked for

`NLLB_ARENA_TRACE=1`, real 600M geometry, f16:

| graph | old formula asked | gallocr compute buffer |
|---|---|---|
| cached decoder step | 81–101 MiB | **0.99 MiB** (1 MiB of it is the 256206-float logits row) |
| encoder, Ts = 24 | 153 MiB | **0.56 MiB** |
| encoder, Ts = 96 | 481 MiB | **2.63 MiB** |
| encoder, Ts = 200 | 1116 MiB | **6.35 MiB** |
| encoder, Ts = 512 | 4160 MiB | **26.0 MiB** |
| encoder, Ts = 1000 | 12353 MiB | **80.6 MiB** |

(The "asked" column is the old `mem` expression without tensor overhead, B = 1.
Step 11's batched encoder multiplied it by B, which is how it reached
`ggml_init` returning null.) Peak RSS of `nllb-bench --src 512`, weights mmap'ed
in both: **2066 MiB before, 1346 MiB after** — about 890 MiB above the weights
before and about 170 MiB after.

**This deletes the arena formulas, the 64 MB constants, the 2.3x headroom and the
reason `enc_max_tokens` existed** (the parked batching branch will want its
encoder slicing re-decided, not re-applied).

### Speed: unchanged, as it should be on the CPU

Real geometry, f16, 2 threads, `--steps 32`, medians: cached step 55.6 ms (new)
vs 57.0 ms (old, reused arena) — inside this sandbox's noise. Everything
outside `compute` is 0.80 ms warm against 0.65 before: the scheduler splits the
graph and runs gallocr every step (~0.27 ms, `alloc`), where the bump arena did
nothing. That is the next fixed cost to remove if it ever matters — graph reuse
across steps of the same shape — and it is not worth doing before Metal numbers
exist.

A **fresh** scheduler per step (the `[arena]` pass, `reuse_scratch = false`)
costs more than a fresh bump arena did: it rebuilds the scheduler's hash sets.
That is a test configuration, not a shipping one, and `nllb-bench` section [1]
now times the reused scheduler, which is the code.

### The corruption matrix for what the port introduced

`tests/corrupt_backend.py` — seven mutations of `src/nllb.cpp`, each compiled
into `/tmp`, each run through `nllb-test-cache` and `nllb-dump-states --diff`:

| mutation | cache gate | dump diff | expected |
|---|---|---|---|
| `no-output-states` — hidden states not `ggml_set_output` | FAIL full,step,chunk | 544/912 | fire |
| `no-output-final` — `dec_out` not marked | pass | identical | **inert** |
| `mask-too-large` — offset +1 (trap 13) | FAIL full,chunk | 176/912 | fire |
| `mask-too-small` — offset −1 | FAIL full,step,chunk | 688/912 | fire |
| `kq-before-write` — trap 15 | FAIL full,step,chunk,arena | 648/912 | fire |
| `mask-through-data` — input via `->data` | pass | identical | report |
| `mask-not-input` — no `ggml_set_input` | pass | identical | report |

**Identical at f32 and at f16.** `mask-too-large` is caught by `full` and `chunk`
and missed by `step` — exactly step 5's row for the same bug. `no-output-final`
is inert because `dec_out` is the last node before the lm_head and nothing
allocated after it can take its memory; it is asserted inert so that a graph
reordering that changes that is noticed.

## Three findings

**1. A shared helper blinded the cache gate, and the matrix is what showed it.**
The first draft built the causal mask in one helper, `nllb_causal_mask`, used
by both the cached step and the un-cached decoder. `mask-too-large` then
**passed the cache gate**: the reference and the candidate were broken
identically and cancelled. That is trap 44 — a control that shares the defect —
arriving through code reuse rather than through a threshold. The fix is to keep
two independent implementations: the un-cached decoder, which is the reference
the cache is diffed against, keeps ggml's own `ggml_diag_mask_inf` (proven
against HF in step 3). The price is that on Metal that one op runs on the CPU
fallback, in the reference path only — which also makes it the one place the
fallback is exercised on purpose. Trap 46.

**2. gallocr makes `ggml_set_output` load-bearing.** The bump arena kept every
intermediate alive, so any tensor could be read back afterwards. gallocr hands
a tensor's memory to a later one as soon as its last consumer has run, so a
hidden state the host reads but did not mark is overwritten — on the CPU as
much as on a GPU. `no-output-states` fails `full`, `step` and `chunk`. Trap 45.

**3. Host memory hides host-pointer bugs, on the M1 too.** Writing an input
through `->data`, or not flagging it as an input, works on the CPU by accident,
and very probably on the M1: the scheduler puts every graph input in the CPU
backend's buffer and copies it across, and Apple silicon's Metal buffers are
shared memory. The machines where these break are discrete GPUs, which nothing
here runs on. No behavioural gate available can see this class, so it is
enforced by reading the source: `ci.yml` refuses any `->data` in a code line of
`src/` or `examples/cli/`. Today there are zero. Trap 47.

## Things to expect on Metal (predictions — confirm on the M1)

* **The un-cached decoder shows 12+ splits** under `NLLB_ARENA_TRACE=1`
  (`ggml_diag_mask_inf` on the CPU, once per layer). **The cached step should
  show `splits 1`.** More than one on the cached step means an op fell back
  that was expected to run on the GPU — find it with `GGML_SCHED_DEBUG`.
* **Metal will not be bit-identical to the CPU.** Different kernels, different
  accumulation order, and ggml-metal fuses `norm+mul+add`. So the cross-backend
  gate is the precision gate: `nllb-test-precision f16.gguf f16.gguf ref/
  --backend cpu:metal` — exact ids plus the per-decision margin audit, the same
  claim f16 made against f32. The f16 file has 1.50x of headroom to spend
  (step 7); Metal's error spends some of it. If ids move, that is information,
  not a bug, and `--allow-id-drift` plus chrF is the product-level answer.
* **`full = exact` in `nllb-test-cache` should still hold on Metal** — both
  sides run on Metal on the same bytes at the same sequence length. If it does
  not, suspect trap 15 through Metal's `graph_optimize` (which reorders nodes;
  it is memory-range aware and should not move the cache write past its read)
  and bisect with `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1`.
* **Speed: modest.** The M1 Air's CPU already draws ~54 GB/s of the SoC's
  ~68 GB/s at f16 (step 7). A single-token step is a matvec; the GPU drinks
  from the same well. Expect somewhere between a small win and a wash at B = 1.
  The case for Metal is portability and batching (the parked step 11), not
  this laptop at B = 1.
* **mmap is zero-copy on Metal too**: the mapped file becomes an `MTLBuffer`
  (`buffer_from_host_ptr`), nothing is uploaded, and `-v` says
  `weights mmap'ed, zero-copy in the backend buffer`.

## The M1 acceptance run, in order

**0. First, before merging this: promote the batching stash** (handoff, Next
steps 4). This port rewrites `nllb_encode`, `nllb_decode`,
`nllb_kv_cache_init` and `nllb_decode_cached` — exactly the functions the stash
touches.

```bash
git stash list && git stash branch step11-batching && git checkout main
```

**1. Build and look.**

```bash
rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/nllb-cli --list-backends          # expect an MTL0 ... (gpu) line
```

**2. The CPU path is bit-identical to the pre-port build, on the real
checkpoint.** Build the dumper against the commit before this step:

```bash
git worktree add ../nllb-pre13 <commit-before-step-13>
cp tests/dump-states.cpp ../nllb-pre13/tests/
(cd ../nllb-pre13 && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target nllb \
 && c++ -std=c++17 -O2 -DNLLB_PRE_BACKEND -Iinclude -Iggml/include tests/dump-states.cpp build/libnllb.a \
        -Lbuild/ggml/src -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build/ggml/src -o dump-pre13)
for f in f32 f16; do
  ../nllb-pre13/dump-pre13  models/nllb-600M-$f.gguf ref/ /tmp/pre-$f.bin -t 8 --lite
  ./build/bin/nllb-dump-states models/nllb-600M-$f.gguf ref/ /tmp/new-$f.bin -t 8 --lite --backend cpu
  ./build/bin/nllb-dump-states --diff /tmp/pre-$f.bin /tmp/new-$f.bin      # expect IDENTICAL
done
```

(If the ggml libraries are under a different directory on macOS, point `-L` and
`-rpath` at wherever `libggml-base.dylib` landed.) `--lite` hashes the logits;
without it the nine goldens are gigabytes.

**3. Every existing gate, unchanged, on the CPU** (`NLLB_BACKEND` unset):
`./scripts/step10-measure.sh`, `nllb-test-tokenizer`, and
`python tests/verify_translation.py ./build/bin/nllb-cli models/nllb-600M-f16.gguf ref/ -t 8 --backend cpu`
— note the `--backend cpu`: the CLI's own default is now `auto`, i.e. Metal.

**4. The cache gate ON Metal — the first real device run.**

```bash
NLLB_ARENA_TRACE=1 NLLB_BACKEND=metal ./build/bin/nllb-test-cache models/nllb-600M-f32.gguf ref/ -t 8 2> arena.txt
NLLB_BACKEND=metal ./build/bin/nllb-test-cache models/nllb-600M-f16.gguf ref/ -t 8
grep "cached step" arena.txt | sort -u     # expect splits 1
```

Expect `full = exact` and `[arena]` bit-identical. Red here is a port bug.

**5. CPU against Metal.**

```bash
./build/bin/nllb-test-precision models/nllb-600M-f16.gguf models/nllb-600M-f16.gguf ref/ -t 8 --backend cpu:metal
./build/bin/nllb-test-precision models/nllb-600M-f32.gguf models/nllb-600M-f32.gguf ref/ -t 8 --backend cpu:metal
```

Record the tightest headroom and the worst logit error for both; they are the
numbers this step exists to produce.

**6. The product, on Metal.**

```bash
python tests/verify_translation.py ./build/bin/nllb-cli models/nllb-600M-f16.gguf ref/ -t 8 --backend metal
NLLB_BACKEND=metal ./build/bin/nllb-test-mmap models/nllb-600M-f16.gguf
NLLB_BACKEND=metal python tests/corrupt_backend.py build models/nllb-600M-f16.gguf ref/ -t 8
```

**7. Speed and memory.**

```bash
NLLB_BACKEND=cpu   ./build/bin/nllb-bench models/nllb-600M-f16.gguf -t 8 --src 24 --steps 64
NLLB_BACKEND=metal ./build/bin/nllb-bench models/nllb-600M-f16.gguf -t 8 --src 24 --steps 64
/usr/bin/time -l ./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --tgt eng_Latn --backend metal < input.txt > /dev/null
```

Compare the CPU step with step 10's 18.52 ms before reading the Metal one, and
read `peak memory footprint`, not RSS (trap 36).

## For the next backend

Step 13 is closed: Metal ships. The work for a second device backend (CUDA,
step 16) is a build flag and this file's acceptance run with the backend name
changed — nothing in `src/nllb.{h,cpp}` is Metal-specific, and
`nllb_pick_device` knows the family names `metal`/`mtl` and `cuda`.

What Metal taught that carries over:

* **`splits 1` on the cached step** or an op fell back — check
  `NLLB_ARENA_TRACE=1` before reading any timing.
* **Trap 48 generalises.** Precision is a property of the kernel the backend
  picks, not of the tensor types. Read its `mul_mat` selection rule and its
  kernel instantiations before trusting a divergence number, and watch for a
  divergence table that splits by sequence length.
* **`full = exact` is the trap-15 assertion** against whatever graph
  reordering the backend does.
* **The two host-pointer mutations may finally fire.** They are `report` in
  `tests/corrupt_backend.py` because the CPU and Apple's unified memory both
  hide that class. A device with non-host-visible buffers is the first machine
  that can catch it — promote them to `fire` there, and name the backend the
  assertion rests on.
* **The scheduler's per-step cost is not Metal's.** 1.59 ms of an 18.4 ms step
  went on re-splitting an unchanged graph; measure it on the new device before
  concluding anything about its arithmetic.

And do not assume a GPU is faster: on the M1 it wins the encoder 9.8x and loses
the one-token decode step at 0.93x. Read `encoder pass` and `with lm_head`
separately — one wall-clock number averages two opposite results.

## What was built

```
src/nllb.{h,cpp}           the port. nllb_model gains backend / backend_cpu /
                           weights_buf / sched; nllb_kv_cache gains buf / sched and
                           loses the step-6 `scratch` arena. nllb_list_backends().
tests/dump-states.cpp      nllb-dump-states: the whole API as raw bytes, --diff,
                           --lite. For comparing two BUILDS - the port, a ggml bump.
tests/corrupt_backend.py   seven mutations of what the port introduced
examples/cli/cli.cpp           --backend / -b, --list-backends; -v prints the backend
tests/test-precision.cpp   --backend A:B, the cross-backend gate
tests/test-mmap.cpp        compares through ggml_backend_tensor_get
tests/verify_translation.py  --backend passthrough; prints which backend ran
examples/bench/bench.cpp     section [1] times the reused scheduler; arena in MiB.2f
.github/workflows/ci.yml   the ->data audit, the no-silent-fallback check, the
                           matrix at f32 and f16, and a macOS Metal job
                           (continue-on-error: unverified on GitHub's runners)
```

## Step 16: CUDA — the build flag, and the blank table

**BUILT AND CI-GUARDED. NO GRAPH HAS RUN ON IT YET** — the acceptance table
below is deliberately empty, and it is the only
part of this section that matters. `-DNLLB_CUDA=ON` configures, compiles the
tree with nvcc in it, registers a device on a machine that has one, and
**does not move the CPU path**: `nllb-dump-states --diff` between a CUDA build
and a plain one is the assertion the `cuda` job in `ci.yml` exists for.

`src/nllb.{h,cpp}` is unchanged. That is what step 13 bought: `nllb_pick_device`'s
family row `{ "cuda" }` resolves ggml's registry `CUDA` and its devices `CUDA0`,
`CUDA1` — by substring on the device name, and by pass 1 on an exact `cuda0`.

Read out of ggml rather than assumed (`456172ec`, 14 Sep 2026):

| fact | where | consequence here |
|---|---|---|
| registry `CUDA`, devices `CUDA0`… | `include/ggml-cuda.h`, `GGML_CUDA_NAME` | `--backend cuda` and `--backend cuda0` both resolve |
| `caps.buffer_from_host_ptr = false` | `ggml-cuda.cu`, device props | the loader drops the mapping and reads; `nllb-test-mmap` is CPU-only here |
| `caps.host_buffer = true`, `async = true` | same struct | pinned staging is available if load time ever matters |
| `DIAG_MASK_INF` implemented | `ggml-cuda.cu` | **the un-cached decoder should NOT split**, unlike Metal |
| `GET_ROWS` on Q8_0 | `supports_op` | `q8emb` runs on device — still unmeasured on Metal |
| every other op this library builds | `ADD MUL MUL_MAT NORM SCALE RELU SOFT_MAX CPY CONT GET_ROWS` | `splits 1` is the expectation for the cached step |

### The acceptance run (unfilled)

| gate | f32 | f16 |
|---|---|---|
| `nllb-cli` translates, right language | | |
| `nllb-test-cache`, `full = exact`, `[arena]`, splits | | |
| `nllb-test-precision --backend cpu:cuda`, ids + headroom | | |
| `verify_translation.py --backend cuda`, 9 cases + chrF | | |
| `corrupt_backend.py`, the same table as the CPU? | | |
| encoder / un-cached prefix / cached step, ms | | |

```bash
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --tgt deu_Latn "Hello, world!"
NLLB_BACKEND=cuda ./build/bin/nllb-test-cache models/nllb-600M-f16.gguf ref/ -t 8
./build/bin/nllb-test-precision models/nllb-600M-f16.gguf models/nllb-600M-f16.gguf \
    ref/ -t 8 --backend cpu:cuda
python tests/verify_translation.py ./build/bin/nllb-cli models/nllb-600M-f16.gguf ref/ \
    -t 8 --backend cuda
NLLB_BACKEND=cuda python tests/corrupt_backend.py build models/nllb-600M-f16.gguf ref/ -t 8
NLLB_ARENA_TRACE=1 ./build/bin/nllb-bench models/nllb-600M-f16.gguf --steps 32
./build/bin/nllb-test-mmap models/nllb-600M-f16.gguf --backend cpu   # CPU, on purpose
```

### Three things this device can test that no machine here could

1. **The two host-pointer mutations should finally fire.**
   `mask-through-data` and `mask-not-input` are `report` rows in
   `tests/corrupt_backend.py` because the CPU and the M1's unified memory both
   hide that class by accident. A non-host-visible buffer is the first machine
   that can catch it. **The interesting outcome is the table CHANGING** — if it
   does not, that is a fact about how the scheduler places graph inputs, and it
   should be written down as one rather than read as a pass.
2. **The uncleared KV cache meets a recycling allocator.**
   `nllb_kv_cache_init` deliberately does not clear its buffer, on the argument
   that nothing reads a slot before it is written (the mask hides every slot
   >= n_kv and the views never reach past it). ggml-cuda pools device buffers,
   so an unwritten slot holds a previous graph's floats instead of a fresh zero
   page — a stronger adversary for that argument than either machine so far.
   `full` and `[arena]` in the cache gate are what would say so.
3. **`splits 1` on the un-cached decoder.** Metal's 12+ splits are
   `ggml_diag_mask_inf` falling back once per layer; CUDA implements it, so the
   fallback that step 13 kept on purpose in the reference path stops being
   exercised here. Do not read the "12 splits" line above as a GPU fact — it is
   a Metal fact.

### Two predictions, written before measuring

* **The scheduler's per-step cost comes back, and bigger in share.** 1.59 ms of
  an 18.4 ms step on the M1 was `ggml_backend_sched_alloc_graph` re-splitting
  and re-placing an unchanged 716-node graph once per token, with
  `graph_optimize` inside it. It is a property of the scheduler, not of Metal.
  Expect a faster `compute` on any recent NVIDIA card and therefore a *larger*
  fraction lost to `alloc`. Graph reuse is Next steps 5, and CUDA gives it a
  second argument: ggml has `GGML_CUDA_GRAPHS`, it is labelled "llama.cpp only"
  and defaults OFF, and the reason is that capturing a CUDA graph needs the
  ggml graph to be unchanged between steps. Reuse first; that flag is inert
  until then.
* **f16 headroom is the number to watch.** `cpu:metal` at f16 came back at
  0.39x with 1/176 unsafe decisions. CUDA picks between cuBLAS and mmq
  (`GGML_CUDA_FORCE_MMQ` forces the latter), so trap 48 applies again with a
  different selection rule: run f32 first as the control, and treat moving ids
  as information — `--allow-id-drift` plus chrF is the product-level answer.
