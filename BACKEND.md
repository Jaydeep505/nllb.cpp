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
Nothing here has run on Metal.** The sandbox has no GPU, and the fallback plan
(Mesa's software Vulkan driver, which has genuine device-memory semantics) was
blocked: the package downloads return 403. Every Metal claim below is either a
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
`src/` or `cli/`. Today there are zero. Trap 47.

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
cp tests/dump_states.cpp ../nllb-pre13/tests/
(cd ../nllb-pre13 && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target nllb \
 && c++ -std=c++17 -O2 -DNLLB_PRE_BACKEND -Isrc -Ivendor/ggml/include tests/dump_states.cpp build/libnllb.a \
        -Lbuild/vendor/ggml/src -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build/vendor/ggml/src -o dump-pre13)
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
`./tools/step10_measure.sh`, `nllb-test-tokenizer`, and
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

## Step 14: OpenCL — the build flag, and what it did NOT prove

**The prediction at the top of the next section was right about the shape of
the work and wrong about one word.** It said a second backend is "a build flag
and this file's acceptance run with the backend name changed". The build flag is
real and `src/nllb.{h,cpp}` needed **no** change for the backend itself. But
`nllb_pick_device` did need one, and this file was the thing asserting it did
not — see "the alias table was one-directional" below.

### What is proven

Sandbox, x86, 2 cores, ggml 0.23.0 (`7840aaba`, the step-13 pin — deliberately
NOT bumped, so that any OpenCL difference is OpenCL's and not the bump's).

| claim | result |
|---|---|
| `-DNLLB_OPENCL=ON` configures, compiles clean under `-Wall -Wextra` | **yes**, all 12 targets |
| the OpenCL backend registers a device | **yes**, `GPUOpenCL - … (gpu)` |
| `--backend opencl` / `ocl` / `OCL` / `GPUOpenCL` / `gpu` / `auto` all resolve | **yes**, after the fix below |
| a named-but-absent backend still FAILS with the device list | **yes** (`metal`, `vk`) |
| **the flag does not move the CPU path** | **bit-identical, 912 records, 0 differ** |
| `nllb-test-cache` on the CPU in the OpenCL build | `full = exact`, `[arena]` exact, PASS |
| `tests/corrupt_backend.py` on the CPU in the OpenCL build | matrix unchanged |

The bit-identity row is the load-bearing one and it is the reason
`nllb-dump-states` exists. A build flag that pulls a second backend library into
the process is exactly the kind of change that "obviously" cannot affect the CPU
path, and the instrument for "obviously" in this repo is a diff:

```
$ ./build/bin/nllb-dump-states    models/shape-small-f32.gguf ref-shape/ /tmp/a.bin -t 2
$ ./build-cl/bin/nllb-dump-states models/shape-small-f32.gguf ref-shape/ /tmp/b.bin -t 2
$ ./build-cl/bin/nllb-dump-states --diff /tmp/a.bin /tmp/b.bin
IDENTICAL - 912 records, 9158944 bytes, 0 differ
```

### What is NOT proven, and it is the whole acceptance run

**No graph has ever executed on OpenCL.** Not one. Every row above is about
registration, naming and the CPU path. The reason is not a defect in this
project and is worth recording, because the next person in a sandbox will hit
it in the same order:

* **pocl 5.0** (Ubuntu noble, `pocl-opencl-icd`) has `cl_khr_subgroups` and
  **not** `cl_khr_fp16`, so ggml drops it at registration:
  `ggml_opencl: device does not support FP16` → `drop unsupported device`. The
  registry then contains no OpenCL device and `--backend opencl` fails with the
  device list — a message about naming, for a cause that is about extensions.
* **Intel's CPU OpenCL runtime** (`pip install intel-opencl-rt intel-cmplr-lib-rt`,
  ICD at `/etc/OpenCL/vendors/`) has `cl_khr_fp16` **and** `cl_intel_subgroups`,
  registers cleanly, reports `device FP16 support: true`, and then **fails to
  compile ggml's kernels**:

  ```
  ggml_opencl: loading OpenCL kernels...........kernel compile error (err=-11)
  CompilerException Failed to lookup symbol kernel_geglu
  JIT session error: Symbols not found: [ _Z6exp_rmDh ]
  ```

**That was isolated to four lines, and the isolation is the useful part.** A
kernel containing nothing but `exp(half)` builds and yields a kernel object with
no build options, and fails with `err=-11` and the same missing symbol under the
options ggml passes:

```
-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations
-cl-finite-math-only -cl-fast-relaxed-math
```

So it is a gap in that runtime's JIT under relaxed math, reproducible against
the raw ICD with no ggml in the picture. Three things follow, and the second is
the one that generalises past OpenCL:

1. **ggml compiles its ENTIRE kernel set at backend init**, not lazily per op.
   NLLB uses no GLU; a missing `exp(half)` in `glu.cl` still takes the whole
   backend down before a single graph is built. `supports_op` never gets a
   chance to be wrong, because init fails first.
2. **`-cl-fast-relaxed-math` is not neutral.** This is trap 48's lesson one
   layer down: a backend's arithmetic is a property of the kernel it picks, and
   a backend's *availability* is a property of the flags it compiles with. Both
   are invisible from the tensor types.
3. **A device that registers is not a device that runs.** ggml's extension check
   at registration (`cl_khr_fp16` + subgroups) is necessary and not sufficient,
   so `--list-backends` showing `GPUOpenCL` proves less than it looks like it
   proves. Read the `ggml_opencl:` lines on stderr.

An OpenCL device that can actually run the kernels is therefore the one thing
step 14 could not obtain, and every number in "the acceptance run" below is
still blank. **Do not read the table above as "OpenCL works".**

### The alias table was one-directional, and this file was asserting otherwise

Measured on the registered device: `--backend opencl` resolved and
**`--backend ocl` did not**, on the same machine, in the same process.

```
opencl   -> resolves          ocl  -> backend 'ocl' is not available in this
                                      build (devices: GPUOpenCL (OpenCL), CPU (CPU))
```

The table mapped a family NAME to the spellings ggml uses, and matched a device
only when the user typed the family's own name. Every other member was reachable
as a *target* and not as an *input*. So the documented `ocl`, `vk` and `hip`
never worked as things to type; `mtl` worked only because ggml's Metal device is
literally called `MTL0` and the plain substring test caught it. This file and
the handoff had both been claiming `opencl`/`ocl` since step 13, because the
table looks like it lists them.

Fixed by making every row enterable from any member. Note what the case shows
about the plain substring test that runs first: `"gpuopencl"` contains
`"opencl"`, so that spelling resolves without the table at all — **luck, not
design.** `"ocl"` is not a substring of `"gpuopencl"` or of `"opencl"`, which is
precisely the case the table exists for and precisely the one that was broken.

This is trap 49 arriving a second time, one level up. Trap 49 was "the name the
user types is not the name the library prints". Step 14's is: **a table that
translates names is a claim, and a claim nothing executes is a comment.** There
is now a CI step that types every documented spelling.

### Naming, for the record

| | |
|---|---|
| registry name | `OpenCL` |
| device name | `GPUOpenCL` — not `OCL0`, not `OpenCL0` |
| device type | `GPU`, **unconditionally** — ggml returns `GGML_BACKEND_DEVICE_TYPE_GPU` without asking the device, so a CPU-backed OpenCL device (pocl, Intel CPU runtime) presents as a GPU to `nllb_pick_device` and to `auto`. ggml prints `warning, the default device is not a GPU` and registers it anyway. |

### Two things OpenCL does differently from Metal, both already handled

* **`buffer_from_host_ptr` is `false`.** Metal's zero-copy weight load — the
  mapped file becoming an `MTLBuffer` with nothing uploaded — is not available.
  `nllb_model_load` already handles this generically: it unmaps and reads
  instead, with `NLLB_LOAD_TRACE=1` printing `cannot wrap host memory -
  reading`. No change was needed. **The consequence is for the GATE, not the
  loader:** `nllb-test-mmap` correctly fails there, because the two loads become
  the same code and the comparison is vacuous (the step-4 SKIP lesson). Its
  message named only the platform and the alignment as causes; it now names the
  backend and this third cause first, since it is the one that is not a defect.
  **`nllb-test-mmap`'s claim lives on the CPU. Run it there.**
* **`diag_mask_inf` is a kernel OpenCL HAS.** Metal does not implement it, which
  is why the cached step builds its mask as data (and why trap 46 exists — the
  mask helper shared between the cache and its reference blinded the gate).
  Nothing should be undone on that account: the two independent implementations
  are what the cache gate is made of, and reuniting them to use an op OpenCL
  happens to have would re-introduce exactly the failure trap 46 records. It
  does suggest `splits 1` may come for free where Metal had to be worked at —
  which is a prediction to check, not a result.

### The acceptance run, unmeasured

Every cell below is blank on purpose. Fill them on a device that compiles the
kernels — an Adreno phone, or a desktop GPU with a vendor ICD.

| check | command | expected | measured |
|---|---|---|---|
| device registers | `nllb-cli --list-backends` | a `GPUOpenCL` line | — |
| nothing falls back | `NLLB_ARENA_TRACE=1 NLLB_BACKEND=opencl nllb-test-cache …` | `splits 1` on the cached step | — |
| trap 15 survives reordering | `NLLB_BACKEND=opencl nllb-test-cache …` | `full = exact`, 0 floats | — |
| trap 16 at equal length | same run | `[arena]` bit-identical | — |
| ids against the CPU | `nllb-test-precision … --backend cpu:opencl` | exact 9/9 | — |
| the product | `verify_translation.py … --backend opencl` | 9/9 byte-identical | — |
| the gate still fires | `NLLB_BACKEND=opencl corrupt_backend.py …` | the CPU's matrix, unchanged | — |
| **the host-pointer pair** | the two `report` rows | **may finally FIRE** | — |

The last row is the one worth going out of your way for. `mask-through-data` and
`mask-not-input` are `report` rather than `fire` because the CPU's buffers are
host memory and Apple's are shared, so neither machine can see that class.
**OpenCL is the first candidate that might not map host memory** — it says so
itself with `buffer_from_host_ptr = false`. If they fire, promote them to `fire`
and name the backend the assertion rests on. If they stay inert, find out
whether that device is mapping host memory anyway before concluding the code is
clean.

And the expectation, since step 13 set it wrong once already: **a device backend
is not automatically faster.** On the M1 the GPU wins the encoder 9.8x and loses
the one-token decode step at 0.93x, because that step is a bandwidth-bound
matvec. Read `encoder pass` and `with lm_head` separately.

## For the next backend (OpenCL, Vulkan, CUDA …)

Step 13 is closed: Metal ships. The work for a second device backend is a build
flag and this file's acceptance run with the backend name changed — nothing in
`src/nllb.{h,cpp}` is Metal-specific, and `nllb_pick_device` already knows the
family names `opencl`/`ocl`, `vulkan`/`vk`, `cuda`, `rocm`/`hip`, `sycl`.
**(Step 14 qualified both halves of that sentence: the library needed nothing,
`nllb_pick_device` needed a fix, and "a build flag" understates getting an
OpenCL implementation that compiles the kernels at all. See above.)**

```bash
cmake -B build -DNLLB_OPENCL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/nllb-cli --list-backends
NLLB_BACKEND=opencl ./build/bin/nllb-test-cache models/nllb-600M-f16.gguf ref/ -t 8
./build/bin/nllb-test-precision models/nllb-600M-f16.gguf models/nllb-600M-f16.gguf \
    ref/ -t 8 --backend cpu:opencl
python tests/verify_translation.py ./build/bin/nllb-cli models/nllb-600M-f16.gguf ref/ \
    -t 8 --backend opencl
NLLB_BACKEND=opencl python tests/corrupt_backend.py build models/nllb-600M-f16.gguf ref/ -t 8
```

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
tests/dump_states.cpp      nllb-dump-states: the whole API as raw bytes, --diff,
                           --lite. For comparing two BUILDS - the port, a ggml bump.
tests/corrupt_backend.py   seven mutations of what the port introduced
cli/nllb-cli.cpp           --backend / -b, --list-backends; -v prints the backend
tests/test_precision.cpp   --backend A:B, the cross-backend gate
tests/test_mmap.cpp        compares through ggml_backend_tensor_get
tests/verify_translation.py  --backend passthrough; prints which backend ran
tests/bench_cached.cpp     section [1] times the reused scheduler; arena in MiB.2f
.github/workflows/ci.yml   the ->data audit, the no-silent-fallback check, the
                           matrix at f32 and f16, and a macOS Metal job
                           (continue-on-error: unverified on GitHub's runners)
```

## Step 16: CUDA — the build flag, and the blank table

**BUILT AND CI-GUARDED. NO GRAPH HAS RUN ON IT YET** — the acceptance table
below is deliberately empty, the same way step 14's was, and it is the only
part of this section that matters. `-DNLLB_CUDA=ON` configures, compiles the
tree with nvcc in it, registers a device on a machine that has one, and
**does not move the CPU path**: `nllb-dump-states --diff` between a CUDA build
and a plain one is the assertion the `cuda` job in `ci.yml` exists for.

`src/nllb.{h,cpp}` is unchanged. That is what step 13 bought, and unlike step
14 it held without a fix: `nllb_pick_device`'s family row `{ "cuda" }` resolves
ggml's registry `CUDA` and its devices `CUDA0`, `CUDA1` — by substring on the
device name, and by pass 1 on an exact `cuda0`. There is only one spelling, so
trap 50's asymmetry has nothing to catch here.

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

## Step 17: Vulkan — and the first device backend to run every gate outside Metal

`-DNLLB_VULKAN=ON`. Unlike steps 14 and 16 this one **ran graphs**: Mesa's
lavapipe (conformant Vulkan 1.4, executing on the CPU, `uma: 0` to ggml) runs
every gate in the sandbox and in CI. ggml 0.23.0 (`7840aaba`), shape fixture
(d_model 256, 4+4 layers, vocab 4000). `GGML_VK_VISIBLE_DEVICES=0` is required:
ggml-vulkan hides CPU-type devices by default.

| check | result |
|---|---|
| the flag does not move the CPU path (`nllb-dump-states --diff`) | **IDENTICAL, 912 records, f32 and f16** |
| `NLLB_KV_CONT=1` forced on the CPU | **IDENTICAL** — the workaround is an exact copy |
| every spelling: `vulkan vk VK Vulkan0 gpu auto` | resolve |
| `splits` | **1 on every graph**, the un-cached decoder included (`DIAG_MASK_INF` exists on Vulkan) |
| **cache gate, first run** | **FAIL** — `full = exact`, `step`/`chunk` wrong from layer 1 (rel 0.16), `[arena]` differs |
| `nllb-probe-kv-ops vulkan` | `mul_mat(K view, q)` BAD at every Tn/n_past; the two cpy writes and the V matmul fine |
| cache gate after the fix, f32 / f16 | **PASS** — `full = exact`, `[arena]` exact; step 6.9e-04 vs control 6.5e-04 (f32) |
| `precision --backend cpu:vulkan` f32 | exact ids 96/96, tightest headroom **47.8x**, state rel 8.7e-04 |
| `precision --backend cpu:vulkan` f16 | exact ids 96/96, tightest headroom **4.6x**, state rel 5.2e-03 |
| `corrupt_backend.py` with `NLLB_BACKEND=vulkan` | 7/7 as expected — the CPU's matrix, unchanged |
| with OpenCL also built in | `auto` → **Vulkan0**. Read from `ggml-backend-reg.cpp` (Vulkan registers before OpenCL); the run only confirms it weakly, since rusticl was dropped |

### The bug (trap 55)

ggml-vulkan's `mul_mat` treats `src0` as contiguous when dims 0 and 1 are
(`ggml_vk_dim01_contiguous`) and then steps between the matrices of a batch by
`ne00*ne01`, not by `nb[2]`. The self-attention K view is `[HD, n_kv, H]` over a
`[HD, n_ctx, H]` cache, so its heads are `n_ctx` rows apart and every head after
the first reads the wrong rows whenever `n_kv < n_ctx`. The V view is not
dim01-contiguous (its rows are `n_ctx` apart), takes ggml's non-contiguous copy
path, and is fine. llama.cpp does not hit this because its K view is permuted
(`nb1 = n_embd_k`), which the same check calls non-contiguous.

**Why `full = exact` stayed green:** that pass builds its cache to exactly the
prefix length, so `n_kv == n_ctx` and there is no gap. `step` and `chunk` use
the same cache after it has room to grow, which is every decode step of
`nllb-cli`. A gate that passes because the fixture happens to fill a buffer
exactly is trap 13's shape — an error that is one-sided in the configuration.

**The fix** is one line in `nllb_decode_cached`: `Kv = ggml_cont(ctx, Kv)` when
`model.kv_k_cont`, which is set for the Vulkan registry only (`NLLB_KV_CONT=0/1`
overrides). Scoped rather than universal because Metal and the CPU read `nb[2]`
correctly and were verified without it; the copy is exact, so turning it on
anywhere moves no float (checked on the CPU). CI runs the gate with it forced
OFF and requires the red, so a ggml bump that fixes the op announces itself.

### Trap 48, on Vulkan

An **f32** file diverges from the CPU at ~9e-4 relative on Vulkan — f16-sized
noise — and the f16 file at ~5e-3. `nllb-probe-kv-ops` shows the same thing per
op: exact at `Tn = 1` (mat-vec) and ~3e-4 at `Tn >= 3` for the contiguous K
matmul. Expect the real 600M's headroom on an Adreno to look like Metal's f16
(0.39x) rather than the CPU's, and read `[margin]` before calling a moved id a
bug.

### Not proven

* Anything on a GPU. lavapipe is real Vulkan but not a real GPU: no Adreno
  shader compiler, no discrete memory, and no speed that means anything.
  `QDC-XELITE.md` is the device run.
* The host-pointer pair (`mask-through-data`, `mask-not-input`) stayed inert
  here too despite `uma: 0`. The likelier explanation now is the scheduler, not
  the device: `ggml_backend_sched` allocates graph inputs in the CPU backend's
  buffer and copies them across, so `->data` on an input is host memory on
  every backend. The `ci.yml` source rule stays the guard.

## Step 17b: OpenCL on the X Elite — still unrun, now packaged

Nothing in this repo changed for OpenCL in step 17 beyond packaging. Two facts
measured on the way: Mesa's rusticl has `cl_khr_fp16` but advertises subgroups
only as the OpenCL-C feature `__opencl_c_subgroups`, not as the extension
string ggml looks for; and at this pin **ggml-opencl accepts only Adreno and
Intel GPUs** (`unsupported GPU … drop unsupported device`), so no software
OpenCL can ever stand in for the device. The step-14 acceptance table is still
blank and `run.ps1` fills it, including `nllb-probe-kv-ops opencl`, which
answers whether OpenCL needs `NLLB_KV_CONT=1` as well.
