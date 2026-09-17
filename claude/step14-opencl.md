# Step 14: OpenCL — the build flag

`-DNLLB_OPENCL=ON`. The backend builds, registers a device, and resolves by
every spelling the docs advertise. **No graph has run on it.** That sentence is
the step, and the rest of this page is why, what was proven instead, and what to
run on the first machine that can.

Repo copy of the detail: the "Step 14" section of `BACKEND.md`. Build recipes:
`BUILD.md`.

## The flag

```bash
cmake -B build -DNLLB_OPENCL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/nllb-cli --list-backends
```

`-DGGML_OPENCL=ON` works too — ggml is `add_subdirectory`'d and reads the same
cache. `NLLB_OPENCL` exists on top of it to change exactly one default, and the
reason is worth stating because it is the kind of thing that produces a build
which configures, compiles, and is then quietly wrong:

**ggml's OpenCL defaults are Adreno defaults.** `GGML_OPENCL_USE_ADRENO_KERNELS`
is ON upstream. Here the portable value is OFF and `-DNLLB_ADRENO=ON` asks for
it back. `GGML_OPENCL_EMBED_KERNELS` stays ON, which is also ggml's default and
is kept deliberately rather than inherited: with it OFF, ggml copies `.cl` files
next to the binary at configure time and loads them by relative path at runtime,
making the binary non-relocatable. One file that carries everything it needs is
the step-8 argument about the grapheme tables; it applies to kernels unchanged.

CMake fails at configure time, by name, if `NLLB_OPENCL=ON` did not produce a
`ggml-opencl` target — ggml's `find_package(OpenCL REQUIRED)` needs headers and
an ICD loader, and a backend that silently did not get built is a runtime
mystery rather than a build error.

## What is proven

Sandbox, x86, 2 cores, ggml **0.23.0 (`7840aaba`)** — the step-13 pin,
deliberately not bumped, so that any OpenCL difference is OpenCL's and not the
bump's. Fixture: `make_shape_model.py` at d_model 256, 4+4 layers, vocab 4000.

| claim | result |
|---|---|
| configures and compiles clean under `-Wall -Wextra`, all 12 targets | yes |
| the OpenCL backend registers a device | yes — `GPUOpenCL - … (gpu)` |
| `opencl` / `ocl` / `OCL` / `OpenCL` / `GPUOpenCL` / `gpu` / `auto` resolve | yes, **after the fix below** |
| a named-but-absent backend FAILS with the device list | yes (`metal`, `vk`) |
| **the flag does not move the CPU path** | **bit-identical, 912 records, 0 differ** |
| `nllb-test-cache` on the CPU in the OpenCL build | `full = exact`, `[arena]` exact, PASS |
| `corrupt_backend.py` on the CPU in the OpenCL build | 7/7, matrix unchanged |

The bit-identity row is the load-bearing one, and it is why `nllb-dump-states`
exists. Pulling a second backend library into the process is exactly the kind of
change that "obviously" cannot touch the CPU path:

```
$ ./build/bin/nllb-dump-states    models/shape-small-f32.gguf ref-shape/ /tmp/a.bin -t 2
$ ./build-cl/bin/nllb-dump-states models/shape-small-f32.gguf ref-shape/ /tmp/b.bin -t 2
$ ./build-cl/bin/nllb-dump-states --diff /tmp/a.bin /tmp/b.bin
IDENTICAL - 912 records, 9158944 bytes, 0 differ
```

## What is NOT proven: everything about running

No OpenCL implementation reachable from the sandbox can execute ggml's kernels,
and the two failures are different in kind. Both are findings about OpenCL
implementations, not about this project.

**pocl 5.0** (Ubuntu noble, `pocl-opencl-icd`) has `cl_khr_subgroups` and **not**
`cl_khr_fp16`, so ggml drops it at registration:

```
ggml_opencl: device does not support FP16
ggml_opencl: drop unsupported device 'cpu-skylake-avx512-…'
```

The registry then holds no OpenCL device, so `--backend opencl` fails with the
device list — **a message about naming, for a cause that is about extensions.**
The `ggml_opencl:` lines on stderr are the only place the real reason appears.

**Intel's CPU OpenCL runtime** (`pip install intel-opencl-rt intel-cmplr-lib-rt`,
ICD written to `/etc/OpenCL/vendors/`) has `cl_khr_fp16` *and*
`cl_intel_subgroups`, registers cleanly, prints `device FP16 support: true`, and
then fails to compile the kernels:

```
ggml_opencl: loading OpenCL kernels...........kernel compile error (err=-11)
CompilerException Failed to lookup symbol kernel_geglu
JIT session error: Symbols not found: [ _Z6exp_rmDh ]
```

**Isolated to four lines, and the isolation is the useful part.** A kernel
containing nothing but `exp(half)` builds and yields a kernel object with no
build options, and fails with `err=-11` and that same missing symbol under the
options ggml passes:

```
-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations
-cl-finite-math-only -cl-fast-relaxed-math
```

No ggml in the picture. Three consequences, and the middle one generalises well
past OpenCL:

1. **ggml compiles its ENTIRE kernel set at backend init**, not lazily per op.
   NLLB uses no GLU; a missing `exp(half)` in `glu.cl` still takes the whole
   backend down before a graph is built. `supports_op` never gets a chance to be
   wrong, because init fails first.
2. **`-cl-fast-relaxed-math` is not neutral.** Trap 48 says a backend's
   *arithmetic* is a property of the kernel it picks. This is one layer down: a
   backend's *availability* is a property of the flags it compiles with. Neither
   is visible from the tensor types you handed it.
3. **A device that registers is not a device that runs.** The extension check at
   registration is necessary and not sufficient, so `--list-backends` printing
   `GPUOpenCL` proves less than it looks like it proves.

An older Intel runtime (2024.2.1) was tried and abandoned: it needs a versioned
`OrtGetApiBase` from an Intel-specific ONNX Runtime build, and a stub for it
segfaults on first call. Stubbing a runtime to make a gate green is the opposite
of what the gates are for.

## The alias table was one-directional, and the docs were asserting otherwise

This is the one source change the backend itself needed, and it is not in the
backend.

Measured on the registered device: `--backend opencl` resolved and
**`--backend ocl` did not**, same machine, same process.

```
opencl  -> resolves
ocl     -> backend 'ocl' is not available in this build
           (devices: GPUOpenCL (OpenCL), CPU (CPU))
```

`nllb_pick_device`'s table mapped a family NAME to the spellings ggml uses, and
matched a device only when the user typed the family's own name. Every other
member was reachable as a *target* and never as an *input*. So documented `ocl`,
`vk` and `hip` never worked as things to type. `mtl` worked only because ggml's
Metal device is literally called `MTL0` and the plain substring test caught it
before the table was consulted.

Fixed by making every row enterable from any member. Note what the case says
about that plain substring test: `"gpuopencl"` contains `"opencl"`, so that
spelling resolves without the table at all — **luck, not design**. `"ocl"` is a
substring of neither `"gpuopencl"` nor `"opencl"`, which is exactly the case the
table exists for and exactly the one that was broken.

**Trap 49 arriving a second time, one level up.** Trap 49 was "the name the user
types is not the name the library prints". This is: **a table that translates
names is a claim, and a claim nothing executes is a comment.** `BACKEND.md` and
`claude/handoff.md` had both been listing `opencl`/`ocl` since step 13 on the
strength of reading the table. `ci.yml` now types every documented spelling.

## Naming, for the record

| | |
|---|---|
| registry name | `OpenCL` |
| device name | `GPUOpenCL` — not `OCL0`, not `OpenCL0` |
| device type | `GPU`, **unconditionally** |

The type is returned without asking the device, so a CPU-backed OpenCL device
presents as a GPU to `nllb_pick_device` and is what `auto` picks. ggml prints
`warning, the default device is not a GPU` and registers it anyway.

**ggml also enumerates devices on ONE platform only.** With two ICDs installed
the other platform's devices are invisible rather than merely unselected —
`GGML_OPENCL_PLATFORM` / `GGML_OPENCL_DEVICE` are the override, and without them
the sandbox picked pocl and dropped it while a usable Intel platform sat unlisted.

## Two things OpenCL does differently from Metal, both already handled

**`buffer_from_host_ptr` is `false`.** Metal's zero-copy weight load — the mapped
file becoming an `MTLBuffer` with nothing uploaded — is not available.
`nllb_model_load` already handles this generically: it unmaps and reads instead,
with `NLLB_LOAD_TRACE=1` printing `cannot wrap host memory - reading`. **No
change was needed in the loader.** The consequence is for the GATE:
`nllb-test-mmap` correctly fails there, because the two loads become the same
code and the comparison is vacuous (the step-4 SKIP lesson, which that gate
already applied). Its failure message named only the platform and the file's
alignment as causes; it now names the backend and this third cause first, since
it is the one that is not a defect. **`nllb-test-mmap`'s claim lives on the CPU.**

**`diag_mask_inf` is a kernel OpenCL HAS.** Metal does not implement it, which is
why the cached step builds its mask as data — and why trap 46 exists, the mask
helper shared between the cache and its reference that blinded the gate. **Do
not undo that on this account.** The two independent implementations are what
the cache gate is made of, and reuniting them to use an op OpenCL happens to
have would re-introduce exactly the failure trap 46 records. It does suggest
`splits 1` may come for free where Metal had to be worked at — a prediction to
check, not a result.

## The acceptance run, unmeasured

Every cell is blank on purpose. Fill them on a device that compiles the kernels.

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

The last row is worth going out of your way for. `mask-through-data` and
`mask-not-input` are `report` rather than `fire` because the CPU's buffers are
host memory and Apple's are shared, so neither machine can see that class.
**OpenCL is the first candidate that might not map host memory — it says so
itself with `buffer_from_host_ptr = false`.** If they fire, promote them to
`fire` and name the backend the assertion rests on. If they stay inert, find out
whether that device is mapping host memory anyway before concluding the code is
clean. (On the sandbox CPU they stayed inert, as predicted — 7/7 as expected.)

Also expect trap 48 with different numbers: read the OpenCL `mul_mat` selection
rule and its kernel instantiations before trusting any divergence figure, and
watch for a divergence table that splits by sequence length the way Metal's did.

And the expectation step 13 set wrong once already: **a device backend is not
automatically faster.** On the M1 the GPU wins the encoder 9.8x and loses the
one-token decode step at 0.93x. Read `encoder pass` and `with lm_head`
separately; one wall-clock number averages two opposite results.

## CI

A new `opencl` job. What it can claim is stated in its own header comment, so a
green tick is not read as more than it is: it configures, compiles the whole
tree with the backend in, asserts the CPU path stays **bit-identical**, runs the
cache gate and the corruption matrix in that build, and **types every backend
spelling the docs advertise**. It cannot run a graph on OpenCL — no hosted
runner has a device that compiles the kernels — and it asserts pocl's dropout
rather than ignoring it, so if pocl ever gains `cl_khr_fp16` that step changes
and the acceptance run becomes runnable in CI.

The spelling loop found a bug in itself on the first local run: it required
`--backend gpu` to resolve, and on a machine with no GPU device `gpu` must
**fail**, because a named backend never silently falls back — only `auto` does.
`cpu` and `auto` are the only two required to resolve. Four harnesses in this
repo have now lied toward green; this one was caught before it shipped by
running it rather than reading it.

## What changed

```
CMakeLists.txt           NLLB_OPENCL / NLLB_ADRENO, the portable defaults, and a
                         configure-time FATAL_ERROR if ggml did not build the backend
src/nllb.cpp             nllb_pick_device: the family table is enterable from any
                         member. The ONLY source change, and not in the backend
tests/test_mmap.cpp      the fallback message names the backend and the third cause
BUILD.md                 the OpenCL section: requirements, the three target recipes
                         (Android and Windows Arm64 TRANSCRIBED, not run)
BACKEND.md               the step-14 section, and the qualification of what the
                         "for the next backend" section had promised
.github/workflows/ci.yml the opencl job
```
