# Step 16: CUDA — the changes, file by file

Four files. `src/nllb.cpp` is not one of them, and the reason is step 13: the
port to `ggml_backend_sched` is what made a second device backend a build flag
rather than a rewrite. Verified against ggml master (`456172ec`, 14 Sep 2026),
not assumed:

| what | where it is | why it matters here |
|---|---|---|
| registry `CUDA`, devices `CUDA0`, `CUDA1` | `include/ggml-cuda.h`, `GGML_CUDA_NAME` | `nllb_pick_device`'s family row `{ "cuda", nullptr, nullptr }` already resolves it — by substring on the device name, and by pass 1 on `cuda0` |
| `props.caps.buffer_from_host_ptr = false` | `ggml-cuda.cu:5041` | the mmap path already tests this cap and falls back to the read path, so mmap switches itself off |
| `host_buffer = true`, `async = true` | same struct | pinned staging is available if the load time ever matters (optional, below) |
| `DIAG_MASK_INF` implemented | `ggml-cuda.cu:2298` | the un-cached decoder should NOT split, unlike Metal |
| `GET_ROWS` on Q8_0 | `ggml-cuda.cu:5195` | `q8emb` runs on device — still unmeasured on Metal |
| every other op in `src/nllb.cpp` | `ADD MUL MUL_MAT NORM SCALE RELU SOFT_MAX CPY CONT GET_ROWS` | all present, so `splits 1` is the expectation for the cached step |

---

## 1. `CMakeLists.txt`

Insert before `add_subdirectory(ggml)`:

```cmake
# ---------------------------------------------------------------------------
# CUDA (step 16).  -DNLLB_CUDA=ON
# ---------------------------------------------------------------------------
# ggml's own switch is GGML_CUDA and it works unchanged; this flag exists to
# say at CONFIGURE time what this build will be able to schedule, so the block
# is short.
#
#   cmake -B build -DNLLB_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake -B build -DNLLB_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ...   # T4
#
# CMAKE_CUDA_ARCHITECTURES: ggml sets "native" when CUDAToolkit >= 11.6 and
# CMake >= 3.24, which is a Colab runtime, and otherwise builds a ladder of
# virtual architectures that costs many minutes of nvcc. PIN IT to the card you
# are on - 75 T4, 80 A100, 86 RTX 3000, 89 L4/RTX 4000, 90 H100 - and the
# build is one architecture's worth of compile.
#
# What the device must have is checked by ggml at runtime, not by this file: a
# machine with the toolkit and no driver configures, compiles, registers NO
# device, and then `--backend cuda` fails with the device list (trap 49's
# message) - and it is correct: `auto` falls back to the CPU and a named backend
# never does.
option(NLLB_CUDA "nllb: build the ggml CUDA backend" OFF)

if(NLLB_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "nllb: CUDA backend ON (arch: ${CMAKE_CUDA_ARCHITECTURES})")
endif()
```

and after `add_subdirectory(ggml)`:

```cmake
if(NLLB_CUDA AND NOT TARGET ggml-cuda)
    message(FATAL_ERROR
        "NLLB_CUDA=ON but ggml did not build its CUDA backend. Its "
        "find_package(CUDAToolkit) found no toolkit: install one and put nvcc "
        "on PATH, or point -DCUDAToolkit_ROOT at it. `nvcc --version` is the "
        "check. See docs/build.md.")
endif()
```

One more line, at the top:

```cmake
cmake_minimum_required(VERSION 3.18)   # was 3.14: ggml-cuda needs CMAKE_CUDA_ARCHITECTURES
```

`ggml/src/ggml-cuda/CMakeLists.txt` declares `cmake_minimum_required(3.18)`
itself, so 3.14 at the top *works* — but it leaves CMP0104 unset in this
directory, and the top-level minimum is a claim about what this project needs.
Raising it is one line and no platform in the matrix ships CMake older than
3.22.

## 2. `examples/cli/cli.cpp`

One line in the `--backend` help, after the `metal` line:

```c
"                          cuda   CUDA; FAILS rather than fall back\n"
```

## 3. `.github/workflows/ci.yml`

A `cuda` job, honest about what a green tick means. It **can**: configure, compile the whole tree with the CUDA backend
linked in, and assert that doing so leaves the CPU path bit-identical
(`nllb-dump-states --diff`). It **cannot**: run one graph — no hosted runner has
a GPU. Its shape:

- `- uses: Jimver/cuda-toolkit@v0.2.19` (or `apt-get install nvidia-cuda-toolkit`)
- `-DNLLB_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75-virtual` (virtual only: nothing
  will execute it, and `-real` needs no GPU either but doubles the compile)
- the spelling loop becomes `for want in cpu auto gpu cuda CUDA Cuda`
- the pocl/fp16 environment assertion is replaced by its CUDA equivalent:
  `--list-backends` must contain **no** CUDA device on a runner with no driver.
  If it ever does, the acceptance run below can move into CI.

## 4. `docs/backend.md` + `docs/build.md`

`docs/backend.md` gets a CUDA section with a **deliberately blank acceptance
table** — the step-14 convention. `docs/build.md` gets the toolkit prerequisite and
the `CMAKE_CUDA_ARCHITECTURES` note.

---

# Colab

```bash
!nvidia-smi --query-gpu=name,compute_cap --format=csv    # T4 -> 7.5
!git clone --recursive https://github.com/<you>/nllb.cpp && cd nllb.cpp
!cmake -B build -DNLLB_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=75
!cmake --build build -j$(nproc)
!./build/bin/nllb-cli --list-backends
```

Expect `CUDA0 - Tesla T4 (gpu)` and `CPU - ... (cpu)`. Then, in this order,
because each one tells you something the next one assumes:

```bash
# 1. does anything run at all, and is it the right language
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --tgt deu_Latn "Hello, world!"

# 2. the cache against its own un-cached reference, ON the device.  full = exact
NLLB_BACKEND=cuda ./build/bin/nllb-test-cache models/nllb-600M-f16.gguf ref/ -t 2

# 3. the cross-backend gate: exact ids + the per-decision margin audit
./build/bin/nllb-test-precision models/nllb-600M-f16.gguf models/nllb-600M-f16.gguf \
    ref/ -t 2 --backend cpu:cuda

# 4. text -> text against HF, 9 cases
python tests/verify_translation.py ./build/bin/nllb-cli --backend cuda ...

# 5. where the splits are
NLLB_ARENA_TRACE=1 ./build/bin/nllb-bench models/nllb-600M-f16.gguf --steps 32
```

`nllb-test-mmap` must be run `--backend cpu`. On CUDA it is not a failure, it is
a no-op with a misleading name: `buffer_from_host_ptr` is false, so the loader
drops the mapping and reads. Same posture as Windows.

## Three things CUDA will test that nothing has tested yet

1. **Trap 47's two "report" rows should become "fire" rows.** `mask-through-data`
   and `mask-not-input` in `tests/corrupt_backend.py` are inert on the CPU and
   on the M1 because both have host-visible graph-input buffers. A discrete GPU
   is the machine that class of bug was predicted to break on. Run
   `NLLB_BACKEND=cuda python tests/corrupt_backend.py build <f16> ref/ -t 2` and
   **the interesting outcome is the table changing**. If it does not change,
   that is a fact about the scheduler's input handling, not a pass.

2. **The uncleared KV cache meets a recycling allocator.** `nllb_kv_cache_init`
   deliberately does not clear the buffer, on the argument that nothing reads a
   slot before it is written. ggml-cuda pools device buffers, so an unwritten
   slot holds a previous graph's floats rather than a fresh zero page — a
   stronger adversary than either machine so far. The `[arena]` and `full`
   passes of the cache gate are what would say so.

3. **`splits 1` on the un-cached decoder, not 12+.** CUDA implements
   `DIAG_MASK_INF`, so the fallback that step 13 kept on purpose in the
   reference path stops being exercised here. That is worth knowing before
   reading any other GPU trace: the "12 splits" line in docs/backend.md is a
   Metal fact, not a GPU fact.

## Two predictions worth writing down before you measure

- **The `alloc` cost comes back.** 1.59 ms/step on Metal was
  `ggml_backend_sched_alloc_graph` splitting and placing a 716-node graph per
  token. It is a property of the scheduler, so expect it on CUDA too, and
  expect it to be a *larger* share: the T4's `compute` should be faster than the
  M1's. Graph reuse (Next steps 5) is the fix, and CUDA gives it a second
  argument — `GGML_CUDA_GRAPHS` exists in ggml but is labelled "llama.cpp only"
  and defaults OFF, because capturing a CUDA graph needs the ggml graph to be
  *unchanged* between steps. Your cached step rebuilds it every token. Reuse
  first, then that flag becomes available rather than inert.
- **f16 headroom is the number to watch.** `precision cpu:metal f16` came back
  at 0.39x with 1/176 unsafe. CUDA's f16 matmul path (cuBLAS or mmq, and
  `GGML_CUDA_FORCE_MMQ` chooses) accumulates differently again. Ids moving is
  information, not a bug — `--allow-id-drift` plus chrF is the product answer,
  and the f32 file is the control to run first.

## Optional, not required

Pinned host staging in `nllb_read_weights`: the 64 MB `std::vector<char>` could
come from `ggml_backend_dev_host_buffer_type(pdev)` when the device advertises
`host_buffer`, which is roughly 2x on the H2D copy and turns a ~2.4 GB load
into a noticeably shorter one. It is a load-time change with no effect on any
float, so it is a separate commit and it needs `nllb-test-mmap`'s "every weight
byte identical" claim run on the CPU either side of it.
