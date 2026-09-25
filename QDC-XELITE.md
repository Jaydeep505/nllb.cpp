# nllb.cpp on a Snapdragon X Elite (Adreno X1-85) via Qualcomm Device Cloud

Target: the **CRD8380X Compute Reference Design, Windows 11 on Arm**, Adreno
X1-85. Two GPU backends are tested in one session: **OpenCL** (with ggml's
Adreno kernels) and **Vulkan**. Step 17 rewrote this page; the older recipe
(MSVC `-A ARM64` cross-compile from the G531) **does not configure** at the
pinned ggml. See "What changed" at the bottom.

---

## The short version

1. Push the repo to GitHub with `vendor/ggml` at **`7840aab`**.
2. **Actions → xelite → Run workflow.** ~30-40 min. It builds natively on a
   `windows-11-arm` runner, tests the CPU path there, and uploads
   **`nllb-xelite.zip`** (~60 MB, no model inside).
3. Start a QDC session on the X Elite CRD (give an SSH key at "Test
   Configuration"), copy the zip over, unzip, and run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1
   ```

4. Copy **`results.zip`** off the device before the session ends. QDC wipes it.

That first session needs **no model**: every gate in it compares two C++ paths
on a 17 MB random fixture, so the weights cancel. Bring the real
`nllb-600M-f16.gguf` (+ `ref\` from the Mac) to a second session, once the
first one is green:

```powershell
.\run.ps1 -Model C:\m\nllb-600M-f16.gguf -Ref C:\m\ref
```

---

## Why the build moved off the G531

| | old recipe (G531, x64 → Arm64) | now |
|---|---|---|
| compiler | MSVC `cl.exe`, `-A ARM64` | **clang** (LLVM 22), MSVC Arm64 libs |
| why it had to change | ggml: `message(FATAL_ERROR "MSVC is not supported for ARM, use clang")` in `src/ggml-cpu/CMakeLists.txt` | — |
| Vulkan | would have had to build `vulkan-shaders-gen` for x64 while building everything else for Arm64 | host = target, nothing to cross |
| CPU flags | `-march=armv8.7-a -ffp-model=fast` (llama.cpp's toolchain file) | `-DGGML_CPU_ARM_ARCH=armv8.7-a`, **no** `-ffp-model=fast` |
| CRT | dynamic (needs the VC++ redist on a wiped device) | **static** (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`) |

GitHub's `windows-11-arm` runners are free on public repos and available to
private ones since January 2026. The image already has VS 2022 with the Arm64
tools, LLVM 22, CMake, Ninja and Python; the workflow adds the LunarG Vulkan SDK
for Windows on Arm and builds the Khronos OpenCL loader from source.

**One trap the workflow is built around:** the runner is a Cobalt 100
(Neoverse N2, Armv9 **with SVE2**); Oryon is Armv8.7 **without SVE**.
`GGML_NATIVE=ON` on the runner would emit SVE into the CPU backend and the
binary would die with an illegal instruction on the X Elite — in the reference
path every GPU number is compared against. Hence `GGML_NATIVE=OFF` and one fixed
`-march` for both machines, which also makes section 3 below an exact test.

If you have Arm64 Windows hardware of your own, `tools/qdc/build-arm64.cmd` is
the same recipe by hand.

---

## What is in the zip

```
cpu\       nllb-cli, -test-cache, -test-precision, -dump-states, -bench, -probe-kv-ops
opencl\    the same, built -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON, + OpenCL.dll + ocl-probe.exe
vulkan\    the same, built -DNLLB_VULKAN=ON
models\    shape-small-f16.gguf, shape-small-f32.gguf   (random weights, NLLB shape)
ref-shape\ structural manifest for them (no goldens)
baseline\  runner-cpu-f16.bin, runner-cpu-f32.bin       (the CI runner's CPU dump)
run.ps1  sentences.txt  BUILD-INFO.txt  QDC-XELITE.md
```

**Three builds, one backend each, on purpose.** A missing `vulkan-1.dll` then
cannot stop the OpenCL run, `cpu\` always starts, and `auto` is never
ambiguous — in a build with both, ggml registers Vulkan *before* OpenCL and
`auto` silently picks Vulkan (registry order in `ggml-backend-reg.cpp`).

`vulkan-1.dll` is **not** shipped: the Adreno driver installs it in System32,
and section 0 of the run says whether it is there. `OpenCL.dll` **is** shipped
(the Khronos ICD loader); it finds the Adreno driver through the registry.

---

## What `run.ps1` does, and what to read in the output

Everything goes to `results\` with one `.txt` per step and a `SUMMARY.txt`.

| § | step | read this | means |
|---|---|---|---|
| 0 | environment | `vulkan-1.dll present`, `OpenCL.dll present` | if a DLL is absent that backend's rows say `DID NOT START` — a driver fact, not a bug |
| 1 | `ocl-probe` | `VERDICT: USABLE`, exit code = usable devices | 0 → ignore every OpenCL row below |
| 1b | `nllb-probe-kv-ops <b>` | the 45-row table | **Vulkan: the raw `K view` rows are EXPECTED to be BAD** (trap 55). **OpenCL: nobody knows yet** — BAD there means OpenCL needs the same workaround |
| 2 | `--list-backends` | `Vulkan0 - Qualcomm(R) Adreno(TM) X1-85 GPU (igpu)`, `GPUOpenCL …` | stderr above it explains any dropped device |
| 3 | CPU dump vs runner | `IDENTICAL` | same binary, different Arm64 CPU. **If this is not identical, stop: no GPU number means anything yet** |
| 4 | cache gate per backend | `PASS`, `full = exact`, `[arena]` exact, `splits 1` on the cached step | traps 15/16 hold on the device. On a FAIL it re-runs once with `NLLB_KV_CONT` flipped and says whether that changed the verdict |
| 5 | `precision cpu:<b>` | `exact ids`, `tightest headroom` | ids moving at f16 is information (trap 48), not automatically a bug |
| 6 | states diff + bench | `worst rel` per record; `encoder pass` vs `with lm_head` | a REPORT. Does the divergence split at 8 columns like Metal's? |
| 7 | real model (`-Model`) | the same gates on the 600M, text CPU vs GPU **byte-identical with every line present**, bench | the product claim |

### What sandbox evidence says to expect (step 17, Mesa lavapipe)

Vulkan has now run every one of these gates on a real Vulkan implementation
(lavapipe: conformant Vulkan 1.4, on the CPU). On the shape fixture:

```
cache gate  vulkan f32/f16    PASS  full = exact, [arena] exact, splits 1 everywhere
precision   cpu:vulkan f32    PASS  exact ids 96/96, tightest headroom 47.8x
precision   cpu:vulkan f16    PASS  exact ids 96/96, tightest headroom  4.6x
states      cpu vs vulkan f32 756/912 records differ, worst rel ~9e-4   <- half-precision
                                                                          matmuls (trap 48)
corrupt_backend.py on vulkan  7/7 as expected, the CPU's matrix unchanged
```

**The f32 file diverges at ~9e-4 on Vulkan** — f16-sized noise from an f32 file,
the same thing trap 48 found on Metal. So the real 600M's headroom on the Adreno
will be closer to Metal's f16 figure (0.39x) than to the CPU's, and a moved id
there is expected rather than alarming. Read `[margin]` before deciding.

---

## Trap 55, found in the sandbox before any Adreno saw it

The first Vulkan run of `nllb-test-cache` went **red**: `full = exact`, but
`step` and `chunk` wrong from layer 1 at rel 0.16, and `[arena]` differing.
`nllb-probe-kv-ops` (new) isolated it to one op:

```
mul_mat(K view, q)    Tn=1  n_past=0   rel 2.6e+00  BAD
mul_mat(cont(K), q)   Tn=1  n_past=0   rel 9.4e-08  ok
```

ggml-vulkan's `mul_mat` calls a tensor contiguous when dims 0 and 1 are
(`ggml_vk_dim01_contiguous`) and then steps between batch matrices by
`ne00*ne01`, **ignoring `nb[2]`**. The self-attention K view is `[HD, n_kv, H]`
over a `[HD, n_ctx, H]` cache: rows contiguous, heads `n_ctx` rows apart. So
every head after the first read the wrong rows whenever `n_kv < n_ctx` — i.e.
on every decode step of `nllb-cli`. `full = exact` could not see it because
that pass sizes the cache to exactly the prefix, where there is no gap.
The same code is in ggml master as of 2026-09-25 (read, not run).

**Fix, in `src/nllb.cpp`:** on Vulkan only, `ggml_cont` the K view before the
matmul (`nllb_model::kv_k_cont`; `NLLB_KV_CONT=0/1` overrides). The copy is
exact: `nllb-dump-states` is IDENTICAL on the CPU with it forced on, and the CPU
path is untouched by default. CI runs the gate with it OFF and requires red, so
the day ggml fixes this the job says so.

---

## Things that go red and are not bugs

* `nllb-test-mmap` — not shipped: Windows has no mmap in this codebase, and
  neither OpenCL nor Vulkan can wrap host memory (`buffer_from_host_ptr = false`).
* `--backend metal` / `cuda` — correct: a named backend never falls back.
* A raw `K view` row BAD in `1b-probe-kv-ops-vulkan.txt` — expected, trap 55.
* `nllb-test-decoder` on an f16 file — trap 19; not in the zip.

## What is still not proven after a green session

* Anything about **speed** beyond this one machine: read `nllb-bench`'s
  `encoder pass` and `with lm_head` lines separately (step 13: a GPU can win the
  encoder and lose the one-token step).
* The **host-pointer pair** in `corrupt_backend.py` (`mask-through-data`,
  `mask-not-input`) stayed inert on lavapipe too, although lavapipe reports
  `uma: 0`. The likely reason is not the device: `ggml_backend_sched` allocates
  graph inputs in the CPU backend's buffer and copies them across itself, so a
  write through `->data` lands in host memory on every backend. The source
  rule in `ci.yml` remains the guard. (The harness needs a compiler, so it does
  not run on the QDC device.)
* The OpenCL backend has **never executed a graph anywhere** in this project.
  At the pinned ggml it accepts only Adreno and Intel GPUs (`unsupported GPU`
  for everything else, measured on Mesa rusticl in step 17), so the X Elite
  session is its first run.

## What changed from the old version of this page

* **Step 1b (MSVC `-A ARM64`) was wrong** — ggml refuses MSVC on Arm at
  configure time. Clang is required.
* The clang toolchain file's **`-ffp-model=fast`** is not used by the new build.
  `cmake/arm64-windows-llvm.cmake` is kept for a cross build if one is ever
  needed, with that flag still flagged as unmeasured.
* The x64-vs-Arm64 diff of step 4b is replaced by **runner-Arm64 vs
  device-Arm64 with the same binary**, which isolates the one variable the old
  comparison could not (it also changed compiler and ISA).
* `run.ps1` is in the repo (`tools/qdc/`), was parse-checked and rehearsed end
  to end under PowerShell 7 against Linux builds on lavapipe, and caught three
  of its own bugs there: a `$p:` interpolation, a trailing `\` on a directory
  argument, and a CPU-vs-GPU text comparison that passed when **both outputs
  were empty**.
