# Building nllb.cpp on Windows, with OpenCL

Written for an **ASUS ROG Strix G531** (Core i7-9750H + Intel UHD 630 iGPU +
an NVIDIA GTX 1650 / 1660 Ti / RTX 2060 depending on the variant), which is
**x86-64, not Arm64**. If you came here from llama.cpp's `docs/build.md`, note
that its OpenCL section is split into *Android* and *Windows Arm64* — the Arm64
recipe is for Snapdragon X Elite machines and its
`-DCMAKE_TOOLCHAIN_FILE=…/arm64-windows-llvm.cmake` line will not work here.
The OpenCL-Headers / OpenCL-ICD-Loader part of it *is* reusable; drop the
toolchain file.

**This is the project's first Windows build.** `claude/handoff.md` lists "the
mmap path on anything but macOS/arm64" under *not proven*, and that is still
true — see "What will fail, and why none of it is a bug" at the end, so you
don't spend an evening chasing an expected red.

---

## The CPU build WORKS — start here

Windows x86-64, CPU backend, verified end to end on a ROG Strix G531. Full
write-up and every gotcha: `claude/step15-windows-cpu.md`.

```powershell
git clone <remote> nllb.cpp && cd nllb.cpp
git submodule update --init --recursive        # a clone does NOT bring ggml

pip install -r requirements-convert.txt        # numpy gguf sentencepiece REGEX
pip install huggingface_hub
hf download facebook/nllb-200-distilled-600M --local-dir models\nllb-600M
python tools\convert_nllb_to_gguf.py models\nllb-600M --outfile models\nllb-600M-f16.gguf --outtype f16

cmake -B build -A x64
cmake --build build --config Release -j

./build/bin/Release/nllb-cli -m models/nllb-600M-f16.gguf --src eng_Latn --tgt deu_Latn "Hello, World"
```

Four things that are not obvious and each cost time:

* **`--config Release` is mandatory** and `-DCMAKE_BUILD_TYPE=Release` is
  ignored — the VS generator is multi-config. Binaries land in
  `build\bin\Release\`, not `build\bin\`.
* **An old MinGW on `PATH` will be picked up and may be 32-bit**, which cannot
  address the 1.24 GB model. Add `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`
  if in doubt.
* **A failed configure poisons `build\`** — `Remove-Item -Recurse -Force build`
  before retrying.
* **PowerShell renders UTF-8 as mojibake** (`möchte` → `m├╢chte`). The bytes are
  fine. `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`, or redirect
  to a file. Step 9's rule holds: for a non-ASCII target, read the bytes, never
  the terminal.

Everything below is about the **OpenCL** build, which is a separate matter.

---

## 0. Read this before installing anything

Your laptop has two GPUs and **the fast one is the one OpenCL cannot use.**

| device | ggml OpenCL? | why |
|---|---|---|
| NVIDIA GTX 1650 / 1660 Ti / RTX 2060 | **almost certainly not** | NVIDIA's OpenCL does not claim conformance for `cl_khr_fp16` and has **no half math builtins**. ggml requires the first and its kernels use the second. |
| Intel UHD 630 (iGPU) | **probably yes** | Intel's driver exposes `cl_khr_fp16` and `cl_intel_subgroups`. ggml's own docs say Intel GPUs work "although the performance is not optimal". |

So the realistic outcome is: **OpenCL runs on your slowest GPU.** ggml's OpenCL
backend is written for Qualcomm Adreno first; Intel is a secondary target and
NVIDIA is not a target at all.

That is worth doing **if** what you want is the OpenCL path itself — a rehearsal
for Adreno/Android, and the numbers that fill in step 14's blank acceptance
table in `BACKEND.md`. It is the wrong thing to do **if** what you want is a
fast nllb.cpp on this laptop: for that the answer is `-DGGML_CUDA=ON` and
`--backend cuda`, which needs no code change here because step 13's port is
device-agnostic and `nllb_pick_device` already resolves `cuda`.

**Step 4 settles this empirically in about five seconds. Do not guess, and do
not skip it.**

---

## 1. A compiler

**VS Code is an editor, not a compiler.** It will not build this on its own.
You need MSVC:

1. Download **Build Tools for Visual Studio 2022** (free — you do not need the
   full Visual Studio IDE) from Microsoft's downloads page.
2. In the installer tick the **"Desktop development with C++"** workload. That
   gives you the MSVC compiler, the Windows SDK, **CMake**, and **Ninja**.
3. Afterwards, use the **"x64 Native Tools Command Prompt for VS 2022"** from
   the Start menu for every command below. A plain `cmd` or PowerShell will not
   have the compiler on `PATH`, and the error when it is missing is unhelpful.

Also install **Git for Windows** if you do not have it.

Keep using VS Code to edit — just build from that terminal, or point VS Code's
CMake Tools extension at the MSVC kit.

---

## 2. OpenCL headers and an ICD loader

These are the *build-time* pieces. They are not the driver — the driver is
already on your machine and registers itself under
`HKLM\SOFTWARE\Khronos\OpenCL\Vendors`. What you are installing here is what
`find_package(OpenCL REQUIRED)` needs in order to compile and link.

### Route A — vcpkg (recommended, one command)

```bat
cd %USERPROFILE%
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg install opencl:x64-windows
```

Then add this to every `cmake -B` line below:

```
-DCMAKE_TOOLCHAIN_FILE=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake
```

### Route B — Khronos, by hand (what your screenshot shows, minus the Arm64 bits)

```bat
mkdir %USERPROFILE%\dev && cd %USERPROFILE%\dev

git clone https://github.com/KhronosGroup/OpenCL-Headers && cd OpenCL-Headers
cmake -B build -G Ninja -DBUILD_TESTING=OFF -DOPENCL_HEADERS_BUILD_TESTING=OFF ^
    -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF ^
    -DCMAKE_INSTALL_PREFIX="%USERPROFILE%\dev\opencl"
cmake --build build --target install

cd %USERPROFILE%\dev
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && cd OpenCL-ICD-Loader
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="%USERPROFILE%\dev\opencl" ^
    -DCMAKE_INSTALL_PREFIX="%USERPROFILE%\dev\opencl"
cmake --build build --target install
```

Then add to every `cmake -B` line below:

```
-DCMAKE_PREFIX_PATH=%USERPROFILE%\dev\opencl
```

Note the differences from the page you screenshotted: **no**
`-DCMAKE_TOOLCHAIN_FILE=…arm64-windows-llvm.cmake`, and `cmake -B build`
instead of `mkdir build && cd build && cmake ..` (same thing, fewer steps).

### Make sure the *driver* side exists too

- **Intel UHD 630** — install the Intel Graphics driver. On a ROG laptop
  Windows Update often ships a minimal display driver *without* the OpenCL
  runtime. If step 4 shows no Intel platform, this is why: get the driver from
  Intel or from ASUS's support page for the G531 rather than from Windows
  Update.
- **NVIDIA** — the GeForce driver installs `nvopencl64.dll` automatically.

---

## 3. The repo

```bat
git clone <your nllb.cpp remote> && cd nllb.cpp
git submodule update --init --recursive
```

`vendor/ggml` must be populated — a `git archive` tarball does not include
submodules, and an empty `vendor/ggml` fails at `add_subdirectory` with a
message that does not mention submodules.

---

## 4. Build the probe FIRST and run it

This is the decision point, and it costs one target instead of a full build.

```bat
cmake -B build-cl -G Ninja -DNLLB_OPENCL=ON -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build-cl --target ocl-probe
.\build-cl\bin\ocl-probe.exe
```

`ocl-probe` links the OpenCL loader **and nothing else** — no ggml, no nllb. It
asks the two questions in the order ggml asks them, per device:

1. **Will ggml register it?** `cl_khr_fp16`, plus subgroups on an OpenCL 3.0
   device.
2. **Will ggml's kernels compile on it?** It builds a half math builtin under
   ggml's *exact* compile options, including `-cl-fast-relaxed-math`.

Question 2 is the one `clinfo` cannot answer and the one that costs an
afternoon, because a device can pass question 1, print `device FP16 support:
true`, and then fail at kernel load. Here is what it looked like on a runtime
that does exactly that:

```
  device 0: Intel(R) Xeon(R) Processor @ 2.80GHz
    cl_khr_fp16                        yes
    cl_khr_subgroups/cl_intel_subgroups yes
    half math builtin under ggml's flags   NO (clBuildProgram -11)
    --- build log ---
    JIT session error: Symbols not found: [ _Z6exp_rmDh ]
    -----------------
    VERDICT: ggml REGISTERS this device and then fails loading kernels.
```

**What to expect on your G531:** the NVIDIA device dropped at question 1 (the
probe says so by name, and says to use CUDA), and the Intel UHD 630 reaching
`VERDICT: USABLE`. If the Intel line is missing entirely, go back to the driver
note in step 2.

If **no** device is usable, stop here and go to step 7 — there is nothing to fix
in this repo, and the probe's exit code is the number of usable devices, so
`ocl-probe.exe & echo %errorlevel%` is scriptable.

---

## 5. The full build

```bat
cmake --build build-cl -j
.\build-cl\bin\nllb-cli.exe --list-backends
```

Expect a `GPUOpenCL - Intel(R) UHD Graphics 630 (gpu)` line. Read the
`ggml_opencl:` lines on **stderr** above it — that is where a dropped device
explains itself.

If you have more than one OpenCL platform (likely: Intel *and* NVIDIA),
**ggml enumerates devices on one platform only**, so the other one is invisible
rather than merely unselected:

```bat
set GGML_OPENCL_PLATFORM=Intel
.\build-cl\bin\nllb-cli.exe --list-backends
```

---

## 6. Test it

You need a GGUF. Copy `models/nllb-600M-f16.gguf` across from the Mac — do not
re-convert, so that the file is provably the same object every existing number
was measured against. **f16, not f32**: f16 is the default and the only
configuration with headroom above 1.

Smoke test:

```bat
.\build-cl\bin\nllb-cli.exe -m models\nllb-600M-f16.gguf --src eng_Latn --tgt deu_Latn ^
    --backend opencl -v "Hello, world!"
```

Then the acceptance run, in order. This is `BACKEND.md`'s step-14 table, and
every cell in it is currently blank:

```bat
:: 1. nothing falls back. >1 split = an op ran on the CPU with copies around it
set NLLB_ARENA_TRACE=1
set NLLB_BACKEND=opencl
.\build-cl\bin\nllb-test-cache.exe models\nllb-600M-f16.gguf ref\ -t 8
set NLLB_ARENA_TRACE=

:: 2. exact ids against the CPU - same file twice, the kernels are the variable
.\build-cl\bin\nllb-test-precision.exe models\nllb-600M-f16.gguf models\nllb-600M-f16.gguf ^
    ref\ -t 8 --backend cpu:opencl

:: 3. THE PRODUCT. 9 translations, byte-identical to HuggingFace
python tests\verify_translation.py .\build-cl\bin\nllb-cli.exe models\nllb-600M-f16.gguf ^
    ref\ -t 8 --backend opencl

:: 4. does the gate still fire? The table must match the CPU's
set NLLB_BACKEND=opencl
python tests\corrupt_backend.py build-cl models\nllb-600M-f16.gguf ref\ -t 8
```

`ref/` has to come across from the Mac too, or be re-dumped — the goldens are
per-checkpoint.

**The row worth going out of your way for** is the last one. In
`tests/corrupt_backend.py`, `mask-through-data` and `mask-not-input` are marked
`report` rather than `fire` because they are invisible on host memory — the CPU
and Apple's unified memory both hide that bug class, so no machine in this
project has ever been able to catch it. OpenCL says `buffer_from_host_ptr =
false`, which makes your iGPU the **first candidate that might actually see
it**. If those two fire, promote them to `fire` in the matrix and name the
backend the assertion rests on.

---

## 7. If OpenCL is a dead end here: CUDA

Nothing in `src/nllb.{h,cpp}` is backend-specific, so this is a build flag:

```bat
cmake -B build-cu -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cu -j
.\build-cu\bin\nllb-cli.exe --list-backends
.\build-cu\bin\nllb-cli.exe -m models\nllb-600M-f16.gguf --tgt deu_Latn --backend cuda -v "Hello!"
```

Needs the **CUDA Toolkit** installed. Then run the same four acceptance
commands with `cuda` in place of `opencl`. This is the path that gives your
laptop real numbers, and a third backend in the calibration ladder next to CPU
and Metal.

Expect trap 48 to reappear with different numbers: a backend's precision is a
property of the **kernel it picks**, not of the tensor types you handed it, so
read the divergence figures before trusting them and watch for a table that
splits by sequence length.

---

## What will fail, and why none of it is a bug

* **`nllb-test-mmap` fails on Windows. Expected.** `NLLB_HAVE_MMAP` is `0` in
  the `_WIN32` branch, so `nllb_mmap_file` returns `nullptr` and the loader
  reads instead — which is correct and always has been. The *gate* then has
  nothing to compare, so it fails rather than printing a green PASS about a run
  in which mmap never happened (the step-4 SKIP lesson). Its message names this
  as cause 2. **That gate's claim lives on macOS/Linux.**
* **`nllb-test-mmap` would also fail under `--backend opencl` for a second,
  different reason**: OpenCL reports `buffer_from_host_ptr = false`, so even
  where mmap exists the loader unmaps on purpose. Two independent reasons, same
  red.
* **`--backend metal` fails.** Correct — a named backend never silently falls
  back to the CPU; only `auto` does.
* **`nllb-test-decoder` against an f16 file fails.** Trap 19, unrelated to any
  of this: its tolerances *are* the f32 noise floor. Use `nllb-test-precision`.
* **ggml prints `warning, the default device is not a GPU`** if it picks a
  CPU-backed OpenCL device. It registers it anyway, and reports its type as
  `GPU` regardless of what the driver says.

If something outside this list goes red, it is worth reporting — and
`nllb-dump-states --diff` against a CPU dump from the Mac is the instrument
that says whether the Windows CPU path itself moved.
