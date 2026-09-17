# Running nllb.cpp on a Snapdragon X Elite (Adreno X85) via Qualcomm Device Cloud

Target: the **CRD8380X Compute Reference Design, Windows on Arm64**, with an
**Adreno X85** — one of the three GPUs llama.cpp's OpenCL backend actually
verifies against. This is the real thing, not a substitute.

**The llama.cpp "Windows Arm64" page you screenshotted at the start is finally
the right recipe.** It was the wrong one for the ROG Strix (x86-64); it is the
right one here.

---

## Read this before you click "Next"

Two facts about QDC shape everything below:

1. **Devices are wiped after every session ends.** Nothing you install
   persists. So the goal is *one self-contained zip* that needs nothing
   installed on the device — no Visual Studio, no Python, no CMake.
2. **You give an SSH public key during session setup.** That is worth doing at
   the "Test Configuration" step you are on now: SSH + `scp` beats driving a
   mirrored screen for anything involving a command line and a log file.

So: **do not start the session yet.** Build the zip on the G531 first. Minutes
only start counting when the session starts, and you have 1000 — but a session
spent discovering you have nothing to run is still a session.

---

## Which machine does what — you need both

The M1 and the G531 are not interchangeable here, and the split is not
arbitrary.

| | job | why that machine |
|---|---|---|
| **G531 (Windows x64)** | builds the Arm64 binaries | A Windows target needs MSVC's headers and import libraries, and those exist on Windows. |
| **M1 Mac** | makes the fixture, holds the reference | It has the real checkpoint, `ref/`, and the verified CPU and Metal baselines. `nllb-dump-states --diff` needs *both* sides and the Mac is the side that already means something. |

**Can you cross-compile for Windows-on-Arm from the Mac instead? Technically
yes, practically no.** Two routes exist — `xwin` (fetches the MSVC CRT and
Windows SDK from Microsoft's CDN, then clang targets `aarch64-pc-windows-msvc`)
and `llvm-mingw` targeting `aarch64-w64-mingw32`. Both are real and maintained.
Neither is a path ggml or this project has ever taken, and picking one means
debugging a novel toolchain at the same time as a new GPU backend, with no way
to tell which layer a failure came from. The G531 route is twenty minutes of
installer checkboxes and is what llama.cpp's own Windows Arm64 page assumes.

**If you would rather build on the Mac, change the DEVICE, not the toolchain.**
QDC also offers the **Dragonwing RB3 Gen 2** — QCS6490, **Adreno 643**, running
Qualcomm Linux. Two things make that attractive from an M1:

* **`docker run --platform linux/arm64` is NATIVE on Apple silicon**, not
  emulated. Building aarch64 Linux binaries on the M1 is a native build at full
  speed, which is far less friction than any Windows cross-compile.
* Adreno 643 is an A6x **on an IoT platform**, which is exactly the category
  llama.cpp's OpenCL docs say is supported ("A6x GPUs with a recent driver and
  compiler are supported; usually found in IoT platforms") — as opposed to A6x
  in phones, which they warn against.

The trade is that Adreno 643 is a much weaker GPU than X85 and is *supported*
rather than *verified*, and the free minutes you were granted appear to be
X-Elite-specific — check the Devices tab before planning around it. If the goal
is "does this backend work on a real Adreno", X85 is the better evidence. If
the goal is "get something running with the least new toolchain", RB3 Gen 2
from an arm64 container on the M1 is the shorter road.

### The split, concretely

On the **Mac**, before touching the G531:

```bash
python tests/make_shape_model.py models/shape-small-f16.gguf --d-model 256 \
    --enc-layers 4 --dec-layers 4 --heads 4 --ffn 1024 --vocab 4000 \
    --n-ctx 64 --outtype f16
python tests/make_shape_ref.py ref-shape/ --vocab 4000 --n-ctx 64 \
    --max-new 24 --lengths 6 12 24 40

# the reference side of the --diff, on a machine whose CPU path is proven
./build/bin/nllb-dump-states models/shape-small-f16.gguf ref-shape/ mac-cpu.bin -t 8
```

Copy `models/shape-small-f16.gguf`, `ref-shape/` and `mac-cpu.bin` to the G531
and into the zip. Then after the session, back on the Mac:

```bash
./build/bin/nllb-dump-states --diff mac-cpu.bin xelite-cpu.bin
```

That one command answers a question nothing else in this plan does: **did the
Arm64 CPU path move?** If it did, no OpenCL number from that binary means
anything yet, and `-ffp-model=fast` in the toolchain file is the first suspect.

---

## Step 1 — cross-compilation toolchain on the G531 (one-off, ~20 min)

You are on x86-64 and the CRD is Arm64, so everything is cross-compiled.

**Build Tools for Visual Studio 2022** — in the installer, tick:

- **Desktop development with C++** (the base workload)
- **MSVC v143 — VS 2022 C++ ARM64/ARM64EC build tools** ← the target libraries
- **C++ Clang Compiler for Windows** ← the compiler

Both of the last two are needed. clang targets `arm64-pc-windows-msvc`, so it
still needs MSVC's ARM64 headers and import libraries; installing clang alone
fails at link time with an error that names a missing library rather than a
missing installer component.

Then OpenCL for the Arm64 target:

```powershell
cd $env:USERPROFILE
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg install opencl:arm64-windows      # NOTE the triplet
```

`opencl:x64-windows` is the wrong one — it would give you x64 import libraries
for an Arm64 binary, and the failure is a link error a long way from the cause.

Work from the **"x64_arm64 Cross Tools Command Prompt for VS 2022"** (Start
menu) so the ARM64 toolchain is on `PATH`.

---

## Step 1b — MSVC `-A ARM64` is the simpler route, and it is the better one here

The toolchain file in `cmake/arm64-windows-llvm.cmake` is llama.cpp's route
(clang, Ninja, `-march=armv8.7-a -ffp-model=fast`). **Try MSVC first.** The
Visual Studio generator cross-compiles to Arm64 with one flag:

```powershell
cmake -B build-arm64 -A ARM64 `
    -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=arm64-windows `
    -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON
cmake --build build-arm64 --config Release -j
```

Three reasons it is preferable, and the third is not about convenience:

1. **It needs no clang and no Ninja** — only the *MSVC v143 ARM64/ARM64EC build
   tools* component. One installer checkbox instead of two plus a toolchain file.
2. **It is the generator you already have working**, so a failure is about
   Arm64 and not about a build system you adopted the same afternoon.
3. **It avoids `-ffp-model=fast`.** That flag is in the clang toolchain file
   because llama.cpp puts it there, and it is a change of float semantics that
   nothing in this project's calibration ladder was measured under. MSVC's
   default is strict-ish IEEE, so the Arm64 CPU path has a far better chance of
   coming out bit-identical — which is exactly what step 4b below checks.

Binaries land in `build-arm64\bin\Release\` (multi-config generator — see
BUILD.md). Keep the clang toolchain file as the fallback: if MSVC's Arm64
codegen miscompiles ggml, switch to it and note in `BACKEND.md` that the
numbers came from a build with relaxed float semantics.

---

## Step 4b — the diff that actually controls for something

The plan used to be "dump on the Mac, dump on the CRD, diff". **That comparison
is confounded** if the two machines are on different ggml commits — and they
are: a Windows checkout reported `ggml version 0.20.2 / 8c63e709` while the
handoff records steps 13-14 as built against `0.23.0 / 7840aab`. A diff across
211 ggml commits measures the bump, not the architecture.

**The controlled comparison is x64-Windows against Arm64-Windows.** Same repo,
same submodule commit, same source, one variable: the architecture and the
compiler targeting it.

```powershell
# on the x64 build you already have working
.\build\bin\Release\nllb-dump-states.exe models\shape-small-f16.gguf ref-shape\ win-x64.bin -t 4

# ... and after the session, on the Arm64 dump brought back from the CRD
.\build\bin\Release\nllb-dump-states.exe --diff win-x64.bin xelite-cpu.bin
```

`IDENTICAL` means the Arm64 CPU path is sound and any OpenCL divergence is
OpenCL's. Anything else and the architecture moved the arithmetic, which has to
be understood before a single GPU number means anything.

Do the Mac diff too if you like, but fix the submodule pin first — otherwise it
answers a different question than the one you asked.

---

## Step 2 — the probe first, because it is 21 KB and answers the question

`ocl-probe` is plain C, links nothing, and `dlopen`s the vendor OpenCL library
at runtime. On Windows that is `LoadLibraryA("OpenCL.dll")`, which the Adreno
driver on the CRD installs. So it needs no ICD loader and no ggml, and it tells
you whether Adreno X85 will work at all.

```powershell
cmake -B build-arm64 -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE=cmake\arm64-windows-llvm.cmake `
    -DCMAKE_PREFIX_PATH="$env:USERPROFILE\vcpkg\installed\arm64-windows" `
    -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm64 --target ocl-probe
```

`-DNLLB_ADRENO=ON` is right here and nowhere else — `NLLB_OPENCL` defaults
ggml's Adreno kernels **off** because upstream's default (on) is wrong on every
other target. This is the target it was written for.

---

## Step 3 — the rest of the build

```powershell
cmake --build build-arm64 -j
```

Kernels are embedded by default, so the binaries are self-contained: no `.cl`
files to ship alongside them.

---

## Step 4 — a fixture, and why you do NOT need the real model yet

**The single most valuable run on a new backend needs no weights at all.**
`nllb-test-cache` compares two C++ paths — the KV-cached decoder against the
un-cached one — so the weights cancel and a random model with the right shape
is a complete test. Its `full = exact` is the trap-15 assertion (the cache
write still precedes its read, through whatever node reordering the backend
does) and the trap-16 one (bit-identity at equal sequence length, on a GPU).

That means a **33 MB** file instead of a 1.24 GB upload:

```powershell
pip install numpy gguf
python tests\make_shape_model.py models\shape-small-f16.gguf --d-model 256 `
    --enc-layers 4 --dec-layers 4 --heads 4 --ffn 1024 --vocab 4000 `
    --n-ctx 64 --outtype f16
python tests\make_shape_ref.py ref-shape\ --vocab 4000 --n-ctx 64 `
    --max-new 24 --lengths 6 12 24 40
```

Save the real `nllb-600M-f16.gguf` for a second session, once the plumbing is
known good.

---

## Step 5 — the zip

```
nllb-xelite.zip
├─ ocl-probe.exe
├─ nllb-cli.exe
├─ nllb-test-cache.exe
├─ nllb-test-precision.exe
├─ nllb-dump-states.exe
├─ models\shape-small-f16.gguf
├─ ref-shape\...
└─ run.ps1
```

`run.ps1` — script it, because the device is wiped each time and you do not
want to retype this into a mirrored screen:

```powershell
$ErrorActionPreference = "Continue"
Start-Transcript -Path "$PSScriptRoot\qdc-log.txt"

Write-Host "=== 1. will Adreno X85 work at all? ==="
& "$PSScriptRoot\ocl-probe.exe"
Write-Host "probe exit code (= usable devices): $LASTEXITCODE"

Write-Host "`n=== 2. what does the backend registry see? ==="
& "$PSScriptRoot\nllb-cli.exe" --list-backends

Write-Host "`n=== 3. THE GATE: cached vs un-cached, on OpenCL ==="
$env:NLLB_ARENA_TRACE = "1"
$env:NLLB_BACKEND     = "opencl"
& "$PSScriptRoot\nllb-test-cache.exe" "$PSScriptRoot\models\shape-small-f16.gguf" "$PSScriptRoot\ref-shape\" -t 4

Write-Host "`n=== 4. CPU vs OpenCL: exact ids and the margin audit ==="
Remove-Item Env:\NLLB_BACKEND
& "$PSScriptRoot\nllb-test-precision.exe" `
    "$PSScriptRoot\models\shape-small-f16.gguf" "$PSScriptRoot\models\shape-small-f16.gguf" `
    "$PSScriptRoot\ref-shape\" -t 4 --backend cpu:opencl

Write-Host "`n=== 5. a CPU dump, to diff against the Mac later ==="
& "$PSScriptRoot\nllb-dump-states.exe" "$PSScriptRoot\models\shape-small-f16.gguf" `
    "$PSScriptRoot\ref-shape\" "$PSScriptRoot\xelite-cpu.bin" -t 4

Stop-Transcript
```

Upload it at the **Test Configuration** step (`.zip` is an accepted Compute
package format), or `scp` it once the session is up if you configured SSH.

---

## Step 6 — what to read in the output, in order

**1. The probe.** `VERDICT: USABLE` and you are in business. Anything else and
stop reading — a driver decides this, not the repo. Expect Adreno X85 to pass,
since llama.cpp verifies against exactly this GPU.

**2. `--list-backends`** must show a `GPUOpenCL` line. Read the `ggml_opencl:`
lines on **stderr** above it — a dropped device explains itself there and
nowhere else.

**3. `splits 1`** in the arena trace on the cached step. More than one means an
op fell back to the CPU with copies around it. Worth watching here: OpenCL
*has* a `diag_mask_inf` kernel where Metal does not, so this may come for free
where Metal had to be worked at. That is a prediction, not a result.

**4. `full = exact`, 0 differing floats.** This is the one that matters. It
says the KV write still precedes its read through the backend's own graph
reordering (trap 15), and that bit-identity at equal sequence length holds on a
GPU (trap 16). Metal passed both; nothing guarantees the next backend does.

**5. `[arena]` bit-identical** — nothing in the graph reads memory it did not
write, under gallocr, on a device.

**6. Exact ids at `cpu:opencl`.** Expect trap 48 to reappear with different
numbers: a backend's precision is a property of the **kernel it picks**, not of
the tensor types. Metal silently runs matmuls with more than 8 columns in half.
If the divergence figures split by sequence length, that is the same effect.

---

## The row worth going out of your way for

`tests/corrupt_backend.py` has two mutations, `mask-through-data` and
`mask-not-input`, marked `report` rather than `fire`. They write a graph input
through `->data` instead of through the backend, and they are **invisible on
every machine this project has ever run on** — the CPU's buffers are host
memory and Apple's Metal buffers are shared, so both hide the bug.

OpenCL reports `buffer_from_host_ptr = false`. **An Adreno is the first real
candidate that can see this class.**

The harness compiles its mutations, so it cannot run on the device. The
adaptation is to **cross-compile the mutants on the G531 and ship them in the
zip**: run `corrupt_backend.py` locally to generate the mutated sources, build
each into its own `.exe` for Arm64, and have `run.ps1` execute all seven and
diff their output against the baseline. If those two fire, promote them to
`fire` in the matrix and name the backend the assertion rests on. If they stay
inert, check whether the Adreno driver is mapping host memory anyway before
concluding the code is clean.

---

## Things that will go red, and are not bugs

* **`nllb-test-mmap`** fails, twice over: Windows has no mmap in this codebase
  (`NLLB_HAVE_MMAP` is 0 in the `_WIN32` branch) *and* OpenCL cannot wrap host
  memory, so the loader unmaps on purpose. That gate's claim lives on the CPU,
  on macOS or Linux.
* **`--backend metal`** fails. Correct — a named backend never silently falls
  back; only `auto` does.
* **`nllb-test-decoder`** against an f16 file fails. Trap 19, unrelated: its
  tolerances *are* the f32 noise floor.
* **`verify_translation.py` and `corrupt_backend.py`** need Python and a
  compiler, so they do not run on the device as-is.

## One thing to verify that nobody has

The toolchain file passes **`-ffp-model=fast`**, inherited from llama.cpp. That
is a change of float semantics, and every number in this project's calibration
ladder was measured without it. `nllb-dump-states --diff` is the instrument:
build twice, once with the flag and once without, and diff. If it is not
IDENTICAL, drop the flag rather than explain the difference away. Step 5 of
`run.ps1` dumps the Arm64 CPU path so you have one side of that comparison
without spending extra session minutes.
