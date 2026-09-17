# Getting an Adreno GPU, and running nllb.cpp on it

## Short answer: you probably already have one, and the fallback is free

| route | cost | what you get |
|---|---|---|
| **The Android phone you already own** | €0 | If it is a Snapdragon, it has an Adreno. Check it in 30 seconds — step 1. |
| **[Qualcomm Device Cloud](https://qdc.qualcomm.com)** | €0 | Real Qualcomm reference devices — **Snapdragon 8 Gen 3 and X Elite**, i.e. exactly the GPUs llama.cpp verifies against. Free minutes on signup, `adb` access, and you can **push and run your own binaries**. This is the answer if your phone turns out to be the wrong chip. |
| A used flagship phone | ~€150-300 | Snapdragon 8 Gen 1/2/3 → Adreno 730/740/750. Get **≥ 8 GB RAM**: the f16 GGUF is 1.24 GB and `q8emb` is 0.996 GB. |
| A Snapdragon X Elite laptop | ~€900+ | Adreno X85, verified by llama.cpp. Only worth it if you want Windows-on-Arm anyway. |

**Do not go looking for the Snapdragon Dev Kit** (the $899 mini-PC). Qualcomm
**cancelled it and refunded every order** — it is not a thing you can buy.

### Which GPU brands support this API at all

"Supports OpenCL" and "works with ggml's OpenCL backend" are different
questions. The second one needs `cl_khr_fp16`, subgroups on an OpenCL 3.0
device, and a compiler that survives `-cl-fast-relaxed-math`.

| vendor | ggml OpenCL | notes |
|---|---|---|
| **Qualcomm Adreno** | **yes — the primary target** | The backend was written for it. 750 / 830 / X85 verified. |
| **AMD** (Radeon, Instinct) | **yes** | Mature OpenCL with real fp16. The best *desktop* OpenCL device, and the easiest to rent. |
| **Intel** (Arc, iGPU) | **yes, but slow** | ggml's own docs: Intel GPUs work "although the performance is not optimal". |
| **Arm Mali / Immortalis** | **maybe — measure it** | Arm ships a real OpenCL driver and Mali does fp16 well. The catch is below. |
| **NVIDIA** | **effectively no** | No conformant `cl_khr_fp16`, no half math builtins. NVIDIA calls it experimental. Use CUDA. |
| **Apple** | n/a | OpenCL deprecated. Metal already ships here (step 13). |
| **Imagination PowerVR** | rarely | Driver exists; almost never exposed on retail Android. |

### Mali: not a dead end, but check before assuming

A Mali phone is **not** automatically out. Arm's OpenCL driver is real and Mali
handles fp16 well — and since `NLLB_OPENCL` defaults `NLLB_ADRENO=OFF`, what
you build is generic OpenCL rather than Adreno-tuned kernels.

The actual obstacle is distribution: **`libOpenCL.so` is optional for device
vendors**, and many ship Vulkan only. This is a long-standing complaint against
Android — Arm's own Compute Library issue tracker has "most of the Android
phones do not have the OpenCL Driver" against Huawei, Xiaomi and Samsung
handsets. It is a packaging decision by the OEM, not a property of the GPU.

So one command decides it:

```powershell
adb shell "ls -l /vendor/lib64/libOpenCL.so"
```

Present → push `ocl-probe` (step 2) and let it answer properly. Absent → this
phone has no OpenCL at all, and no amount of work on this repo changes that.

**And if you have a Mali, Vulkan is probably the better backend anyway.** ggml's
Vulkan backend is broadly portable rather than vendor-tuned, Mali's Vulkan
driver is always present, and people do run llama.cpp on Mali that way. It is
the same shape of change as step 14 — a build flag — and `nllb_pick_device`
already resolves `vulkan` and `vk`:

```
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
```

That is a separate step and nothing in this repo has tried it.

### Which Adrenos actually work

llama.cpp's OpenCL backend lists as verified:

| GPU | SoC |
|---|---|
| Adreno 750 | Snapdragon 8 Gen 3 |
| Adreno 830 | Snapdragon 8 Elite |
| Adreno X85 | Snapdragon X Elite |

A6x GPUs are supported "with a recent driver and compiler, usually found in IoT
platforms" — the docs explicitly warn that **older A6x variants in phones lack
the driver and compiler support**. Adreno 610 fails outright: it does not answer
the `CL_DEVICE_OPENCL_C_ALL_VERSIONS` query the backend makes, and the report
was closed stale.

So: a recent flagship Snapdragon is the target. A mid-range or older phone will
probably not work, and step 2 tells you which you have without building
anything.

---

## Step 1 — what is in the phone you already have

You already have `adb` and NDK 27.2 installed for the Speech app, so this is
free. PowerShell:

```powershell
adb shell getprop ro.soc.manufacturer
adb shell getprop ro.soc.model
adb shell getprop ro.hardware
adb shell "ls -l /vendor/lib64/libOpenCL.so"
```

- `ro.soc.manufacturer` = **QTI / Qualcomm** → you have an Adreno. Good.
- The last line is the one that matters: **`/vendor/lib64/libOpenCL.so` is the
  driver.** If it is missing, this SoC has no OpenCL at all — Mali (MediaTek)
  and Xclipse (recent Exynos) devices often ship Vulkan only.

Map `ro.soc.model` to a GPU against the table above. `SM8650` = 8 Gen 3 =
Adreno 750, which is the best case.

---

## Step 2 — the two-minute test, before you build anything

`tools/ocl-probe.c` answers "will ggml's OpenCL backend work on this device"
without building ggml, without an ICD loader, and without root. It **dlopen()s
the vendor's `libOpenCL.so` at runtime**, so it is a single self-contained
binary — 21 KB, linked against libc and nothing else.

That matters here specifically: llama.cpp's Android recipe has you
cross-compile the Khronos ICD loader *before* you can find out whether the GPU
is usable, which is backwards. This skips it.

Headers only — no build, they are header-only:

```powershell
cd $env:USERPROFILE\dev
git clone https://github.com/KhronosGroup/OpenCL-Headers
```

Compile with the NDK's clang directly. No CMake needed for one file:

```powershell
$NDK = "$env:LOCALAPPDATA\Android\Sdk\ndk\27.2.12479018"
$CC  = "$NDK\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android28-clang.cmd"

& $CC -O2 -I"$env:USERPROFILE\dev\OpenCL-Headers" `
      tools\ocl-probe.c -o ocl-probe-arm64

adb push ocl-probe-arm64 /data/local/tmp/
adb shell chmod +x /data/local/tmp/ocl-probe-arm64
adb shell /data/local/tmp/ocl-probe-arm64
```

It prints, per device, the two questions in the order ggml asks them:

1. **Will ggml register it?** `cl_khr_fp16`, plus subgroups on an OpenCL 3.0
   device. Fail here and the device is dropped at startup.
2. **Will ggml's kernels compile on it?** It builds a half math builtin under
   ggml's *exact* options, `-cl-fast-relaxed-math` included.

Question 2 is the one `clinfo` cannot answer, and it is the expensive one — a
device can pass question 1, make ggml print `device FP16 support: true`, and
then fail at kernel load. That is not hypothetical; it is what happened on
Intel's CPU runtime during step 14:

```
    cl_khr_fp16                        yes
    half math builtin under ggml's flags   NO (clBuildProgram -11)
    JIT session error: Symbols not found: [ _Z6exp_rmDh ]
    VERDICT: ggml REGISTERS this device and then fails loading kernels.
```

**`VERDICT: USABLE` on your phone is the green light for step 3.** Anything else
and you want Qualcomm Device Cloud instead — no amount of work on this repo
changes a driver.

*(The probe is verified on x86-64 against two deliberately-broken runtimes,
where it called both correctly at the right stage. The arm64 cross-compile
above is **transcribed, not run** — no NDK was reachable from the sandbox it
was written in.)*

---

## Step 3 — the full Android build

Only worth doing once step 2 says USABLE.

### OpenCL headers into the NDK sysroot, and an ICD loader

ggml itself *does* need the loader at link time (only the probe avoids it).

```powershell
$NDK = "$env:LOCALAPPDATA\Android\Sdk\ndk\27.2.12479018"
$SYS = "$NDK\toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\include"

cd $env:USERPROFILE\dev\OpenCL-Headers
Copy-Item -Recurse -Force CL $SYS

cd $env:USERPROFILE\dev
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader
cmake -B build_ndk -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE="$NDK\build\cmake\android.toolchain.cmake" `
    -DOPENCL_ICD_LOADER_HEADERS_DIR="$SYS" `
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24 `
    -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=Release
cmake --build build_ndk
Copy-Item build_ndk\libOpenCL.so "$NDK\toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\aarch64-linux-android"
```

### nllb.cpp

```powershell
cd <your nllb.cpp>
cmake -B build-android -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE="$NDK\build\cmake\android.toolchain.cmake" `
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 `
    -DBUILD_SHARED_LIBS=OFF `
    -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON `
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j
```

**`-DNLLB_ADRENO=ON` is the one flag that differs from every other target.**
`NLLB_OPENCL` defaults the Adreno kernels OFF because ggml's upstream default
(ON) is wrong everywhere except here. This is the one place you want them back.

Kernels are embedded by default, so the binary is self-contained — no `.cl`
files to push alongside it.

---

## Step 4 — run it

```powershell
adb push build-android\bin\nllb-cli /data/local/tmp/
adb push models\nllb-600M-f16.gguf  /data/local/tmp/
adb shell chmod +x /data/local/tmp/nllb-cli

adb shell "cd /data/local/tmp && ./nllb-cli -m nllb-600M-f16.gguf --src eng_Latn --tgt deu_Latn --backend opencl -v 'Hello, world!'"
```

No root, no APK — `/data/local/tmp` is executable for the shell user. Copy the
GGUF across from the Mac rather than re-converting, so the file is provably the
same object every existing number was measured against.

Then the acceptance run. This is `BACKEND.md`'s step-14 table, and **every cell
in it is currently blank**:

```powershell
adb push ref /data/local/tmp/ref
adb push build-android\bin\nllb-test-cache     /data/local/tmp/
adb push build-android\bin\nllb-test-precision /data/local/tmp/

# 1. nothing falls back. >1 split = an op ran on the CPU with copies around it
adb shell "cd /data/local/tmp && NLLB_ARENA_TRACE=1 NLLB_BACKEND=opencl ./nllb-test-cache nllb-600M-f16.gguf ref/ -t 4"

# 2. exact ids against the CPU - same file twice, the kernels are the variable
adb shell "cd /data/local/tmp && ./nllb-test-precision nllb-600M-f16.gguf nllb-600M-f16.gguf ref/ -t 4 --backend cpu:opencl"
```

`verify_translation.py` and `corrupt_backend.py` need Python and a compiler, so
run those against a device over `adb` from the PC rather than on the phone —
or leave them for Qualcomm Device Cloud, where the same binaries apply.

---

## What to watch for, and two of these are your own lessons

* **Thermal throttling makes every performance number a lie.** You already know
  this from the Speech app — aggressive `STEP_MS`/`n_threads` settings throttle
  mid-range devices. A decode benchmark on a phone is a *sustained* load, so
  run it twice with a cool-down and quote the second, and treat any single
  number from a phone in a case as provisional. `nllb-bench` reports a noise
  floor rather than a number when the difference is not real — trust that.
* **16 KB page size.** Your Play Store note applies if any of this ever ships
  inside the app: `-Wl,-z,max-page-size=16384`, AGP 8.3+, NDK 27+. It does not
  matter for a binary you only `adb push`.
* **`splits 1` on the cached step, or an op fell back** to the CPU with copies
  around it. Read `NLLB_ARENA_TRACE=1` before reading any timing.
  Interesting here: OpenCL *has* a `diag_mask_inf` kernel where Metal does not,
  so this may come for free where Metal had to be worked at. That is a
  prediction, not a result.
* **Trap 48 will reappear with different numbers.** A backend's precision is a
  property of the kernel it picks, not of the tensor types you hand it — Metal
  silently runs matmuls with more than 8 columns in half. Read Adreno's
  `mul_mat` selection rule before trusting any divergence figure, and watch for
  a table that splits by sequence length.
* **The host-pointer pair may finally fire.** `mask-through-data` and
  `mask-not-input` in `tests/corrupt_backend.py` are marked `report` rather than
  `fire` because host memory hides that bug class — the CPU and Apple's unified
  memory both do, so no machine in this project has ever been able to catch
  them. OpenCL reports `buffer_from_host_ptr = false`, which makes an Adreno the
  **first real candidate**. If they fire, promote them to `fire` and name the
  backend the assertion rests on. If they stay inert, check whether the device
  is mapping host memory anyway before concluding the code is clean.
* **`nllb-test-mmap` will fail, twice over and for two independent reasons:**
  OpenCL cannot wrap host memory, so the loader unmaps on purpose. Expected, not
  a bug. That gate's claim lives on the CPU.
* **Do not expect a GPU to be automatically faster.** On the M1, Metal wins the
  encoder 9.8x and *loses* the one-token decode step at 0.93x, because that step
  is a bandwidth-bound matvec. Read `encoder pass` and `with lm_head`
  separately; one wall-clock number averages two opposite results. On a phone
  the encoder win is the one to expect first.

---

## Qualcomm Device Cloud, if the phone is the wrong chip

[qdc.qualcomm.com](https://qdc.qualcomm.com) — sign up with a Qualcomm ID, free
minutes on first login. It gives real QRD/MTP reference devices, `adb` for
mobile targets and SSH for compute ones, and it explicitly supports uploading
and executing your own binaries via `adb push`. Minutes are only counted once
the session actually starts, not while you wait for a device.

Everything in steps 3 and 4 applies unchanged — you are still cross-compiling
with the NDK on your PC and pushing a binary over `adb`. The only difference is
which device is on the other end. Which also makes it the cheapest way to get
the **Adreno 830 / X85** numbers rather than whatever your phone happens to
have.
