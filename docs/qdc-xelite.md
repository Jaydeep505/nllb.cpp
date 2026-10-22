# nllb.cpp on a Snapdragon X Elite (Adreno X1-85) via Qualcomm Device Cloud

Target: the **Snapdragon X Elite Compute Reference Design, Windows 11 on Arm**,
Adreno X1-85 GPU, through the **Vulkan** backend. The CPU path is tested in the
same session as the reference every GPU number is compared against.

## The short version

1. Push the branch to GitHub (ggml submodule at the pin in `xelite.yml`,
   currently `ffa4e8b`).
2. **Actions → xelite → Run workflow** (pick the branch). It builds natively on
   a `windows-11-arm` runner, tests the CPU path there, and uploads
   **`nllb-xelite.zip`** (no model inside).
3. Start a QDC session on the X Elite device, copy the zip over, unzip, and run
   from PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1
   ```

4. Copy **`results.zip`** off the device before the session ends. QDC wipes it.

The first session needs **no model**: every gate in it compares two C++ paths on
a small random fixture, so the weights cancel. Once it is green, bring the real
model to a second session (download it on the device with
`.\models\download-nllb-model.ps1 f16` from a repo checkout, or copy it over),
plus `ref\` from the Mac if you want the real-model cache/precision gates:

```powershell
.\run.ps1 -Model C:\m\nllb-600M-f16.gguf -Ref C:\m\ref
```

Without `-Ref` section 7 still runs the CPU-vs-GPU text comparison and the bench.

## Why it is built on a windows-11-arm runner

| | cross-compile from x64 | native on Arm64 (what the workflow does) |
|---|---|---|
| compiler | MSVC `-A ARM64` → **ggml refuses it**: "MSVC is not supported for ARM, use clang" | **clang** with MSVC's Arm64 libraries |
| Vulkan | `vulkan-shaders-gen` must be built for x64 while everything else targets Arm64 | host = target, nothing to cross |
| CPU flags | — | `GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8.7-a` |
| CRT | — | **static**, so a wiped device needs no VC++ redistributable |

**The trap the workflow is built around:** the runner is a Cobalt 100 (Neoverse
N2, Armv9 **with SVE2**); Oryon is Armv8.7 **without SVE**. A native build on the
runner would put SVE into the CPU backend and die with an illegal instruction on
the X Elite. One fixed `-march` for both machines also makes section 3 below an
exact test.

On your own Arm64 Windows machine, `scripts/qdc/build-arm64.cmd` is the same
recipe by hand.

## What is in the zip

```
cpu\       nllb-cli, -test-cache, -test-precision, -test-batch, -dump-states, -bench, -probe-kv-ops
vulkan\    the same, built -DNLLB_VULKAN=ON
models\    shape-small-f16.gguf, shape-small-f32.gguf   (random weights, NLLB shape)
ref-shape\ structural manifest for them (no goldens)
baseline\  runner-cpu-f16.bin, runner-cpu-f32.bin       (the CI runner's CPU dump)
run.ps1  sentences.txt  BUILD-INFO.txt  qdc-xelite.md
```

Two builds, one backend each: `cpu\` always starts, even if the driver's
`vulkan-1.dll` is missing. `vulkan-1.dll` is **not** shipped - the Adreno driver
installs it in System32, and section 0 says whether it is there.

## What `run.ps1` does, and what to read

Everything goes to `results\`, one `.txt` per step plus `SUMMARY.txt`.

| § | step | read this | means |
|---|---|---|---|
| 0 | environment | `vulkan-1.dll present`, GPU name and driver version | absent → the Vulkan rows say `DID NOT START`: a driver fact, not a bug |
| 1 | `nllb-probe-kv-ops vulkan` | 36 rows, none `BAD` | `half` is fine (f16 kernel, trap 48). `BAD` on `mul_mat(K view, q)` = trap 55 on this driver |
| 2 | `--list-backends` | `Vulkan0 - Qualcomm(R) Adreno(TM) X1-85 GPU` | stderr above it explains a dropped device |
| 3 | CPU dump vs runner | `IDENTICAL` | same binary, different Arm64 CPU. **If not identical, stop: no GPU number means anything yet** |
| 4 | cache gate | `PASS`, `full = exact`, `[arena]` exact, `splits 1` on the cached step | traps 15/16 hold on the device. On FAIL it re-runs once with `NLLB_KV_CONT=1` and says whether that changed the verdict |
| 5 | precision `cpu:vulkan`, batch gate | `exact ids`, `tightest headroom`; batch `PASS` | ids moving at f16 is information (trap 48), not automatically a bug |
| 6 | states diff + bench | `worst rel`; `encoder pass` vs `with lm_head` | a REPORT, not a gate |
| 7 | real model (`-Model`) | text CPU vs GPU **byte-identical, every line present**; bench | the product claim |

## Sandbox evidence (Mesa lavapipe, ggml `ffa4e8b`)

Measured before any Adreno saw it, on the small shape fixture:

```
nllb-probe-kv-ops vulkan         PASS  0 of 36 disagree (K view rel 9.4e-08 at Tn=1)
cache gate vulkan f32 / f16      PASS  full = exact, [arena] exact, splits 1 on every cached step
precision cpu:vulkan f32         PASS  exact ids 96/96, tightest headroom 47.8x
precision cpu:vulkan f16         PASS  exact ids 96/96, tightest headroom 3.9x
batch gate vulkan f16            PASS  [equal] [perm] [mask] 0 differing floats
corrupt_backend.py on vulkan     PASS  7/7 as expected - the CPU's matrix
CPU path, Vulkan build vs none   IDENTICAL, 912 records (f32 and f16)
run.ps1 rehearsal (pwsh 7)       every section PASS
```

The `ci.yml` `vulkan` job runs all of it on every push.

## Trap 55 is fixed upstream

At the old ggml pin (`7840aab`) ggml-vulkan's `mul_mat` treated a tensor as
contiguous when dims 0 and 1 were, then stepped between batch matrices by
`ne00*ne01`, ignoring `nb[2]`. The self-attention K view `[HD, n_kv, H]` over a
`[HD, n_ctx, H]` cache has heads `n_ctx` rows apart, so every head after the
first read the wrong rows on every decode step. The workaround was a
`ggml_cont` on that view, Vulkan only.

At the current pin the probe reports the raw K view correct and the cache gate
is identical with and without the copy, so the workaround is now **off by
default** and kept as an opt-in diagnostic: `NLLB_KV_CONT=1`. The CI probe is
the canary: if a ggml bump brings the bug back, the `vulkan` job goes red.

## Trap 61: the embedding table is too large for the Adreno's buffer limit

First real-model run on the X1-85 aborted in `ggml_backend_sched`:
`pre-allocated tensor (token_embd.weight) in a buffer (Vulkan0) that cannot run
the operation (NONE)`. `token_embd.weight` is 256206 x 1024 f16 = **525 MB in one
tensor**, over the Adreno's `maxStorageBufferRange`. ggml-vulkan refuses every op
on a tensor over that limit, and a weight already in device memory cannot be moved,
so the scheduler aborts. The small test model (2 MB table) and lavapipe (huge limits)
could not show it.

Fix: at load time each weight is checked with `ggml_backend_dev_supports_op`, the
same question the scheduler asks, and any the device refuses is kept in CPU memory.
The embedding lookup and the tied lm_head then run on the CPU (`splits 3` on the
cached step) and every other op stays on the GPU. `nllb-cli -v` prints
`weights kept in CPU memory (too large for Vulkan0): token_embd.weight (500.3 MiB)`.
CI forces small limits on lavapipe (`GGML_VK_FORCE_MAX_BUFFER_SIZE`) to keep this
path tested. Follow-up for speed: split the lm_head into row chunks the GPU accepts.

## Things that go red and are not bugs

* `--backend metal` / `cuda` on this machine - correct: a named backend never
  falls back to the CPU.
* `nllb-test-mmap` - not shipped: Windows has no mmap in this codebase.
* `half` rows in section 1 - the GPU picked an f16 matmul kernel (trap 48).

## What a green session still does not prove

* Speed on any other machine. Read `nllb-bench`'s `encoder pass` and
  `with lm_head` lines separately: on the M1 the GPU won the encoder 9.8x and
  lost the one-token decode step, and the X Elite also shares memory between
  CPU and GPU.
* Other Vulkan drivers (Intel, AMD, Mali). Each is a new driver; run section
  1 and 4 there before trusting it.
