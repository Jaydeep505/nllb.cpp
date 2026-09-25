# Step 17: Vulkan, and the Snapdragon X Elite package (OpenCL + Vulkan)

**Read this before `claude/handoff.md`'s "Where things stand" for anything
backend-related: it supersedes the Windows-on-Arm build recipe there and in the
old `QDC-XELITE.md`.** Repo copies: `BACKEND.md` "Step 17", `QDC-XELITE.md`,
`BUILD.md` "Vulkan (step 17)".

## Where things stand

* **Vulkan is built, and it RAN.** `-DNLLB_VULKAN=ON`. On Mesa lavapipe
  (conformant Vulkan 1.4 on the CPU; `GGML_VK_VISIBLE_DEVICES=0` required)
  every gate passes at f32 and f16 on the shape fixture: `full = exact`,
  `[arena]` exact, `splits 1` everywhere, exact ids vs the CPU (headroom 47.8x
  f32 / 4.6x f16), `corrupt_backend.py` 7/7 with the CPU's matrix. The CPU path
  is bit-identical with the flag on. A `vulkan` CI job runs all of it.
* **The first Vulkan run found a real bug, in ggml** (trap 55), worked around in
  one line of `src/nllb.cpp`, scoped to Vulkan, proven arithmetic-free on the CPU.
* **The X Elite session is packaged, not run.** `.github/workflows/xelite.yml`
  builds cpu / opencl / vulkan natively on `windows-11-arm`, tests the CPU path
  there, and uploads `nllb-xelite.zip` with `run.ps1`. The script was
  parse-checked and rehearsed end to end under PowerShell 7 on lavapipe.
  **The workflow itself has not run yet** — first run may need fixes in the
  Vulkan-SDK or OpenCL-loader steps.
* **OpenCL has still never executed a graph.** At the pin, ggml-opencl accepts
  only Adreno and Intel GPUs, so the X Elite session is its first run.

## Files

```
CMakeLists.txt                NLLB_VULKAN + configure-time FATAL_ERROR; nllb-probe-kv-ops
src/nllb.{h,cpp}              nllb_model::kv_k_cont; ggml_cont on the K view (Vulkan only)
cli/nllb-cli.cpp              --backend help: vulkan/vk, opencl/ocl, auto order
tests/probe_kv_ops.cpp        NEW. The 4 KV-cache ops (+ the workaround) device vs CPU,
                              Tn 1/3/9 x n_past 0/5/20, three bands: ok / half / BAD
.github/workflows/ci.yml      NEW `vulkan` job - runs graphs on lavapipe
.github/workflows/xelite.yml  NEW. fixtures (ubuntu) -> windows-11-arm build x3 -> zip
tools/qdc/run.ps1             NEW. the QDC acceptance run, sections 0-7
tools/qdc/build-arm64.cmd     NEW. same recipe by hand on Arm64 Windows
tools/qdc/sentences.txt       NEW. 9 English lines for the CPU-vs-GPU text check
QDC-XELITE.md                 REWRITTEN. BACKEND.md, BUILD.md, README.md updated
```

## Commands

```bash
# Linux, no GPU
sudo apt install libvulkan-dev glslc spirv-headers mesa-vulkan-drivers
cmake -B build-vk -DNLLB_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-vk -j2
export GGML_VK_VISIBLE_DEVICES=0
./build-vk/bin/nllb-probe-kv-ops vulkan
NLLB_BACKEND=vulkan NLLB_ARENA_TRACE=1 ./build-vk/bin/nllb-test-cache models/shape-small-f16.gguf ref-shape/ -t 2
./build-vk/bin/nllb-test-precision models/shape-small-f16.gguf models/shape-small-f16.gguf ref-shape/ -t 2 --backend cpu:vulkan
NLLB_BACKEND=vulkan python tests/corrupt_backend.py build-vk models/shape-small-f16.gguf ref-shape/ -t 2
NLLB_KV_CONT=0 NLLB_BACKEND=vulkan ./build-vk/bin/nllb-test-cache ...   # must go RED at layer 1

# X Elite: Actions -> xelite -> Run workflow; on the QDC device:
powershell -ExecutionPolicy Bypass -File .\run.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Model C:\m\nllb-600M-f16.gguf -Ref C:\m\ref
```

## New traps (55-59), continuing the handoff's numbering

55. **ggml-vulkan's `mul_mat` ignores a gapped batch stride.** It calls `src0`
    contiguous when dims 0 and 1 are (`ggml_vk_dim01_contiguous`) and steps
    between batch matrices by `ne00*ne01`, not `nb[2]`. The self-attention K
    view `[HD, n_kv, H]` over a `[HD, n_ctx, H]` cache has heads `n_ctx` rows
    apart, so every head after the first read wrong rows whenever
    `n_kv < n_ctx` — every decode step. Measured: `nllb-probe-kv-ops` rel 2.6
    on the raw view, 9.4e-08 after `ggml_cont`. **`full = exact` was blind to it
    because that pass sizes the cache to exactly the prefix** — a gate green
    because a fixture fills a buffer exactly. Fix: `kv_k_cont`, Vulkan only,
    `NLLB_KV_CONT=0/1` overrides; exact copy (dump-states IDENTICAL on the CPU
    with it forced on). Same code in ggml master 2026-09-25 (read, not run).
    Worth an upstream issue; `tests/probe_kv_ops.cpp` is a self-contained repro.
56. **ggml refuses MSVC on Arm** (`src/ggml-cpu/CMakeLists.txt`: "MSVC is not
    supported for ARM, use clang"). The old QDC recipe's MSVC `-A ARM64` step
    could never have configured. And ggml-vulkan builds and RUNS a host tool,
    so a cross build needs two toolchains. Build natively (windows-11-arm).
57. **`GGML_NATIVE=ON` on the CI runner would ship SVE to a CPU without it.**
    windows-11-arm is Cobalt 100 (Neoverse N2, SVE2); Oryon has no SVE. The
    crash would be in the CPU backend — the reference. `GGML_NATIVE=OFF`,
    `GGML_CPU_ARM_ARCH=armv8.7-a`, and the same binary tested on both, which
    turns "runner CPU vs device CPU" into an exact assertion.
58. **Two more harnesses lied toward green, both caught by running them.**
    `nllb-probe-kv-ops vulkan` matched devices case-sensitively, found none,
    printed an empty table, and the CI greps over it passed (trap 49 in a tool
    written the same day as re-reading trap 49). And `run.ps1`'s CPU-vs-GPU text
    check passed when BOTH CLIs had failed, because two empty files are
    byte-identical. Both now assert the process did the work (device line + 45
    rows; every input line present and non-empty).
59. **Registry order decides `auto`.** ggml registers Vulkan before OpenCL
    (`ggml-backend-reg.cpp`, read; not yet run with two live devices), so a
    build with both picks Vulkan for `auto`/`gpu`. The QDC zip ships one backend
    per build so nothing depends on it. Related: ggml-vulkan hides CPU-type
    devices without `GGML_VK_VISIBLE_DEVICES`; ggml-opencl drops every GPU that
    is not Adreno or Intel.

## Predictions for the X Elite session, written before it

* §3 runner-vs-device CPU dump: IDENTICAL.
* Vulkan: probe raw `K view` BAD (trap 55 is in ggml, not in lavapipe);
  cache gate PASS; f32 file diverging at ~1e-3 (trap 48); real-600M f16
  headroom < 1 somewhere, as on Metal.
* OpenCL: unknown whether its mul_mat reads `nb[2]`; the probe answers it. If
  the cache gate fails there, `run.ps1` re-runs it with `NLLB_KV_CONT=1`.
* Speed: GPU wins the encoder, the one-token step is closer (bandwidth shared
  with the CPU on a unified-memory SoC, as on the M1).

## Next

1. Run `xelite.yml` once; fix whatever the first Windows-on-Arm build hits.
2. QDC session 1 (fixture) → fill the OpenCL (step 14) and Vulkan tables.
3. Session 2 with the real 600M.
4. File the ggml-vulkan issue with `tests/probe_kv_ops.cpp` as the repro.
5. Bumping ggml past `7840aab` brings "tune KHR cooperative matrix support for
   Adreno" (`0d1fb478`) for Vulkan — but a bump is `nllb-dump-states --diff`
   plus every gate again, so only after the pinned run is recorded.
