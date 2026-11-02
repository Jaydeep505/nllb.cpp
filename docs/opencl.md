# OpenCL backend: the Snapdragon X Elite (Adreno X1-85) test

nllb.cpp's OpenCL backend (`-DNLLB_OPENCL=ON`) is ggml's OpenCL backend, which
is written for Qualcomm Adreno GPUs. At the pinned ggml (`ffa4e8b`) it accepts
**Adreno and Intel GPUs only**, and needs `cl_khr_fp16`.

**Status: built and CI-guarded, not yet run on hardware.** The results table
below is filled in by the QDC session described here.

## Build

```bash
# Linux: headers + ICD loader to link against, Python 3 to embed the kernels
sudo apt install opencl-headers ocl-icd-opencl-dev
cmake -B build -DNLLB_OPENCL=ON                # portable kernels
cmake -B build -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON   # Adreno-tuned kernels
cmake --build build -j --config Release
./build/bin/nllb-cli --list-backends           # expect a GPUOpenCL line
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --backend opencl --tgt deu_Latn "Hello, world!"
```

* `NLLB_ADRENO` decides `GGML_OPENCL_USE_ADRENO_KERNELS`. ggml defaults it ON;
  nllb.cpp defaults it OFF, because ggml warns and drops a non-Adreno device
  when it is on. Turn it on for Snapdragon.
* Kernels are always **embedded** in the binary, so it is relocatable: no
  `.cl` files to ship.
* `--backend opencl` and `--backend ocl` both work. ggml names the device
  `GPUOpenCL` and the registry `OpenCL`. A named backend that is not there
  **fails** and prints the devices it did find; only `auto` falls back to the CPU.
* In a build with both Vulkan and OpenCL, `auto` picks Vulkan, because ggml
  registers it first. Name the backend.

On Windows on Arm, build natively with clang, the same as for Vulkan. The
`xelite` workflow does it: it builds the Khronos headers and ICD loader for
Arm64 (static CRT) and ships `OpenCL.dll` next to the `.exe` files.

## Running it on QDC

This is the same flow as [vulkan.md](vulkan.md). The zip now holds three
builds: `cpu\`, `vulkan\` and `opencl\`.

1. Push the patches, then **Actions -> xelite -> Run workflow** (~40 min).
   Download `nllb-xelite.zip`.
2. QDC: start a session on the **Snapdragon X Elite CRD** with your SSH key.
   Upload the zip (it lands as `C:\Temp\file`, with no extension):

   ```powershell
   Copy-Item C:\Temp\file C:\Temp\nllb-xelite.zip
   Expand-Archive C:\Temp\nllb-xelite.zip C:\nllb
   cd C:\nllb
   ```

3. **Session 1, no model** (~10 min). Runs every gate on random-weight fixtures:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1 -Backends opencl
   ```

   Drop `-Backends opencl` to run Vulkan and OpenCL side by side.

4. **Session 2, real model.** Copy `nllb-600M-f16.gguf` and `ref\` over, then:

   ```powershell
   .\run.ps1 -Backends opencl,vulkan -Model C:\m\nllb-600M-f16.gguf -Ref C:\m\ref
   ```

5. Copy `results.zip` off the device before the session ends. QDC wipes it.

### What to read in `results\`

| § | file | what green looks like | if it is not green |
|---|---|---|---|
| 0 | `SUMMARY.txt` top | `OpenCL ICD ...` names a Qualcomm driver; `opencl\OpenCL.dll shipped` | no ICD = the driver has no OpenCL; nothing below can run |
| 1 | `1-probe-kv-ops-opencl.txt` | `device: GPUOpenCL`, no `BAD` row | a BAD `mul_mat(K view, q)` row is trap 55 on OpenCL: section 4 retries with `NLLB_KV_CONT=1` |
| 2 | `2-list-backends-opencl.txt` | `GPUOpenCL - ...Adreno...` | the `ggml_opencl:` lines above it say why a device was dropped |
| 3 | cpu device vs runner | `IDENTICAL` | stop: no GPU number means anything until this is identical |
| 4 | `4-cache-opencl-*.txt` | `PASS`, `full = exact`, `[arena]` exact, `splits 1` on the cached step | `splits` > 1 means an op fell back to the CPU; the trace names it |
| 5 | `5-precision-cpu-opencl-*.txt` | exact ids, headroom > 1 | ids moving at f16 can be trap 48 (half-precision matmul kernels). Read `[margin]` first |
| 5 | `5-batch-opencl-f16.txt` | `[equal]` `[perm]` `[mask]` bit-identical | |
| 6 | `6-diff-cpu-vs-opencl-f32.txt` | a report, not a gate: `worst rel` | ~1e-3 from an f32 file means f16 kernels inside |
| 7 | `REAL text cpu vs opencl` | byte-identical, 9/9/9 lines | |
| 7 | `7-bench-opencl-real.txt` | read `encoder pass` and `with lm_head` separately | |

### Things specific to OpenCL on this machine

* **The 525 MB `token_embd` tensor.** ggml-opencl limits a single buffer to
  `CL_DEVICE_MAX_MEM_ALLOC_SIZE`. It does not refuse the tensor through
  `supports_op` the way ggml-vulkan does (trap 61), so the loader now also
  checks each weight against the buffer type's max size. A weight over it is
  kept in CPU memory and its ops run on the CPU. `-v` prints
  `weights kept in CPU memory (too large for GPUOpenCL): ...` when this
  happens. If `token_embd` fits, it stays on the GPU, the lm_head runs there,
  and the cached step should show `splits 1`. Either outcome is information.
* **No zero-copy load.** OpenCL cannot wrap host memory
  (`buffer_from_host_ptr = false`), so the weights are copied into device
  buffers. `nllb-test-mmap` is CPU-only and is not in the zip.
* **Kernel compilation at startup.** ggml compiles its whole kernel set when
  the backend initialises, not per op. The first OpenCL call takes seconds.
  ggml caches compiled programs (`cl-program-cache`), so time the second run.
* **A device that registers is not a device that runs.** Step 14 saw a runtime
  register, print `FP16 support: true`, and then fail to compile the kernels
  under `-cl-fast-relaxed-math`. If `--list-backends` shows `GPUOpenCL` but
  section 4 fails at init, look for `kernel compile error` in its log.
* **Expect the CPU to win batch-1 decoding again.** Vulkan's step on this
  machine was 0.68x the CPU's (see [vulkan.md](vulkan.md)): the step is a
  bandwidth-bound matvec and Oryon reads the same LPDDR5X. ggml-opencl has
  Adreno-specific GEMV paths that ggml-vulkan lacks, so the gap may close.
  That is the measurement to make.

## Results

*To be filled in from the QDC session.*

| Check | Model | Result |
|---|---|---|
| OpenCL ICD / device | - | |
| KV-cache ops, device vs CPU | - | |
| Cache gate f32 / f16 | fixture | |
| CPU vs OpenCL ids + margins f32 / f16 | fixture | |
| Batching gate | fixture | |
| Translation text, CPU vs OpenCL | real 600M f16 | |

| | CPU (Oryon) | Vulkan | OpenCL |
|---|---|---|---|
| Decoder step, with lm_head | 10.5 ms | 15.4 ms | |
| Decoder step, no lm_head | 5.4 ms | 8.9 ms | |
| Encoder, 24 tokens (warm) | 72 ms | | |
