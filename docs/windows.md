# Building nllb.cpp on Windows

Written for an **ASUS ROG Strix G531** (Core i7-9750H + an NVIDIA GTX 1650 /
1660 Ti / RTX 2060 depending on the variant), which is **x86-64**. Supported
backends here: the CPU, and CUDA on the NVIDIA dGPU.

**This is the project's first Windows build.** `docs/dev/handoff.md` lists "the
mmap path on anything but macOS/arm64" under *not proven*, and that is still
true — see "What will fail, and why none of it is a bug" at the end, so you
don't spend an evening chasing an expected red.

---

## The CPU build WORKS — start here

Windows x86-64, CPU backend, verified end to end on a ROG Strix G531. Full
write-up and every gotcha: `docs/dev/step15-windows-cpu.md`.

```powershell
git clone <remote> nllb.cpp && cd nllb.cpp
git submodule update --init --recursive        # a clone does NOT bring ggml

pip install -r models/requirements-convert.txt        # numpy gguf sentencepiece REGEX
pip install huggingface_hub
hf download facebook/nllb-200-distilled-600M --local-dir models\nllb-600M
python models\convert-nllb-to-gguf.py models\nllb-600M --outfile models\nllb-600M-f16.gguf --outtype f16

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

---

## CUDA (NVIDIA dGPU)

Nothing in `src/nllb.{h,cpp}` is backend-specific, so this is a build flag:

```bat
cmake -B build-cu -G Ninja -DNLLB_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cu -j
.\build-cu\bin\nllb-cli.exe --list-backends
.\build-cu\bin\nllb-cli.exe -m models\nllb-600M-f16.gguf --tgt deu_Latn --backend cuda -v "Hello!"
```

Needs the **CUDA Toolkit** installed (`nvcc --version`). `75` is Turing (GTX
16xx / RTX 20xx); `nvidia-smi --query-gpu=name,compute_cap --format=csv` is the
lookup. Then run the acceptance commands in `docs/backend.md` (step 16) with `cuda`.

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
* **`--backend metal` fails.** Correct — a named backend never silently falls
  back to the CPU; only `auto` does.
* **`nllb-test-decoder` against an f16 file fails.** Trap 19, unrelated to any
  of this: its tolerances *are* the f32 noise floor. Use `nllb-test-precision`.

If something outside this list goes red, it is worth reporting — and
`nllb-dump-states --diff` against a CPU dump from the Mac is the instrument
that says whether the Windows CPU path itself moved.
