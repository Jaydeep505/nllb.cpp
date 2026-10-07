# Step 15: Windows x86-64, CPU backend — COMPLETE

**`nllb.cpp` builds and translates on Windows.** First time the project has run
on anything but macOS and the Linux sandbox. CPU backend only, which is the
configuration every claim in the calibration ladder was measured on.

Machine: ASUS ROG Strix G531, Intel Core i7-9750H, Windows, MSVC v143 via
Build Tools for Visual Studio 2022, Visual Studio generator.

```
> ./build/bin/Release/nllb-cli -m models/nllb-600M-f32.gguf --src eng_Latn --tgt deu_Latn "Its non of your business"
Das geht dich nichts an .

> ./build/bin/Release/nllb-cli -m models/nllb-600M-f32.gguf --src eng_Latn --tgt deu_Latn "It si better to be jobless and poor rather than working in stressfull environment"
Es ist besser, arbeitslos und arm zu sein, als in einem stressigen Umfeld zu arbeiten
```

Both at f32 and f16. Note the second input carries a typo (`It si`) and the
translation is still correct German — eyeball evidence, not a gate, but the
right kind of eyeball evidence.

---

## The build, start to finish

```powershell
# 1. the repo
git clone <remote> nllb.cpp
cd nllb.cpp

# 2. the ggml submodule - a clone does NOT bring it down
git submodule update --init --recursive
git submodule status                     # verify: an empty ggml fails
                                         # later with a message that never
                                         # mentions submodules

# 3. the checkpoint, and the GGUF
pip install -r models/requirements-convert.txt  # numpy gguf sentencepiece REGEX
pip install huggingface_hub
hf download facebook/nllb-200-distilled-600M --local-dir models\nllb-600M

python models\convert-nllb-to-gguf.py models\nllb-600M `
    --outfile models\nllb-600M-f32.gguf --outtype f32
python models\convert-nllb-to-gguf.py models\nllb-600M `
    --outfile models\nllb-600M-f16.gguf --outtype f16

# 4. configure
cmake -B build -A x64

# 5. build
cmake --build build --config Release -j

# 6. run
./build/bin/Release/nllb-cli -m models/nllb-600M-f16.gguf `
    --src eng_Latn --tgt deu_Latn "Hello, World"
```

No torch at any point — the converter parses the safetensors container itself.
Budget ~6.5 GB of disk for the snapshot plus both GGUFs.

---

## What bit, in the order it bit

Recorded because every one of these cost time and none is in any README.

**1. `git clone` does not bring the submodule.** Git stores a gitlink — a
pointer to one ggml commit — not the contents. `--recurse-submodules` on the
clone, or `git submodule update --init --recursive` after. That pointer is what
pins ggml, which is the point.

**2. VS Code is not a compiler.** It installs no toolchain. **Build Tools for
Visual Studio 2022** with *Desktop development with C++* is the actual
requirement, and it brings CMake and Ninja along with MSVC.

**3. An old MinGW on `PATH` will be picked up, and it is 32-bit.** CMake found
`i686-8.1.0-posix-dwarf / mingw32` — GCC 8.1.0 from 2018 — and failed on
`THREAD_POWER_THROTTLING_STATE`, which is Windows 8+ and absent from those
headers. Two things matter here and the compile error is the lesser one:
**a 32-bit process has ~2 GB of user address space**, and the f16 GGUF alone is
1.24 GB before the ~101 MiB graph arena. That build would have linked and then
died on the real checkpoint. Pass `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`
if anything ambiguous is on `PATH`.

**4. `cmake -B build -G Ninja` needs a developer shell; `cmake -B build -A x64`
does not.** The Visual Studio generator locates MSVC through the registry, so it
works from plain PowerShell. Ninja needs `cl.exe` on `PATH`, which only the
*Developer PowerShell for VS 2022* provides. Both are fine; the VS generator has
fewer moving parts on a fresh machine.

**5. THE VS GENERATOR IS MULTI-CONFIG, and two consequences follow.**
`-DCMAKE_BUILD_TYPE=Release` is **silently ignored** — you must pass
`--config Release` at build time, or you get a Debug build, which is ~20x slower
and makes any bench number meaningless. And binaries land in
**`build\bin\Release\`**, not `build\bin\`, because all four configurations
share one build tree and their outputs cannot collide. Every path in this
project's docs assumes the flat single-config layout. Ninja on Windows restores
it.

**6. A failed configure poisons the build directory.** CMake caches the
generator and the compiler it detected, so fixing `PATH` and re-running in the
same folder keeps failing. `Remove-Item -Recurse -Force build` first. This is
why out-of-source builds are worth the extra flag.

**7. `regex` must be importable when converting**, or the converter prints
`! no grapheme tables in this file` **to stderr** and writes a GGUF anyway. That
file works, and its ids become a function of the GGUF *plus* whichever Unicode
release the binary was built against, instead of the GGUF alone. It is in
`models/requirements-convert.txt`; read the converter's stderr rather than only its
exit code. Then check:

```powershell
nllb-cli -m models\nllb-600M-f16.gguf --list-langs | Measure-Object -Line   # 202
nllb-cli -m models\nllb-600M-f16.gguf --tgt deu_Latn -v "Hello!"
#   -v must say: grapheme tables from THE GGUF   (not "the compiled-in fallback")
```

---

## PowerShell renders UTF-8 as mojibake, and that is the console

`möchte` came out as `m├╢chte`, `für` as `f├╝r`. The **bytes are correct**;
PowerShell 5.1's console defaults to a legacy code page and misreads UTF-8. The
first two translations looked clean only because they contained no umlauts.

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# or
chcp 65001
```

This is step 9's lesson arriving on a new platform. There, Hindi output appeared
to carry an en dash where HF had a hyphen; `hexdump -C` read `20 2d 20`, a plain
hyphen, and the difference was the terminal font. **For any non-ASCII target,
read the bytes or read the file, never the terminal.** Redirect to a file and
open it in an editor that states its encoding.

---

## An open question worth one command: does a double space change the ids?

A long, typo-laden two-sentence input produced degenerate repetition —
`CPU-Back-Back-Back-…` to the token cap. Re-running it translated correctly.
The user's reading was that a **double space** caused it.

**Three things differed between those two runs, not one:** a double space, a
trailing space, and `ap` vs `api`. So the double space is a hypothesis, not a
finding — and it is an interesting one, because the tokenizer's normalizer has a
`Replace(" {2,}" -> " ")` rule whose job is to collapse exactly that. If two
spaces genuinely move the ids, that rule is not doing what step 8 says it does,
and `tests/corrupt_tokenizer.py` has a mutation (`drop Replace(" {2,}" -> " ")`)
that exists for precisely this.

One command settles it, and it needs no weights:

```powershell
nllb-test-tokenizer models\nllb-600M-f16.gguf --text "now i just  want to close"
nllb-test-tokenizer models\nllb-600M-f16.gguf --text "now i just want to close"
```

Identical ids → the double space was innocent and `ap`/`api` did it, which is
ordinary out-of-distribution behaviour for a sentence-level model on a
malformed multi-sentence input (the eos tendency in `docs/cli.md`, and the parked
sentence-segmentation item). **Different ids → that is a real tokenizer
finding** and belongs in `docs/tokenizer.md` rather than being filed under "NLLB is
like that".

Nobody has run it.

---

## Proven, and not

**Proven on Windows x86-64:** the tree configures and compiles clean with MSVC
v143; the converter produces both GGUFs without torch; `nllb-cli` translates
English to German correctly at f32 and f16; `--list-langs` reports 202.

**Not proven, and none of it blocks the next step:**

* **No gate has been run on Windows.** `nllb-test-cache`, `nllb-test-generate`,
  `nllb-test-precision` and `verify_translation.py` have all run on macOS and in
  the sandbox, never here. The cheapest one needs no weights and no goldens:
  `make_shape_model.py` + `make_shape_ref.py` then `nllb-test-cache`, which
  asserts `full = exact`.
* **`nllb-test-mmap` will FAIL on Windows, by design.** `NLLB_HAVE_MMAP` is `0`
  in the `_WIN32` branch, so `nllb_mmap_file` returns null and the loader reads
  instead — correct, and always has been. The gate then has nothing to compare
  and fails rather than printing green about a run in which mmap never happened.
  **That gate's claim lives on macOS and Linux.** This is the first time the
  Windows branch has been executed at all, which retires "the mmap path on
  anything but macOS/arm64" only in the weak sense: the fallback works, the
  mapping still does not exist.
* **THE SUBMODULE PIN IS 211 COMMITS BEHIND the Mac.** The Windows configure
  reports `ggml version 0.20.2 / commit 8c63e709`; steps 13-14 are recorded
  against `0.23.0 / 7840aab`. Both carry
  `buffer_from_host_ptr`, so 0.20.2 compiles — but it means a ggml bump made on
  the Mac was never committed. **Any `nllb-dump-states --diff` between these two
  machines measures the bump, not the platform.** Fix the pin before drawing a
  cross-platform conclusion.
* **No `--diff` against the Mac has been run**, so whether the MSVC x64 CPU path
  is bit-identical to clang on arm64 macOS is unknown. It is one command per
  side once the pin is settled.

---

## Next

**CUDA next.** The G531's NVIDIA dGPU is the fast device in it, and
`nllb_pick_device` already resolves `cuda` with no code change.

1. **CUDA (`-DGGML_CUDA=ON`, `--backend cuda`)** — the G531's dGPU. Needs the
   CUDA Toolkit. Then `docs/backend.md`'s acceptance run with `cuda` in place of
   `metal`: `splits 1`, `full = exact`, exact ids at `cpu:cuda`, the corruption
   matrix unchanged. Expect trap 48 with different numbers — a backend's
   precision is a property of the kernel it picks, not of the tensor types.
   **And CUDA on a discrete GPU is the first machine in this project that can
   see the host-pointer class** (`mask-through-data`, `mask-not-input`, both
   marked `report` because the CPU and Apple's unified memory hide them). That
   row is worth going out of the way for.
