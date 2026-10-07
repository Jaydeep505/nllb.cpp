# iOS: nllb.xcframework and the SwiftUI example (step 18)

**Status: BUILT, NOT YET RUN ON A DEVICE.** The C facade and the generation hook
are verified on Linux, and the framework script and app were written against
whisper.cpp's working equivalents. No Xcode build has happened yet.
`.github/workflows/ios.yml` does the first one; the acceptance run is on an
iPhone.

```bash
git submodule update --init --recursive
./build-xcframework.sh                        # -> build-apple/nllb.xcframework
open examples/nllb.swiftui/NllbTranslate.xcodeproj
```

## What was added

| file | what |
|---|---|
| `include/nllb-c.h`, `src/nllb-c.cpp` | Plain-C facade: `nllb_ctx_init`, `nllb_translate`, `nllb_ctx_request_abort`, language list. Part of `libnllb`. |
| `build-xcframework.sh` | whisper.cpp's script, adapted. Slices: iOS, iOS simulator, macOS. |
| `CMakeLists.txt` | `NLLB_BUILD_EXAMPLES`, `NLLB_BUILD_TESTS` (default ON). Both OFF builds the library alone. |
| `nllb_generate_params::on_token` | Optional per-token hook; returning false stops with `NLLB_STOP_ABORTED`. |
| `examples/nllb.swiftui/` | The app: split screen, streaming, cancel-on-type. |
| `.github/workflows/ios.yml` | Builds the framework and compiles the app for the simulator. |

## Why the facade, and why it changes no arithmetic

`nllb.h` is C++, and Swift cannot import `std::vector`. `nllb_translate()` makes
the same calls as `nllb-cli`'s `translate_one()`: `nllb_vocab_encode`, then
`nllb_generate` with the cache on and f32 KV, then
`nllb_vocab_decode(skip_special)`. It runs these once per input line, like the
CLI's stdin mode. So `tests/verify_translation.py` also covers this path, line
for line. The framework exports only this header, so the C++ API is not part of
the Swift surface and can keep changing.

## The hook, and the evidence that it is inert

`on_token` defaults to `nullptr`, so no gate sees a different code path.
Measured on a synthetic shape model (d_model 256, 2+2 layers, vocab 4000),
cached, 20 tokens:

```
no hook: 21 ids stop=1 | hook: 21 ids stop=1 calls=20 | identical=yes
abort after 5: ok=1 stop=3 (3=ABORTED) ids=6 prefix-of-full=yes
```

The hook runs AFTER the eos check. So a sequence that finished reports EOS even
if the hook would also have asked to stop, and an aborted sequence is always a
prefix of the full one. The hook reads ids and never touches a tensor.

## Framework decisions

* **One header in the module map.** `nllb-c.h` compiles under
  `-std=c99 -pedantic -Werror`, which CI checks, so `import nllb` works with no
  bridging header.
* **`GGML_BLAS=OFF`.** nllb schedules the primary device plus the CPU and never
  BLAS (docs/backend.md). The CPU backend still uses Accelerate's vDSP.
* **`GGML_METAL_EMBED_LIBRARY=ON`.** The shaders are inside the binary, so there
  is no `.metallib` to ship or find.
* **Libraries are found by name, and a missing one is fatal.** ggml sits at
  `ggml`, not whisper's `ggml/`, so hard-coded paths do not carry over.
  A framework without `libggml-metal.a` links fine, and `auto` then quietly means
  CPU. CI also checks `_ggml_backend_metal_reg` in the binary.

## Things to expect on the device

* **The simulator runs on the CPU, on purpose.** Its Metal is a translation
  layer, and nothing measured there transfers.
* **Memory.** mmap plus zero-copy is what brought `phys_footprint` to 0.25 GB
  on the M1 (trap 36). This is not measured on iOS, where jetsam enforces that
  number. First thing to read: Xcode's memory gauge with f16 loaded.
* **Speed.** On the M1 the encoder was 9.8x faster on Metal and the one-token
  step was 0.93x (step 13). The A-series should look similar in shape. The
  lm_head matvec is bandwidth-bound, and a phone has less bandwidth than an M1.
  Expect ms/token higher than the M1's 18 ms, and read the footer's tok/s.
* **A paragraph on one line stops at the first sentence** (the step 9 eos
  tendency). The app splits on lines only. Sentence segmentation is still the
  "when time permits" item.

## Acceptance run (to do, on an iPhone)

1. Framework builds; the app launches; the footer says `MTL0 · zero-copy`.
2. The nine golden English sentences into German, one per line, match
   `ref/translations.txt` byte for byte. Metal on the M1 did (step 13).
3. Memory gauge with f16 loaded, and whether jetsam allows it without the
   increased-memory-limit entitlement.
4. tok/s on a long sentence, compared with `nllb-bench` on the M1.
