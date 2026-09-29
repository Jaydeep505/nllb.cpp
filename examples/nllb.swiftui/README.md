# nllb.swiftui — on-device translation for iOS

A minimal SwiftUI app on top of `nllb.xcframework`. The screen is split in two:
type in the upper half, and the translation streams into the lower half as it
is generated. Everything runs on the device, on the GPU through ggml's Metal
backend. No network.

```
┌──────────────────────────┐
│ English ▾             ⓧ  │
│ Hello, how are you?      │   <- TextEditor (source)
│                          │
├────────────⇅─────────────┤   <- swap languages
│ German ▾           ⟳  ⧉  │
│ Hallo, wie geht es dir?  │   <- translation, streamed per token
│                          │
│ MTL0 · zero-copy   12 tok · 0.31 s │
└──────────────────────────┘
```

## Build

From the repository root, on a Mac with Xcode 16 or later and CMake:

```bash
git submodule update --init --recursive
./build-xcframework.sh                  # -> build-apple/nllb.xcframework
```

Then open `examples/nllb.swiftui/NllbTranslate.xcodeproj`. The project already
references `../../build-apple/nllb.xcframework` and embeds it. Pick your team
under *Signing & Capabilities* and run on a device.

## The model

The app loads the first `.gguf` it finds, preferring `f16`, then `q8emb`:

1. **In the app bundle.** Drop `nllb-600M-f16.gguf` into the `NllbTranslate/`
   folder next to the Swift files. The folder is a synchronized group, so Xcode
   picks it up with no project edit. This is the simplest option, but the
   1.24 GB file is reinstalled with every build.
2. **In the app's Documents.** Connect the phone, open Finder → *iPhone* →
   *Files* → *NllbTranslate*, and drag the file in. You can also use the Files
   app. It survives rebuilds. `UIFileSharingEnabled` is already on.

The file must come from the current converter, because the tokenizer lives in
the GGUF (step 8). A file converted earlier is refused at load, with the name of
the missing key.

| file | size | notes |
|---|---|---|
| `nllb-600M-f16.gguf` | 1.24 GB | the default; byte-identical to HF on the goldens |
| `nllb-600M-q8emb.gguf` | 0.996 GB | 1.33x faster on the M1 CPU; chrF 96.57 |

## How it works

* `NllbContext.swift` wraps `nllb-c.h` (the framework's only header). All C
  calls run on one private serial queue, because an `nllb_ctx` must not be
  used from two threads at once.
* `Translator.swift` handles typing. Each edit cancels the running translation,
  which stops at the next token through `nllb_ctx_request_abort` plus a cancel
  flag checked in the per-token callback. It then starts a new one after a
  350 ms pause. Only the latest request can update the UI.
* The text is translated **line by line**. NLLB is a sentence-level model, and
  a whole paragraph on one line tends to stop at the first sentence's eos (see
  the step 9 notes in the handoff). If you want paragraphs translated fully,
  put one sentence per line.
* The simulator runs on the CPU (`#if targetEnvironment(simulator)`). Use a
  device for Metal and for any timing.

## Memory

The weights are `mmap`ed and wrapped zero-copy in a Metal buffer; the footer
shows `zero-copy` when that happened. On the M1, the same setup measured
0.25 GB of `phys_footprint` against 2.7 GB with reading (trap 36), and jetsam
kills by footprint. Nobody has measured this on an iPhone yet. If a smaller
device kills the app, add the *Increased Memory Limit* capability
(`com.apple.developer.kernel.increased-memory-limit`) or use `q8emb`.
