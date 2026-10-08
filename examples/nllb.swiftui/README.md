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

