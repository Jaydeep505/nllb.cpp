# Vulkan backend: results on the Snapdragon X Elite (Adreno X1-85)

nllb.cpp's Vulkan backend (`-DNLLB_VULKAN=ON`) tested on real GPU hardware: the
**Snapdragon X Elite Compute Reference Design (X1E80100), Windows 11 on Arm,
Adreno X1-85**, through Qualcomm Device Cloud (QDC). The CPU path is tested in the
same session as the reference every GPU number is compared against. CI runs the
same gates on Mesa lavapipe on every pull request.

## Results

Device: Adreno X1-85, driver 31.0.160.0, `vulkan-1.dll` 1.3.300.0; ggml reports
`uma: 1 | fp16: 1 | int dot: 1 | warp size: 64 | matrix cores: none`.
Build: ggml `ffa4e8b`, clang (MSVC ABI), `armv8.7-a`, static, Vulkan SDK 1.4.363.0.

### Correctness: everything matches the CPU

| Check | Model | Result |
|---|---|---|
| KV-cache ops, device vs CPU (`nllb-probe-kv-ops`) | - | PASS, 0 of 36 disagree |
| Device CPU vs CI-runner CPU, same binary | test fixture | IDENTICAL (f16 and f32) |
| Cache gate (`nllb-test-cache`) | test fixture | PASS, `full = exact`, `[arena]` exact (f32, f16) |
| CPU vs Vulkan ids + margin audit (`nllb-test-precision`) | test fixture | PASS, 96/96 exact; headroom 37.7x (f32), 5.2x (f16) |
| Batching (`nllb-test-batch`) | test fixture | PASS, `[equal]` `[perm]` `[mask]` bit-identical |
| **Translation text, CPU vs Vulkan** | **real 600M f16** | **9/9 sentences byte-identical** |

```
Hallo, Welt!
Der schnelle braune Fuchs springt über den faulen Hund.
Der Ausschuß wird seinen Bericht am Dienstag nach der Abstimmung veröffentlichen.
```

### Speed: the CPU is faster on this machine at batch 1

Real 600M f16, 8 threads, `nllb-bench --src 24 --steps 64`, medians:

| | CPU (Oryon, 12 cores) | Vulkan (Adreno X1-85) |
|---|---|---|
| Decoder step (one token) | **10.5 ms** | 15.4 ms (0.68x) |
| Decoder step without the lm_head | **5.4 ms** | 8.9 ms |
| Encoder, 24 source tokens | **72 ms** | 476 ms (single run, includes shader compilation on first use) |

At batch 1 each decoder step is a memory-bound matrix-vector product. The Oryon
CPU reads the same unified memory as the GPU and ggml's CPU kernels are well
tuned for it, while the Adreno through ggml-vulkan has no matrix cores and pays a
per-dispatch cost on ~700 graph nodes per step. Use `--backend cpu` for the fastest
single-sentence translation on this laptop; Vulkan is there for GPUs and machines
where the CPU is the weaker side, and gives the same text. Not yet measured: batched
decoding (`--batch N`), where the GPU has matrix-matrix work to do.

