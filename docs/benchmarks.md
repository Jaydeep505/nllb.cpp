# Benchmarks: nllb.cpp vs Hugging Face transformers

| | Apple M1 Air (nllb.cpp Metal vs PyTorch MPS) | NVIDIA Tesla T4 (CUDA) |
|---|---|---|
| **Speed vs HF f16 (end-to-end)** | **1.44x** faster | **1.79x** faster |
| **Peak memory vs HF f16** | **3.1x** less (1362 vs 4261 MiB) | GPU: same (1333 vs 1341 MiB)<br>host RAM: **13x** less (464 vs 6148 MiB) |
| **Load (excl. torch import)** | 963 ms vs 10441 ms | 1412 ms vs 4137 ms (HF f32)† |
| **CPU vs CPU (nllb.cpp vs HF f32)** | **4.4x** faster, **2.3x** less RAM | **1.7x** faster, **2.4x** less RAM |

`facebook/nllb-200-distilled-600M` (nllb.cpp uses `nllb-600M-f16.gguf`). `"Hello, how are you today?"` eng_Latn → deu_Latn, greedy, batch 1, median of 20 runs. All implementations produced the identical translation.

These numbers are for a single short sentence. Longer inputs and batched translation (`--batch N`) have not been benchmarked yet.

† The HF f16 load on the T4 included downloading the model, so HF f32 is used for the load comparison.
