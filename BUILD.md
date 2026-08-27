# Step 2: the C++ encoder on ggml

```
src/nllb.h            model + hparams + result structs
src/nllb.cpp          GGUF loading and the encoder graph
tests/test_encoder.cpp  golden diff against ref/, with per-layer bisect
CMakeLists.txt        vendors ggml, builds bin/nllb-test
```

Later steps add `bin/nllb-test-decoder` (see `DECODER.md`),
`bin/nllb-test-generate` (see `GENERATE.md`) and `bin/nllb-test-cache`
(see `CACHE.md`) to the same build. The last of those needs no reference dump —
it compares the KV-cached decoder against the un-cached one.

## Build

ggml is vendored as a git submodule, so the tree stays dependency-free:

```bash
git submodule add https://github.com/ggml-org/ggml vendor/ggml
git submodule update --init --recursive

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On your Mac ggml will pick up Accelerate and Metal automatically. The encoder
path here is CPU-only f32; that is deliberate for now, since the point is
correctness, not speed.

## Run

The golden format changed in this step (raw per-layer `.bin` files plus a
`manifest.txt` that C++ can parse without a JSON library), so regenerate:

```bash
python tools/dump_reference.py models/nllb-600M --out ref/ --lang eng_Latn --per-layer
./build/bin/nllb-test models/nllb-600M-f32.gguf ref/ -t 8
```

Expect `max|d|` in the 1e-6 .. 1e-5 range. Default tolerance is 2e-4, which is
loose enough for ggml-vs-torch kernel differences across 12 layers and far
tighter than any real bug, which lands at 1e-2 or worse.

## The bisect

When a case fails, the test walks the per-layer goldens and stops at the first
one that diverges. Two real examples, both produced by deliberately corrupting
a working GGUF:

Position table shifted by one row:

```
  00_hello_world  T=4  max|d|=1.337e+00  FAIL
      layer_00  max|d|=9.373e-01  FAIL   (embeddings)
```

A transposed K weight in block 1:

```
  00_hello_world  T=4  max|d|=2.409e-02  FAIL
      layer_00  max|d|=1.192e-07  ok   (embeddings)
      layer_01  max|d|=2.384e-07  ok
      layer_02  max|d|=1.604e-02  FAIL
```

`layer_01` is the input to block 1 and `layer_02` is its output, so the first
red line names the broken block directly.

Indexing, which is easy to get wrong:

```
layer_00        embeddings (scale + positions) = INPUT to block 0
layer_i         input to block i = output of block i-1
layer_n_layer   output of the last block AFTER the final encoder layer_norm
```

There is no dump of the last block's raw pre-norm output; HF does not expose it.

## Graph notes

`nllb_encode` keeps every intermediate alive (`no_alloc = false`, no graph
allocator reuse) so the per-layer states can be read back after compute. That
costs memory and is the right trade while debugging. Once the encoder is green,
switching to `ggml_gallocr` with only `enc_out` marked as an output cuts the
working set substantially - do that as a separate, verifiable change.

For the DECODER that change was measured and dropped: step 6 found the whole of
what a graph allocator targets to be 1.6-1.7 ms of a ~38 ms cached token, most
of which is recovered by simply keeping the arena across steps
(`nllb_kv_cache::scratch`). The encoder runs once per sentence, so the working
set is the argument there, not the time. See `PROFILE.md`.

Attention is unmasked: every golden case is a single unpadded sequence.
`inp_pos` is already computed the general way (`cumsum(mask)*mask + pad_id`),
so padded batches need only a mask tensor passed to `ggml_soft_max_ext`.

Verified end-to-end on a synthetic 3-layer NLLB: 9/9 cases, worst
`max|d| = 1.371e-06`.
