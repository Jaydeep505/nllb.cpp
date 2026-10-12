# The gates. Each one is a standalone binary in build/bin/ that takes a model
# and (most of them) a ref/ directory of HF goldens; scripts/step10-measure.sh
# and .github/workflows/ci.yml are what run them. See the header of the
# top-level CMakeLists.txt for what each one claims.

function(nllb_add_test name source)
    nllb_add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE nllb)
endfunction()

nllb_add_test(nllb-test           test-encoder.cpp)
nllb_add_test(nllb-test-decoder   test-decoder.cpp)
nllb_add_test(nllb-test-generate  test-generate.cpp)
nllb_add_test(nllb-test-batch     test-batch.cpp)
nllb_add_test(nllb-test-cache     test-cache.cpp)
nllb_add_test(nllb-test-precision test-precision.cpp)
nllb_add_test(nllb-test-mmap      test-mmap.cpp)

# The tokenizer gate also reads src/nllb-grapheme-data.h, an internal header.
nllb_add_test(nllb-test-tokenizer test-tokenizer.cpp)
target_include_directories(nllb-test-tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Not a gate on its own: dumps everything the public API computes as raw bytes,
# so two BUILDS can be compared with `nllb-dump-states --diff a.bin b.bin`. That
# is how the step-13 backend port was held to bit-identity against the old CPU
# path, and it is the instrument for a ggml submodule bump.
nllb_add_test(nllb-dump-states dump-states.cpp)

# The cached step's four KV-cache ops (the two cpy writes into the cache views,
# the two mul_mats over them) one at a time, device vs CPU, at Tn = 1, 3, 9.
# Needs no model, and links ggml alone. The first thing to run on a new device
# backend (CUDA, Metal) when the cache gate goes red: it localises the failure
# to one op.
nllb_add_executable(nllb-probe-kv-ops probe-kv-ops.cpp)
target_link_libraries(nllb-probe-kv-ops PRIVATE ggml)
