# nllb.cpp - NLLB-200 inference on ggml, no dependencies beyond the submodule.
#
# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
#   git submodule update --init --recursive        # vendor/ggml, first time only
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
#   cmake --build build --target nllb-cli -j       # one target
#   rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release   # after moving
#                                                  # or bumping the submodule
#
# Release is the default if CMAKE_BUILD_TYPE is unset (see below), but pass it
# anyway: a Debug build of this is ~20x slower and the first thing you would
# notice is a bench number that means nothing.
#
# On macOS the configure step finds Accelerate, BLAS and Metal and links them.
# Since step 13 Metal IS used: every graph goes through ggml_backend_sched, and
# nllb-cli defaults to --backend auto (Metal when present). The LIBRARY default
# is still the CPU, so every gate means what it meant unless NLLB_BACKEND=metal
# is set. BLAS is linked and never scheduled. See BACKEND.md.
#
# ---------------------------------------------------------------------------
# WHAT GETS BUILT, and what each one claims
# ---------------------------------------------------------------------------
#   bin/nllb-cli             THE PRODUCT. text in, translation out.
#   bin/nllb-quantize        f32/f16 GGUF -> quantised, via ggml_quantize_chunk.
#                            Required for the K-quants; gguf-py cannot write
#                            them (trap 33).  --list-mixes
#
#   bin/nllb-test            encoder vs HF goldens
#   bin/nllb-test-decoder    decoder vs HF goldens, per-layer bisect. f32 ONLY -
#                            its tolerances ARE the f32 noise floor (trap 19)
#   bin/nllb-test-generate   greedy ids vs HF generate(), exact
#   bin/nllb-test-cache      cached vs un-cached. full = exact.  --kv-f16
#   bin/nllb-test-precision  two precisions: ids, margins, [topk].
#                            --allow-id-drift  --kv f32:f16
#   bin/nllb-test-tokenizer  text -> ids vs HF, exact, 5 passes
#   bin/nllb-test-mmap       mmap vs read: every weight byte identical
#   bin/nllb-bench           where the cached step goes.  --kv-f16
#   bin/nllb-dump-states     everything the API computes, as bytes; --diff two
#                            BUILDS. How step 13 was held to bit-identity.
#
# The gates run in one command: ./tools/step10_measure.sh
# Everything above is documented in claude/handoff.md; the trap numbers refer
# to its "traps" section.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)
project(nllb.cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# vendored ggml - no external dependencies
set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/ggml)

# Warnings on OUR targets only - vendor/ggml is added above and keeps its own
# flags. This is not housekeeping: the [arena] report in tests/test_cache.cpp
# shipped with a printf missing its first argument, which is a segfault on the
# failure path and invisible on the passing one. -Wformat (via -Wall) catches
# exactly that, and a gate that crashes instead of reporting is worse than no
# gate.
if(MSVC)
    set(NLLB_WARN /W4)
else()
    set(NLLB_WARN -Wall -Wextra -Wno-unused-parameter)
endif()

add_library(nllb STATIC src/nllb.cpp src/nllb-vocab.cpp)
target_compile_options(nllb PRIVATE ${NLLB_WARN})
target_include_directories(nllb PUBLIC src)
target_link_libraries(nllb PUBLIC ggml)

# The CLI. This is the only binary here that is a *product* rather than a
# gate: text in, translation out, no goldens and no reference dumps. It is
# also the first thing in the repo to cross the text -> text seam, which is
# why tests/verify_translation.py exists.
add_executable(nllb-cli cli/nllb-cli.cpp)
target_link_libraries(nllb-cli PRIVATE nllb)
target_compile_options(nllb-cli PRIVATE ${NLLB_WARN})
set_target_properties(nllb-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test tests/test_encoder.cpp)
target_link_libraries(nllb-test PRIVATE nllb)
target_compile_options(nllb-test PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-decoder tests/test_decoder.cpp)
target_link_libraries(nllb-test-decoder PRIVATE nllb)
target_compile_options(nllb-test-decoder PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-decoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-generate tests/test_generate.cpp)
target_link_libraries(nllb-test-generate PRIVATE nllb)
target_compile_options(nllb-test-generate PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-cache tests/test_cache.cpp)
target_link_libraries(nllb-test-cache PRIVATE nllb)
target_compile_options(nllb-test-cache PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-cache PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-bench tests/bench_cached.cpp)
target_link_libraries(nllb-bench PRIVATE nllb)
target_compile_options(nllb-bench PRIVATE ${NLLB_WARN})
set_target_properties(nllb-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-tokenizer tests/test_tokenizer.cpp)
target_link_libraries(nllb-test-tokenizer PRIVATE nllb)
target_compile_options(nllb-test-tokenizer PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-mmap tests/test_mmap.cpp)
target_link_libraries(nllb-test-mmap PRIVATE nllb)
target_compile_options(nllb-test-mmap PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-mmap PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# The quantiser. A TOOL, not a gate: it writes files, it asserts about the file
# it wrote (trap 32), and it makes no claim about the model. It links ggml
# directly rather than through libnllb because it never builds a graph - it
# reads a GGUF, calls ggml_quantize_chunk and writes another one.
add_executable(nllb-quantize tools/nllb-quantize.cpp)
target_link_libraries(nllb-quantize PRIVATE ggml)
target_compile_options(nllb-quantize PRIVATE ${NLLB_WARN})
set_target_properties(nllb-quantize PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-precision tests/test_precision.cpp)
target_link_libraries(nllb-test-precision PRIVATE nllb)
target_compile_options(nllb-test-precision PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-precision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Not a gate on its own: dumps everything the public API computes as raw bytes,
# so two BUILDS can be compared with `nllb-dump-states --diff a.bin b.bin`. That
# is how the step-13 backend port was held to bit-identity against the old CPU
# path, and it is the instrument for a ggml submodule bump.
add_executable(nllb-dump-states tests/dump_states.cpp)
target_link_libraries(nllb-dump-states PRIVATE nllb)
target_compile_options(nllb-dump-states PRIVATE ${NLLB_WARN})
set_target_properties(nllb-dump-states PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
