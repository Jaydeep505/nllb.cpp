# nllb.cpp - NLLB-200 inference on ggml, no dependencies beyond the submodule.
#
# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
#   git submodule update --init --recursive        # ggml, first time only
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
# Device backends are OFF by default and each is one flag:
#   -DNLLB_CUDA=ON     step 16   (+ -DCMAKE_CUDA_ARCHITECTURES=<sm> — pin it)
# Metal needs no flag: it is found on macOS and, since step 13, used.
#
# On macOS the configure step finds Accelerate, BLAS and Metal and links them.
# Since step 13 Metal IS used: every graph goes through ggml_backend_sched, and
# nllb-cli defaults to --backend auto (Metal when present). The LIBRARY default
# is still the CPU, so every gate means what it meant unless NLLB_BACKEND=metal
# is set. BLAS is linked and never scheduled. See docs/backend.md.
#
# ---------------------------------------------------------------------------
# LAYOUT (same shape as whisper.cpp)
# ---------------------------------------------------------------------------
#   include/      public headers: nllb.h (C++ API), nllb-vocab.h, nllb-c.h (C API)
#   src/          the library (libnllb)                     src/CMakeLists.txt
#   examples/     cli, server, quantize, bench, nllb.swiftui examples/CMakeLists.txt
#   tests/        the gates + their Python harnesses        tests/CMakeLists.txt
#   models/       download + convert-nllb-to-gguf.py
#   scripts/      reference dumpers, measurement, publishing
#   docs/         per-step write-ups; docs/dev/ is the dev log
#   ggml/         git submodule
#
#   -DNLLB_BUILD_EXAMPLES=OFF / -DNLLB_BUILD_TESTS=OFF skip those directories;
#   both OFF builds the library alone (what build-xcframework.sh does).
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
# The gates run in one command: ./scripts/step10-measure.sh
# Everything above is documented in docs/dev/handoff.md; the trap numbers refer
# to its "traps" section.
# ---------------------------------------------------------------------------
# 3.18 and not 3.14 since step 16: ggml/src/ggml-cuda/CMakeLists.txt
# declares that minimum itself for CMAKE_CUDA_ARCHITECTURES, and a top-level
# minimum lower than what a component needs leaves CMP0104 unset in this
# directory. No platform in the matrix ships CMake older than 3.22.
cmake_minimum_required(VERSION 3.18)
project(nllb.cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# vendored ggml - no external dependencies
set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# CUDA (step 16).  -DNLLB_CUDA=ON
# ---------------------------------------------------------------------------
# ggml's own switch is GGML_CUDA and it works unchanged: ggml is
# add_subdirectory'd below and reads its options out of the same cache. This
# flag exists on top of it to say at CONFIGURE time what this build will be
# able to schedule (see the TARGET check below), and that is all it does.
#
#   cmake -B build -DNLLB_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake -B build -DNLLB_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ...   # T4
#
# PIN CMAKE_CUDA_ARCHITECTURES. ggml sets it to "native" when CUDAToolkit >=
# 11.6 and CMake >= 3.24 — which a Colab runtime satisfies — and otherwise
# builds a ladder of virtual architectures that costs many minutes of nvcc for
# code no machine here will run. 75 T4, 80 A100, 86 RTX 3000, 89 L4/RTX 4000,
# 90 H100. `nvidia-smi --query-gpu=name,compute_cap --format=csv` is the
# lookup.
#
# What the device must have is checked by ggml at RUNTIME and not by this file:
# a machine with the toolkit and no usable driver configures, compiles,
# registers no device, and then `--backend cuda` fails with the device list
# (trap 49's message). That is the contract, not a defect — `auto` falls back
# to the CPU and a named backend never does.
option(NLLB_CUDA "nllb: build the ggml CUDA backend" OFF)

if(NLLB_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "nllb: CUDA backend ON (arch ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "nllb: CUDA backend ON, arch NOT pinned - ggml will pick 'native' or a virtual ladder. Pass -DCMAKE_CUDA_ARCHITECTURES=<sm> (75 T4, 80 A100, 86, 89, 90).")
    endif()
endif()

# Local ggml fixes pending upstream (patches/ggml-*.patch). Each is applied once:
# if it already reverse-applies, it is in the tree (applied earlier, or merged
# upstream after a pin bump), and is skipped.
option(NLLB_GGML_PATCHES "nllb: apply patches/ggml-*.patch to ggml" ON)
if(NLLB_GGML_PATCHES)
    find_package(Git REQUIRED)
    file(GLOB NLLB_GGML_PATCH_FILES "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-*.patch")
    foreach(p ${NLLB_GGML_PATCH_FILES})
        execute_process(COMMAND ${GIT_EXECUTABLE} apply --reverse --check "${p}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ggml"
            RESULT_VARIABLE nllb_already OUTPUT_QUIET ERROR_QUIET)
        if(NOT nllb_already EQUAL 0)
            execute_process(COMMAND ${GIT_EXECUTABLE} apply "${p}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ggml"
                RESULT_VARIABLE nllb_rc)
            if(NOT nllb_rc EQUAL 0)
                message(FATAL_ERROR "nllb: ${p} does not apply to ggml - the pin moved; refresh or drop the patch")
            endif()
            message(STATUS "nllb: applied ${p}")
        endif()
    endforeach()
endif()

add_subdirectory(ggml)

# ggml registers its backends at runtime and a missing one is a runtime
# failure, not a link error - so say at CONFIGURE time what this build will and
# will not be able to schedule. The step-13 failure that cost the most time was
# a device that was present and not FOUND, and the fix started with printing
# what was there.
if(NLLB_CUDA AND NOT TARGET ggml-cuda)
    message(FATAL_ERROR
        "NLLB_CUDA=ON but ggml did not build its CUDA backend. Its "
        "find_package(CUDAToolkit) found no toolkit: install one and put nvcc "
        "on PATH, or point -DCUDAToolkit_ROOT at it. `nvcc --version` is the "
        "check. See docs/build.md.")
endif()

# Warnings on OUR targets only - ggml is added above and keeps its own
# flags. This is not housekeeping: the [arena] report in tests/test-cache.cpp
# shipped with a printf missing its first argument, which is a segfault on the
# failure path and invisible on the passing one. -Wformat (via -Wall) catches
# exactly that, and a gate that crashes instead of reporting is worse than no
# gate.
if(MSVC)
    set(NLLB_WARN /W4)
else()
    set(NLLB_WARN -Wall -Wextra -Wno-unused-parameter)
endif()

# Shared by every subdirectory: our warning policy and one bin/ for every
# executable, so every doc and script can say ./build/bin/<name>.
function(nllb_add_executable name)
    add_executable(${name} ${ARGN})
    target_compile_options(${name} PRIVATE ${NLLB_WARN})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

add_subdirectory(src)

# Neither is needed for the library. build-xcframework.sh turns both OFF: an iOS
# target has no use for a command-line binary and the Xcode generator would try
# to bundle each one.
option(NLLB_BUILD_EXAMPLES "nllb: build examples/ (nllb-cli, nllb-server, nllb-quantize, nllb-bench)" ON)
option(NLLB_BUILD_TESTS    "nllb: build tests/ (the gates)" ON)

if(NLLB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NLLB_BUILD_TESTS)
    add_subdirectory(tests)
endif()
