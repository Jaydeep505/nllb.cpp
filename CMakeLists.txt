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
# Device backends are OFF by default and each is one flag:
#   -DNLLB_OPENCL=ON   step 14   (+ -DNLLB_ADRENO=ON on Snapdragon)
#   -DNLLB_CUDA=ON     step 16   (+ -DCMAKE_CUDA_ARCHITECTURES=<sm> — pin it)
#   -DNLLB_VULKAN=ON   step 17   (needs glslc at BUILD time; see BUILD.md)
# Metal needs no flag: it is found on macOS and, since step 13, used.
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
# 3.18 and not 3.14 since step 16: vendor/ggml/src/ggml-cuda/CMakeLists.txt
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
# OPENCL (step 14).  -DNLLB_OPENCL=ON
# ---------------------------------------------------------------------------
# ggml's own switch is GGML_OPENCL and it works unchanged, because ggml is
# add_subdirectory'd above and reads its options out of the same cache. This
# flag exists on top of it for one reason: ggml's OPENCL DEFAULTS ARE ADRENO
# DEFAULTS, and taking them on anything else produces a build that configures,
# compiles, and then either refuses the device at runtime or runs kernels
# written for a GPU this is not.  GGML_OPENCL_USE_ADRENO_KERNELS is ON by
# default upstream; here the portable value is OFF and -DNLLB_ADRENO=ON is how
# you ask for it back.
#
#   cmake -B build -DNLLB_OPENCL=ON -DCMAKE_BUILD_TYPE=Release   # portable
#   cmake -B build -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON ...         # Snapdragon
#
# What the device must have, checked by ggml at registration and NOT by this
# file: cl_khr_fp16, and - on an OpenCL 3.0 device, where subgroups became
# optional - cl_khr_subgroups or cl_intel_subgroups. A device missing either is
# dropped with a warning on stderr and the registry then has no OpenCL device
# in it, so `--backend opencl` fails with the device list (trap 49's message)
# rather than with anything about extensions. Read the ggml_opencl: lines.
# Measured: pocl 5.0 has the subgroups and NOT fp16, so it is dropped; Intel's
# CPU runtime has both and is picked up. See BACKEND.md.
#
# Kernels are EMBEDDED by default (ggml's default too, kept deliberately): with
# GGML_OPENCL_EMBED_KERNELS=OFF ggml copies .cl files next to the binary at
# configure time and loads them by relative path at runtime, which makes the
# binary non-relocatable. This project ships one file that carries everything
# it needs - that is the step-8 argument about the grapheme tables, and it
# applies to kernels for the same reason.
option(NLLB_OPENCL "nllb: build the ggml OpenCL backend"            OFF)
option(NLLB_ADRENO "nllb: use ggml's Adreno-optimised OpenCL kernels" OFF)

if(NLLB_OPENCL)
    set(GGML_OPENCL                    ON          CACHE BOOL   "" FORCE)
    set(GGML_OPENCL_EMBED_KERNELS      ON          CACHE BOOL   "" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS ${NLLB_ADRENO} CACHE BOOL "" FORCE)
    set(GGML_OPENCL_TARGET_VERSION     "300"       CACHE STRING "" FORCE)
    message(STATUS "nllb: OpenCL backend ON (adreno kernels: ${NLLB_ADRENO})")
endif()

# ---------------------------------------------------------------------------
# CUDA (step 16).  -DNLLB_CUDA=ON
# ---------------------------------------------------------------------------
# ggml's own switch is GGML_CUDA and it works unchanged, for the same reason
# GGML_OPENCL does: ggml is add_subdirectory'd below and reads its options out
# of the same cache. This flag exists on top of it to say at CONFIGURE time
# what this build will be able to schedule (see the TARGET check below) —
# unlike OpenCL there are no Adreno-shaped defaults to undo, so that is all it
# does.
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

# ---------------------------------------------------------------------------
# VULKAN (step 17).  -DNLLB_VULKAN=ON
# ---------------------------------------------------------------------------
# ggml's own switch is GGML_VULKAN and works unchanged. This flag exists for
# the same reason as NLLB_OPENCL / NLLB_CUDA: to fail at CONFIGURE time, by
# name, when the backend was not built (see the TARGET check below).
#
#   cmake -B build -DNLLB_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
#   cmake -B build -DNLLB_VULKAN=ON -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON ...  # X Elite
#
# BUILD needs: the Vulkan headers + loader import library, glslc (shaders are
# compiled to SPIR-V at build time and embedded), and SPIRV-Headers' CMake
# config. The LunarG SDK has all three (set VULKAN_SDK); on Ubuntu:
#   apt install libvulkan-dev glslc spirv-headers
# RUN needs only the driver's vulkan-1.dll / libvulkan.so.1.
#
# ggml builds a HOST tool (vulkan-shaders-gen) during the build and runs it,
# and ggml refuses cl.exe on Arm outright ("MSVC is not supported for ARM, use
# clang"). So a Windows-on-Arm build is done NATIVELY with clang (a
# windows-11-arm runner, or Arm64 hardware), not cross-compiled with -A ARM64
# from x64. See QDC-XELITE.md and .github/workflows/xelite.yml.
#
# Two runtime facts that are not visible from here, found in step 17:
#  * ggml registers Vulkan BEFORE OpenCL (ggml-backend-reg.cpp), so with both, `auto` and
#    `gpu` pick Vulkan0. Name `opencl` to get the other one.
#  * ggml-vulkan HIDES CPU-type devices (Mesa lavapipe, SwiftShader) unless
#    GGML_VK_VISIBLE_DEVICES names them. That is how CI runs graphs on it.
option(NLLB_VULKAN "nllb: build the ggml Vulkan backend" OFF)

if(NLLB_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "nllb: Vulkan backend ON")
endif()

# Local ggml fixes pending upstream (patches/ggml-*.patch). Each is applied once:
# if it already reverse-applies, it is in the tree (applied earlier, or merged
# upstream after a pin bump), and is skipped.
option(NLLB_GGML_PATCHES "nllb: apply patches/ggml-*.patch to vendor/ggml" ON)
if(NLLB_GGML_PATCHES)
    find_package(Git REQUIRED)
    file(GLOB NLLB_GGML_PATCH_FILES "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-*.patch")
    foreach(p ${NLLB_GGML_PATCH_FILES})
        execute_process(COMMAND ${GIT_EXECUTABLE} apply --reverse --check "${p}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml"
            RESULT_VARIABLE nllb_already OUTPUT_QUIET ERROR_QUIET)
        if(NOT nllb_already EQUAL 0)
            execute_process(COMMAND ${GIT_EXECUTABLE} apply "${p}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml"
                RESULT_VARIABLE nllb_rc)
            if(NOT nllb_rc EQUAL 0)
                message(FATAL_ERROR "nllb: ${p} does not apply to vendor/ggml - the pin moved; refresh or drop the patch")
            endif()
            message(STATUS "nllb: applied ${p}")
        endif()
    endforeach()
endif()

add_subdirectory(vendor/ggml)

# ggml registers its backends at runtime and a missing one is a runtime
# failure, not a link error - so say at CONFIGURE time what this build will and
# will not be able to schedule. The step-13 failure that cost the most time was
# a device that was present and not FOUND, and the fix started with printing
# what was there.
if(NLLB_OPENCL AND NOT TARGET ggml-opencl)
    message(FATAL_ERROR
        "NLLB_OPENCL=ON but ggml did not build its OpenCL backend. Its "
        "find_package(OpenCL REQUIRED) needs headers and an ICD loader: "
        "apt install opencl-headers ocl-icd-opencl-dev (Linux), or point "
        "-DOpenCL_INCLUDE_DIR / -DOpenCL_LIBRARY at them. See BUILD.md.")
endif()

if(NLLB_VULKAN AND NOT TARGET ggml-vulkan)
    message(FATAL_ERROR
        "NLLB_VULKAN=ON but ggml did not build its Vulkan backend. Its "
        "find_package(Vulkan COMPONENTS glslc) needs the headers, the loader "
        "library and glslc: install the LunarG SDK and set VULKAN_SDK, or on "
        "Ubuntu `apt install libvulkan-dev glslc spirv-headers`. See BUILD.md.")
endif()

if(NLLB_CUDA AND NOT TARGET ggml-cuda)
    message(FATAL_ERROR
        "NLLB_CUDA=ON but ggml did not build its CUDA backend. Its "
        "find_package(CUDAToolkit) found no toolkit: install one and put nvcc "
        "on PATH, or point -DCUDAToolkit_ROOT at it. `nvcc --version` is the "
        "check. See BUILD.md.")
endif()

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

# nllb-c.cpp is the plain-C facade (include/nllb-c.h) that Swift and other FFIs
# import; build-xcframework.sh puts only that header in the framework.
add_library(nllb STATIC src/nllb.cpp src/nllb-vocab.cpp src/nllb-c.cpp)
target_compile_options(nllb PRIVATE ${NLLB_WARN})
target_include_directories(nllb PUBLIC include src)
target_link_libraries(nllb PUBLIC ggml)

# NLLB_BUILD_TOOLS=OFF builds the library alone - no CLI, no gates, no tools.
# That is what build-xcframework.sh passes: an iOS target has no use for a
# command-line binary and the Xcode generator would try to bundle each one.
option(NLLB_BUILD_TOOLS "nllb: build nllb-cli, the gates and the tools" ON)

if(NLLB_BUILD_TOOLS)

# The CLI. This is the only binary here that is a *product* rather than a
# gate: text in, translation out, no goldens and no reference dumps. It is
# also the first thing in the repo to cross the text -> text seam, which is
# why tests/verify_translation.py exists.
add_executable(nllb-cli cli/nllb-cli.cpp)
target_link_libraries(nllb-cli PRIVATE nllb)
target_compile_options(nllb-cli PRIVATE ${NLLB_WARN})
set_target_properties(nllb-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# The HTTP server (examples/server). A product binary like nllb-cli, so it is
# built with the tools; NLLB_BUILD_SERVER=OFF skips it (it pulls in two large
# single headers and a thread dependency the gates do not need).
option(NLLB_BUILD_SERVER "nllb: build examples/server (nllb-server)" ON)
if(NLLB_BUILD_SERVER)
    find_package(Threads REQUIRED)
    add_subdirectory(examples/server)
endif()

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

# ocl-probe. NOT a gate and not linked against nllb or ggml - it links the
# OpenCL ICD loader alone, on purpose, so that when it disagrees with ggml the
# fault is in ggml and when it agrees the fault is in the runtime. Built only
# with the backend, since without it there is no OpenCL to find.
#
# It answers the two questions step 14 needed and clinfo cannot: will ggml
# REGISTER this device (extensions), and will ggml's kernels COMPILE on it
# (a half math builtin under -cl-fast-relaxed-math). Those fail at different
# stages and the second one looks like success until it does not.
if(NLLB_OPENCL)
    # ggml's own find_package(OpenCL) ran inside add_subdirectory, so its
    # result variables are in THAT directory's scope and are empty here. Ask
    # again - it is cached, so this costs nothing and is not a second search.
    find_package(OpenCL REQUIRED)
    add_executable(ocl-probe tools/ocl-probe.c)
    # HEADERS ONLY - no link against the ICD loader. The probe dlopen()s the
    # loader at runtime, so it can be cross-compiled for a phone without
    # first cross-compiling an ICD loader for that phone, which is the
    # dependency you would otherwise have to build BEFORE finding out whether
    # the device works at all. On Linux/Android that needs libdl.
    target_include_directories(ocl-probe PRIVATE ${OpenCL_INCLUDE_DIRS})
    if(NOT WIN32)
        target_link_libraries(ocl-probe PRIVATE ${CMAKE_DL_LIBS})
    endif()
    target_compile_options(ocl-probe PRIVATE ${NLLB_WARN})
    set_target_properties(ocl-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Step 17. The cached step's four KV-cache ops (the two cpy writes into the
# cache views, the two mul_mats over them) one at a time, device vs CPU, at
# Tn = 1, 3, 9. Needs no model. It is how the Vulkan cache-gate failure was
# localised to one op, and the first thing to run on any new device backend.
add_executable(nllb-probe-kv-ops tests/probe_kv_ops.cpp)
target_link_libraries(nllb-probe-kv-ops PRIVATE ggml)
target_compile_options(nllb-probe-kv-ops PRIVATE ${NLLB_WARN})
set_target_properties(nllb-probe-kv-ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

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

endif() # NLLB_BUILD_TOOLS
