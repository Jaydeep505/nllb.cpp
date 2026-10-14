# nllb.cpp - NLLB-200 inference on ggml. Build instructions: see README.md.

cmake_minimum_required(VERSION 3.18)
project(nllb.cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

option(NLLB_CUDA "nllb: build the ggml CUDA backend" OFF)

if(NLLB_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "nllb: CUDA backend ON (arch ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "nllb: CUDA backend ON, arch NOT pinned - ggml will pick 'native' or a virtual ladder. Pass -DCMAKE_CUDA_ARCHITECTURES=<sm> (75 T4, 80 A100, 86, 89, 90).")
    endif()
endif()

# Vulkan: -DNLLB_VULKAN=ON. BUILD needs the Vulkan headers + loader, glslc and
# SPIRV-Headers (LunarG SDK with VULKAN_SDK set, or on Ubuntu
# `apt install libvulkan-dev glslc spirv-headers`); RUN needs only the driver's
# vulkan-1.dll / libvulkan.so.1. ggml-vulkan builds and RUNS a host tool
# (vulkan-shaders-gen), and ggml refuses MSVC on Arm, so Windows on Arm (the
# Snapdragon X Elite) is built NATIVELY with clang - see docs/qdc-xelite.md.
# ggml-vulkan hides CPU-type devices (Mesa lavapipe) unless
# GGML_VK_VISIBLE_DEVICES names them; that is how CI runs graphs without a GPU.
option(NLLB_VULKAN "nllb: build the ggml Vulkan backend" OFF)

if(NLLB_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "nllb: Vulkan backend ON")
endif()

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

if(NLLB_CUDA AND NOT TARGET ggml-cuda)
    message(FATAL_ERROR
        "NLLB_CUDA=ON but ggml did not build its CUDA backend. Its "
        "find_package(CUDAToolkit) found no toolkit: install one and put nvcc "
        "on PATH, or point -DCUDAToolkit_ROOT at it. `nvcc --version` is the "
        "check.")
endif()

if(NLLB_VULKAN AND NOT TARGET ggml-vulkan)
    message(FATAL_ERROR
        "NLLB_VULKAN=ON but ggml did not build its Vulkan backend. Its "
        "find_package(Vulkan COMPONENTS glslc) needs the headers, the loader "
        "library and glslc: install the LunarG SDK and set VULKAN_SDK, or on "
        "Ubuntu `apt install libvulkan-dev glslc spirv-headers`.")
endif()

if(MSVC)
    set(NLLB_WARN /W4)
else()
    set(NLLB_WARN -Wall -Wextra -Wno-unused-parameter)
endif()

function(nllb_add_executable name)
    add_executable(${name} ${ARGN})
    target_compile_options(${name} PRIVATE ${NLLB_WARN})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

add_subdirectory(src)

option(NLLB_BUILD_EXAMPLES "nllb: build examples/ (nllb-cli, nllb-server, nllb-quantize, nllb-bench)" ON)
option(NLLB_BUILD_TESTS    "nllb: build tests/" ON)

if(NLLB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NLLB_BUILD_TESTS)
    add_subdirectory(tests)
endif()