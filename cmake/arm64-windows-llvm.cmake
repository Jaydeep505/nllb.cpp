# Cross-compile from Windows x64 to Windows on Arm64 (Snapdragon X Elite).
#
#   cmake -B build-arm64 -G Ninja `
#       -DCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-llvm.cmake `
#       -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON -DCMAKE_BUILD_TYPE=Release
#
# Transcribed from llama.cpp's cmake/arm64-windows-llvm.cmake, which is the
# file its Windows Arm64 OpenCL recipe uses, with this project's warning
# policy left alone - see the note at the bottom.
#
# WHAT YOU NEED INSTALLED (Build Tools for Visual Studio 2022):
#   * "MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools"   <- the target libs
#   * "C++ Clang Compiler for Windows"                      <- the compiler
# clang targets arm64-pc-windows-msvc, so it still needs MSVC's ARM64 headers
# and import libraries. Installing clang alone is not enough, and the error
# when the ARM64 libs are missing names a link failure rather than a missing
# component.
#
# NOTE this is `clang`, not `clang-cl`: the command line is GNU-style, so
# CMake leaves MSVC false and this project's -Wall -Wextra apply unchanged.
# That is deliberate. The warning flags earned their place (a printf missing
# an argument, which was a segfault on the failure path and invisible on the
# passing one) and a cross build is exactly where you want them on.

set( CMAKE_SYSTEM_NAME      Windows )
set( CMAKE_SYSTEM_PROCESSOR arm64 )

set( target arm64-pc-windows-msvc )

set( CMAKE_C_COMPILER   clang )
set( CMAKE_CXX_COMPILER clang++ )

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

# -march=armv8.7-a is llama.cpp's choice for the X Elite (Oryon).
#
# -ffp-model=fast IS A CHANGE OF ARITHMETIC, and this project has an
# instrument for that: build with it, dump, and diff against a CPU dump from a
# build without it -
#
#   nllb-dump-states models\shape.gguf ref\ a.bin -t 4
#   nllb-dump-states --diff a.bin b.bin
#
# If that is not IDENTICAL then every number in the calibration ladder was
# measured under different float semantics than this binary uses, and the
# honest move is to drop the flag rather than to explain the difference away.
# Nobody has run that diff yet. It is the first thing to do if a gate goes red
# on Arm64 and you cannot see why.
set( arch_c_flags "-march=armv8.7-a -fvectorize -ffp-model=fast -fno-finite-math-only" )

# llama.cpp additionally silences -Wno-format / -Wno-unused-variable /
# -Wno-unused-function here. Those are NOT copied: they would switch off
# exactly the warnings CMakeLists.txt turns on for this project's own targets,
# and vendor/ggml keeps its own flags either way.
set( CMAKE_C_FLAGS_INIT   "${arch_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags}" )
