@echo off
rem Native Windows-on-Arm build of the three QDC flavours (cpu / opencl / vulkan).
rem The SAME recipe as .github/workflows/xelite.yml - use that workflow if you
rem can; this script is for building ON an Arm64 Windows machine by hand.
rem
rem Needs (Arm64 host): VS 2022 Build Tools with "MSVC v143 ARM64 build tools",
rem LLVM (clang on PATH), CMake, Ninja, the LunarG Vulkan SDK for Windows on
rem Arm (VULKAN_SDK set), and OpenCL headers + ICD loader installed under
rem %OPENCL_PREFIX% (see the workflow's "OpenCL headers + ICD loader" step).
rem
rem It does NOT cross-compile from x64: ggml refuses cl.exe on Arm ("MSVC is not
rem supported for ARM, use clang") and ggml-vulkan runs a host tool it builds.

if "%OPENCL_PREFIX%"=="" set OPENCL_PREFIX=%USERPROFILE%\opencl-arm64
if "%VULKAN_SDK%"=="" (echo VULKAN_SDK is not set - install the LunarG SDK for Windows on Arm & exit /b 1)

call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 2>nul || ^
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" arm64 || exit /b 1

rem GGML_NATIVE=OFF + one -march: the binary must run on the X Elite (Armv8.7,
rem no SVE) whatever machine built it.
set COMMON=-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.7-a -DGGML_OPENMP=OFF

cmake -B build-cpu %COMMON% || exit /b 1
cmake --build build-cpu -j || exit /b 1
cmake -B build-opencl %COMMON% -DNLLB_OPENCL=ON -DNLLB_ADRENO=ON -DCMAKE_PREFIX_PATH=%OPENCL_PREFIX% || exit /b 1
cmake --build build-opencl -j || exit /b 1
cmake -B build-vulkan %COMMON% -DNLLB_VULKAN=ON || exit /b 1
cmake --build build-vulkan -j || exit /b 1

for %%b in (cpu opencl vulkan) do build-%%b\bin\nllb-cli.exe --list-backends
