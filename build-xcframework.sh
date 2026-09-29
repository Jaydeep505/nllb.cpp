#!/bin/bash
#
# build-xcframework.sh - package libnllb + ggml (Metal) as nllb.xcframework.
#
# Modeled on whisper.cpp's script of the same name. Differences, all deliberate:
#   * ggml lives at vendor/ggml, so its static libraries are FOUND by name
#     under each build directory rather than hard-coded by path;
#   * the framework exports ONE header, src/nllb-c.h. nllb.h is C++ and Swift
#     cannot import it; the C facade is the whole app-facing API;
#   * slices: iOS device, iOS simulator, macOS. visionOS/tvOS are one more
#     cmake block each (see whisper.cpp) and nothing here has been run on them;
#   * arm64 ONLY, simulator and macOS included (Apple silicon). Intel slices
#     came out without ggml's symbols on the first real run; nothing here
#     targets an Intel Mac, so they are dropped rather than debugged.
#   * GGML_BLAS=OFF: nllb schedules the primary device + the CPU and never the
#     BLAS backend (BACKEND.md), so linking it only adds size. The CPU backend
#     still uses Accelerate's vDSP (GGML_ACCELERATE, on by default).
#
# Usage (macOS, Xcode + CMake):
#   git submodule update --init --recursive
#   ./build-xcframework.sh                               # -> build-apple/nllb.xcframework
#   BUILD_STATIC_XCFRAMEWORK=ON ./build-xcframework.sh   # static variant
#
# Then drag build-apple/nllb.xcframework into the app target ("Embed & Sign"
# for the default dynamic variant) and `import nllb` in Swift.

# Options
IOS_MIN_OS_VERSION=16.4
MACOS_MIN_OS_VERSION=13.3

BUILD_SHARED_LIBS=OFF
GGML_METAL=ON
GGML_METAL_EMBED_LIBRARY=ON     # shaders inside the binary: no .metallib to ship or find
GGML_METAL_USE_BF16=ON
GGML_BLAS=OFF
GGML_OPENMP=OFF                 # no libomp on iOS; ggml's own thread pool is used
BUILD_STATIC_XCFRAMEWORK=${BUILD_STATIC_XCFRAMEWORK:-OFF}

FRAMEWORK_NAME="nllb"

COMMON_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
COMMON_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"

# Common options for all builds
COMMON_CMAKE_ARGS=(
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=""
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    -DCMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT="dwarf-with-dsym"
    -DCMAKE_XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS=YES
    -DCMAKE_XCODE_ATTRIBUTE_COPY_PHASE_STRIP=NO
    -DCMAKE_XCODE_ATTRIBUTE_STRIP_INSTALLED_PRODUCT=NO
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=ggml
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DNLLB_BUILD_TOOLS=OFF
    -DGGML_METAL=${GGML_METAL}
    -DGGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY}
    -DGGML_METAL_USE_BF16=${GGML_METAL_USE_BF16}
    -DGGML_BLAS=${GGML_BLAS}
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=${GGML_OPENMP}
    -DGGML_BUILD_TESTS=OFF
    -DGGML_BUILD_EXAMPLES=OFF
)

XCODE_VERSION=$(xcodebuild -version 2>/dev/null | head -n1 | awk '{ print $2 }')
echo "Detected Xcode version: $XCODE_VERSION"

check_required_tool() {
    local tool=$1
    local install_message=$2

    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not found."
        echo "$install_message"
        exit 1
    fi
}
echo "Checking for required tools..."
check_required_tool "cmake" "Please install CMake 3.28.0 or later (brew install cmake)"
check_required_tool "xcodebuild" "Please install Xcode and Xcode Command Line Tools (xcode-select --install)"
check_required_tool "libtool" "Please install libtool which should be available with Xcode Command Line Tools (CLT). Make sure Xcode CLT is installed (xcode-select --install)"
check_required_tool "dsymutil" "Please install Xcode and Xcode Command Line Tools (xcode-select --install)"

set -e

# The upload/clone trap from the handoff: a `git archive` or a plain clone
# leaves vendor/ggml EMPTY, and cmake's error about it is not obvious.
if [ ! -f vendor/ggml/CMakeLists.txt ]; then
    echo "Error: vendor/ggml is empty. Run: git submodule update --init --recursive"
    exit 1
fi

## Clean up previous builds
rm -rf build-apple
rm -rf build-ios-sim
rm -rf build-ios-device
rm -rf build-macos

# Setup the xcframework build directory structure
setup_framework_structure() {
    local build_dir=$1
    local min_os_version=$2
    local platform=$3  # "ios" or "macos"
    local framework_name="${FRAMEWORK_NAME}"

    echo "Creating ${platform}-style framework structure for ${build_dir}"

    if [[ "$platform" == "macos" ]]; then
        # macOS versioned structure uses versioned directories
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Versions/A/Headers
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Versions/A/Modules
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Versions/A/Resources

        # Create symbolic links
        ln -sf A ${build_dir}/framework/${framework_name}.framework/Versions/Current
        ln -sf Versions/Current/Headers ${build_dir}/framework/${framework_name}.framework/Headers
        ln -sf Versions/Current/Modules ${build_dir}/framework/${framework_name}.framework/Modules
        ln -sf Versions/Current/Resources ${build_dir}/framework/${framework_name}.framework/Resources
        ln -sf Versions/Current/${framework_name} ${build_dir}/framework/${framework_name}.framework/${framework_name}

        local header_path=${build_dir}/framework/${framework_name}.framework/Versions/A/Headers/
        local module_path=${build_dir}/framework/${framework_name}.framework/Versions/A/Modules/
    else
        # iOS uses a flat structure
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Headers
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Modules
        rm -rf ${build_dir}/framework/${framework_name}.framework/Versions

        local header_path=${build_dir}/framework/${framework_name}.framework/Headers/
        local module_path=${build_dir}/framework/${framework_name}.framework/Modules/
    fi

    # ONE header. It is pure C (checked with -std=c99 -pedantic), so the
    # module imports into Swift with no bridging header.
    cp src/nllb-c.h ${header_path}

    cat > ${module_path}module.modulemap << EOF
framework module ${framework_name} {
    header "nllb-c.h"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

    local platform_name=""
    local sdk_name=""
    local supported_platform=""

    case "$platform" in
        "ios")
            platform_name="iphoneos"
            sdk_name="iphoneos${min_os_version}"
            supported_platform="iPhoneOS"
            local plist_path="${build_dir}/framework/${framework_name}.framework/Info.plist"
            local device_family='    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>'
            ;;
        "macos")
            platform_name="macosx"
            sdk_name="macosx${min_os_version}"
            supported_platform="MacOSX"
            local plist_path="${build_dir}/framework/${framework_name}.framework/Versions/A/Resources/Info.plist"
            local device_family=""
            ;;
    esac

    cat > ${plist_path} << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${framework_name}</string>
    <key>CFBundleIdentifier</key>
    <string>org.ggml.${framework_name}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${framework_name}</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${min_os_version}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>${supported_platform}</string>
    </array>${device_family}
    <key>DTPlatformName</key>
    <string>${platform_name}</string>
    <key>DTSDKName</key>
    <string>${sdk_name}</string>
</dict>
</plist>
EOF
}

# Find exactly one built static library by name inside one build directory's
# Release products. Required libraries that are missing are a hard error with
# the name in it - a framework that silently lacks libggml-metal.a links fine
# and then has no GPU at runtime ("auto" quietly means CPU).
find_lib() {
    local build_dir="$1" release_dir="$2" name="$3" required="$4"
    local hit
    hit=$(find "${build_dir}" -name "${name}" -path "*/${release_dir}/*" -not -path "*/temp/*" | head -n1)
    if [[ -z "$hit" && "$required" == "required" ]]; then
        echo "Error: ${name} not found under ${build_dir} (*/${release_dir}/*)" >&2
        exit 1
    fi
    echo "$hit"
}

# Create dynamic libraries from static libraries.
combine_static_libraries() {
    local build_dir="$1"
    local release_dir="$2"
    local platform="$3"  # "ios" or "macos"
    local is_simulator="$4"
    local base_dir="$(pwd)"
    local framework_name="${FRAMEWORK_NAME}"

    local output_lib=""
    if [[ "$platform" == "macos" ]]; then
        output_lib="${build_dir}/framework/${framework_name}.framework/Versions/A/${framework_name}"
    else
        output_lib="${build_dir}/framework/${framework_name}.framework/${framework_name}"
    fi

    local libs=()
    local name hit
    for name in libnllb.a libggml.a libggml-base.a libggml-cpu.a libggml-metal.a; do
        hit=$(find_lib "${build_dir}" "${release_dir}" "${name}" required) || exit 1
        libs+=("${base_dir}/${hit}")
    done
    echo "Combining: ${libs[*]##*/}"

    local temp_dir="${base_dir}/${build_dir}/temp"
    mkdir -p "${temp_dir}"

    # Every input must carry every architecture this slice links for. If one
    # does not, libtool still "succeeds" and the dylib link then fails with
    # hundreds of undefined ggml/gguf symbols for the missing arch - so check
    # here, and name the library.
    local want_archs
    case "$platform" in
        "ios")   want_archs="arm64" ;;
        "macos") want_archs="arm64" ;;
    esac
    local lib a
    for lib in "${libs[@]}"; do
        echo "  $(lipo -archs "$lib")  ${lib#${base_dir}/}"
        for a in $want_archs; do
            if ! lipo -archs "$lib" | tr ' ' '\n' | grep -qx "$a"; then
                echo "Error: ${lib#${base_dir}/} has no $a slice (has: $(lipo -archs "$lib"))" >&2
                exit 1
            fi
        done
    done

    libtool -static -o "${temp_dir}/combined.a" "${libs[@]}"

    local sdk=""
    local archs=""
    local min_version_flag=""
    local install_name=""
    local frameworks="-framework Foundation -framework Metal -framework Accelerate"

    case "$platform" in
        "ios")
            if [[ "$is_simulator" == "true" ]]; then
                sdk="iphonesimulator"
                archs="arm64"
                min_version_flag="-mios-simulator-version-min=${IOS_MIN_OS_VERSION}"
            else
                sdk="iphoneos"
                archs="arm64"
                min_version_flag="-mios-version-min=${IOS_MIN_OS_VERSION}"
            fi
            install_name="@rpath/${framework_name}.framework/${framework_name}"
            ;;
        "macos")
            sdk="macosx"
            archs="arm64"
            min_version_flag="-mmacosx-version-min=${MACOS_MIN_OS_VERSION}"
            install_name="@rpath/${framework_name}.framework/Versions/Current/${framework_name}"
            ;;
    esac

    local arch_flags=""
    for arch in $archs; do
        arch_flags+=" -arch $arch"
    done

    if [[ "${BUILD_STATIC_XCFRAMEWORK}" == "ON" ]]; then
        echo "Packaging static framework for ${platform}."
        mkdir -p "$(dirname "${base_dir}/${output_lib}")"
        cp "${temp_dir}/combined.a" "${base_dir}/${output_lib}"
        rm -rf "${temp_dir}"
        return
    fi

    echo "Creating dynamic library for ${platform}."
    xcrun -sdk $sdk clang++ -dynamiclib \
        -isysroot $(xcrun --sdk $sdk --show-sdk-path) \
        $arch_flags \
        $min_version_flag \
        -Wl,-force_load,"${temp_dir}/combined.a" \
        $frameworks \
        -install_name "$install_name" \
        -o "${base_dir}/${output_lib}"

    if [[ "$is_simulator" == "false" && "$platform" == "ios" ]]; then
        if command -v xcrun vtool &>/dev/null; then
            echo "Marking binary as a framework binary for iOS..."
            xcrun vtool -set-build-version ios ${IOS_MIN_OS_VERSION} ${IOS_MIN_OS_VERSION} -replace \
                -output "${base_dir}/${output_lib}" "${base_dir}/${output_lib}"
        else
            echo "Warning: vtool not found. Binary may not pass App Store validation."
        fi
    fi

    echo "Creating properly formatted dSYM..."
    mkdir -p "${base_dir}/${build_dir}/dSYMs"

    if [[ "$platform" == "ios" ]]; then
        xcrun dsymutil "${base_dir}/${output_lib}" -o "${base_dir}/${build_dir}/dSYMs/${framework_name}.dSYM"
        cp "${base_dir}/${output_lib}" "${temp_dir}/binary_to_strip"
        xcrun strip -S "${temp_dir}/binary_to_strip" -o "${temp_dir}/stripped_lib"
        mv "${temp_dir}/stripped_lib" "${base_dir}/${output_lib}"
    else
        xcrun strip -S "${base_dir}/${output_lib}" -o "${temp_dir}/stripped_lib"
        xcrun dsymutil "${base_dir}/${output_lib}" -o "${base_dir}/${build_dir}/dSYMs/${framework_name}.dSYM"
        mv "${temp_dir}/stripped_lib" "${base_dir}/${output_lib}"
    fi

    # A dSYM inside the framework fails App Store validation (Invalid Bundle Structure).
    if [ -d "${base_dir}/${output_lib}.dSYM" ]; then
        rm -rf "${base_dir}/${output_lib}.dSYM"
    fi

    rm -rf "${temp_dir}"
}

echo "Building for iOS simulator..."
cmake -B build-ios-sim -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_MIN_OS_VERSION} \
    -DIOS=ON \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphonesimulator \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -S .
cmake --build build-ios-sim --config Release -- -quiet

echo "Building for iOS devices..."
cmake -B build-ios-device -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -S .
cmake --build build-ios-device --config Release -- -quiet

echo "Building for macOS..."
cmake -B build-macos -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -S .
cmake --build build-macos --config Release -- -quiet

echo "Setting up framework structures..."
setup_framework_structure "build-ios-sim"    ${IOS_MIN_OS_VERSION}   "ios"
setup_framework_structure "build-ios-device" ${IOS_MIN_OS_VERSION}   "ios"
setup_framework_structure "build-macos"      ${MACOS_MIN_OS_VERSION} "macos"

echo "Creating dynamic libraries from static libraries..."
combine_static_libraries "build-ios-sim"    "Release-iphonesimulator" "ios"   "true"
combine_static_libraries "build-ios-device" "Release-iphoneos"        "ios"   "false"
combine_static_libraries "build-macos"      "Release"                 "macos" "false"

echo "Creating XCFramework..."

if [[ "${BUILD_STATIC_XCFRAMEWORK}" == "ON" ]]; then
    xcodebuild -create-xcframework \
        -framework $(pwd)/build-ios-sim/framework/${FRAMEWORK_NAME}.framework \
        -framework $(pwd)/build-ios-device/framework/${FRAMEWORK_NAME}.framework \
        -framework $(pwd)/build-macos/framework/${FRAMEWORK_NAME}.framework \
        -output $(pwd)/build-apple/${FRAMEWORK_NAME}.xcframework
    exit 0
fi

xcodebuild -create-xcframework \
    -framework $(pwd)/build-ios-sim/framework/${FRAMEWORK_NAME}.framework \
    -debug-symbols $(pwd)/build-ios-sim/dSYMs/${FRAMEWORK_NAME}.dSYM \
    -framework $(pwd)/build-ios-device/framework/${FRAMEWORK_NAME}.framework \
    -debug-symbols $(pwd)/build-ios-device/dSYMs/${FRAMEWORK_NAME}.dSYM \
    -framework $(pwd)/build-macos/framework/${FRAMEWORK_NAME}.framework \
    -debug-symbols $(pwd)/build-macos/dSYMs/${FRAMEWORK_NAME}.dSYM \
    -output $(pwd)/build-apple/${FRAMEWORK_NAME}.xcframework

echo "Done: build-apple/${FRAMEWORK_NAME}.xcframework"
