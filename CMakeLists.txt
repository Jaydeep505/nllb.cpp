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

add_executable(nllb-test-precision tests/test_precision.cpp)
target_link_libraries(nllb-test-precision PRIVATE nllb)
target_compile_options(nllb-test-precision PRIVATE ${NLLB_WARN})
set_target_properties(nllb-test-precision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
