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

add_library(nllb STATIC src/nllb.cpp)
target_include_directories(nllb PUBLIC src)
target_link_libraries(nllb PUBLIC ggml)

add_executable(nllb-test tests/test_encoder.cpp)
target_link_libraries(nllb-test PRIVATE nllb)
set_target_properties(nllb-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(nllb-test-decoder tests/test_decoder.cpp)
target_link_libraries(nllb-test-decoder PRIVATE nllb)
set_target_properties(nllb-test-decoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
