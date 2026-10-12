# nllb-server: HTTP front end (examples/server/README.md).
# httplib.h and json.hpp are vendored single headers (MIT), as in whisper.cpp.
set(TARGET nllb-server)
add_executable(${TARGET} server.cpp)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE nllb Threads::Threads)
set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

install(TARGETS ${TARGET} RUNTIME)

# nllb-server-mock: the same server with the model replaced by a stand-in
# (-DNLLB_SERVER_MOCK, see server.cpp). It needs no weights, so the HTTP,
# validation and batching logic can be tested on any CI runner
# (tests/server/, .github/workflows/server.yml). Off by default.
option(NLLB_BUILD_SERVER_MOCK "nllb: also build nllb-server-mock (no model; for tests/server)" OFF)
if (NLLB_BUILD_SERVER_MOCK)
    add_executable(nllb-server-mock server.cpp)
    target_compile_definitions(nllb-server-mock PRIVATE NLLB_SERVER_MOCK)
    target_compile_features(nllb-server-mock PRIVATE cxx_std_17)
    target_link_libraries(nllb-server-mock PRIVATE Threads::Threads)
    set_target_properties(nllb-server-mock PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if (WIN32)
        target_link_libraries(nllb-server-mock PRIVATE ws2_32)
    endif()
endif()
