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
