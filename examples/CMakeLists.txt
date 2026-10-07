# Product binaries and tools built on libnllb. Each lands in build/bin/.

add_subdirectory(cli)
add_subdirectory(quantize)
add_subdirectory(bench)

# The HTTP server. NLLB_BUILD_SERVER=OFF skips it (it pulls in two large single
# headers and a thread dependency nothing else needs).
option(NLLB_BUILD_SERVER "nllb: build examples/server (nllb-server)" ON)
if(NLLB_BUILD_SERVER)
    find_package(Threads REQUIRED)
    add_subdirectory(server)
endif()
