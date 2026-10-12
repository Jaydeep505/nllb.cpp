# libnllb. nllb-c.cpp is the plain-C facade (include/nllb-c.h) that Swift and
# other FFIs import; build-xcframework.sh puts only that header in the framework.
#
# include/ is PUBLIC (nllb.h, nllb-vocab.h, nllb-c.h); src/ is PRIVATE, so
# nllb-grapheme-data.h and anything else in here stays an implementation detail.
add_library(nllb STATIC
    nllb.cpp
    nllb-vocab.cpp
    nllb-c.cpp)
target_compile_options(nllb PRIVATE ${NLLB_WARN})
target_include_directories(nllb
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nllb PUBLIC ggml)
