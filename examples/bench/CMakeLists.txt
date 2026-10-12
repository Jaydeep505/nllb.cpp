# nllb-bench: where the cached decoder step goes.  --kv-f16
nllb_add_executable(nllb-bench bench.cpp)
target_link_libraries(nllb-bench PRIVATE nllb)
