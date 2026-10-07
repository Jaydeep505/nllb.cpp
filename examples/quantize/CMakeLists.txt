# nllb-quantize. A TOOL, not a gate: it writes files, it asserts about the file
# it wrote (trap 32), and it makes no claim about the model. It links ggml
# directly rather than through libnllb because it never builds a graph - it
# reads a GGUF, calls ggml_quantize_chunk and writes another one.
nllb_add_executable(nllb-quantize quantize.cpp)
target_link_libraries(nllb-quantize PRIVATE ggml)
