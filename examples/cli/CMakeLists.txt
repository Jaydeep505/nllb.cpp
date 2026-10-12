# nllb-cli. The only binary here that is a *product* rather than a gate: text
# in, translation out, no goldens and no reference dumps. It is also the first
# thing in the repo to cross the text -> text seam, which is why
# tests/verify_translation.py exists.
nllb_add_executable(nllb-cli cli.cpp)
target_link_libraries(nllb-cli PRIVATE nllb)
