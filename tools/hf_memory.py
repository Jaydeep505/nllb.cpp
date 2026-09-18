#!/usr/bin/env python3
"""
hf_memory.py - what HuggingFace costs in RAM and in TIME, for comparison with
nllb-cli. Prints the same two quantities nllb-cli's `-v` reports (load ms,
generate ms/token) plus the peak RSS /usr/bin/time -l reports, and a breakdown
neither of those can give you.

    python tools/hf_memory.py models/nllb-600M f32 deu_Latn eng_Latn
    python tools/hf_memory.py models/nllb-600M f16 deu_Latn eng_Latn

Cross-check the RSS against the OS:

    /usr/bin/time -l python tools/hf_memory.py models/nllb-600M f32 \
        deu_Latn eng_Latn 2>&1 | grep -E "PEAK|maximum resident"

CPU IS FORCED, and asserted. `pipeline()` moves the model to mps in place on
Apple silicon (see "the translation dumper moved the model to MPS" in the
handoff); mps is a different arithmetic path, so a number taken there is not
comparable with a CPU f32 C++ run, and a greedy decision with a small margin
can land the other way.

WHAT MAPS ONTO WHAT, so the two sides can actually be compared:

    nllb-cli  `load:   N ms`      = nllb_model_load + nllb_vocab_load_from_gguf
    here      `LOAD TOTAL`        = torch import + from_pretrained + tokenizer

    nllb-cli  `timing: generate`  = wall clock over new tokens, encoder pass
    here      `generate`            included. Comparable to each other and NOT
                                    to nllb-bench, which excludes one-time cost.

The torch import is inside the load total on purpose: it is time the user
waits before a translation appears, and the C++ binary does not pay it. It is
reported separately so it can be subtracted if you want the narrower
weights-only comparison.
"""

from __future__ import annotations

import resource
import sys
import time

T_START = time.perf_counter()


def ms() -> float:
    return (time.perf_counter() - T_START) * 1000.0


def rss_gb() -> float:
    """Peak RSS in GB. macOS ru_maxrss is BYTES, Linux is KILOBYTES."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/nllb-600M"
    dtype_arg = sys.argv[2] if len(sys.argv) > 2 else "f32"
    src_lang  = sys.argv[3] if len(sys.argv) > 3 else "deu_Latn"
    tgt_lang  = sys.argv[4] if len(sys.argv) > 4 else "eng_Latn"
    in_file   = sys.argv[5] if len(sys.argv) > 5 else "input.txt"

    print(f"model:  {model_dir}  dtype={dtype_arg}")
    print(f"run:    {src_lang} -> {tgt_lang} | num_beams=1 | do_sample=False\n")

    hdr = f"{'stage':<34}{'ms':>10}{'cumulative ms':>15}{'RSS GB':>10}"
    print(hdr)
    print("-" * len(hdr))

    # --- torch import. Time the C++ binary does not pay at all. -------------
    t0 = time.perf_counter()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    t_import = (time.perf_counter() - t0) * 1000.0
    print(f"{'import torch + transformers':<34}{t_import:10.1f}{ms():15.1f}{rss_gb():10.2f}")

    dtype = {"f32": torch.float32, "f16": torch.float16,
             "bf16": torch.bfloat16}[dtype_arg]

    # --- tokenizer ----------------------------------------------------------
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang)
    t_tok = (time.perf_counter() - t0) * 1000.0
    print(f"{'AutoTokenizer.from_pretrained':<34}{t_tok:10.1f}{ms():15.1f}{rss_gb():10.2f}")

    # --- weights ------------------------------------------------------------
    t0 = time.perf_counter()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=dtype)
    model.to("cpu")
    model.eval()
    t_model = (time.perf_counter() - t0) * 1000.0
    print(f"{'from_pretrained + to(cpu) + eval':<34}{t_model:10.1f}{ms():15.1f}{rss_gb():10.2f}")

    dev = next(model.parameters()).device.type
    assert dev == "cpu", f"model is on {dev}, not cpu - the numbers are not comparable"

    # named_parameters() de-duplicates shared storage, so a tied lm_head is
    # counted ONCE - which is what the GGUF stores too, so the two are
    # comparable. NLLB ties model.shared / encoder / decoder / lm_head.
    params = sum(p.numel() * p.element_size() for _, p in model.named_parameters())
    bufs   = sum(b.numel() * b.element_size() for _, b in model.named_buffers())

    t_load_total = ms()
    print("-" * len(hdr))
    print(f"{'LOAD TOTAL':<34}{t_load_total:10.1f}")
    print(f"{'  of which torch import':<34}{t_import:10.1f}"
          f"   <- the C++ binary pays none of this")
    print(f"{'  weights + tokenizer only':<34}{t_load_total - t_import:10.1f}"
          f"   <- compare THIS with nllb-cli's `load:` line")
    print(f"\nparameters (unique storage): {params / 1e9:.3f} GB   "
          f"buffers: {bufs / 1e6:.1f} MB")

    # --- generate -----------------------------------------------------------
    with open(in_file, encoding="utf-8") as f:
        text = f.read().strip()
    enc = tok(text, return_tensors="pt")
    bos = tok.convert_tokens_to_ids(tgt_lang)

    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model.generate(**enc, forced_bos_token_id=bos,
                             num_beams=1, do_sample=False, max_new_tokens=256)
        t_gen = (time.perf_counter() - t0) * 1000.0

    n_new = int(out.shape[1]) - 1
    print(f"\n{tok.batch_decode(out, skip_special_tokens=True)[0]}\n")
    print(f"src tokens: {int(enc['input_ids'].shape[1])}   new tokens: {n_new}")
    print(f"generate:   {t_gen:.0f} ms  ({t_gen / max(n_new, 1):.1f} ms/token)")
    print(f"PEAK RSS:   {rss_gb():.2f} GB   "
          f"(torch+python floor was {0.0:.2f} GB - see the table's first row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())