#!/usr/bin/env python3
"""
bench_cuda_vs_hf.py - time ONE sentence on CUDA: HuggingFace transformers vs nllb.cpp.

    python tools/bench_cuda_vs_hf.py \
        --hf models/nllb-600M \
        --gguf models/nllb-600M-f16.gguf \
        --cli build/bin/nllb-cli \
        --src eng_Latn --tgt deu_Latn \
        --text "The quick brown fox jumps over the lazy dog." \
        --runs 20 --warmup 3 --hf-dtype f16,f32

WHAT IS TIMED, so both sides measure the same thing:

    per run   tokenize  ->  generate (encoder + greedy decode, KV cache on)  ->  detokenize
              HF:   tok(...) -> model.generate(...) -> batch_decode, each bracketed
                    by torch.cuda.synchronize()
              cpp:  the `timing:` line nllb-cli -v prints for each stdin line

    load      HF:   from_pretrained + .to("cuda") (torch import reported separately)
              cpp:  the `load:` line (nllb_model_load + vocab, includes upload to GPU)

Both sides: greedy (num_beams=1, do_sample=False), same max_new_tokens, KV cache on,
batch size 1. The first --warmup runs are discarded (CUDA context, cuBLAS handles,
kernel autotune / graph allocation all land there).

nllb.cpp is run ONCE with the sentence repeated on stdin (warmup + runs lines), so the
model is loaded once and every line after the first reuses the warm device - the
same situation as the HF loop. Each line still gets its own encoder pass and a
fresh KV cache (see CLI.md), so no work is shared between runs.

Needs: torch with CUDA, transformers, sentencepiece; nllb-cli built with
-DNLLB_CUDA=ON. Either side can be skipped with --skip-hf / --skip-cpp.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time


# --------------------------------------------------------------------------- stats

def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    p90 = xs[min(n - 1, int(round(0.9 * (n - 1))))]
    return {
        "n": n,
        "median": st.median(xs),
        "mean": st.fmean(xs),
        "min": xs[0],
        "p90": p90,
        "stdev": st.stdev(xs) if n > 1 else 0.0,
    }


# --------------------------------------------------------------------------- HF

def bench_hf(args, dtype_name: str) -> dict:
    t0 = time.perf_counter()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    t_import = (time.perf_counter() - t0) * 1e3

    if not torch.cuda.is_available():
        sys.exit("torch.cuda.is_available() is False - this script compares CUDA runs.")

    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    dev = torch.device(args.device)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.hf, src_lang=args.src)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.hf, torch_dtype=dtype)
    model.to(dev).eval()
    torch.cuda.synchronize(dev)
    t_load = (time.perf_counter() - t0) * 1e3

    p = next(model.parameters())
    assert p.device.type == "cuda" and p.dtype == dtype, (p.device, p.dtype)

    tgt_id = tok.convert_tokens_to_ids(args.tgt)
    gen_kw = dict(forced_bos_token_id=tgt_id, num_beams=1, do_sample=False,
                  max_new_tokens=args.max_new, use_cache=True)

    t_tok, t_gen, t_detok, t_total, n_new = [], [], [], [], None
    text_out = None
    torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        for i in range(args.warmup + args.runs):
            torch.cuda.synchronize(dev)
            a = time.perf_counter()
            enc = tok(args.text, return_tensors="pt").to(dev)
            torch.cuda.synchronize(dev)
            b = time.perf_counter()
            out = model.generate(**enc, **gen_kw)
            torch.cuda.synchronize(dev)
            c = time.perf_counter()
            text_out = tok.batch_decode(out, skip_special_tokens=True)[0]
            d = time.perf_counter()

            if i >= args.warmup:
                t_tok.append((b - a) * 1e3)
                t_gen.append((c - b) * 1e3)
                t_detok.append((d - c) * 1e3)
                t_total.append((d - a) * 1e3)
            n_new = int(out.shape[1]) - 1     # same convention as nllb-cli: minus dec start

    return {
        "name": f"HF transformers ({dtype_name}, {torch.cuda.get_device_name(dev)})",
        "import_ms": t_import,
        "load_ms": t_load,
        "peak_vram_mb": torch.cuda.max_memory_allocated(dev) / 2**20,
        "src_tokens": int(enc["input_ids"].shape[1]),
        "new_tokens": n_new,
        "output": text_out,
        "tokenize": summarize(t_tok),
        "generate": summarize(t_gen),
        "detokenize": summarize(t_detok),
        "total": summarize(t_total),
    }


# --------------------------------------------------------------------------- nllb.cpp

RE_TIMING = re.compile(r"timing: tokenize ([\d.]+) ms \| generate ([\d.]+) ms .*?\| detokenize ([\d.]+) ms")
RE_STOP = re.compile(r"stop: (.+?) \| (\d+) new tokens \| cache (\w+)")
RE_LOAD = re.compile(r"load:\s+([\d.]+) ms")
RE_BACKEND = re.compile(r"backend: (.+)")
RE_SRC = re.compile(r"src ids \((\d+)\)")


def bench_cpp(args) -> dict:
    if "\n" in args.text or "\r" in args.text:
        sys.exit("--text must be a single line (nllb-cli reads stdin line by line).")
    if not os.path.isfile(args.cli):
        sys.exit(f"nllb-cli not found at {args.cli} (build with -DNLLB_CUDA=ON, or pass --cli)")

    n = args.warmup + args.runs
    cmd = [args.cli, "-m", args.gguf, "--src", args.src, "--tgt", args.tgt,
           "-b", args.backend, "-n", str(args.max_new), "-t", str(args.threads), "-v"]
    if args.kv_f16:
        cmd.append("--kv-f16")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, input=(args.text + "\n") * n, capture_output=True,
                          text=True, encoding="utf-8")
    wall = (time.perf_counter() - t0) * 1e3

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        sys.exit(f"nllb-cli exited with {proc.returncode}: {' '.join(cmd)}")

    err = proc.stderr
    timings = [tuple(map(float, m)) for m in RE_TIMING.findall(err)]
    stops = RE_STOP.findall(err)
    outs = proc.stdout.splitlines()
    if len(timings) != n or len(outs) != n:
        sys.stderr.write(err[-4000:])
        sys.exit(f"expected {n} timing lines / outputs, got {len(timings)} / {len(outs)}")

    backend = (RE_BACKEND.search(err) or [None, "?"])[1].strip()
    if args.backend.lower().startswith("cuda") and "cuda" not in backend.lower():
        sys.exit(f"nllb-cli is not running on CUDA (backend: {backend})")
    if len(set(outs)) != 1:
        print("WARNING: nllb.cpp produced different outputs across runs:", set(outs))

    keep = timings[args.warmup:]
    load = RE_LOAD.search(err)
    src = RE_SRC.search(err)
    return {
        "name": f"nllb.cpp ({os.path.basename(args.gguf)}, backend {backend})",
        "load_ms": float(load[1]) if load else float("nan"),
        "process_wall_ms": wall,
        "src_tokens": int(src[1]) if src else None,
        "new_tokens": int(stops[-1][1]) if stops else None,
        "cache": stops[-1][2] if stops else "?",
        "output": outs[-1],
        "tokenize": summarize([t[0] for t in keep]),
        "generate": summarize([t[1] for t in keep]),
        "detokenize": summarize([t[2] for t in keep]),
        "total": summarize([sum(t) for t in keep]),
    }


# --------------------------------------------------------------------------- report

def report(results: list[dict]) -> None:
    print()
    for r in results:
        print(f"== {r['name']}")
        print(f"   output     : {r['output']}")
        print(f"   tokens     : src {r['src_tokens']}  new {r['new_tokens']}")
        extra = f"  (+ torch import {r['import_ms']:.0f} ms)" if "import_ms" in r else ""
        print(f"   load       : {r['load_ms']:.1f} ms{extra}")
        if "peak_vram_mb" in r:
            print(f"   peak VRAM  : {r['peak_vram_mb']:.0f} MiB (torch allocator)")
        g, t = r["generate"], r["total"]
        per_tok = g["median"] / max(r["new_tokens"] or 1, 1)
        print(f"   generate   : median {g['median']:.2f} ms  mean {g['mean']:.2f}  "
              f"min {g['min']:.2f}  p90 {g['p90']:.2f}  sd {g['stdev']:.2f}  "
              f"-> {per_tok:.2f} ms/token")
        print(f"   end-to-end : median {t['median']:.2f} ms  "
              f"(tokenize {r['tokenize']['median']:.2f} + detok {r['detokenize']['median']:.2f})")
        print()

    # side-by-side table
    hdr = f"{'implementation':<58}{'load ms':>10}{'gen ms':>10}{'ms/tok':>9}{'e2e ms':>10}{'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))
    base = next((r for r in results if r["name"].startswith("HF")), results[0])
    for r in results:
        e2e = r["total"]["median"]
        sp = base["total"]["median"] / e2e if e2e else float("nan")
        per_tok = r["generate"]["median"] / max(r["new_tokens"] or 1, 1)
        print(f"{r['name'][:57]:<58}{r['load_ms']:10.0f}{r['generate']['median']:10.2f}"
              f"{per_tok:9.2f}{e2e:10.2f}{sp:8.2f}x")
    print(f"(speedup = end-to-end median relative to: {base['name']})")

    outs = {r["output"] for r in results}
    if len(outs) > 1:
        print("\nNOTE: translations differ between implementations - with f16 arithmetic a "
              "close greedy decision can flip. Compare ids with nllb-cli -v if it matters.")
    toks = {r["new_tokens"] for r in results}
    if len(toks) > 1:
        print("NOTE: different numbers of generated tokens - compare ms/token, not gen ms.")


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", default="facebook/nllb-200-distilled-600M", help="HF model dir or hub id")
    ap.add_argument("--gguf", default="models/nllb-600M-f16.gguf")
    ap.add_argument("--cli", default="build/bin/nllb-cli")
    ap.add_argument("--src", default="eng_Latn")
    ap.add_argument("--tgt", default="deu_Latn")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, "
                                      "and nobody in the village seemed to notice.")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--hf-dtype", default="f16", help="comma list of f16,f32,bf16")
    ap.add_argument("--device", default="cuda:0", help="torch device for HF")
    ap.add_argument("--backend", default="cuda", help="nllb-cli -b value (cuda, cuda0, ...)")
    ap.add_argument("--threads", type=int, default=4, help="nllb-cli -t (CPU fallback ops only)")
    ap.add_argument("--kv-f16", action="store_true", help="pass --kv-f16 to nllb-cli")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-cpp", action="store_true")
    ap.add_argument("--json", help="also write all numbers to this file")
    args = ap.parse_args()

    if args.runs < 1 or args.warmup < 0:
        ap.error("--runs >= 1 and --warmup >= 0")

    print(f"sentence : {args.text!r}")
    print(f"pair     : {args.src} -> {args.tgt} | greedy | max_new {args.max_new} | "
          f"{args.warmup} warmup + {args.runs} timed runs")

    results = []
    if not args.skip_cpp:
        print("running nllb.cpp ...", flush=True)
        results.append(bench_cpp(args))
    if not args.skip_hf:
        for dt in [d.strip() for d in args.hf_dtype.split(",") if d.strip()]:
            print(f"running HF ({dt}) ...", flush=True)
            results.append(bench_hf(args, dt))
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()

    if not results:
        ap.error("nothing to run")
    report(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
