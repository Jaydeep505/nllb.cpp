#!/usr/bin/env python3
"""
Dump HF encoder hidden states as golden files the C++ side can diff against
offline (no python, no torch, at test time).

Run this once on a machine that can reach huggingface.co:

    python tools/dump_reference.py facebook/nllb-200-distilled-600M \
        --out ref/ --lang eng_Latn

Produces, per prompt:
    ref/<slug>.ids.bin      int32[T]        token ids  (source lang code first, </s> last)
    ref/<slug>.enc.bin      float32[T,1024] encoder last_hidden_state, row-major
    ref/<slug>.json         shapes, ids, checksums
and ref/manifest.json tying them together.

The .bin files are raw little-endian - fread them straight into a buffer.
Also dumps per-layer states with --per-layer, which is what you actually want
the first time the final output is wrong: it tells you WHICH layer diverged
instead of just that something did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

PROMPTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "",                                        # degenerate: lang code + </s> only
    "a",                                       # single token
    "Translation quality depends on the encoder producing exactly the right "
    "hidden states, so this sentence is deliberately long enough to span "
    "several attention patterns and exercise the position table.",
    "Zwölf Boxkämpfer jagen Eva quer über den großen Sylter Deich.",  # non-ASCII
    "日本語のテキストもトークナイザを通します。",                        # CJK
]


def slug(s: str, i: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", s.lower())[:32].strip("_")
    return f"{i:02d}_{base or 'empty'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="facebook/nllb-200-distilled-600M", nargs="?")
    ap.add_argument("--out", type=Path, default=Path("ref"))
    ap.add_argument("--lang", default="eng_Latn", help="src_lang for the tokenizer")
    ap.add_argument("--per-layer", action="store_true",
                    help="also dump every encoder layer's output (bisect tool)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, src_lang=args.lang)
    model = M2M100ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()

    print(f"d_model={model.config.d_model} layers={model.config.encoder_layers} "
          f"heads={model.config.encoder_attention_heads} "
          f"vocab={model.config.vocab_size}")
    print(f"src_lang={args.lang} -> id {tok.convert_tokens_to_ids(args.lang)}")

    manifest = {
        "model": args.model,
        "src_lang": args.lang,
        "d_model": model.config.d_model,
        "encoder_layers": model.config.encoder_layers,
        "encoder_attention_heads": model.config.encoder_attention_heads,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
        "cases": [],
    }

    for i, text in enumerate(PROMPTS):
        name = slug(text, i)
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"]

        with torch.no_grad():
            out = model.model.encoder(
                input_ids=ids,
                attention_mask=enc["attention_mask"],
                output_hidden_states=args.per_layer,
            )
        h = out.last_hidden_state[0].numpy().astype(np.float32)
        idn = ids[0].numpy().astype(np.int32)

        (args.out / f"{name}.ids.bin").write_bytes(idn.tobytes())
        (args.out / f"{name}.enc.bin").write_bytes(np.ascontiguousarray(h).tobytes())

        case = {
            "name": name,
            "text": text,
            "ids": idn.tolist(),
            "tokens": tok.convert_ids_to_tokens(idn.tolist()),
            "shape": list(h.shape),
            "enc_sha256": hashlib.sha256(h.tobytes()).hexdigest()[:16],
            "enc_mean": float(h.mean()),
            "enc_std": float(h.std()),
            "enc_absmax": float(np.abs(h).max()),
        }

        if args.per_layer:
            hs = [x[0].numpy().astype(np.float32) for x in out.hidden_states]
            np.savez(args.out / f"{name}.layers.npz",
                     **{f"layer_{j:02d}": a for j, a in enumerate(hs)})
            case["layers"] = len(hs)

        with open(args.out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)
        manifest["cases"].append(case)
        print(f"  {name:38s} T={h.shape[0]:<3d} std={h.std():.4f}")

    with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(PROMPTS)} cases to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
