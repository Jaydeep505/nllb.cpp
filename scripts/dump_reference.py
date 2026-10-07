#!/usr/bin/env python3
"""
Dump HF encoder hidden states as golden files the C++ side diffs against.

    python scripts/dump_reference.py models/nllb-600M --out ref/ --lang eng_Latn --per-layer

Per case, all raw little-endian so C++ can fread them straight into a buffer:
    ref/<name>.ids.bin          int32[T]
    ref/<name>.enc.bin          float32[T, d_model]   final encoder output
    ref/<name>.layer_NN.bin     float32[T, d_model]   with --per-layer
    ref/<name>.json             ids, tokens, shapes, checksums (for humans)
    ref/manifest.txt            "<name> <T> <d_model> <n_layers>" per line
    ref/manifest.json           the same, richer

IMPORTANT - what layer_NN actually contains. HF returns n_layers+1 hidden
states and the indexing is not what you would guess:

    layer_00            embeddings (scale + positions), i.e. INPUT to block 0
    layer_01 .. NN-1    input to block i, i.e. output of block i-1
    layer_NN            output of the last block AFTER the final encoder
                        layer_norm  (== enc.bin)

So compare your C++ block-i output against layer_{i+1} for i in 0..n_layers-2,
and your finished output against layer_NN. There is no dump of the last
block's raw pre-norm output. Getting this off by one will make you think
block 0 is broken when it is fine.

--random-ids skips the tokenizer entirely (deterministic pseudo-random ids),
which is how you generate goldens for a model that has no tokenizer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration

# (name, text) - names are explicit so non-ASCII prompts do not collapse to
# the same slug. Learned that one the hard way.
PROMPTS = [
    ("00_hello_world",   "Hello, world!"),
    ("01_quick_brown",   "The quick brown fox jumps over the lazy dog."),
    ("02_empty",         ""),
    ("03_single_char",   "a"),
    ("04_long",          "Translation quality depends on the encoder producing "
                         "exactly the right hidden states, so this sentence is "
                         "deliberately long enough to span several attention "
                         "patterns and exercise the position table."),
    ("05_german",        "Zwölf Boxkämpfer jagen Eva quer über den großen Sylter Deich."),
    ("06_japanese",      "日本語のテキストもトークナイザを通します。"),
    ("07_arabic",        "الترجمة الآلية تعتمد على جودة المُرمِّز."),
    ("08_repeated",      "test test test test test test test test"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="facebook/nllb-200-distilled-600M",
                    help="local snapshot dir, or a hub id")
    ap.add_argument("--out", type=Path, default=Path("ref"))
    ap.add_argument("--lang", default="eng_Latn")
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--random-ids", action="store_true",
                    help="skip the tokenizer; use deterministic pseudo-random ids")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = M2M100ForConditionalGeneration.from_pretrained(args.model,
                                                           dtype=torch.float32)
    model.eval()
    cfg = model.config
    print(f"d_model={cfg.d_model} layers={cfg.encoder_layers} "
          f"heads={cfg.encoder_attention_heads} vocab={cfg.vocab_size}")

    tok = None
    if not args.random_ids:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, src_lang=args.lang)
        print(f"src_lang={args.lang} -> id {tok.convert_tokens_to_ids(args.lang)}")

    rng = np.random.default_rng(20240819)
    manifest = {
        "model": str(args.model),
        "src_lang": args.lang if tok else None,
        "d_model": cfg.d_model,
        "encoder_layers": cfg.encoder_layers,
        "encoder_attention_heads": cfg.encoder_attention_heads,
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_token_id,
        "per_layer": bool(args.per_layer),
        "cases": [],
    }
    lines = []

    for name, text in PROMPTS:
        if tok is not None:
            enc = tok(text, return_tensors="pt")
            ids_t, am_t = enc["input_ids"], enc["attention_mask"]
        else:
            T = max(1, len(text.split()) + 2)
            ids = rng.integers(4, cfg.vocab_size, size=T).astype(np.int64)
            ids[-1] = cfg.eos_token_id
            ids_t = torch.tensor(ids[None, :], dtype=torch.long)
            am_t = torch.ones_like(ids_t)

        with torch.no_grad():
            out = model.model.encoder(input_ids=ids_t, attention_mask=am_t,
                                      output_hidden_states=args.per_layer)

        h = np.ascontiguousarray(out.last_hidden_state[0].numpy().astype(np.float32))
        idn = np.ascontiguousarray(ids_t[0].numpy().astype(np.int32))

        (args.out / f"{name}.ids.bin").write_bytes(idn.tobytes())
        (args.out / f"{name}.enc.bin").write_bytes(h.tobytes())

        case = {
            "name": name, "text": text, "ids": idn.tolist(),
            "tokens": tok.convert_ids_to_tokens(idn.tolist()) if tok else None,
            "shape": list(h.shape),
            "enc_sha256": hashlib.sha256(h.tobytes()).hexdigest()[:16],
            "enc_mean": float(h.mean()), "enc_std": float(h.std()),
            "enc_absmax": float(np.abs(h).max()),
        }

        n_layers_dumped = 0
        if args.per_layer:
            for j, hs in enumerate(out.hidden_states):
                a = np.ascontiguousarray(hs[0].numpy().astype(np.float32))
                (args.out / f"{name}.layer_{j:02d}.bin").write_bytes(a.tobytes())
            n_layers_dumped = len(out.hidden_states)
            case["n_hidden_states"] = n_layers_dumped

        with open(args.out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)
        manifest["cases"].append(case)
        lines.append(f"{name} {h.shape[0]} {h.shape[1]} {n_layers_dumped}")
        print(f"  {name:20s} T={h.shape[0]:<3d} std={h.std():.4f}"
              f"{'  +' + str(n_layers_dumped) + ' layers' if n_layers_dumped else ''}")

    (args.out / "manifest.txt").write_text("\n".join(lines) + "\n")
    with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(PROMPTS)} cases to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
