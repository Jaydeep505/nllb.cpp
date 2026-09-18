#!/usr/bin/env python3
"""Tiny randomly-initialised NLLB in HF snapshot layout. No download.

    python tests/make_tiny_model.py models/tiny-nllb
    python tests/make_tiny_model.py models/deep-nllb --layers 12 --heads 16 \
        --d-model 256 --ffn 512 --vocab 2000

The default is deliberately minimal - fast enough to sit in the edit/build/test
loop. The knobs exist because a 3-layer / 4-head model does not exercise
everything the real 600M does: 12 layers accumulate float error the way the real
one will, 16 heads catch reshape/permute mistakes that 4 heads let slide, and
--untied is the only way to reach the `output.weight` path, since NLLB itself
ties the lm_head and that branch would otherwise never run.

Positional args stay backwards compatible: <outdir> [safetensors|bin].
"""

import argparse
from pathlib import Path

import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="models/tiny-nllb", type=Path)
ap.add_argument("fmt", nargs="?", default="safetensors",
                choices=["safetensors", "bin"])
ap.add_argument("--layers", type=int, default=3, help="encoder layers")
ap.add_argument("--dec-layers", type=int, default=None,
                help="decoder layers (defaults to --layers, or 2 in the default preset)")
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--ffn", type=int, default=128)
ap.add_argument("--vocab", type=int, default=512)
ap.add_argument("--max-pos", type=int, default=64)
ap.add_argument("--untied", action="store_true",
                help="untie the lm_head so the GGUF carries output.weight")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dec_layers = args.dec_layers if args.dec_layers is not None else (
    2 if args.layers == 3 else args.layers)

cfg = M2M100Config(
    vocab_size=args.vocab, d_model=args.d_model,
    encoder_layers=args.layers, decoder_layers=dec_layers,
    encoder_attention_heads=args.heads, decoder_attention_heads=args.heads,
    encoder_ffn_dim=args.ffn, decoder_ffn_dim=args.ffn,
    max_position_embeddings=args.max_pos,
    activation_function="relu", scale_embedding=True,
    tie_word_embeddings=not args.untied,
    dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
    encoder_layerdrop=0.0, decoder_layerdrop=0.0,
    pad_token_id=1, bos_token_id=0, eos_token_id=2)

torch.manual_seed(args.seed)
model = M2M100ForConditionalGeneration(cfg)
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2: p.normal_(0.0, 0.05)
        elif "layer_norm" in name and name.endswith("weight"): p.normal_(1.0, 0.05)
        else: p.normal_(0.0, 0.02)
    if args.untied:
        # tie_word_embeddings=False unties FOUR things in transformers, not one:
        # model.shared, encoder.embed_tokens, decoder.embed_tokens and lm_head
        # all become independent parameters, and the loop above just gave each a
        # different random value. Real NLLB never looks like that - the two
        # towers always share an embedding - and a model where they differ is
        # not convertible to this GGUF schema at all. So re-tie the three
        # embeddings by value and untie only the lm_head, which is the case we
        # actually want to cover.
        shared = model.model.shared.weight
        model.model.encoder.embed_tokens.weight.copy_(shared)
        model.model.decoder.embed_tokens.weight.copy_(shared)
        # Distinct from the embedding, or the converter drops output.weight as
        # redundant and the untied path stays untested.
        model.lm_head.weight.normal_(0.0, 0.05)
model.eval()
args.out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=(args.fmt == "safetensors"))
print(f"wrote {args.out} ({args.fmt}): d_model={args.d_model} "
      f"enc={args.layers} dec={dec_layers} heads={args.heads} "
      f"vocab={args.vocab} lm_head={'untied' if args.untied else 'tied'}")
