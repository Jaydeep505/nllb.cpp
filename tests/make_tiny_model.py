#!/usr/bin/env python3
"""
Build a tiny randomly-initialised NLLB and save it in HF snapshot layout.

Same architecture class (M2M100), same structural choices as the real
600M - pre-layernorm, relu, sinusoidal positions, scaled embeddings, tied
lm_head - just small enough to iterate on in seconds and with no download.

    python tests/make_tiny_model.py models/tiny-nllb
"""

import sys
from pathlib import Path

import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/tiny-nllb")

cfg = M2M100Config(
    vocab_size=512,
    d_model=64,
    encoder_layers=3,
    decoder_layers=2,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
    max_position_embeddings=64,
    activation_function="relu",   # NLLB uses relu, not gelu
    scale_embedding=True,
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    encoder_layerdrop=0.0,
    decoder_layerdrop=0.0,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
)

torch.manual_seed(0)
model = M2M100ForConditionalGeneration(cfg)

# Default init is near-zero-variance for some params; nudge every weight so a
# bug in any single tensor actually shows up in the output diff.
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.05)
        elif "layer_norm" in name and name.endswith("weight"):
            p.normal_(1.0, 0.05)
        else:
            p.normal_(0.0, 0.02)

model.eval()
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
print(f"wrote {out}")
print(f"  d_model={cfg.d_model} enc_layers={cfg.encoder_layers} "
      f"heads={cfg.encoder_attention_heads} vocab={cfg.vocab_size}")
