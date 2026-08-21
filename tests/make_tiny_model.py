#!/usr/bin/env python3
"""Tiny randomly-initialised NLLB in HF snapshot layout. No download."""
import sys
from pathlib import Path
import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/tiny-nllb")
fmt = sys.argv[2] if len(sys.argv) > 2 else "safetensors"

cfg = M2M100Config(
    vocab_size=512, d_model=64, encoder_layers=3, decoder_layers=2,
    encoder_attention_heads=4, decoder_attention_heads=4,
    encoder_ffn_dim=128, decoder_ffn_dim=128, max_position_embeddings=64,
    activation_function="relu", scale_embedding=True,
    dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
    encoder_layerdrop=0.0, decoder_layerdrop=0.0,
    pad_token_id=1, bos_token_id=0, eos_token_id=2)

torch.manual_seed(0)
model = M2M100ForConditionalGeneration(cfg)
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2: p.normal_(0.0, 0.05)
        elif "layer_norm" in name and name.endswith("weight"): p.normal_(1.0, 0.05)
        else: p.normal_(0.0, 0.02)
model.eval()
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=(fmt == "safetensors"))
print(f"wrote {out} ({fmt})")
