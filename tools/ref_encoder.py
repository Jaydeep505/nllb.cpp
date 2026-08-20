#!/usr/bin/env python3
"""
NLLB encoder forward pass in pure numpy, reading ONLY the GGUF file.

Two jobs:
  1. Proves the GGUF is self-sufficient - if this runs, every number the C++
     encoder needs is in the file and named correctly.
  2. Is the executable spec for the C++ port. Every line here maps to a ggml op.
     When ggml disagrees with HF, bisect against THIS, not against transformers.

    python tools/ref_encoder.py models/nllb-f32.gguf --ids 256047 1234 5678 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader


class GGUFModel:
    def __init__(self, path: str | Path):
        self.r = GGUFReader(str(path))
        self.t: dict[str, np.ndarray] = {}
        for t in self.r.tensors:
            # ReaderTensor.shape is ggml ne (fastest dim first). numpy wants the
            # reverse. For a torch Linear this restores (out_features, in_features).
            ne = [int(x) for x in t.shape]
            arr = np.asarray(t.data)
            self.t[t.name] = arr.reshape(tuple(reversed(ne))).astype(np.float32)

        self.kv: dict[str, object] = {}
        for key, field in self.r.fields.items():
            try:
                self.kv[key] = field.contents()
            except Exception:
                self.kv[key] = None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.t[name]

    def has(self, name: str) -> bool:
        return name in self.t

    def k(self, key: str, default=None):
        v = self.kv.get(key, None)
        return default if v is None else v


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    # torch LayerNorm uses the biased (population) variance
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def encode(m: GGUFModel, ids: np.ndarray, pad_id: int | None = None,
           trace: bool = False) -> np.ndarray:
    """ids: (T,) int array for a single sequence. Returns (T, d_model) float32."""
    arch = "nllb"
    d_model = int(m.k(f"{arch}.embedding_length"))
    n_head  = int(m.k(f"{arch}.encoder.attention.head_count"))
    n_layer = int(m.k(f"{arch}.encoder.block_count"))
    eps     = float(m.k(f"{arch}.attention.layer_norm_epsilon", 1e-5))
    scale   = float(m.k(f"{arch}.embedding_scale", np.sqrt(d_model)))
    pos_off = int(m.k(f"{arch}.position_offset", 2))
    if pad_id is None:
        pad_id = int(m.k("tokenizer.ggml.padding_token_id", 1))

    head_dim = d_model // n_head
    attn_scale = head_dim ** -0.5
    T = len(ids)

    # --- embeddings ------------------------------------------------------
    # HF: inputs_embeds = embed_tokens(ids) * embed_scale; then += positions.
    # The scale applies to the TOKEN embedding only.
    x = m["token_embd.weight"][ids] * scale

    # HF position ids: cumsum(mask)*mask + padding_idx, padding_idx == 1,
    # table offset == 2. For an unpadded sequence that is simply j + 2.
    mask = (ids != pad_id).astype(np.int64)
    positions = np.cumsum(mask) * mask + pad_id
    positions = np.where(mask == 1, positions, pad_id)
    assert pos_off == 2, "table was baked with a different offset"
    x = x + m["position_embd.weight"][positions]

    if trace:
        print(f"  after embed:  mean={x.mean():+.6f} std={x.std():.6f}")

    # --- encoder blocks (pre-layernorm) ----------------------------------
    for i in range(n_layer):
        p = f"enc.blk.{i}."

        residual = x
        h = layer_norm(x, m[p + "attn_norm.weight"], m[p + "attn_norm.bias"], eps)

        # HF scales Q, not the logits. Keep it here so intermediate tensors
        # line up when you dump and diff them.
        q = (h @ m[p + "attn_q.weight"].T + m[p + "attn_q.bias"]) * attn_scale
        k =  h @ m[p + "attn_k.weight"].T + m[p + "attn_k.bias"]
        v =  h @ m[p + "attn_v.weight"].T + m[p + "attn_v.bias"]

        q = q.reshape(T, n_head, head_dim).transpose(1, 0, 2)   # (H, T, D)
        k = k.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        v = v.reshape(T, n_head, head_dim).transpose(1, 0, 2)

        logits = q @ k.transpose(0, 2, 1)                        # (H, T, T)
        if mask.sum() != T:                                      # padding present
            logits = logits + np.where(mask[None, None, :] == 1, 0.0, -np.inf)
        a = softmax(logits, axis=-1) @ v                         # (H, T, D)

        a = a.transpose(1, 0, 2).reshape(T, d_model)
        a = a @ m[p + "attn_out.weight"].T + m[p + "attn_out.bias"]
        x = residual + a

        residual = x
        h = layer_norm(x, m[p + "ffn_norm.weight"], m[p + "ffn_norm.bias"], eps)
        h = np.maximum(h @ m[p + "ffn_up.weight"].T + m[p + "ffn_up.bias"], 0.0)  # relu
        h = h @ m[p + "ffn_down.weight"].T + m[p + "ffn_down.bias"]
        x = residual + h

        if trace:
            print(f"  after blk {i:2d}: mean={x.mean():+.6f} std={x.std():.6f}")

    # --- final norm ------------------------------------------------------
    x = layer_norm(x, m["enc.output_norm.weight"], m["enc.output_norm.bias"], eps)
    if trace:
        print(f"  after norm:   mean={x.mean():+.6f} std={x.std():.6f}")
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    m = GGUFModel(args.gguf)
    print(f"{len(m.t)} tensors, arch={m.k('general.architecture')}")
    out = encode(m, np.array(args.ids, dtype=np.int64), trace=args.trace)
    print(f"\nhidden states {out.shape}")
    print(out[:, :6])
    if args.save:
        np.save(args.save, out)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
