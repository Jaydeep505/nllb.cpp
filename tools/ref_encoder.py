#!/usr/bin/env python3
"""NLLB encoder forward pass in pure numpy, reading ONLY the GGUF file.
This is the executable spec for the C++ port: each line maps to one ggml op."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader


class GGUFModel:
    def __init__(self, path):
        self.r = GGUFReader(str(path))
        self.t: dict[str, np.ndarray] = {}
        for t in self.r.tensors:
            ne = [int(x) for x in t.shape]
            self.t[t.name] = np.asarray(t.data).reshape(
                tuple(reversed(ne))).astype(np.float32)
        self.kv = {}
        for key, field in self.r.fields.items():
            try:
                self.kv[key] = field.contents()
            except Exception:
                self.kv[key] = None

    def __getitem__(self, name): return self.t[name]
    def has(self, name): return name in self.t

    def k(self, key, default=None):
        v = self.kv.get(key, None)
        return default if v is None else v


def layer_norm(x, w, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)   # biased, like torch
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def encode(m, ids, pad_id=None, trace=False):
    a = "nllb"
    d_model = int(m.k(f"{a}.embedding_length"))
    n_head  = int(m.k(f"{a}.encoder.attention.head_count"))
    n_layer = int(m.k(f"{a}.encoder.block_count"))
    eps     = float(m.k(f"{a}.attention.layer_norm_epsilon", 1e-5))
    scale   = float(m.k(f"{a}.embedding_scale", np.sqrt(d_model)))
    if pad_id is None:
        pad_id = int(m.k("tokenizer.ggml.padding_token_id", 1))

    head_dim, T = d_model // n_head, len(ids)
    attn_scale = head_dim ** -0.5

    x = m["token_embd.weight"][ids] * scale
    mask = (ids != pad_id).astype(np.int64)
    positions = np.cumsum(mask) * mask + pad_id
    positions = np.where(mask == 1, positions, pad_id)
    x = x + m["position_embd.weight"][positions]
    if trace:
        print(f"  after embed:  mean={x.mean():+.6f} std={x.std():.6f}")

    for i in range(n_layer):
        p = f"enc.blk.{i}."
        r = x
        h = layer_norm(x, m[p + "attn_norm.weight"], m[p + "attn_norm.bias"], eps)
        q = (h @ m[p + "attn_q.weight"].T + m[p + "attn_q.bias"]) * attn_scale
        k =  h @ m[p + "attn_k.weight"].T + m[p + "attn_k.bias"]
        v =  h @ m[p + "attn_v.weight"].T + m[p + "attn_v.bias"]
        q = q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        k = k.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        v = v.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        logits = q @ k.transpose(0, 2, 1)
        if mask.sum() != T:
            logits = logits + np.where(mask[None, None, :] == 1, 0.0, -np.inf)
        att = softmax(logits, axis=-1) @ v
        att = att.transpose(1, 0, 2).reshape(T, d_model)
        x = r + (att @ m[p + "attn_out.weight"].T + m[p + "attn_out.bias"])

        r = x
        h = layer_norm(x, m[p + "ffn_norm.weight"], m[p + "ffn_norm.bias"], eps)
        h = np.maximum(h @ m[p + "ffn_up.weight"].T + m[p + "ffn_up.bias"], 0.0)
        x = r + (h @ m[p + "ffn_down.weight"].T + m[p + "ffn_down.bias"])
        if trace:
            print(f"  after blk {i:2d}: mean={x.mean():+.6f} std={x.std():.6f}")

    x = layer_norm(x, m["enc.output_norm.weight"], m["enc.output_norm.bias"], eps)
    if trace:
        print(f"  after norm:   mean={x.mean():+.6f} std={x.std():.6f}")
    return x


def main():
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
