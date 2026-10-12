#!/usr/bin/env python3
"""Write a random GGUF with the *shape* of a real NLLB checkpoint, no torch.

`tests/make_tiny_model.py` needs torch and transformers and builds a small
model, which is the right tool for correctness work. It is the wrong tool for a
performance measurement: on the 600M the lm_head alone is 256206 x 1024, which
is 1.05 GB read per step and dominates everything the cached decoder does. A
2000-word vocabulary reproduces none of that, so a profile taken on the tiny
fixture would answer a different question.

This writes the weights straight through gguf.GGUFWriter with numpy, so it
needs neither torch nor a download, and defaults to
facebook/nllb-200-distilled-600M's exact geometry.

    python tests/make_shape_model.py models/shape-600M-f32.gguf

The numbers in it are random. It is for timing and for the cache gate - which
compares two C++ paths and so cancels the weights out entirely - and NOT for
anything that reads what it emits.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import gguf

ARCH = "nllb"


def sinusoidal(num_positions: int, dim: int, padding_idx: int) -> np.ndarray:
    # Same as models/convert-nllb-to-gguf.py: (half - 1) denominator, sin|cos
    # concatenated, padding row zeroed.
    half = dim // 2
    step = math.log(10000.0) / (half - 1)
    inv = np.exp(np.arange(half, dtype=np.float64) * -step)
    pos = np.arange(num_positions, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([np.sin(pos), np.cos(pos)], axis=1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros((num_positions, 1))], axis=1)
    emb[padding_idx, :] = 0.0
    return emb.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outfile", type=Path)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--enc-layers", type=int, default=12)
    ap.add_argument("--dec-layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=256206)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outtype", choices=("f32", "f16"), default="f32",
                    help="f16 mirrors models/convert-nllb-to-gguf.py: norms, biases "
                         "and the position table stay f32, everything else halves")
    a = ap.parse_args()

    D, F, V, H = a.d_model, a.ffn, a.vocab, a.heads
    rng = np.random.default_rng(a.seed)
    a.outfile.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(a.outfile), ARCH)

    # Same rule as models/convert-nllb-to-gguf.py's should_keep_f32: norms,
    # biases and the position table are tiny and are exactly where precision
    # loss compounds, so they stay f32 even in f16 mode.
    def cast(name: str, arr: np.ndarray) -> np.ndarray:
        keep = (name.endswith(".bias") or "_norm." in name
                or name.startswith("position_embd"))
        return arr if (a.outtype == "f32" or keep) else arr.astype(np.float16)

    _add = w.add_tensor
    w.add_tensor = lambda name, arr, **kw: _add(name, cast(name, arr), **kw)

    w.add_name("shape-only random NLLB")
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if a.outtype == "f32"
                    else gguf.LlamaFileType.MOSTLY_F16)
    w.add_context_length(a.n_ctx)
    w.add_embedding_length(D)
    w.add_vocab_size(V)
    w.add_uint32(f"{ARCH}.encoder.block_count", a.enc_layers)
    w.add_uint32(f"{ARCH}.encoder.attention.head_count", H)
    w.add_uint32(f"{ARCH}.encoder.feed_forward_length", F)
    w.add_uint32(f"{ARCH}.decoder.block_count", a.dec_layers)
    w.add_uint32(f"{ARCH}.decoder.attention.head_count", H)
    w.add_uint32(f"{ARCH}.decoder.feed_forward_length", F)
    w.add_decoder_start_token_id(2)
    w.add_layer_norm_eps(1e-5)
    w.add_float32(f"{ARCH}.embedding_scale", math.sqrt(D))
    w.add_uint32(f"{ARCH}.position_offset", 2)
    w.add_string(f"{ARCH}.activation", "relu")
    w.add_bool(f"{ARCH}.pre_layernorm", True)
    w.add_bool(f"{ARCH}.attention.qkv_bias", True)
    w.add_bool(f"{ARCH}.encoder_only", False)
    w.add_bos_token_id(0)
    w.add_eos_token_id(2)
    w.add_pad_token_id(1)

    def lin(out_dim: int, in_dim: int, name: str):
        # torch stores (out, in); gguf reverses on write, giving ggml
        # ne = [in, out], which is what ggml_mul_mat wants.
        std = 1.0 / math.sqrt(in_dim)
        w.add_tensor(name, rng.normal(0.0, std, (out_dim, in_dim)).astype(np.float32))
        w.add_tensor(name.replace(".weight", ".bias"),
                     np.zeros(out_dim, dtype=np.float32))

    def norm(name: str):
        w.add_tensor(name + ".weight", np.ones(D, dtype=np.float32))
        w.add_tensor(name + ".bias", np.zeros(D, dtype=np.float32))

    w.add_tensor("token_embd.weight",
                 rng.normal(0.0, 0.02, (V, D)).astype(np.float32))
    w.add_tensor("position_embd.weight", sinusoidal(a.n_ctx + 2, D, 1))

    for i in range(a.enc_layers):
        p = f"enc.blk.{i}."
        norm(p + "attn_norm")
        for t in ("attn_q", "attn_k", "attn_v", "attn_out"):
            lin(D, D, p + t + ".weight")
        norm(p + "ffn_norm")
        lin(F, D, p + "ffn_up.weight")
        lin(D, F, p + "ffn_down.weight")
    norm("enc.output_norm")

    for i in range(a.dec_layers):
        p = f"dec.blk.{i}."
        norm(p + "attn_norm")
        for t in ("attn_q", "attn_k", "attn_v", "attn_out"):
            lin(D, D, p + t + ".weight")
        norm(p + "cross_attn_norm")
        for t in ("cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out"):
            lin(D, D, p + t + ".weight")
        norm(p + "ffn_norm")
        lin(F, D, p + "ffn_up.weight")
        lin(D, F, p + "ffn_down.weight")
    norm("dec.output_norm")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.outfile} "
          f"({a.outfile.stat().st_size / (1 << 30):.2f} GiB) "
          f"d_model={D} enc={a.enc_layers} dec={a.dec_layers} heads={H} "
          f"ffn={F} vocab={V} outtype={a.outtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
