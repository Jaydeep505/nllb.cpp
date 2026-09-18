#!/usr/bin/env python3
"""NLLB decoder forward pass in pure numpy, reading ONLY the GGUF file.

The executable spec for the C++ port, exactly as tools/ref_encoder.py is for
the encoder: each line maps to one ggml op. Teacher-forced, no KV cache - the
cache is a separate, separately-verifiable change and mixing the two makes the
first failing diff ambiguous.

    python tools/ref_decoder.py models/nllb-600M-f32.gguf \
        --src-ids 256047 1 2 --dec-ids 2 256042 --trace

Decoder block, in order (pre-layernorm throughout, relu FFN):

    r = x;  x = LN(x, attn_norm);        x = self_attn(x, causal);   x = r + x
    r = x;  x = LN(x, cross_attn_norm);  x = cross_attn(x, enc_out); x = r + x
    r = x;  x = LN(x, ffn_norm);         x = fc2(relu(fc1(x)));      x = r + x

then one closing LN(x, dec.output_norm), then the tied lm_head.

Three things that differ from the encoder and are easy to get wrong:

1. Self-attention is CAUSAL. Position i may not see j > i. Cross-attention is
   NOT - it sees the whole source, masked only by source padding.
2. Cross-attention K and V come from the ENCODER output, not from x. They do
   not depend on the decoder step at all, which is exactly why they are worth
   caching once per sequence later.
3. The lm_head is TIED to token_embd.weight and has no bias. The GGUF only
   carries `output.weight` when the checkpoint actually untied them.

Position ids follow the same rule as the encoder - cumsum(mask)*mask + pad_id,
against a table built with offset 2 - with `past` added for incremental
decoding. With no cache, `past` is 0 and token j reads row j + 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ref_encoder import GGUFModel, encode, layer_norm, softmax


def positions_for(ids, pad_id, past=0):
    """HF create_position_ids_from_input_ids, with past_key_values_length."""
    mask = (ids != pad_id).astype(np.int64)
    pos = (np.cumsum(mask) + past) * mask + pad_id
    return np.where(mask == 1, pos, pad_id)


def decode(m, enc_out, dec_ids, src_ids=None, pad_id=None, past=0, trace=False):
    """enc_out: [Ts, d_model] from ref_encoder.encode. dec_ids: [Td] int ids.

    Returns (hidden_states, logits) where hidden_states is a list of
    n_layer + 1 arrays matching HF's decoder_hidden_states exactly:
    entry i (< n_layer) is the INPUT to block i, entry n_layer is the output
    after dec.output_norm.
    """
    a = "nllb"
    d_model = int(m.k(f"{a}.embedding_length"))
    n_head  = int(m.k(f"{a}.decoder.attention.head_count"))
    n_layer = int(m.k(f"{a}.decoder.block_count"))
    eps     = float(m.k(f"{a}.attention.layer_norm_epsilon", 1e-5))
    scale   = float(m.k(f"{a}.embedding_scale", np.sqrt(d_model)))
    if pad_id is None:
        pad_id = int(m.k("tokenizer.ggml.padding_token_id", 1))

    head_dim = d_model // n_head
    Td, Ts = len(dec_ids), enc_out.shape[0]
    attn_scale = head_dim ** -0.5

    # Causal mask over the decoder, additive. Built once, shared by every block.
    causal = np.triu(np.full((Td, Td), -np.inf, dtype=np.float32), k=1)

    # Source padding mask for cross-attention. Nothing to mask when the source
    # is a single unpadded sequence, which is every golden case today - so this
    # branch is untested until padded batches land. Written now so the shape of
    # the fix is obvious later.
    cross_bias = None
    if src_ids is not None:
        smask = (np.asarray(src_ids) != pad_id).astype(np.int64)
        if smask.sum() != Ts:
            cross_bias = np.where(smask[None, :] == 1, 0.0, -np.inf).astype(np.float32)

    def heads(t, T):
        return t.reshape(T, n_head, head_dim).transpose(1, 0, 2)

    x = m["token_embd.weight"][dec_ids] * scale
    x = x + m["position_embd.weight"][positions_for(np.asarray(dec_ids), pad_id, past)]
    hs = []
    if trace:
        print(f"  dec embed:    mean={x.mean():+.6f} std={x.std():.6f}")

    for i in range(n_layer):
        p = f"dec.blk.{i}."
        hs.append(x.copy())

        # ---- causal self-attention ----
        r = x
        h = layer_norm(x, m[p + "attn_norm.weight"], m[p + "attn_norm.bias"], eps)
        q = (h @ m[p + "attn_q.weight"].T + m[p + "attn_q.bias"]) * attn_scale
        k =  h @ m[p + "attn_k.weight"].T + m[p + "attn_k.bias"]
        v =  h @ m[p + "attn_v.weight"].T + m[p + "attn_v.bias"]
        q, k, v = heads(q, Td), heads(k, Td), heads(v, Td)
        att = softmax(q @ k.transpose(0, 2, 1) + causal[None], axis=-1) @ v
        att = att.transpose(1, 0, 2).reshape(Td, d_model)
        x = r + (att @ m[p + "attn_out.weight"].T + m[p + "attn_out.bias"])

        # ---- cross-attention against the frozen encoder output ----
        r = x
        h = layer_norm(x, m[p + "cross_attn_norm.weight"],
                       m[p + "cross_attn_norm.bias"], eps)
        q = (h @ m[p + "cross_attn_q.weight"].T
             + m[p + "cross_attn_q.bias"]) * attn_scale
        k = enc_out @ m[p + "cross_attn_k.weight"].T + m[p + "cross_attn_k.bias"]
        v = enc_out @ m[p + "cross_attn_v.weight"].T + m[p + "cross_attn_v.bias"]
        q, k, v = heads(q, Td), heads(k, Ts), heads(v, Ts)
        logits = q @ k.transpose(0, 2, 1)
        if cross_bias is not None:
            logits = logits + cross_bias[None]
        att = softmax(logits, axis=-1) @ v
        att = att.transpose(1, 0, 2).reshape(Td, d_model)
        x = r + (att @ m[p + "cross_attn_out.weight"].T
                 + m[p + "cross_attn_out.bias"])

        # ---- feed forward ----
        r = x
        h = layer_norm(x, m[p + "ffn_norm.weight"], m[p + "ffn_norm.bias"], eps)
        h = np.maximum(h @ m[p + "ffn_up.weight"].T + m[p + "ffn_up.bias"], 0.0)
        x = r + (h @ m[p + "ffn_down.weight"].T + m[p + "ffn_down.bias"])
        if trace:
            print(f"  after blk {i:2d}: mean={x.mean():+.6f} std={x.std():.6f}")

    x = layer_norm(x, m["dec.output_norm.weight"], m["dec.output_norm.bias"], eps)
    hs.append(x.copy())
    if trace:
        print(f"  after norm:   mean={x.mean():+.6f} std={x.std():.6f}")

    # lm_head: tied to the input embedding unless the checkpoint untied it.
    w_out = m["output.weight"] if m.has("output.weight") else m["token_embd.weight"]
    return hs, x @ w_out.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--src-ids", type=int, nargs="+", required=True)
    ap.add_argument("--dec-ids", type=int, nargs="+", required=True)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    m = GGUFModel(args.gguf)
    print(f"{len(m.t)} tensors, arch={m.k('general.architecture')}")
    src = np.array(args.src_ids, dtype=np.int64)
    enc_out = encode(m, src, trace=args.trace)
    hs, logits = decode(m, enc_out, np.array(args.dec_ids, dtype=np.int64),
                        src_ids=src, trace=args.trace)
    print(f"\ndecoder states {hs[-1].shape}, logits {logits.shape}")
    print(hs[-1][:, :6])
    print(f"argmax at last step: {int(logits[-1].argmax())}  "
          f"max logit {logits[-1].max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
