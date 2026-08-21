#!/usr/bin/env python3
"""The milestone-1 gate: does the GGUF encoder reproduce HuggingFace?"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ref_encoder import GGUFModel, encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tol", type=float, default=2e-5)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    hf = M2M100ForConditionalGeneration.from_pretrained(
        args.hf_dir, torch_dtype=torch.float32)
    hf.eval()
    enc, vocab = hf.model.encoder, hf.config.vocab_size
    m = GGUFModel(args.gguf)
    print(f"gguf: {len(m.t)} tensors  hf: d_model={hf.config.d_model} "
          f"layers={hf.config.encoder_layers}\n")

    rng = np.random.default_rng(1234)
    worst, failures = 0.0, 0

    for n, T in enumerate([1, 2, 7, 17]):
        ids = rng.integers(4, vocab, size=T)
        ids[-1] = hf.config.eos_token_id
        t = torch.tensor(ids[None, :], dtype=torch.long)
        with torch.no_grad():
            ref = enc(input_ids=t, attention_mask=torch.ones_like(t)
                      ).last_hidden_state[0].numpy()
        got = encode(m, ids, pad_id=hf.config.pad_token_id,
                     trace=args.trace and n == 0)
        d = np.abs(ref - got)
        den = np.linalg.norm(ref) * np.linalg.norm(got)
        cos = float((ref * got).sum() / den) if den else 1.0
        worst = max(worst, d.max())
        ok = d.max() <= args.tol
        failures += not ok
        print(f"  T={T:<3d} max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  "
              f"cos={cos:.10f}  {'ok' if ok else 'FAIL'}")

    T = 12
    ids = rng.integers(4, vocab, size=T)
    ids[-4:] = hf.config.pad_token_id
    am = (ids != hf.config.pad_token_id).astype(np.int64)
    with torch.no_grad():
        ref = enc(input_ids=torch.tensor(ids[None, :], dtype=torch.long),
                  attention_mask=torch.tensor(am[None, :], dtype=torch.long)
                  ).last_hidden_state[0].numpy()
    got = encode(m, ids, pad_id=hf.config.pad_token_id)
    mx = np.abs(ref - got)[am == 1].max()
    worst = max(worst, mx)
    ok = mx <= args.tol
    failures += not ok
    print(f"  padded  max|d|={mx:.3e}  (non-pad positions)  {'ok' if ok else 'FAIL'}")

    print(f"\nworst max|d| = {worst:.3e}  tol = {args.tol:.1e}")
    if failures:
        print("MISMATCH - the GGUF does not reproduce HuggingFace")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
