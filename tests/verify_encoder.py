#!/usr/bin/env python3
"""
The milestone-1 gate: does the GGUF encoder reproduce HuggingFace?

Runs HF's M2M100 encoder and the numpy-from-GGUF encoder on identical token ids
and diffs the hidden states. Works unchanged on the tiny synthetic model and on
the real NLLB-200-distilled-600M.

    python tests/verify_encoder.py models/tiny-nllb models/tiny-nllb-f32.gguf

Exits non-zero if the diff exceeds tolerance, so you can wire it into CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ref_encoder import GGUFModel, encode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tol", type=float, default=2e-5,
                    help="max allowed absolute difference")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--trace", action="store_true",
                    help="print per-layer stats on the first sequence")
    args = ap.parse_args()

    hf = M2M100ForConditionalGeneration.from_pretrained(
        args.hf_dir, torch_dtype=torch.float32)
    hf.eval()
    encoder = hf.model.encoder
    vocab = hf.config.vocab_size

    m = GGUFModel(args.gguf)
    print(f"gguf: {len(m.t)} tensors  hf: d_model={hf.config.d_model} "
          f"layers={hf.config.encoder_layers}\n")

    rng = np.random.default_rng(1234)
    lengths = [1, 2, 7, 17, 33][: args.seqs] or [7]
    worst = 0.0
    failures = 0

    for n, T in enumerate(lengths):
        # ids avoid pad(1) so the position arithmetic is the simple j+2 case;
        # a padded case is covered separately below.
        ids = rng.integers(4, vocab, size=T)
        ids[-1] = hf.config.eos_token_id
        ids_t = torch.tensor(ids[None, :], dtype=torch.long)

        with torch.no_grad():
            ref = encoder(input_ids=ids_t,
                          attention_mask=torch.ones_like(ids_t)
                          ).last_hidden_state[0].numpy()

        got = encode(m, ids, pad_id=hf.config.pad_token_id,
                     trace=args.trace and n == 0)

        diff = np.abs(ref - got)
        mx, mean = diff.max(), diff.mean()
        denom = np.linalg.norm(ref) * np.linalg.norm(got)
        cos = float((ref * got).sum() / denom) if denom else 1.0
        worst = max(worst, mx)
        ok = mx <= args.tol
        failures += not ok
        print(f"  T={T:<3d} max|d|={mx:.3e}  mean|d|={mean:.3e}  "
              f"cos={cos:.10f}  {'ok' if ok else 'FAIL'}")

    # padded batch: exercises the mask and the cumsum position formula
    T = 12
    ids = rng.integers(4, vocab, size=T)
    ids[-4:] = hf.config.pad_token_id
    am = (ids != hf.config.pad_token_id).astype(np.int64)
    with torch.no_grad():
        ref = encoder(input_ids=torch.tensor(ids[None, :], dtype=torch.long),
                      attention_mask=torch.tensor(am[None, :], dtype=torch.long)
                      ).last_hidden_state[0].numpy()
    got = encode(m, ids, pad_id=hf.config.pad_token_id)
    mx = np.abs(ref - got)[am == 1].max()   # pad rows are undefined either side
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
