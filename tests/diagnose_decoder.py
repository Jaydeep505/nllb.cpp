#!/usr/bin/env python3
"""Is a decoder mismatch a real bug, or the float32 noise floor at NLLB's
activation magnitudes?

    python tests/diagnose_decoder.py models/nllb-600M models/nllb-600M-f32.gguf

Prints, per hidden state, the magnitude alongside the error. The number that
matters is `rel` and `ulps`, not `max|d|`:

  * NLLB is pre-layernorm, so the residual stream is NEVER normalised between
    blocks. Values run into the thousands on the real 600M. An absolute
    tolerance calibrated on the encoder's post-layernorm output (O(1)) is
    meaningless against them.
  * `ulps` is max|d| divided by the float32 ULP at that magnitude - the
    smallest difference representable at all. 1-2 ULP is bit noise; there is no
    closer two f32 computations can be without being identical.

Reading it:

    rel ~ 1e-7, ulps ~ 1-4    float noise. Not a bug, whatever max|d| says.
    rel ~ 1e-3+, ulps huge    a real bug. Bisect from the first bad state.

A real bug also destroys the logits and flips the argmax. Clean logits plus
broken-looking hidden states is the signature of a tolerance problem, not a
graph problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ref_encoder import GGUFModel, encode      # noqa: E402
from ref_decoder import decode                 # noqa: E402


def ulp(x: float) -> float:
    """Size of one float32 ULP at magnitude x."""
    if x == 0.0:
        return float(np.finfo(np.float32).tiny)
    return float(np.spacing(np.float32(abs(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--ts", type=int, default=11)
    ap.add_argument("--td", type=int, default=7)
    ap.add_argument("--seed", type=int, default=4321)
    args = ap.parse_args()

    hf = M2M100ForConditionalGeneration.from_pretrained(args.hf_dir, dtype=torch.float32)
    hf.eval()
    cfg = hf.config
    m = GGUFModel(args.gguf)

    rng = np.random.default_rng(args.seed)
    src = rng.integers(4, cfg.vocab_size, size=args.ts); src[-1] = cfg.eos_token_id
    dec = rng.integers(4, cfg.vocab_size, size=args.td); dec[0] = cfg.decoder_start_token_id

    src_t = torch.tensor(src[None, :], dtype=torch.long)
    dec_t = torch.tensor(dec[None, :], dtype=torch.long)
    with torch.no_grad():
        out = hf(input_ids=src_t, attention_mask=torch.ones_like(src_t),
                 decoder_input_ids=dec_t, output_hidden_states=True, use_cache=False)
    ref_hs = [h[0].numpy() for h in out.decoder_hidden_states]
    ref_logits = out.logits[0].numpy()

    enc_out = encode(m, src, pad_id=cfg.pad_token_id)
    got_hs, got_logits = decode(m, enc_out, dec, src_ids=src, pad_id=cfg.pad_token_id)

    print(f"\nTs={args.ts} Td={args.td}  d_model={cfg.d_model} "
          f"dec_layers={cfg.decoder_layers}\n")
    print(f"  {'state':<22s} {'max|ref|':>10s} {'max|d|':>10s} "
          f"{'rel':>10s} {'ulps':>7s}")
    print("  " + "-" * 63)

    worst_rel, worst_ulps = 0.0, 0.0
    for i, (a, b) in enumerate(zip(ref_hs, got_hs)):
        mag = float(np.abs(a).max())
        d = float(np.abs(a - b).max())
        rel = d / mag if mag else 0.0
        u = d / ulp(mag) if mag else 0.0
        worst_rel, worst_ulps = max(worst_rel, rel), max(worst_ulps, u)
        label = ("embeddings" if i == 0
                 else "after dec.output_norm" if i == len(ref_hs) - 1
                 else f"output of block {i - 1}")
        print(f"  {i:<2d} {label:<19s} {mag:10.3f} {d:10.3e} {rel:10.3e} {u:7.1f}")

    mag = float(np.abs(ref_logits).max())
    d = float(np.abs(ref_logits - got_logits).max())
    print("  " + "-" * 63)
    print(f"  {'logits':<22s} {mag:10.3f} {d:10.3e} {d / mag:10.3e} "
          f"{d / ulp(mag):7.1f}")
    same = bool((ref_logits.argmax(-1) == got_logits.argmax(-1)).all())
    print(f"  argmax: {'same' if same else 'DIFFERS'}")

    print(f"\nworst relative error {worst_rel:.3e}, worst {worst_ulps:.1f} ULP")
    if worst_rel < 1e-5 and same:
        print("VERDICT: float noise at NLLB's activation magnitudes, not a bug.\n"
              "         The absolute tolerance is the thing that is wrong.")
    else:
        print("VERDICT: this looks like a real divergence. Bisect from the "
              "first state whose rel is large.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
