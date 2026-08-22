#!/usr/bin/env python3
"""The step-3 gate: does the GGUF decoder reproduce HuggingFace?

Same ladder as verify_encoder.py, one rung up. Run this green BEFORE starting
on the ggml graph - if numpy-from-GGUF already disagrees with torch, the bug is
in the converter or in your reading of the architecture, and no amount of
staring at ggml ops will find it.

    python tests/verify_decoder.py models/nllb-600M models/nllb-600M-f32.gguf

TOLERANCES ARE RELATIVE, and that is not a detail. NLLB is pre-layernorm, so
the residual stream is never normalised between blocks: on the real 600M it
reaches ~4000 by block 4, while the encoder's post-layernorm output - what
verify_encoder.py compares - is O(1). One float32 ULP at magnitude 4000 is
~5e-4, so verify_encoder's absolute 2e-5 threshold is not merely tight here,
it is below the smallest difference float32 can represent. This gate shipped
with that absolute threshold and failed all five cases on the real model at
0.5-3 ULP of agreement. Measure relative, or measure nothing.

Checks, per random case:
  * every hidden state     vs  decoder_hidden_states[i]   (catches which block)
  * final decoder output   vs  the last of those
  * logits                 vs  model.logits              (catches the tied head)
  * argmax agreement       - the only thing greedy decoding actually needs

Run tests/diagnose_decoder.py if anything here goes red: it prints magnitude
and ULPs per state, which is what separates a real divergence from the noise
floor.
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


def rel_err(ref: np.ndarray, got: np.ndarray) -> tuple[float, float, float]:
    """(relative error, absolute error, magnitude of the reference).

    Denominator is floored at 1.0 so that a state which happens to be tiny
    cannot inflate the ratio into a spurious failure.
    """
    mag = float(np.abs(ref).max())
    abs_d = float(np.abs(ref - got).max())
    return abs_d / max(mag, 1.0), abs_d, mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tol", type=float, default=2e-5,
                    help="RELATIVE tolerance on hidden states. Observed worst "
                         "on the real 600M: 1.0e-06. A real bug lands at 1e-2, "
                         "so this sits ~20x above the noise and ~500x below "
                         "anything broken.")
    ap.add_argument("--logit-tol", type=float, default=5e-5,
                    help="RELATIVE tolerance on logits. They are a projection "
                         "onto a 256k-row matrix, so they carry more "
                         "accumulated error than the state they come from. "
                         "Observed worst on the real 600M: 1.6e-06.")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    hf = M2M100ForConditionalGeneration.from_pretrained(
        args.hf_dir, dtype=torch.float32)
    hf.eval()
    cfg = hf.config
    pad, eos = cfg.pad_token_id, cfg.eos_token_id
    dec_start = cfg.decoder_start_token_id
    m = GGUFModel(args.gguf)
    print(f"gguf: {len(m.t)} tensors  hf: d_model={cfg.d_model} "
          f"dec_layers={cfg.decoder_layers} dec_start={dec_start}")
    print(f"tolerances are RELATIVE: states {args.tol:.1e}, "
          f"logits {args.logit_tol:.1e}\n")

    if not m.has("dec.output_norm.weight"):
        print("MISMATCH - this GGUF has no decoder "
              "(converted with --encoder-only?)")
        return 1

    rng = np.random.default_rng(4321)
    worst_h, worst_l, worst_mag, failures, n = 0.0, 0.0, 0.0, 0, 0

    # (Ts, Td). Td=1 is the first greedy step and the tightest causal-mask
    # edge case: a 1x1 softmax must come out as exactly 1.0.
    for Ts, Td in [(4, 1), (4, 3), (11, 7), (23, 19), (2, 2)]:
        src = rng.integers(4, cfg.vocab_size, size=Ts)
        src[-1] = eos
        dec = rng.integers(4, cfg.vocab_size, size=Td)
        dec[0] = dec_start

        src_t = torch.tensor(src[None, :], dtype=torch.long)
        dec_t = torch.tensor(dec[None, :], dtype=torch.long)
        with torch.no_grad():
            out = hf(input_ids=src_t, attention_mask=torch.ones_like(src_t),
                     decoder_input_ids=dec_t, output_hidden_states=True,
                     use_cache=False)
        ref_hs = [h[0].numpy() for h in out.decoder_hidden_states]
        ref_logits = out.logits[0].numpy()

        enc_out = encode(m, src, pad_id=pad)
        got_hs, got_logits = decode(m, enc_out, dec, src_ids=src, pad_id=pad,
                                    trace=args.trace and n == 0)

        if len(got_hs) != len(ref_hs):
            print(f"  Ts={Ts:<3d} Td={Td:<3d} FAIL - {len(got_hs)} hidden states "
                  f"vs HF's {len(ref_hs)}")
            failures += 1
            n += 1
            continue

        # Walk the blocks and name the first one that diverges, rather than
        # reporting only that the end result is wrong.
        first_bad, dh, mag = None, 0.0, 0.0
        for i, (a, b) in enumerate(zip(ref_hs, got_hs)):
            r, _, mg = rel_err(a, b)
            dh, mag = max(dh, r), max(mag, mg)
            if r > args.tol and first_bad is None:
                first_bad = i

        dl, _, lmag = rel_err(ref_logits, got_logits)
        same_argmax = bool((ref_logits.argmax(-1) == got_logits.argmax(-1)).all())
        worst_h, worst_l = max(worst_h, dh), max(worst_l, dl)
        worst_mag = max(worst_mag, mag)
        ok = dh <= args.tol and dl <= args.logit_tol and same_argmax
        failures += not ok
        n += 1

        print(f"  Ts={Ts:<3d} Td={Td:<3d} rel|dh|={dh:.3e}  rel|dlogit|={dl:.3e}  "
              f"|h|max={mag:8.1f}  argmax={'same' if same_argmax else 'DIFFERS'}  "
              f"{'ok' if ok else 'FAIL'}")
        if first_bad is not None:
            what = ("embeddings" if first_bad == 0
                    else "after dec.output_norm" if first_bad == len(ref_hs) - 1
                    else f"output of block {first_bad - 1}")
            print(f"       first divergence at hidden state {first_bad}  ({what})")

    print(f"\n{n} cases, {failures} failed. worst rel|dh| = {worst_h:.3e} "
          f"(tol {args.tol:.1e}), worst rel|dlogit| = {worst_l:.3e} "
          f"(tol {args.logit_tol:.1e})")
    print(f"largest activation seen: {worst_mag:.1f} "
          f"(one float32 ULP there is {np.spacing(np.float32(worst_mag)):.2e})")
    if failures:
        print("MISMATCH - the GGUF decoder does not reproduce HuggingFace")
        print("  run tests/diagnose_decoder.py before touching the graph: it "
              "reports ULPs,\n  which is what tells a real divergence from the "
              "float32 noise floor")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())