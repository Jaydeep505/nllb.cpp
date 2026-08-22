#!/usr/bin/env python3
"""Deliberately break one tensor in a GGUF, to prove the tests can find it.

A green test suite only tells you the suite did not fire. This tells you it
fires on the right line - the failure modes below are the ones that actually
happened while porting, and each should be caught by the per-layer bisect and
attributed to the correct block.

    cp models/tiny.gguf /tmp/bad.gguf
    python tests/corrupt_gguf.py /tmp/bad.gguf --tensor dec.blk.1.cross_attn_k.weight --mode transpose
    ./build/bin/nllb-test-decoder /tmp/bad.gguf ref/

    --mode transpose   square weights only: the classic (out,in) vs (in,out) slip
    --mode shift       roll rows by one: the position-table off-by-one
    --mode scale       multiply by 1.01: a subtle drift rather than a blow-up
    --mode zero        zero it: proves the tensor is used at all

Writes in place, so work on a copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--mode", choices=["transpose", "shift", "scale", "zero"],
                    default="transpose")
    ap.add_argument("--factor", type=float, default=1.01)
    ap.add_argument("--list", action="store_true", help="just print tensor names")
    args = ap.parse_args()

    r = GGUFReader(str(args.gguf), "r+")
    if args.list:
        for t in r.tensors:
            print(f"{t.name:44s} {[int(x) for x in t.shape]}")
        return 0

    hit = next((t for t in r.tensors if t.name == args.tensor), None)
    if hit is None:
        raise SystemExit(f"no tensor named {args.tensor!r} "
                         f"(use --list to see them)")

    ne = [int(x) for x in hit.shape]
    a = hit.data.reshape(tuple(reversed(ne)))   # gguf reverses shape on write
    if args.mode == "transpose":
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise SystemExit(f"--mode transpose needs a square 2-D tensor, "
                             f"{args.tensor} is {a.shape}")
        a[:] = a.T.copy()
    elif args.mode == "shift":
        a[:] = np.roll(a, 1, axis=0)
    elif args.mode == "scale":
        a *= args.factor
    else:
        a[:] = 0.0

    del r  # flush the memmap
    print(f"corrupted {args.tensor} {a.shape} with --mode {args.mode} "
          f"in {args.gguf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
