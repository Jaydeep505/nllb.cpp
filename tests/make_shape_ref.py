#!/usr/bin/env python3
"""A STRUCTURAL ref/ directory for tests/make_shape_model.py. No goldens.

The shape model has the real 600M's geometry and random weights, which is
enough for nllb-test-cache and nllb-test-precision - both compare two C++ paths
against each other, so the weights cancel - and it is what lets step 10's
plumbing be exercised in a sandbox with no checkpoint. What it has never had is
a ref/ directory: every gate reads `manifest_gen.txt` and `<name>.ids.bin` for
its source ids and generation settings, and those come from
scripts/dump_decoder_reference.py, which needs torch and the real model.

This writes that much and nothing more:

    python tests/make_shape_ref.py ref-shape/ --vocab 256206 --n-ctx 1024
    ./build/bin/nllb-test-cache     models/shape-600M-f32.gguf ref-shape/ -t 2
    ./build/bin/nllb-test-precision models/shape-600M-f32.gguf \\
                                    models/shape-600M-f16.gguf ref-shape/ -t 2

WHAT THIS FIXTURE CAN AND CANNOT SUPPORT, because the distinction is the whole
reason it is a separate file with a loud docstring rather than a flag on
make_shape_model.py:

  CAN   the cache gate's `full = exact`, `step`, `chunk`, `grow`, `[arena]` and
        the new `[kv16]` - every one of them is a C++/C++ comparison.
  CAN   nllb-test-precision's [gen], [state], [margin] and [topk] AS PLUMBING:
        that they run, that they report, that they go red when something is
        broken.
  CANNOT any headroom number. Random weights give O(1) activations; the real
        checkpoint's residual stream reaches ~4200 and its f16 headroom is
        1.50x against this fixture's 38.5x. Only the RATIOS transfer, and only
        as lower bounds - trap 29.
  CANNOT anything the tokenizer or the CLI touches. The shape model has no
        vocabulary at all.
  CANNOT anything about what the model SAYS. There are no goldens here. The ids
        below are random and mean nothing; `.gen.bin` is deliberately not
        written, so nllb-test-generate refuses to run rather than passing
        against a target it invented.

That last one is the step-4 SKIP lesson applied in advance: a fixture that
silently satisfies a gate it cannot support is worse than a missing one.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=Path("ref-shape"))
    ap.add_argument("--vocab", type=int, default=256206)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--eos", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 12, 24, 40])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # A forced target code that is a real row of the embedding table and is not
    # a special token. On the real model this is a FLORES id; here it only has
    # to be in range, because trap 10 is about the POSITION it lands at, which
    # the C++ side decides and this file must not.
    forced_bos = a.vocab - 100

    lines = []
    for i, T in enumerate(a.lengths):
        name = f"shape_{i:02d}_len{T}"
        # ids 3.. so pad (1), eos (2) and bos (0) never appear in the source:
        # nllb_decode REJECTS a source containing pad rather than returning
        # plausible numbers, and a stray eos would make the length claim about
        # something other than T.
        ids = rng.integers(3, a.vocab - 200, size=T - 1, dtype=np.int32)
        ids = np.concatenate([ids, np.array([a.eos], dtype=np.int32)])
        (a.out / f"{name}.ids.bin").write_bytes(ids.astype("<i4").tobytes())
        # name  Tg  forced_bos  max_new  dec_start
        # Tg is the golden target length. There is no golden, so it is recorded
        # as the cap - the gates read it for bookkeeping and compare against
        # what they generate, never against a stored sequence.
        lines.append(f"{name} {a.max_new + 1} {forced_bos} {a.max_new} {a.eos}")

    (a.out / "manifest_gen.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}/manifest_gen.txt with {len(lines)} case(s) "
          f"and {len(lines)} .ids.bin files")
    print(f"  vocab={a.vocab} n_ctx={a.n_ctx} forced_bos={forced_bos} "
          f"dec_start={a.eos} max_new={a.max_new}")
    print("  NO GOLDENS. This fixture supports plumbing and C++/C++ "
          "comparisons only - see the docstring.")
    _ = struct  # kept for the reader: the .ids.bin format is little-endian i32
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
