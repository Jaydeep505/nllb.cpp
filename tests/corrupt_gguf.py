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
    --mode quantize    round-trip ONE tensor through a quantisation type and
                       write the result back in the ORIGINAL dtype

Writes in place, so work on a copy.

--mode quantize, AND WHY IT IS NOT THE SAME AS models/convert-nllb-to-gguf.py
-----------------------------------------------------------------
Quantised runs replace exact id equality with two assertions - top-k agreement in
nllb-test-precision and chrF in verify_translation.py - and a gate whose
sensitivity nobody has measured is not a gate. The other harnesses in this
repo can break something on purpose and watch the red appear;  this is that tool
for the quantisation-shaped failure.

models/convert-nllb-to-gguf.py cannot do this job, for two reasons:

  * it changes the FILE's types, so the two files being compared differ in
    dtype as well as in content, and nllb-test-precision's own "both files
    store the lm_head as X" guard starts having opinions;
  * it changes EVERY selected tensor at once, so a red says "quantisation" and
    not "this tensor".

--mode quantize dequantises one tensor to f32, quantises it, dequantises it
again and writes it back AS THE ORIGINAL DTYPE. The file's shape, types and
metadata are untouched; the only thing that moved is the numbers in one tensor,
and they moved by exactly the amount that quantisation would have moved them.
That gives a sensitivity LADDER rather than a single red:

    cp models/nllb-600M-f16.gguf /tmp/probe.gguf
    python tests/corrupt_gguf.py /tmp/probe.gguf --tensor token_embd.weight \
        --mode quantize --qtype q8_0        # small: the gate should stay green
    ./build/bin/nllb-test-precision models/nllb-600M-f32.gguf /tmp/probe.gguf ref/

    cp models/nllb-600M-f16.gguf /tmp/probe.gguf
    python tests/corrupt_gguf.py /tmp/probe.gguf --tensor token_embd.weight \
        --mode quantize --qtype q4_0        # large: the gate MUST go red
    ./build/bin/nllb-test-precision models/nllb-600M-f32.gguf /tmp/probe.gguf ref/ \
        --allow-id-drift

A [topk] gate that stays green at q4_0 on token_embd is not measuring anything,
whatever it prints at q8_0. Note which tensor: token_embd is the
one where the error lands at the decoder INPUT and is amplified through twelve
pre-layernorm blocks, so it is both the most sensitive probe and the one whose
result transfers least to the others. Probe a mid-stack ffn_down as well.

gguf-py can DEQUANTISE every type and can only QUANTISE Q4_0/Q4_1/Q5_0/Q5_1/
Q8_0/BF16, which is exactly the asymmetry that makes this easy to get
wrong; --qtype is restricted to what round-trips and says so on a bad value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf import quants

# What gguf-py can quantise, not what it can read. The dequantise side
# is complete, which is how a missing quantiser goes unnoticed until it raises.
QTYPES = {
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q4_0": GGMLQuantizationType.Q4_0,
    "bf16": GGMLQuantizationType.BF16,
    "f16":  GGMLQuantizationType.F16,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--mode",
                    choices=["transpose", "shift", "scale", "zero", "quantize"],
                    default="transpose")
    ap.add_argument("--factor", type=float, default=1.01)
    ap.add_argument("--qtype", choices=sorted(QTYPES), default="q8_0",
                    help="--mode quantize only: the type to round-trip through")
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

    # Every mode below indexes the tensor as a plain array of numbers, which is
    # only what .data is for the unpacked types. A quantised tensor's .data is
    # blocks of bytes, and reshaping it to (rows, cols) silently produces
    # garbage that still writes back cleanly - the worst kind of failure for a
    # tool whose whole job is to be a known-good corruption.
    if hit.tensor_type not in (GGMLQuantizationType.F32,
                               GGMLQuantizationType.F16,
                               GGMLQuantizationType.BF16):
        raise SystemExit(
            f"{args.tensor} is stored as {hit.tensor_type.name}. This tool "
            f"only edits F32/F16/BF16 tensors - a quantised tensor's .data is "
            f"packed blocks, not numbers. Corrupt the f32 or f16 file and "
            f"requantise it, or use --mode quantize on the unpacked file, "
            f"which is what it is for.")

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
    elif args.mode == "quantize":
        # Round trip through the quantiser and back into the ORIGINAL dtype.
        # The last step is what keeps the file's type map untouched: writing
        # q8_0 blocks here would change the tensor's declared type, which is a
        # different file, not a corrupted one.
        qt = QTYPES[args.qtype]
        n_per_row = a.shape[-1]
        blk = 32 if args.qtype in ("q8_0", "q5_0", "q5_1", "q4_0", "q4_1") else 1
        if n_per_row % blk:
            raise SystemExit(f"{args.tensor} has {n_per_row} elements per row, "
                             f"not a multiple of {args.qtype}'s block size {blk}")
        # IN ROW BLOCKS, and that is not tidiness. token_embd is
        # 256206 x 1024, so the whole-tensor version of this loop holds the
        # f32 copy, the packed blocks and the dequantised result at once -
        # about 2.5 GB - and it was OOM-killed on the first real attempt. The
        # kill was then INVISIBLE, because the calling script piped this
        # program into `head` and `set -e` inspects only the last command of a
        # pipeline: the gate ran against an UNCORRUPTED file and printed PASS.
        # A harness lying toward green - the cheapest fix for the harness is to check the
        # exit code, which is now in scripts/step10-measure.sh's probe section.
        # Block quantisers work per row, so chunking rows changes no byte.
        CHUNK = 4096
        n_per = a.shape[-1]
        flat = a.reshape(-1, n_per)
        n_rows = flat.shape[0]
        max_d = 0.0
        sum_d = 0.0
        n_el = 0
        mag = 0.0
        for i in range(0, n_rows, CHUNK):
            src = np.ascontiguousarray(flat[i:i + CHUNK], dtype=np.float32)
            try:
                packed = quants.quantize(src, qt)
            except NotImplementedError:
                raise SystemExit(f"gguf-py cannot quantise to {args.qtype} - "
                                 f"use the C++ examples/quantize "
                                 f"for K-quants.")
            back = quants.dequantize(packed, qt).reshape(src.shape)
            d = np.abs(back.astype(np.float32) - src)
            max_d = max(max_d, float(d.max()))
            sum_d += float(d.sum())
            n_el += d.size
            mag = max(mag, float(np.abs(src).max()))
            flat[i:i + CHUNK] = back.astype(a.dtype)
        mag = mag or 1.0
        # Printed, because the point of this mode is the SIZE of the error, and
        # a run that reports only "corrupted" leaves the ladder unquantified.
        print(f"  round-tripped through {args.qtype}: max|d|={max_d:.6e} "
              f"mean|d|={sum_d / n_el:.6e} over |w| up to {mag:.4e} "
              f"(relative {max_d / mag:.3e})")
    else:
        a[:] = 0.0

    del r  # flush the memmap
    print(f"corrupted {args.tensor} {a.shape} with --mode {args.mode}"
          f"{' --qtype ' + args.qtype if args.mode == 'quantize' else ''} "
          f"in {args.gguf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
