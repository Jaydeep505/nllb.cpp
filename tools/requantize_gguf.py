#!/usr/bin/env python3
"""
requantize_gguf.py - read a GGUF, rewrite it with selected tensors quantised.

Step 10, section 7.0: the MEASUREMENT tool. It exists so the question "what
does Q8_0 do to the per-decision headroom" can be answered before any C++ is
written, using gguf-py's quantiser (which can do Q8_0 / Q4_0 / Q5_0 / BF16 and
NOT the K-quants - see trap 33).

The C++ nllb-quantize replaces this. Two things here transfer to it verbatim:

  * EVERY metadata key is copied by iterating the reader, never by enumerating
    keys by hand.  Since step 8 the GGUF carries the tokenizer - charsmap,
    merge hash, added tokens, UAX#29 grapheme tables - and nllb_vocab_load
    (hence nllb-cli) REFUSES a file missing any of them.  See trap 32.
  * The key set of the output is asserted equal to the input's, minus nothing,
    plus only general.quantization_version / general.file_type.  An assertion,
    not a comment.

Usage
-----
  # the step 7.0 file: token_embd at Q8_0, everything else left alone
  python requantize_gguf.py nllb-600M-f32.gguf nllb-600M-q8emb.gguf --quantize token_embd.weight --type q8_0

  # every 2-D weight, norms and biases left as they are
  python requantize_gguf.py nllb-600M-f32.gguf nllb-600M-q8all.gguf --quantize '*.weight' --type q8_0 --min-rank 2

  # what would happen, without writing 1 GB
  python requantize_gguf.py in.gguf out.gguf --quantize '*' --type q8_0 --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf import quants
from gguf.constants import GGML_QUANT_SIZES, Keys

# gguf-py can only QUANTISE these. Q2_K..Q6_K and the IQ family raise
# NotImplementedError on the quantise path while dequantising fine, which is
# exactly how you fail to notice. Trap 33.
WRITABLE = {
    "f32":  GGMLQuantizationType.F32,
    "f16":  GGMLQuantizationType.F16,
    "bf16": GGMLQuantizationType.BF16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q4_0": GGMLQuantizationType.Q4_0,
}

# keys the writer produces itself; copying them would duplicate them
WRITER_OWNED = {
    Keys.General.ARCHITECTURE,
    Keys.General.QUANTIZATION_VERSION,
    Keys.General.FILE_TYPE,
    "general.alignment",
}

# NOT metadata. GGUFReader synthesises these three into .fields from the file
# HEADER, so a loop that copies every field writes them out as real keys and
# the next reader chokes on "Duplicate GGUF.version". Caught by the round-trip
# check below, which is the argument for that check existing.
READER_SYNTHETIC = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}

SKIP_KEYS = WRITER_OWNED | READER_SYNTHETIC


# The dominant target type decides general.file_type, the way llama.cpp's
# MOSTLY_* names do: a mix is not a format, so the key is a label rather than a
# specification and the tensor types in the file are the truth. ALL_F32 when
# nothing was quantised at all.
_FTYPE_FOR = {
    GGMLQuantizationType.F32:  "ALL_F32",
    GGMLQuantizationType.F16:  "MOSTLY_F16",
    GGMLQuantizationType.BF16: "MOSTLY_BF16",
    GGMLQuantizationType.Q8_0: "MOSTLY_Q8_0",
    GGMLQuantizationType.Q5_1: "MOSTLY_Q5_1",
    GGMLQuantizationType.Q5_0: "MOSTLY_Q5_0",
    GGMLQuantizationType.Q4_1: "MOSTLY_Q4_1",
    GGMLQuantizationType.Q4_0: "MOSTLY_Q4_0",
}


def _file_type_for(plan):
    from collections import Counter
    from gguf import LlamaFileType
    counts = Counter(to for _, _, to, _ in plan)
    if not counts:
        return LlamaFileType.ALL_F32
    dominant, _ = counts.most_common(1)[0]
    name = _FTYPE_FOR.get(dominant)
    return getattr(LlamaFileType, name, LlamaFileType.MOSTLY_Q8_0) if name \
        else LlamaFileType.MOSTLY_Q8_0


def dequant_to_f32(t) -> np.ndarray:
    """Reader tensor -> f32 rows, shape (nrows, n_per_row)."""
    if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                         GGMLQuantizationType.BF16):
        a = t.data
        if t.tensor_type == GGMLQuantizationType.BF16:
            a = a.view(np.uint16).astype(np.uint32) << 16
            a = a.view(np.float32)
        return np.asarray(a, dtype=np.float32).reshape(-1, t.shape[0])
    return quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1, t.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", type=Path)
    ap.add_argument("outfile", type=Path)
    ap.add_argument("--type", default="q8_0", choices=sorted(WRITABLE),
                    help="target type for matched tensors (default q8_0)")
    ap.add_argument("--quantize", action="append", default=[], metavar="GLOB",
                    help="tensor-name glob to quantise; repeatable")
    ap.add_argument("--keep", action="append", default=[], metavar="GLOB",
                    help="never quantise these, even if --quantize matches; repeatable")
    ap.add_argument("--min-rank", type=int, default=2,
                    help="skip tensors with fewer dims (default 2: never touch "
                         "norms or biases)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the sizes, write nothing")
    args = ap.parse_args()

    if not args.quantize:
        ap.error("nothing selected: pass at least one --quantize GLOB")

    target = WRITABLE[args.type]
    blck, _ = GGML_QUANT_SIZES[target]

    reader = GGUFReader(args.infile, "r")

    arch_f = reader.get_field(Keys.General.ARCHITECTURE)
    arch = arch_f.contents() if arch_f else "llama"

    # ---- plan -------------------------------------------------------------
    plan = []
    for t in reader.tensors:
        name = str(t.name)
        n_per_row = int(t.shape[0])
        rank = int(sum(1 for d in t.shape if d > 1)) if t.shape.size else 0
        rank = int(np.count_nonzero(np.array(t.shape) != 1)) or 1

        why = None
        if not any(fnmatch.fnmatch(name, g) for g in args.quantize):
            why = "not selected"
        elif any(fnmatch.fnmatch(name, g) for g in args.keep):
            why = "--keep"
        elif rank < args.min_rank:
            why = f"rank {rank} < {args.min_rank}"
        elif n_per_row % blck != 0:
            # trap 34: legal for NLLB (every row is 1024 or 4096) but never assume
            why = f"row {n_per_row} not a multiple of {blck}"
        elif t.tensor_type == target:
            why = "already"

        plan.append((t, name, target if why is None else t.tensor_type, why))

    # ---- report -----------------------------------------------------------
    w = max(len(n) for _, n, _, _ in plan)
    in_bytes = out_bytes = 0
    n_q = 0
    print(f"{'tensor':<{w}}  {'shape':>18}  {'from':>6} -> {'to':<6}  {'MB':>8}  note")
    print("-" * (w + 60))
    for t, name, to, why in plan:
        nelem = int(np.prod(t.shape))
        fb, fs = GGML_QUANT_SIZES[t.tensor_type]
        tb, ts = GGML_QUANT_SIZES[to]
        ib = nelem // fb * fs
        ob = nelem // tb * ts
        in_bytes += ib
        out_bytes += ob
        if to != t.tensor_type:
            n_q += 1
        shape = "x".join(str(int(d)) for d in reversed(t.shape) if d > 1) or "1"
        note = "" if why is None else why
        if why in ("not selected",):
            note = ""
        print(f"{name:<{w}}  {shape:>18}  {t.tensor_type.name:>6} -> "
              f"{to.name:<6}  {ob/1e6:8.1f}  {note}")

    print("-" * (w + 60))
    print(f"{n_q} tensor(s) quantised to {target.name}")
    print(f"tensor bytes: {in_bytes/1e9:.3f} GB -> {out_bytes/1e9:.3f} GB "
          f"({in_bytes/out_bytes:.2f}x smaller)")

    if args.dry_run:
        return 0

    # ---- write ------------------------------------------------------------
    writer = GGUFWriter(args.outfile, arch, use_temp_file=False)

    copied = []
    for key, field in reader.fields.items():
        if key in SKIP_KEYS:
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub)
        copied.append(key)
    print(f"\ncopied {len(copied)} metadata key(s) verbatim")

    # THE ASSERTION (trap 32). Not a comment: a check.
    expected = set(reader.fields) - SKIP_KEYS
    got = set(copied)
    missing = expected - got
    extra = got - expected
    assert not missing, f"metadata keys LOST: {sorted(missing)}"
    assert not extra, f"metadata keys invented: {sorted(extra)}"

    from gguf.constants import GGML_QUANT_VERSION
    writer.add_quantization_version(GGML_QUANT_VERSION)

    # general.file_type. It is in WRITER_OWNED - i.e. excluded from the copy
    # loop because the writer is supposed to produce it - and for a long time
    # NOBODY TOLD THE WRITER TO. The output files simply had no file_type, and
    # the trap-32 assertion above could not see it, because it checks
    # `set(reader.fields) - SKIP_KEYS` and FILE_TYPE is inside SKIP_KEYS.
    #
    # Nothing in src/nllb.cpp reads the key, so no file ever broke - which is
    # exactly why it survived. Found by diffing the key sets of an input and an
    # output by hand while writing tools/nllb-quantize.cpp. The lesson is the
    # one trap 32 already states and this code did not obey: a key you exclude
    # from a copy because something else owns it is a key you must then check
    # that something else actually wrote.
    writer.add_file_type(_file_type_for(plan))

    # reader.shape is ne order (n_per_row first); reader.data is already numpy
    # order (nrows, n_per_row), or (nrows, bytes_per_row) for a quantised
    # source. add_tensor infers the shape from the array, so pass neither.
    for t, name, to, _ in plan:
        if to == t.tensor_type:
            writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
            continue
        rows = dequant_to_f32(t)
        q = quants.quantize(rows, to)
        rt = quants.dequantize(q, to).astype(np.float32)
        rms = float(np.sqrt(np.mean((rows - rt) ** 2)))
        scale = float(np.sqrt(np.mean(rows ** 2)))
        writer.add_tensor(name, q, raw_dtype=to)
        print(f"  quantised {name}  rms={rms:.4e}  rms/rms(w)={rms / scale:.4e}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    size = args.outfile.stat().st_size
    print(f"\nwrote {args.outfile}  {size/1e9:.3f} GB")

    # ---- verify the round trip --------------------------------------------
    back = GGUFReader(args.outfile, "r")
    drop = {Keys.General.QUANTIZATION_VERSION, Keys.General.FILE_TYPE} | READER_SYNTHETIC
    bk = set(back.fields) - drop
    ik = set(reader.fields) - drop
    assert bk == ik, (f"key set changed on write: lost {sorted(ik - bk)}, "
                      f"gained {sorted(bk - ik)}")
    # names surviving is not the claim - the VALUES are. A charsmap that comes
    # back one byte short still has its key.
    for key in sorted(ik):
        a, b = reader.get_field(key), back.get_field(key)
        assert a.types == b.types, f"{key}: type changed {a.types} -> {b.types}"
        va, vb = a.contents(), b.contents()
        if isinstance(va, list):
            assert len(va) == len(vb), f"{key}: length {len(va)} -> {len(vb)}"
            assert va == vb, f"{key}: values changed"
        else:
            assert va == vb, f"{key}: {va!r} -> {vb!r}"

    assert len(back.tensors) == len(reader.tensors), "tensor count changed"
    for a, b in zip(reader.tensors, back.tensors):
        assert str(a.name) == str(b.name), f"tensor order changed: {a.name} != {b.name}"
        assert tuple(a.shape) == tuple(b.shape), f"{a.name}: shape changed"
        if b.tensor_type == a.tensor_type:
            assert np.array_equal(a.data, b.data), f"{a.name}: unquantised data moved"

    print(f"[verify] {len(bk)} metadata keys survive with identical types AND "
          f"values; {len(back.tensors)} tensors keep name, order and shape; "
          f"every untouched tensor is byte-identical   ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
