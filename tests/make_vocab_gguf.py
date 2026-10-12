#!/usr/bin/env python3
"""Write a VOCAB-ONLY GGUF from an HF snapshot (or a bare tokenizer.json).

The tokenizer gate has nothing to do with the 2.46 GB of weights, and making
people convert the full checkpoint before they can run it is how a gate stops
being run. This writes the same `tokenizer.ggml.*` keys the real converter
writes - through the very same load_vocab(), so there is one code path and not
two - plus one dummy tensor, in about a second.

    python tests/make_vocab_gguf.py models/nllb-600M models/nllb-vocab.gguf
    python tests/make_vocab_gguf.py path/to/tokenizer.json models/nllb-vocab.gguf

What it CANNOT do is tell you anything about the model: `nllb-test-tokenizer`
takes a vocab-only file happily, every other binary in this repo will refuse it
for want of tensors. That is the intended division.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))   # gen_grapheme_tables, for the converter

import numpy as np                                             # noqa: E402

try:
    import gguf  # noqa: F401
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")

# models/convert-nllb-to-gguf.py has dashes in its name (whisper.cpp's
# convention for models/convert-*.py), so it is loaded by path, not imported.
import importlib.util                                          # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "convert_nllb_to_gguf", ROOT / "models" / "convert-nllb-to-gguf.py")
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)
ARCH, grapheme_tables, load_vocab = _conv.ARCH, _conv.grapheme_tables, _conv.load_vocab


BREAKAGES = {
    "charsmap": "omit tokenizer.ggml.precompiled_charsmap",
    "hash":     "write a wrong tokenizer.nllb.merges_hash",
    "shift":    "rotate the vocabulary by one, the classic fairseq-offset bug",
    "added":    "omit the AddedVocabulary tokens",
    "single-word": "flag a language code single_word, which is unimplemented",
    "tables":   "omit the grapheme tables, forcing the compiled-in fallback",
    "gcb":      "corrupt the grapheme tables by dropping every Extend range",
}


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--break")]
    brk = None
    for a in sys.argv[1:]:
        if a.startswith("--break="):
            brk = a.split("=", 1)[1]
            if brk not in BREAKAGES:
                sys.exit("--break must be one of: " +
                         ", ".join(f"{k} ({v})" for k, v in BREAKAGES.items()))
    if len(argv) != 2:
        sys.exit(__doc__)
    src, outfile = Path(argv[0]), Path(argv[1])

    tmp = None
    if src.is_file():                       # a bare tokenizer.json
        tmp = Path(tempfile.mkdtemp())
        shutil.copy(src, tmp / "tokenizer.json")
        model_dir = tmp
    else:
        model_dir = src

    v = load_vocab(model_dir)
    if v is None:
        sys.exit(f"no tokenizer files under {model_dir}")
    tokens, scores, toktypes, meta = v

    # Deliberate damage, for tests/corrupt_tokenizer.py and for anyone who
    # wants to see the loader refuse rather than take it on trust.
    if brk == "shift":
        tokens = ["[SHIFTED]"] + tokens[:-1]
        toktypes = [toktypes[0]] + toktypes[:-1]
    if brk == "charsmap":
        meta["charsmap"] = b""
    if brk == "hash":
        meta["merges_hash"] ^= 1
    if brk == "added":
        meta["added_tokens"], meta["added_flags"] = [], []
    if brk == "single-word":
        meta["added_flags"] = [f | 1 for f in meta["added_flags"]]
    if brk:
        print(f"BREAKING: {brk} - {BREAKAGES[brk]}")

    w = GGUFWriter(str(outfile), ARCH)
    w.add_name("nllb-vocab-only")
    w.add_tokenizer_model("spm")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)
    w.add_bos_token_id(0)
    w.add_eos_token_id(2)
    w.add_pad_token_id(1)
    w.add_unk_token_id(3)
    w.add_add_bos_token(False)
    w.add_add_eos_token(True)
    w.add_string(f"tokenizer.{ARCH}.vocab_source", meta["source"])
    if meta.get("charsmap"):
        w.add_precompiled_charsmap(meta["charsmap"])
    w.add_array(f"tokenizer.{ARCH}.added_tokens", meta["added_tokens"])
    w.add_array(f"tokenizer.{ARCH}.added_flags", meta["added_flags"])
    gt = None if brk == "tables" else grapheme_tables()
    if gt:
        gcb, ext, ucd = gt
        if brk == "gcb":
            # Drop every Extend range. Combining marks stop joining their base,
            # so "A U+0300" becomes two clusters and normalizes differently.
            gcb = [v for i in range(0, len(gcb), 3)
                   for v in gcb[i:i + 3] if gcb[i + 2] != 4]
            print(f"BREAKING: gcb - dropped the Extend ranges "
                  f"({len(gcb) // 3} left)")
        w.add_array(f"tokenizer.{ARCH}.gcb_ranges", gcb)
        w.add_array(f"tokenizer.{ARCH}.extpict_ranges", ext)
        w.add_string(f"tokenizer.{ARCH}.ucd_version", ucd)
    if meta.get("merges_list") is not None:
        w.add_array(f"tokenizer.{ARCH}.merges", meta["merges_list"])
    w.add_uint64(f"tokenizer.{ARCH}.merges_hash", meta["merges_hash"])
    w.add_uint32(f"tokenizer.{ARCH}.n_merges", meta["n_merges"])
    w.add_string(f"tokenizer.{ARCH}.merges_hash_source",
                 meta.get("merges_hash_source", "tokenizer.json"))

    # Marks the file as deliberately weightless, so a loader that wants tensors
    # can say "this is a vocab-only GGUF" instead of "tensor not found".
    w.add_bool(f"{ARCH}.vocab_only", True)
    w.add_tensor("vocab_only.marker", np.zeros((1,), dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {outfile}  ({outfile.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(tokens)} tokens, {meta['n_merges']} merges, "
          f"{len(meta.get('charsmap') or b'')} charsmap bytes")

    if tmp:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
