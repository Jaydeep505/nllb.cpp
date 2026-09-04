#!/usr/bin/env python3
"""
Dump HuggingFace TRANSLATIONS - text in, text out - as goldens for the CLI.

    python tools/dump_translation_reference.py models/nllb-600M --out ref/ \
        --src-lang eng_Latn --tgt-lang deu_Latn --max-new-tokens 64

Writes ref/translations.txt: a header of `# key<TAB>value` lines, then one
case per line,

    <name><TAB><source text><TAB><translation><TAB><generated ids>

with the three variable fields JSON-encoded so a newline or a tab in a prompt
cannot break the format. tests/verify_translation.py reads it back and needs
no torch.

WHY THIS EXISTS WHEN ref/*.gen.bin ALREADY DOES.
Those are id goldens, and every gate in this repo stops at ids: the tokenizer
gate proves text -> ids, the decoder/generate/cache/precision gates prove
ids -> ids. Nothing has ever compared text to text. The seams that only a
text -> text comparison can see are the target code's POSITION (it belongs at
the first generated position, not in the encoder input), the source code's
presence in the encoder input, Metaspace detokenization (a lost or doubled
leading space), and whether the eos survives into the printed string.

TWO THINGS THIS DUMPER CHECKS ABOUT ITSELF, both recorded in the header:

  * `pipeline(...)` vs `generate()` + `batch_decode(skip_special_tokens=True)`.
    The goldens are the second - the exact operation the CLI performs - and
    the first is compared against it so that a pipeline post-processing quirk
    cannot be attributed to nllb.cpp. `pipeline_agrees` says whether they
    matched on every case.

  * `generation_config.num_beams`. claude/handoff.md flags this as an open
    question: if NLLB ships a default above 1, then every "exact against
    generate()" result in this project is exact against a NON-default
    configuration. This dumper forces num_beams=1 (that is what nllb.cpp
    implements) and records what the checkpoint's own default was, so the
    answer is written down rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

# The SAME nine prompts the encoder goldens use. Imported rather than copied:
# two lists that must agree and are maintained separately eventually do not.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_reference import PROMPTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--out", type=Path, default=Path("ref"))
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--tgt-lang", default="deu_Latn")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="counted from the decoder start token, like HF")
    ap.add_argument("--outfile", default="translations.txt")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, src_lang=args.src_lang)
    model = M2M100ForConditionalGeneration.from_pretrained(args.model,
                                                           dtype=torch.float32)
    model.eval()

    tgt_id = tok.convert_tokens_to_ids(args.tgt_lang)
    src_id = tok.convert_tokens_to_ids(args.src_lang)
    if tgt_id is None or tgt_id == tok.unk_token_id:
        raise SystemExit(f"unknown target language code {args.tgt_lang!r}")
    if src_id is None or src_id == tok.unk_token_id:
        raise SystemExit(f"unknown source language code {args.src_lang!r}")

    # What the checkpoint would do if nobody passed anything. See the note above.
    gc = model.generation_config
    cfg_beams = getattr(gc, "num_beams", None)
    cfg_max_length = getattr(gc, "max_length", None)
    cfg_sample = getattr(gc, "do_sample", None)
    print(f"generation_config: num_beams={cfg_beams} max_length={cfg_max_length} "
          f"do_sample={cfg_sample}")
    if cfg_beams not in (None, 1):
        print(f"  NOTE: this checkpoint's DEFAULT is num_beams={cfg_beams}. These\n"
              f"  goldens force num_beams=1, which is what nllb.cpp implements, so\n"
              f"  they differ from what `pipeline(...)` gives out of the box.",
              file=sys.stderr)

    # The reference operation, matching the CLI line for line: tokenize with
    # src_lang set, generate greedily with the target code forced, decode with
    # skip_special_tokens.
    # device=-1 is NOT optional. `pipeline(...)` picks the best device it can
    # find and moves the model IN PLACE - on an M-series Mac that is mps, and
    # since it is handed the very model object generate() is about to use, the
    # weights leave the CPU while the inputs stay on it:
    #
    #   Device set to use mps:0
    #   RuntimeError: Placeholder storage has not been allocated on MPS device!
    #
    # Even with the tensors moved to match, mps would be the wrong answer here.
    # Every other reference dump in this repo is CPU float32, the C++ side is
    # CPU float32, and mps is a different arithmetic path - a greedy decision
    # whose margin is small enough could land the other way, and these goldens
    # would then encode a device rather than a model. A cross-check that
    # relocates the thing it is checking is worse than no cross-check.
    pipe = None
    try:
        from transformers import pipeline
        pipe = pipeline("translation", model=model, tokenizer=tok,
                        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=1, do_sample=False,
                        device=-1)
    except Exception as e:  # pragma: no cover - informational only
        print(f"  (pipeline unavailable, skipping the cross-check: {e})",
              file=sys.stderr)

    # Belt and braces: whatever the pipeline did, the goldens are dumped from a
    # model on the CPU. Asserted rather than assumed, because the failure this
    # guards against is a silent one - a wrong device gives numbers, not an
    # error, once the inputs happen to be on the same device.
    model.to("cpu")
    dev = next(model.parameters()).device
    if dev.type != "cpu":
        raise SystemExit(f"refusing to dump goldens from device {dev}")
    print(f"device: {dev} (float32)")

    rows = []
    pipeline_agrees = True
    for name, text in PROMPTS:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**enc,
                                 forced_bos_token_id=tgt_id,
                                 max_new_tokens=args.max_new_tokens,
                                 num_beams=1, do_sample=False)
        out_ids = ids[0].tolist()
        translation = tok.batch_decode(ids, skip_special_tokens=True)[0]

        mark = ""
        if pipe is not None:
            # The cross-check must never cost us the dump: this is a sanity
            # check on HuggingFace's own two paths, not the reference itself.
            try:
                p = pipe(text)[0]["translation_text"]
            except Exception as e:
                pipe = None
                print(f"  (pipeline cross-check disabled after an error: {e})",
                      file=sys.stderr)
                p = translation
            if p != translation:
                pipeline_agrees = False
                mark = "  <- pipeline differs: " + json.dumps(p, ensure_ascii=False)

        rows.append((name, text, translation, out_ids))
        print(f"  {name:20s} T={len(out_ids):<3d} "
              f"{json.dumps(translation, ensure_ascii=False)}{mark}")

    header = {
        "model": str(args.model),
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "src_lang_id": int(src_id),
        "tgt_lang_id": int(tgt_id),
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "do_sample": False,
        "skip_special_tokens": True,
        "generation_config_num_beams": cfg_beams,
        "generation_config_max_length": cfg_max_length,
        "pipeline_agrees": pipeline_agrees,
        "n_cases": len(rows),
    }

    path = args.out / args.outfile
    with open(path, "w", encoding="utf-8") as f:
        f.write("# nllb.cpp text -> text goldens. See "
                "tools/dump_translation_reference.py\n")
        for k, v in header.items():
            f.write(f"# {k}\t{json.dumps(v, ensure_ascii=False)}\n")
        f.write("# name\tsource\ttranslation\tids\n")
        for name, text, translation, out_ids in rows:
            f.write("\t".join([
                name,
                json.dumps(text, ensure_ascii=False),
                json.dumps(translation, ensure_ascii=False),
                json.dumps(out_ids),
            ]) + "\n")

    print(f"\nwrote {len(rows)} cases to {path}")
    if not pipeline_agrees:
        print("WARNING: pipeline() and generate()+batch_decode() disagreed on at "
              "least one case.\n"
              "  The goldens are generate()+batch_decode(), which is what the CLI "
              "does.\n"
              "  Look at the marked lines before trusting a red gate.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
