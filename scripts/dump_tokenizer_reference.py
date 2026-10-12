#!/usr/bin/env python3
"""HF tokenizer -> ref/tokenizer_corpus.txt, the corpus the C++ gate diffs against.

Nine golden cases are nine cases. The tokenizer's failure modes are not spread
evenly over sentences - they are concentrated in normalization, which is a
1.1-million-codepoint function with a 224711-entry trie behind it, and in
grapheme clustering, which only bites on combining marks. So this writes a
corpus big enough to actually cover that: the encoder's own nine prompts, a
hand-picked set that pins each known quirk, every codepoint that the charsmap
touches, and a deterministic pseudo-random fuzz.

Needs NO torch and no model weights - `transformers` + `tokenizers` and a
tokenizer.json is the whole dependency:

    pip install transformers tokenizers
    python scripts/dump_tokenizer_reference.py models/nllb-600M --out ref/
    python scripts/dump_tokenizer_reference.py path/to/tokenizer.json --out ref/

Format, one case per line, so that text containing tabs and newlines survives:

    base64(utf-8 text) \\t id,id,id \\t base64(utf-8 decoded)

`ids` are WITHOUT the language code and eos - they are
`tokenizer(text, add_special_tokens=False)` - because that isolates the
tokenizer from the wrapping rule, which ref/manifest.json already covers.
`decoded` is `tokenizer.decode(ids, skip_special_tokens=True)`.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

# The encoder goldens' prompts, kept in sync by hand with scripts/dump_reference.py.
# Duplicated rather than imported because that module imports torch.
PROMPTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "",
    "a",
    "Translation quality depends on the encoder producing exactly the right "
    "hidden states, so this sentence is deliberately long enough to span "
    "several attention patterns and exercise the position table.",
    "Zwölf Boxkämpfer jagen Eva quer über den großen Sylter Deich.",
    "日本語のテキストもトークナイザを通します。",
    "الترجمة الآلية تعتمد على جودة المُرمِّز.",
    "test test test test test test test test",
]

# One line per thing that can go wrong. If you break the implementation, these
# are the cases that should tell you WHICH part you broke.
PINNED = [
    # --- Precompiled charsmap ------------------------------------------------
    "ﬁ ﬄ ½ ㎐ ㏿ Ⅰ Ⅷ",   # NFKC expansions
    "Ｈｅｌｌｏ ９９",        # fullwidth -> ASCII
    " \t\r\n  ",                     # every space-ish thing
    "​‌‍",               # controls and joiners
    "— – … “”",                  # dashes, quotes
    # --- shortest-match, whole-cluster consumption ---------------------------
    "Ầ",            # -> A-circumflex; the U+0300 is DROPPED
    "À́",            # -> A-grave
    "\r\n",                     # -> ONE space, not two
    "é q́",          # one composes, one does not
    "ẛ̣",             # 5 bytes, two codepoints, composes
    # --- grapheme boundaries -------------------------------------------------
    "\U0001f1fa\U0001f1f8\U0001f1e6",         # regional indicators, odd count
    "\U0001f468‍\U0001f469‍\U0001f467",  # emoji ZWJ family
    "각 각",             # Hangul jamo vs syllable
    "กำ क़्ष",       # Thai sara am, Devanagari
    "؀١",                                # Prepend
    # --- Metaspace -----------------------------------------------------------
    "  leading and   internal  runs ",
    " ",
    "▁literal metaspace▁",
    # --- BPE / fuse_unk ------------------------------------------------------
    "\U000e0001\U000e0002\U000e0003",   # a run of unknowns -> ONE unk
    "x\U000e0001y",
    # --- AddedVocabulary: these are cut out BEFORE normalization ------------
    "eng_Latn is a token and also a word here",
    "x eng_Latn y",                 # the space before becomes a bare U+2581
    "zho_Hans中文",
    "eng_Latneng_Latn",
    "hello</s>world",
    "a<unk>b",
    "x<pad>y",
    # <mask> is the only one with lstrip, and the only one matched AFTER
    # normalization - so it eats the space before it and not the one after.
    "a<mask>b",
    "abc <mask> def",
    "  <mask>  x",
    "<mask>abc",
    "ﬁ<mask>ﬁ",
]


def build_charsmap_probes(tok_json_path: Path) -> list[str]:
    """Every codepoint that appears as the first character of a charsmap key.

    These are the inputs the normalizer actually does something to; the other
    ~1.1M codepoints all take the identity path and one sweep of them (which
    the C++ gate does separately) is enough.
    """
    tj = json.loads(tok_json_path.read_text(encoding="utf-8"))
    norm = tj.get("normalizer") or {}
    seq = norm.get("normalizers", [norm])
    blob = base64.b64decode(seq[0]["precompiled_charsmap"])
    import struct
    trie_bytes = struct.unpack("<I", blob[:4])[0]
    arr = struct.unpack("<%dI" % (trie_bytes // 4), blob[4:4 + trie_bytes])

    def label(u):  return u & ((1 << 31) | 0xFF)
    def offset(u): return (u >> 10) << ((u & (1 << 9)) >> 6)

    # Walk one level to collect the first bytes, then reconstruct codepoints by
    # asking the trie which lead bytes it accepts. Simpler and just as good:
    # probe every codepoint and keep the ones the normalizer changes.
    out = []
    node0 = 0 ^ offset(arr[0])
    leads = set()
    for c in range(1, 256):
        p = node0 ^ c
        if p < len(arr) and label(arr[p]) == c:
            leads.add(c)
    for cp in range(1, 0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        b = chr(cp).encode("utf-8")[0]
        if b in leads:
            out.append(chr(cp))
    return out


MARKS = list(range(0x300, 0x370)) + [
    0x483, 0x591, 0x610, 0x64B, 0x6D6, 0x730, 0x7A6, 0x93C, 0x94D,
    0xE31, 0xE4D, 0x200D, 0x200C, 0xFE0F, 0x20E3,
]
SPACING = [0x903, 0x93B, 0x93E, 0xE33, 0x9BE, 0x1B04]
PREPEND = [0x600, 0x601, 0x605, 0x6DD, 0x110BD]
HANGUL = [0x1100, 0x1161, 0x11A8, 0xAC00, 0xAC01]
RI = [0x1F1E6, 0x1F1E8, 0x1F1FA, 0x1F1F8]
BASES = list("abcdeXYZ019 .,!?-'\"\t\n\r") + [
    chr(c) for c in (0xC0, 0x1E9B, 0x915, 0xE01, 0x4E00, 0x3042, 0x627, 0x5D0,
                     0x1F600, 0xFF21, 0xFB01, 0xBD, 0x3390, 0x2160, 0x2026,
                     0x2014, 0xA0, 0x2028, 0x85, 0x07)
]


def fuzz(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        chars = []
        for _ in range(rng.randrange(0, 14)):
            r = rng.random()
            if r < 0.45:
                chars.append(rng.choice(BASES))
            elif r < 0.62:
                chars.append(chr(rng.choice(MARKS)))
            elif r < 0.68:
                chars.append(chr(rng.choice(SPACING)))
            elif r < 0.72:
                chars.append(chr(rng.choice(PREPEND)))
            elif r < 0.76:
                chars.append(chr(rng.choice(HANGUL)))
            elif r < 0.79:
                chars.append(chr(rng.choice(RI)))
            else:
                while True:
                    c = rng.randrange(1, 0x110000)
                    if not (0xD800 <= c < 0xE000):
                        break
                chars.append(chr(c))
        out.append("".join(chars))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF snapshot dir, or a bare tokenizer.json")
    ap.add_argument("--out", type=Path, default=Path("ref"))
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--fuzz", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the every-codepoint hash (takes ~2 minutes)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = Path(args.model)
    tok_json = src if src.is_file() else src / "tokenizer.json"
    if not tok_json.exists():
        sys.exit(f"no tokenizer.json at {tok_json}")

    from transformers import NllbTokenizerFast
    tok = NllbTokenizerFast(tokenizer_file=str(tok_json), src_lang=args.src_lang)

    texts = list(PROMPTS) + list(PINNED)
    texts += build_charsmap_probes(tok_json)
    texts += fuzz(random.Random(args.seed), args.fuzz)

    # de-duplicate but keep order, so a failure line number is stable
    seen, uniq = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    b64 = lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")
    path = args.out / "tokenizer_corpus.txt"
    with open(path, "w", encoding="ascii") as f:
        f.write(f"# nllb.cpp tokenizer corpus, src_lang={args.src_lang}, "
                f"{len(uniq)} cases\n")
        f.write("# base64(text) \\t ids \\t base64(decoded)\n")
        for t in uniq:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            dec = tok.decode(ids, skip_special_tokens=True)
            f.write(f"{b64(t)}\t{','.join(map(str, ids))}\t{b64(dec)}\n")

    print(f"wrote {path}  ({len(uniq)} cases, {path.stat().st_size/1e6:.1f} MB)")
    print(f"  {len(PROMPTS)} encoder prompts, {len(PINNED)} pinned quirks, "
          f"{len(texts) - len(PROMPTS) - len(PINNED) - args.fuzz} charsmap probes, "
          f"{args.fuzz} fuzz")

    # ---- the sweep --------------------------------------------------------
    # The corpus above only contains inputs this script chose. A C++ trie walk
    # that wrongly MATCHES something the reference leaves alone would not be in
    # it, and there are 1.1M codepoints to be wrong about. So: tokenize every
    # single one, and reduce the result to one number the C++ can reproduce.
    # Total coverage for 30 lines and one line of output file.
    if not args.no_sweep:
        h = 0xCBF29CE484222325
        n = 0
        for cp in range(1, 0x110000):
            if 0xD800 <= cp < 0xE000:
                continue
            ids = tok(chr(cp), add_special_tokens=False)["input_ids"]
            for b in ("%d:%s\n" % (cp, ",".join(map(str, ids)))).encode("ascii"):
                h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            n += 1
        sweep = args.out / "tokenizer_sweep.txt"
        sweep.write_text("n=%d hash=%#018x\n" % (n, h))
        print(f"wrote {sweep}  (every codepoint: n={n}, hash={h:#018x})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
