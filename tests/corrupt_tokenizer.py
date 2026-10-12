#!/usr/bin/env python3
"""Break the tokenizer on purpose, to find out whether the gate would notice.

`tests/corrupt_gguf.py` does this for the model by damaging a tensor. The
tokenizer has no tensors: its failure modes live in the ALGORITHM - the
shortest-match rule, the grapheme boundaries, add_prefix_space, fuse_unk,
added-token extraction. So this mutates the source instead. Each mutation is a
plausible implementation of the same function - the version you would write if
you had read sentencepiece rather than spm_precompiled, or codepoints rather
than graphemes - not a random edit.

    python tests/corrupt_tokenizer.py build models/nllb-vocab.gguf ref/
    python tests/corrupt_tokenizer.py build models/nllb-vocab.gguf ref/ --only longest-match

Reads a configured build tree only to find libggml-base; it compiles the
mutated tokenizer into /tmp and never touches your tree.

What to look for is not "everything fails". It is WHICH pass fails: a mutation
that only the [sweep] catches is a mutation the corpus is blind to, and that is
the argument for the sweep existing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name, what it models, [(before, after), ...])
MUTATIONS = [
    ("longest-match",
     "take the LONGEST trie match, as sentencepiece itself does, instead of "
     "spm_precompiled's shortest",
     [("""        if (da_has_leaf(unit)) {
            if (node >= arr.size()) return false;
            value = da_value(arr[node]);
            return true;
        }
    }
    return false;""",
       """        if (da_has_leaf(unit)) {
            if (node >= arr.size()) return found;
            value = da_value(arr[node]);
            found = true;
        }
    }
    return found;"""),
      ("    if (arr.empty()) return false;\n    uint32_t node = 0;",
       "    if (arr.empty()) return false;\n    bool found = false;\n    uint32_t node = 0;"),
      ("        if (da_label(unit) != c) return false;",
       "        if (da_label(unit) != c) return found;"),
      ("        if (node >= arr.size()) return false;\n        unit = arr[node];",
       "        if (node >= arr.size()) return found;\n        unit = arr[node];")]),

    ("codepoints-not-graphemes",
     "iterate codepoints instead of UAX#29 grapheme clusters - i.e. skip "
     "src/nllb-grapheme-data.h entirely",
     [("        const size_t end = next_grapheme(v, text, i);",
       "        size_t end = i + utf8_len((unsigned char) text[i]);\n"
       "        if (end > text.size()) end = text.size();")]),

    ("no-prefix-space",
     "Metaspace without add_prefix_space",
     [("    if (s.compare(0, 3, MS) != 0) s = std::string(MS) + s;", "")]),

    ("no-fuse-unk",
     "emit one unk per unknown character instead of one per RUN",
     [("            if (!syms.empty() && syms.back().id == v.unk_id) { i += cl; continue; }", "")]),

    ("no-added-tokens",
     "skip AddedVocabulary - tokenize 'eng_Latn' as letters",
     [("    if (set.empty()) { out.push_back({-1, 0, s.size()}); return out; }",
       "    { out.push_back({-1, 0, s.size()}); return out; }")]),

    ("merge-highest-rank",
     "apply the HIGHEST-ranked merge first",
     [("            return rank != o.rank ? rank > o.rank : pos > o.pos;",
       "            return rank != o.rank ? rank < o.rank : pos > o.pos;")]),

    ("merge-ties-rightmost",
     "break equal-rank ties at the RIGHTMOST position instead of the leftmost",
     [("            return rank != o.rank ? rank > o.rank : pos > o.pos;",
       "            return rank != o.rank ? rank > o.rank : pos < o.pos;")]),

    # Expected to be INERT, and it is worth keeping for that: a split that
    # lands mid-character produces a substring that is not valid UTF-8 and so
    # is never in the vocabulary, which means the boundary test is a speed
    # optimisation and not a correctness requirement. If this ever starts being
    # caught, something has changed about the vocabulary.
    ("byte-splits",
     "derive merges by splitting pieces at every BYTE rather than at character "
     "boundaries (expected inert)",
     [("            if (((unsigned char) piece[i] & 0xC0) == 0x80) continue; // mid-char", "")],
     "inert"),

    ("no-collapse-spaces",
     "drop the Replace(' {2,}' -> ' ') half of the normalizer stack",
     [("""        if (out[k] == ' ') {
            size_t run = 0;
            while (k + run < out.size() && out[k + run] == ' ') run++;
            collapsed.push_back(' ');
            k += run;
        } else {""",
       """        if (false) {
        } else {""")]),

    ("mask-no-lstrip",
     "ignore the lstrip flag on <mask>",
     [("        if (best->lstrip) b = ws_extend_left(s, b);", "")]),
]


def build(tmp: Path, build_dir: Path) -> Path:
    lib = None
    for p in build_dir.rglob("libggml-base.*"):
        lib = p.parent
        break
    if lib is None:
        sys.exit(f"no libggml-base under {build_dir}; configure and build first")
    out = tmp / "gate"
    cmd = ["g++", "-std=c++17", "-O1", "-w",
           f"-I{tmp}", f"-I{ROOT / 'ggml/include'}",
           "-o", str(out),
           str(tmp / "test-tokenizer.cpp"), str(tmp / "nllb-vocab.cpp"),
           f"-L{lib}", "-lggml-base", f"-Wl,-rpath,{lib}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:])
        sys.exit("compile of the mutated tokenizer failed")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("gguf")
    ap.add_argument("ref")
    ap.add_argument("--only", default=None)
    ap.add_argument("--cc", default="g++")
    args = ap.parse_args()

    src = (ROOT / "src/nllb-vocab.cpp").read_text()
    rows = []

    for row in MUTATIONS:
        name, why, edits = row[0], row[1], row[2]
        expect = row[3] if len(row) > 3 else "caught"
        if args.only and args.only != name:
            continue
        mutated = src
        for before, after in edits:
            if before not in mutated:
                print(f"  {name}: PATCH NO LONGER APPLIES - the source moved "
                      f"under this mutation. Fix it or delete it; a mutation "
                      f"that silently stops applying reads as a gate that "
                      f"caught nothing.")
                rows.append((name, "STALE", "STALE", ""))
                mutated = None
                break
            mutated = mutated.replace(before, after, 1)
        if mutated is None:
            continue

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for f in ("include/nllb-vocab.h", "src/nllb-grapheme-data.h"):
                shutil.copy(ROOT / f, tmp / Path(f).name)
            shutil.copy(ROOT / "tests/test-tokenizer.cpp", tmp / "test-tokenizer.cpp")
            (tmp / "nllb-vocab.cpp").write_text(mutated)
            gate = build(tmp, Path(args.build_dir))
            r = subprocess.run([str(gate), args.gguf, args.ref, "--max-report", "1"],
                               capture_output=True, text=True)
            out = r.stdout

        # Sections are delimited by a '[' at the START of a line; the failure
        # reports contain '[' too, and slicing on those hid every pass but the
        # last one the first time this script was written.
        caught = []
        if "load failed" in r.stderr:
            caught.append("load")
        cur = None
        for line in out.splitlines():
            if line.startswith("["):
                cur = line[1:line.find("]")].strip()
            elif cur and "FAIL" in line and cur not in caught:
                caught.append(cur)

        got = "caught" if caught else "MISSED"
        want = "caught" if expect == "caught" else "MISSED"
        verdict = "ok" if got == want else ("MISSED" if expect == "caught"
                                            else "UNEXPECTED")
        rows.append((name, got, verdict, " ".join(caught)))
        print(f"  {name:26s} {got:7s} {' '.join(caught)}")
        print(f"      ({why})")

    print("\n%-26s %-7s %-11s %s" % ("mutation", "result", "verdict",
                                     "passes that fired"))
    for n, s, ver, c in rows:
        print("%-26s %-7s %-11s %s" % (n, s, ver, c))
    bad = [n for n, _, ver, _ in rows if ver != "ok"]
    print("\n%d of %d mutations behaved as expected" % (len(rows) - len(bad),
                                                        len(rows)))
    if bad:
        print("NOT AS EXPECTED:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
