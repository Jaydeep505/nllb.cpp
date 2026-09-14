#!/usr/bin/env python3
"""
A STAND-IN for nllb-cli that implements its contract by looking the answer up,
so tests/verify_translation.py can be tested WITHOUT the 2.46 GB checkpoint.

This tests the GATE, not the CLI. It is the translation-layer equivalent of
tests/corrupt_gguf.py and tests/corrupt_tokenizer.py: a gate whose failure
modes have never been exercised is a gate nobody knows the sensitivity of, and
this one is the first in the repo that can only be run where the real weights
are - which is exactly where a broken gate is least likely to be noticed.

    # write synthetic goldens (no torch, no model, no download)
    python tests/fake_cli.py --make-goldens ref-fake/

    # the gate should pass against an honest stand-in
    python tests/verify_translation.py tests/fake_cli.py MODEL ref-fake/

    # ... and fire on each deliberate breakage
    python tests/fake_cli.py --matrix ref-fake/

Every breakage below is a mistake the real CLI could plausibly make, and each
is one of the failure modes claude/handoff.md predicted for step 9 - the ones
no existing gate can see, because every other gate in this repo stops at ids.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BREAKAGES = {
    "swap":         "use --tgt as the source code and --src as the forced one",
    "ignore-tgt":   "always force the goldens' target, whatever --tgt says",
    "keep-eos":     "leave </s> in the printed string",
    "keep-lang":    "leave the target language code in the printed string",
    "lead-space":   "keep Metaspace's leading space",
    "stdout-noise": "print the banner on stdout instead of stderr",
    "drop-empty":   "skip empty input lines in stdin mode",
    "no-ids":       "--verbose without the `src ids` / `out ids` lines",
    "shrug":        "accept an unknown --tgt and translate anyway",
    "join-argv":    "join extra positional arguments with a space",
}

# Synthetic goldens: the shape of the real thing, none of the weight. The
# sources are the nine prompt names from tools/dump_reference.py so the file
# reads like the real one; the translations are invented.
FAKE = [
    ("00_hello_world", "Hello, world!", "Hallo, Welt!"),
    ("01_quick_brown", "The quick brown fox jumps over the lazy dog.",
     "Der schnelle braune Fuchs springt über den faulen Hund."),
    ("02_empty", "", ""),
    ("03_single_char", "a", "a"),
    ("04_long", "Translation quality depends on the encoder.",
     "Die Übersetzungsqualität hängt vom Encoder ab."),
    ("05_german", "Zwölf Boxkämpfer jagen Eva quer über den großen Sylter Deich.",
     "Zwölf Boxkämpfer jagen Eva quer über den großen Sylter Deich."),
    ("06_japanese", "日本語のテキストもトークナイザを通します。",
     "Japanischer Text wird ebenfalls durch den Tokenizer geleitet."),
    ("07_arabic", "الترجمة الآلية تعتمد على جودة المُرمِّز.",
     "Maschinelle Übersetzung hängt von der Qualität des Encoders ab."),
    ("08_repeated", "test test test test", "Test Test Test Test"),
]

SRC_LANG, TGT_LANG = "eng_Latn", "deu_Latn"
SRC_ID,   TGT_ID   = 256047, 256042
EOS, PAD = 2, 1


def fake_ids(text: str, lang_id: int) -> list[int]:
    """Deterministic pseudo-ids. Structure is what the gate checks, not values."""
    return [lang_id] + [4 + (ord(c) % 2000) for c in text] + [EOS]


def make_goldens(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    header = {
        "model": "tests/fake_cli.py (synthetic, NOT a real checkpoint)",
        "src_lang": SRC_LANG, "tgt_lang": TGT_LANG,
        "src_lang_id": SRC_ID, "tgt_lang_id": TGT_ID,
        "max_new_tokens": 64, "num_beams": 1, "do_sample": False,
        "skip_special_tokens": True,
        "generation_config_num_beams": 1, "generation_config_max_length": 200,
        "pipeline_agrees": True, "n_cases": len(FAKE),
    }
    path = out / "translations.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC goldens from tests/fake_cli.py --make-goldens\n")
        for k, v in header.items():
            f.write(f"# {k}\t{json.dumps(v, ensure_ascii=False)}\n")
        f.write("# name\tsource\ttranslation\tids\n")
        for name, src, tgt in FAKE:
            ids = [EOS, TGT_ID] + fake_ids(tgt, TGT_ID)[1:]
            f.write("\t".join([name,
                               json.dumps(src, ensure_ascii=False),
                               json.dumps(tgt, ensure_ascii=False),
                               json.dumps(ids)]) + "\n")
    print(f"wrote {len(FAKE)} synthetic cases to {path}")


def load(out: Path) -> dict:
    table = {}
    for line in (out / "translations.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        table[json.loads(f[1])] = (json.loads(f[2]), json.loads(f[3]))
    return table


# --------------------------------------------------------------------------
# the stand-in itself
# --------------------------------------------------------------------------

def serve(argv: list[str]) -> int:
    brk = os.environ.get("FAKE_CLI_BREAK", "")
    ref = Path(os.environ.get("FAKE_CLI_REF", "ref-fake"))
    table = load(ref)

    src, tgt, text, have_text = SRC_LANG, "", "", False
    verbose = list_langs = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-m", "--model", "-t", "--threads", "-n", "--max-new-tokens",
                 "-b", "--backend"):
            i += 2
            continue
        if a == "--src":
            src = argv[i + 1]; i += 2; continue
        if a == "--tgt":
            tgt = argv[i + 1]; i += 2; continue
        if a in ("-v", "--verbose"):
            verbose = True; i += 1; continue
        if a == "--list-langs":
            list_langs = True; i += 1; continue
        if a == "--no-cache":
            i += 1; continue
        if a.startswith("-") and a != "-":
            print(f"unknown option: {a}", file=sys.stderr)
            return 2
        if not have_text:
            text, have_text = a, True
        elif brk == "join-argv":
            text += " " + a
        else:
            print("unexpected extra argument", file=sys.stderr)
            return 2
        i += 1

    if list_langs:
        for c in (SRC_LANG, TGT_LANG, "fra_Latn", "spa_Latn"):
            print(c)
        return 0

    known = {SRC_LANG, TGT_LANG, "fra_Latn", "spa_Latn"}
    if not tgt:
        print("--tgt is required", file=sys.stderr)
        return 2
    if src not in known:
        print(f"unknown source language code {src!r}", file=sys.stderr)
        return 2
    if tgt not in known and brk != "shrug":
        print(f"unknown target language code {tgt!r}", file=sys.stderr)
        return 2

    def one(line: str) -> str:
        want, ids = table.get(line, ("", [EOS, TGT_ID, EOS]))

        # The wiring the gate checks by position.
        eff_src, eff_tgt = src, tgt
        if brk == "swap":
            eff_src, eff_tgt = tgt, src
        if brk == "ignore-tgt":
            eff_tgt = TGT_LANG
        code = {SRC_LANG: SRC_ID, TGT_LANG: TGT_ID,
                "fra_Latn": 256057, "spa_Latn": 256163}
        s_ids = fake_ids(line, code.get(eff_src, SRC_ID))
        o_ids = [EOS, code.get(eff_tgt, TGT_ID)] + ids[2:]

        out = want
        if brk == "swap" or brk == "ignore-tgt":
            # a wrong code at position 1 does not change the stand-in's text;
            # only [wiring] can see this one, which is the point of that pass
            pass
        if brk == "keep-eos":
            out += "</s>"
        if brk == "keep-lang":
            out = TGT_LANG + out
        if brk == "lead-space":
            out = " " + out

        if verbose:
            banner = (f"src ids ({len(s_ids)}): " + " ".join(map(str, s_ids)) + "\n"
                      f"out ids ({len(o_ids)}): " + " ".join(map(str, o_ids)) + "\n")
            if brk == "no-ids":
                banner = "model: synthetic\ntiming: 0 ms\n"
            if brk == "stdout-noise":
                sys.stdout.write(banner)
            else:
                sys.stderr.write(banner)
        return out

    if have_text:
        print(one(text))
        return 0

    for line in sys.stdin:
        line = line.rstrip("\n").rstrip("\r")
        if brk == "drop-empty" and not line:
            continue
        print(one(line))
    return 0


# --------------------------------------------------------------------------
# does the gate fire?
# --------------------------------------------------------------------------

def _run_gate(gate: Path, here: Path, ref: Path, brk: str, extra):
    env = dict(os.environ, FAKE_CLI_BREAK=brk, FAKE_CLI_REF=str(ref))
    p = subprocess.run([sys.executable, str(gate), str(here), "MODEL",
                        str(ref)] + list(extra),
                       capture_output=True, text=True, env=env)
    # Attribute each FAIL to the [pass] whose header last appeared. The
    # summary line ("FAIL - n check(s)") is NOT inside a section, and
    # counting it would make every mutation look like it fired the last
    # pass - which is the same mistake tests/corrupt_tokenizer.py made
    # when it sliced stdout on '[' and concluded only [sweep] ever fired.
    fired = set()
    section = None
    for line in p.stdout.splitlines():
        if line.startswith("["):
            section = line.split("]")[0][1:]
        elif line.startswith("FAIL -") or line.startswith("PASS"):
            section = None
        elif "FAIL" in line and section:
            fired.add(section)
    return p.returncode, ", ".join(sorted(fired)) or "-"


def matrix(ref: Path) -> int:
    """Every mutation, under BOTH assertions.

    The second column is the one worth having. --allow-id-drift replaces byte
    equality with chrF, and a metric that tolerates a changed token also
    tolerates some real breakages - chrF strips whitespace, so a stray leading
    space scores 100.00 against the golden and [golden] cannot see it at all.
    That is not an argument against the flag; it is the reason [wiring] exists
    and the reason this column is printed. A mutation caught only in the strict
    column is a mutation that a quantised run would ship.
    """
    here = Path(__file__).resolve()
    gate = here.parent / "verify_translation.py"
    rows = []
    for brk in [""] + list(BREAKAGES):
        rc_s, fired_s = _run_gate(gate, here, ref, brk, [])
        rc_d, fired_d = _run_gate(gate, here, ref, brk, ["--allow-id-drift"])
        rows.append((brk or "(none - honest stand-in)",
                     rc_s, fired_s, rc_d, fired_d))

    print(f"\n{'mutation':22s} {'rc':>3s}  {'strict: passes that fired':32s}"
          f" {'rc':>3s}  --allow-id-drift")
    print("-" * 110)
    n_missed = 0
    n_drift_only = 0
    for brk, rc_s, fired_s, rc_d, fired_d in rows:
        if brk.startswith("("):
            ok = rc_s == 0 and rc_d == 0
            note = "" if ok else "  <- the HONEST stand-in must pass BOTH"
        else:
            ok = rc_s != 0
            note = "" if ok else "  <- MISSED"
            if not ok:
                n_missed += 1
            if ok and rc_d == 0:
                note = "  <- STRICT ONLY: invisible once ids may drift"
                n_drift_only += 1
        print(f"{brk:22s} {rc_s:>3d}  {fired_s:32s} {rc_d:>3d}  {fired_d}{note}")
    print()
    if rows[0][1] != 0 or rows[0][3] != 0:
        print("FAIL - the gate rejects an honest stand-in")
        return 1
    if n_missed:
        print(f"FAIL - {n_missed} mutation(s) went undetected")
        return 1
    if n_drift_only:
        print(f"NOTE - {n_drift_only} mutation(s) are caught ONLY by the strict "
              f"assertion. That is a\n"
              f"       measured property of --allow-id-drift, not a defect: "
              f"keep byte equality as the\n"
              f"       default and turn the flag on only for the checkpoint "
              f"that has earned it.")
    print(f"PASS - {len(BREAKAGES)} mutations, all caught by the strict gate")
    return 0


def main() -> int:
    if "--make-goldens" in sys.argv:
        i = sys.argv.index("--make-goldens")
        out = Path(sys.argv[i + 1]) if len(sys.argv) > i + 1 else Path("ref-fake")
        make_goldens(out)
        return 0
    if "--matrix" in sys.argv:
        i = sys.argv.index("--matrix")
        ref = Path(sys.argv[i + 1]) if len(sys.argv) > i + 1 else Path("ref-fake")
        return matrix(ref)
    if "--list-breakages" in sys.argv:
        for k, v in BREAKAGES.items():
            print(f"{k:14s} {v}")
        return 0
    return serve(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
