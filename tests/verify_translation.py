#!/usr/bin/env python3
"""
THE TEXT -> TEXT GATE. Runs the actual CLI binary and compares its stdout to
HuggingFace's translations, string for string.

    python tests/verify_translation.py ./build/bin/nllb-cli \
        models/nllb-600M-f16.gguf ref/ -t 8

Needs no torch and no HuggingFace: the goldens are a text file written by
tools/dump_translation_reference.py.

WHY A SUBPROCESS AND NOT A C++ GATE. The claim being tested is "typing a
sentence prints the right translation", and that claim is about the binary's
stdout - argument parsing, the src/tgt wiring, detokenization and what does
and does not get printed. A C++ gate linking libnllb would re-implement the
CLI and then test the re-implementation.

FOUR PASSES, because they have different blind spots:

  [golden]  one process per case, text as an argument. The claim itself.
  [stdin]   all cases in one process, one line each. Proves the stdin path
            and that the lines stay in correspondence.
  [wiring]  --src and --tgt are read with -v and checked by POSITION:
            src_ids[0] must be the source code, out_ids[1] must be the target
            code, and swapping the two flags must swap exactly those two.
            This is the only pass that works on a random model, because it
            asserts about structure rather than about output quality - and it
            is the one that catches --src and --tgt being silently swapped or
            one of them being ignored.
  [errors]  a bad --tgt, an unquoted sentence and a missing --tgt must each
            exit non-zero. A CLI that shrugs at those translates something
            other than what was asked for.

A MISSING GOLDEN FILE IS A FAILURE, not a skip - see the step-4 lesson in
claude/handoff.md, where a SKIP that was not a FAIL meant deleting eight of
nine goldens still printed PASS. Use --no-golden / --no-stdin / --no-wiring /
--no-errors to leave a pass out on purpose.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

IDS_RE = re.compile(r"^(src|out) ids \((\d+)\):(.*)$", re.M)


def load_goldens(path: Path):
    if not path.exists():
        raise SystemExit(
            f"no {path}\n"
            f"  Dump it first (this is the only step that needs torch):\n"
            f"    python tools/dump_translation_reference.py <snapshot> "
            f"--out {path.parent}/ --tgt-lang deu_Latn --max-new-tokens 64")
    header, cases = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line[1:].strip().split("\t", 1)
            if len(parts) == 2:
                try:
                    header[parts[0].strip()] = json.loads(parts[1])
                except json.JSONDecodeError:
                    pass
            continue
        f = line.split("\t")
        if len(f) < 3:
            raise SystemExit(f"malformed golden line: {line!r}")
        cases.append({
            "name": f[0],
            "source": json.loads(f[1]),
            "translation": json.loads(f[2]),
            "ids": json.loads(f[3]) if len(f) > 3 else None,
        })
    if not cases:
        raise SystemExit(f"{path} has a header but no cases")
    n = header.get("n_cases")
    if n is not None and n != len(cases):
        raise SystemExit(f"{path} says n_cases={n} but has {len(cases)}")
    return header, cases


def parse_ids(stderr: str):
    """--verbose writes `src ids (N): ...` and `out ids (N): ...` to stderr."""
    out = {}
    for kind, n, rest in IDS_RE.findall(stderr):
        ids = [int(x) for x in rest.split()]
        if len(ids) != int(n):
            raise SystemExit(f"{kind} ids: header says {n}, listed {len(ids)}")
        out[kind] = ids
    return out


def run(cli, model, args, stdin=None, threads=4):
    cmd = [cli, "-m", model, "-t", str(threads)] + args
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return p


def diagnose(cli, model, case, base_args, threads, got):
    """A mismatch is worth a bisect: did the ids move, or only the string?"""
    p = run(cli, model, base_args + ["-v", case["source"]], threads=threads)
    ids = parse_ids(p.stderr)
    lines = []
    want_ids = case["ids"]
    got_ids = ids.get("out")
    if want_ids is None or got_ids is None:
        lines.append("      (no ids to compare)")
    elif got_ids == want_ids:
        lines.append("      ids are IDENTICAL -> the divergence is in "
                     "DETOKENIZATION, not the model.")
        lines.append("      Look at nllb_vocab_decode: Metaspace, the leading "
                     "space, skip_special.")
    else:
        k = next((i for i, (a, b) in enumerate(zip(got_ids, want_ids)) if a != b),
                 min(len(got_ids), len(want_ids)))
        lines.append(f"      ids DIFFER first at index {k} "
                     f"(got {got_ids[k:k + 4]}, want {want_ids[k:k + 4]}); "
                     f"len got={len(got_ids)} want={len(want_ids)}")
        if k == 0:
            lines.append("      index 0 is the decoder start token (eos, not "
                         "bos - trap 8).")
        elif k == 1:
            lines.append("      index 1 is the FORCED TARGET CODE (trap 10). "
                         "Check --tgt and forced_bos_id.")
        else:
            lines.append("      this is an ids-level divergence: run "
                         "nllb-test-generate, which owns that claim.")
        src = ids.get("src")
        if src:
            lines.append(f"      src ids: {src[:8]}{' ...' if len(src) > 8 else ''}")
    lines.append(f"      got  {json.dumps(got, ensure_ascii=False)}")
    lines.append(f"      want {json.dumps(case['translation'], ensure_ascii=False)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cli", help="path to the built nllb-cli")
    ap.add_argument("model", help="the SAME GGUF the goldens were dumped from")
    ap.add_argument("ref", type=Path, nargs="?", default=Path("ref"))
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--goldens", default="translations.txt")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--no-stdin", action="store_true")
    ap.add_argument("--no-wiring", action="store_true")
    ap.add_argument("--no-errors", action="store_true")
    args = ap.parse_args()

    if not Path(args.cli).exists():
        raise SystemExit(f"no such binary: {args.cli}  (cmake --build build -j)")

    header, cases = load_goldens(args.ref / args.goldens)
    src_lang = header.get("src_lang", "eng_Latn")
    tgt_lang = header.get("tgt_lang", "deu_Latn")
    max_new = header.get("max_new_tokens", 64)
    base = ["--src", src_lang, "--tgt", tgt_lang, "-n", str(max_new)]

    print(f"cli:      {args.cli}")
    print(f"model:    {args.model}")
    print(f"goldens:  {args.ref / args.goldens}  ({len(cases)} cases, "
          f"{src_lang} -> {tgt_lang}, max_new={max_new})")
    print(f"dumped from: {header.get('model')}  num_beams="
          f"{header.get('num_beams')} (checkpoint default: "
          f"{header.get('generation_config_num_beams')})")
    if header.get("pipeline_agrees") is False:
        print("WARNING: the dumper recorded that pipeline() and "
              "generate()+batch_decode() disagreed.")
    print()

    n_fail = 0

    # ---------------------------------------------------------------- golden
    if not args.no_golden:
        print("[golden] one process per case, text as an argument")
        for c in cases:
            p = run(args.cli, args.model, base + [c["source"]],
                    threads=args.threads)
            got = p.stdout[:-1] if p.stdout.endswith("\n") else p.stdout
            if p.returncode != 0:
                print(f"  {c['name']:20s} ERROR rc={p.returncode} "
                      f"{p.stderr.strip()[:200]}")
                n_fail += 1
                continue
            if p.stdout.count("\n") != 1:
                print(f"  {c['name']:20s} FAIL stdout is not exactly one line "
                      f"({p.stdout.count(chr(10))} newlines) - only the "
                      f"translation belongs on stdout")
                n_fail += 1
                continue
            if got == c["translation"]:
                print(f"  {c['name']:20s} ok")
            else:
                print(f"  {c['name']:20s} FAIL")
                print(diagnose(args.cli, args.model, c, base, args.threads, got))
                n_fail += 1
    else:
        print("[golden] SKIPPED on purpose (--no-golden)")

    # ----------------------------------------------------------------- stdin
    if not args.no_stdin:
        print("\n[stdin] all cases in one process, one line each")
        if any("\n" in c["source"] for c in cases):
            print("  skipped: a prompt contains a newline, which the "
                  "line-oriented stdin path cannot represent")
        else:
            p = run(args.cli, args.model, base,
                    stdin="\n".join(c["source"] for c in cases) + "\n",
                    threads=args.threads)
            lines = p.stdout.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if len(lines) != len(cases):
                print(f"  FAIL got {len(lines)} output lines for "
                      f"{len(cases)} input lines - the line correspondence is "
                      f"the whole contract of this mode")
                n_fail += 1
            else:
                bad = [c["name"] for c, l in zip(cases, lines)
                       if l != c["translation"]]
                if bad:
                    print(f"  FAIL {len(bad)} line(s) differ: {', '.join(bad)}")
                    for c, l in zip(cases, lines):
                        if l != c["translation"]:
                            print(f"    {c['name']:20s} "
                                  f"got  {json.dumps(l, ensure_ascii=False)}")
                            print(f"    {'':20s} want "
                                  f"{json.dumps(c['translation'], ensure_ascii=False)}")
                    n_fail += 1
                else:
                    print(f"  ok  {len(lines)} lines, identical to the "
                          f"argument path")
    else:
        print("\n[stdin] SKIPPED on purpose (--no-stdin)")

    # ---------------------------------------------------------------- wiring
    # Structural, so it holds on a random model too: the codes are checked by
    # POSITION rather than by whether the translation reads well.
    if not args.no_wiring:
        print("\n[wiring] --src and --tgt, by position, plus the swap")
        probe = next((c for c in cases if c["source"].strip()), cases[0])
        src_id = header.get("src_lang_id")
        tgt_id = header.get("tgt_lang_id")

        p = run(args.cli, args.model, base + ["-v", probe["source"]],
                threads=args.threads)
        ids = parse_ids(p.stderr)
        checks = []
        if "src" in ids and "out" in ids:
            s, o = ids["src"], ids["out"]
            if src_id is not None:
                checks.append((f"src_ids[0] == {src_lang} ({src_id})",
                               s[0] == src_id))
            checks.append(("src_ids[-1] == eos (2)", s[-1] == 2))
            checks.append(("out_ids[0] == decoder start (eos, 2)", o[0] == 2))
            if tgt_id is not None:
                checks.append((f"out_ids[1] == {tgt_lang} ({tgt_id}), the "
                               f"FORCED position", len(o) > 1 and o[1] == tgt_id))
            # rstrip the line terminator ONLY. .strip() here would remove the
            # leading space this very check is looking for, which is how a
            # gate quietly stops testing one of its four claims.
            text = p.stdout.rstrip("\n")
            checks.append(("printed text has no U+2581", "▁" not in text))
            checks.append(("printed text has no </s>", "</s>" not in text))
            checks.append((f"printed text has no {tgt_lang} code",
                           tgt_lang not in text))
            checks.append(("printed text has no leading space",
                           not text.startswith(" ")))

            # The swap. If either flag were ignored, or if they were used for
            # each other's job, exactly these two positions would not move.
            sw = run(args.cli, args.model,
                     ["--src", tgt_lang, "--tgt", src_lang, "-n", str(max_new),
                      "-v", probe["source"]], threads=args.threads)
            sids = parse_ids(sw.stderr)
            if "src" in sids and "out" in sids:
                if tgt_id is not None:
                    checks.append((f"swapped: src_ids[0] == {tgt_lang}",
                                   sids["src"][0] == tgt_id))
                if src_id is not None:
                    checks.append((f"swapped: out_ids[1] == {src_lang}",
                                   len(sids["out"]) > 1
                                   and sids["out"][1] == src_id))
            else:
                checks.append(("swapped run produced ids", False))
        else:
            checks.append(("-v printed `src ids` and `out ids` on stderr", False))

        # --list-langs is the only way a user finds out what to type, and it
        # must not need the weights (nllb_vocab_load reads metadata only).
        pl = run(args.cli, args.model, ["--list-langs"], threads=args.threads)
        codes = [l for l in pl.stdout.split("\n") if l.strip()]
        checks.append(("--list-langs exits 0", pl.returncode == 0))
        checks.append((f"--list-langs lists {src_lang}", src_lang in codes))
        checks.append((f"--list-langs lists {tgt_lang}", tgt_lang in codes))
        checks.append((f"--list-langs printed {len(codes)} codes (>= 2)",
                       len(codes) >= 2))

        for what, ok in checks:
            print(f"  {'ok  ' if ok else 'FAIL'} {what}")
            if not ok:
                n_fail += 1
    else:
        print("\n[wiring] SKIPPED on purpose (--no-wiring)")

    # ---------------------------------------------------------------- errors
    if not args.no_errors:
        print("\n[errors] bad input must not translate something else")
        bad = [
            (["--src", src_lang, "--tgt", "xxx_Xxxx", "hi"],
             "unknown --tgt"),
            (["--src", "nope", "--tgt", tgt_lang, "hi"],
             "unknown --src"),
            (["--src", src_lang, "hi"],
             "missing --tgt"),
            (["--src", src_lang, "--tgt", tgt_lang, "two", "words"],
             "unquoted text (two positionals)"),
        ]
        for a, what in bad:
            p = run(args.cli, args.model, a, threads=args.threads)
            ok = p.returncode != 0 and p.stdout.strip() == ""
            print(f"  {'ok  ' if ok else 'FAIL'} {what}: rc={p.returncode}, "
                  f"stdout={'empty' if not p.stdout.strip() else 'NOT EMPTY'}")
            if not ok:
                n_fail += 1
    else:
        print("\n[errors] SKIPPED on purpose (--no-errors)")

    print()
    if n_fail:
        print(f"FAIL - {n_fail} check(s)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
