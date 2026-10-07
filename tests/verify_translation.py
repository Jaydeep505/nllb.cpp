#!/usr/bin/env python3
"""
THE TEXT -> TEXT GATE. Runs the actual CLI binary and compares its stdout to
HuggingFace's translations, string for string.

    python tests/verify_translation.py ./build/bin/nllb-cli \
        models/nllb-600M-f16.gguf ref/ -t 8

Needs no torch and no HuggingFace: the goldens are a text file written by
scripts/dump_translation_reference.py.

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
docs/dev/handoff.md, where a SKIP that was not a FAIL meant deleting eight of
nine goldens still printed PASS. Use --no-golden / --no-stdin / --no-wiring /
--no-errors to leave a pass out on purpose.

--allow-id-drift, AND WHY IT IS NOT A LOOSER TOLERANCE
------------------------------------------------------
Byte equality against HuggingFace is an f16-era assertion, exactly as exact id
equality is in nllb-test-precision (trap 30). At Q8_0 one greedy decision on the
real 600M has already flipped, and the sentence that came out was fine - better,
even, though see the handoff for why not to read anything into n = 1. A gate
that fails on that reports "quantisation happened", which is not a finding.

The reply is NOT to compare strings with a fuzzy string comparison. It is to
state a DIFFERENT claim and make the operator ask for it by name:

    --allow-id-drift   the translation need not be byte-identical to HF's; it
                       must stay within chrF of it, per segment and over the
                       corpus.

Without the flag nothing changes: byte equality, and every f32/f16 result in
docs/dev/handoff.md reproduces. The flag exists so that a green line always names
the claim it is green about - "identical to HF" and "close to HF" are different
products and a log six months old cannot tell them apart otherwise.

chrF (Popovic 2015) is the right distance here rather than BLEU: it is character
n-gram F-score, so it degrades smoothly when one token changes, it needs no
tokenizer of its own - which matters when the target script is Devanagari or
Gujarati and a word tokenizer would be a second thing to be wrong - and it is
the standard metric for exactly this model family. The implementation below is
this file's own, deliberately: this gate's whole point is that it runs with no
dependencies. When sacrebleu HAPPENS to be importable it is used as a CROSS-
CHECK and the difference is printed, which is the same instinct as [tables] in
the tokenizer gate - measure the agreement, do not assume it.

THE THRESHOLDS ARE THE WEAK PART OF THIS GATE and should be read as such. 95.0
corpus / 75.0 worst-segment are set to separate "rounding reordered a near-tie"
(one word changed in a long sentence measures chrF 96.4) from "the output
changed meaningfully" (a dropped clause, 39; the wrong language, 0). They are
not derived from anything. Tighten them for a pair you have measured; the
numbers are printed on every run so the drift can be watched rather than
guessed at.

THERE ARE TWO FLOORS BECAUSE ONE IS NOT ENOUGH, and that is measured rather than
argued: under the `lead-space` mutation this gate reports corpus chrF 100.00
with a worst segment of 16.67. Summed statistics are dominated by long segments,
so a short one can be destroyed and vanish into the corpus figure. The
per-segment floor is what fires there.

AND chrF CANNOT SEE WHITESPACE AT ALL - it strips it, which is what lets it work
on Japanese and Devanagari without a tokenizer. So a stray leading space scores
100.00 against the golden and [golden] is blind to it. [wiring] catches it,
structurally, and tests/fake_cli.py --matrix prints both columns so that this
kind of gap is a measured row in a table rather than a thing nobody checked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

IDS_RE = re.compile(r"^(src|out) ids \((\d+)\):(.*)$", re.M)

CHRF_ORDER = 6
CHRF_BETA = 2.0


def chrf_stats(hyp: str, ref: str, order: int = CHRF_ORDER):
    """Per-order (n_hyp, n_ref, n_match) character n-gram counts.

    Whitespace is removed first, which is chrF's own default and the reason it
    needs no tokenizer: for Japanese or Thai there is no whitespace to remove
    and for German there is, and the metric does not have to know which.

    Returned as STATISTICS rather than a score so the corpus figure can be
    computed by summing them. Averaging per-sentence chrF is a different
    quantity and a noisier one - a four-character sentence would carry the same
    weight as a hundred-character one.
    """
    h = "".join(hyp.split())
    r = "".join(ref.split())
    out = []
    for n in range(1, order + 1):
        hc, rc = {}, {}
        for i in range(len(h) - n + 1):
            g = h[i:i + n]
            hc[g] = hc.get(g, 0) + 1
        for i in range(len(r) - n + 1):
            g = r[i:i + n]
            rc[g] = rc.get(g, 0) + 1
        match = sum(min(v, rc.get(g, 0)) for g, v in hc.items())
        out.append((sum(hc.values()), sum(rc.values()), match))
    return out


def chrf_score(stats, beta: float = CHRF_BETA) -> float:
    """chrF2 from summed statistics, 0..100. Byte-identical to sacrebleu.

    Precision and recall are averaged over the orders and the F-score is taken
    once at the end. The subtlety, and the reason this was written by reading
    sacrebleu's source rather than by recalling the paper, is EFFECTIVE ORDER:
    an order in which either side has no n-grams at all does not contribute and
    is not counted in the divisor. Averaging over all six instead caps a
    three-character segment at 50, because orders 4-6 are empty for it - which
    is what the first draft here did, and it disagreed with sacrebleu by 50
    points on a corpus of nine short sentences before the cross-check caught it.

    The consequence to know: two EMPTY strings have no orders at all, so the
    effective order is zero and the score is 0.0, not 100. That is sacrebleu's
    answer and this function keeps it rather than special-casing the metric.
    `02_empty` is one of the nine goldens, so the GATE handles it instead - a
    byte-identical segment is counted as identical and is not held to the
    per-segment floor, because "close but wrong" is not a thing it can be.
    """
    eps = 1e-16
    if not stats:
        return 0.0
    avg_p = avg_r = 0.0
    effective = 0
    for n_hyp, n_ref, n_match in stats:
        if n_hyp <= 0 or n_ref <= 0:
            continue
        avg_p += n_match / n_hyp
        avg_r += n_match / n_ref
        effective += 1
    if effective == 0:
        return 0.0
    avg_p /= effective
    avg_r /= effective
    if avg_p + avg_r < eps:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1.0 + b2) * avg_p * avg_r / (b2 * avg_p + avg_r)


def chrf(hyp: str, ref: str) -> float:
    return chrf_score(chrf_stats(hyp, ref))


def sum_stats(all_stats):
    if not all_stats:
        return []
    return [tuple(sum(st[n][i] for st in all_stats) for i in range(3))
            for n in range(len(all_stats[0]))]


def sacrebleu_crosscheck(pairs):
    """Same numbers out of sacrebleu, if it is installed. Never required.

    This is the [tables] habit from the tokenizer gate: when a second
    implementation is lying around, diff against it and print the difference
    rather than assuming the reimplementation is right. Returns None when
    sacrebleu is not importable, which is the normal case.
    """
    try:
        from sacrebleu.metrics import CHRF          # type: ignore
    except Exception:
        return None
    try:
        m = CHRF(char_order=CHRF_ORDER, word_order=0, beta=int(CHRF_BETA))
        return float(m.corpus_score([h for h, _ in pairs],
                                    [[r for _, r in pairs]]).score)
    except Exception:
        return None


def load_goldens(path: Path):
    if not path.exists():
        raise SystemExit(
            f"no {path}\n"
            f"  Dump it first (this is the only step that needs torch):\n"
            f"    python scripts/dump_translation_reference.py <snapshot> "
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


# --backend NAME, passed to every invocation. Empty means the CLI's own
# default, which since step 13 is "auto" - i.e. Metal on a Mac built with it.
# That is deliberate (this gate tests the PRODUCT), but it means the same
# command tests different kernels on different machines, so the header line
# prints which backend actually ran and step 13's acceptance runs it both ways.
BACKEND: list[str] = []


def run(cli, model, args, stdin=None, threads=4):
    cmd = [cli, "-m", model, "-t", str(threads)] + BACKEND + args
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
    ap.add_argument("--backend", default="",
                    help="passed to nllb-cli --backend (cpu, metal, auto). "
                         "Default: the CLI's own default, auto")
    ap.add_argument("--goldens", default="translations.txt")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--no-stdin", action="store_true")
    ap.add_argument("--batch", type=int, default=0,
                    help="run the [stdin] pass through nllb-cli --batch N. Its output "
                         "must be BYTE-IDENTICAL to the same goldens the unbatched "
                         "run matches - that is the batched product claim.")
    ap.add_argument("--no-wiring", action="store_true")
    ap.add_argument("--no-errors", action="store_true")
    ap.add_argument("--allow-id-drift", action="store_true",
                    help="the translation need not be byte-identical to HF's; "
                         "chrF becomes the assertion. Required at Q8_0 and "
                         "below - see the module docstring.")
    ap.add_argument("--min-chrf", type=float, default=95.0,
                    help="corpus chrF floor under --allow-id-drift (default 95)")
    ap.add_argument("--min-seg-chrf", type=float, default=75.0,
                    help="worst-segment chrF floor under --allow-id-drift "
                         "(default 75)")
    args = ap.parse_args()

    if args.backend:
        BACKEND[:] = ["--backend", args.backend]

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
    # Which kernels this run is actually testing (step 13). One -v call; the
    # CLI prints `backend: NAME` on stderr. A stand-in that does not is fine.
    if cases:
        pb = run(args.cli, args.model, base + ["-v", cases[0]["source"]], threads=args.threads)
        be = re.search(r"^backend: (.*)$", pb.stderr, re.M)
        print(f"backend:  {be.group(1) if be else '(not reported by this binary)'}"
              f"{'  [--backend ' + args.backend + ']' if args.backend else '  [CLI default]'}")
    if header.get("pipeline_agrees") is False:
        print("WARNING: the dumper recorded that pipeline() and "
              "generate()+batch_decode() disagreed.")
    if args.allow_id_drift:
        print(f"MODE: --allow-id-drift. The assertion is chrF, not byte "
              f"equality:\n"
              f"      corpus chrF >= {args.min_chrf:.1f} and every segment "
              f">= {args.min_seg_chrf:.1f}.\n"
              f"      Byte-identical segments are still counted and reported - "
              f"that count,\n"
              f"      not the chrF, is the number to watch move.")
    print()

    n_fail = 0
    seg_stats = []      # per-case chrF statistics, summed for the corpus figure
    seg_pairs = []      # (hypothesis, reference), for the sacrebleu cross-check
    seg_identical = []  # parallel to seg_stats; see the [chrF] block
    n_identical = 0

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
            st = chrf_stats(got, c["translation"])
            seg_stats.append(st)
            seg_pairs.append((got, c["translation"]))
            seg_identical.append(got == c["translation"])
            if got == c["translation"]:
                n_identical += 1
                print(f"  {c['name']:20s} ok   (identical)")
            elif not args.allow_id_drift:
                print(f"  {c['name']:20s} FAIL")
                print(diagnose(args.cli, args.model, c, base, args.threads, got))
                n_fail += 1
            else:
                sc = chrf_score(st)
                if sc >= args.min_seg_chrf:
                    print(f"  {c['name']:20s} ok   chrF={sc:.2f} (differs from "
                          f"HF, within the floor)")
                    print(f"    {'':18s} got  "
                          f"{json.dumps(got, ensure_ascii=False)}")
                    print(f"    {'':18s} want "
                          f"{json.dumps(c['translation'], ensure_ascii=False)}")
                else:
                    print(f"  {c['name']:20s} FAIL chrF={sc:.2f} < "
                          f"{args.min_seg_chrf:.1f}")
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
            stdin_args = base + (["--batch", str(args.batch)] if args.batch > 1 else [])
            if args.batch > 1:
                print(f"  through the BATCHED path: --batch {args.batch}")
            p = run(args.cli, args.model, stdin_args,
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
                # Under --allow-id-drift the stdin claim is NOT "matches HF" -
                # [golden] already made that one. It is that stdin and the
                # argument path agree with EACH OTHER, which is the only thing
                # this pass has ever been the sole catcher of (the `drop-empty`
                # mutation). Comparing it to HF a second time would make it fail
                # for a reason [golden] has already accepted.
                if args.allow_id_drift:
                    ref_lines = [h for h, _ in seg_pairs] if seg_pairs else None
                    if ref_lines is None or len(ref_lines) != len(lines):
                        print("  skipped: [golden] did not run, so there is no "
                              "argument-path output to compare against")
                    else:
                        diff = [c["name"] for c, l, g in zip(cases, lines, ref_lines)
                                if l != g]
                        if diff:
                            print(f"  FAIL {len(diff)} line(s) differ from the "
                                  f"ARGUMENT path: {', '.join(diff)}")
                            n_fail += 1
                        else:
                            print(f"  ok  {len(lines)} lines, identical to the "
                                  f"argument path")
                elif bad:
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

    # ------------------------------------------------------------------ chrF
    # Printed on EVERY run, drift allowed or not. In strict mode it is a free
    # number that says how far a failure was, and it is what makes a later
    # quantised run comparable to today's exact one.
    if seg_stats:
        corpus = chrf_score(sum_stats(seg_stats))
        per_seg = [chrf_score(st) for st in seg_stats]
        # Byte-identical segments are exempt from the worst-segment figure.
        # They are exact, and chrF scores two EMPTY strings 0.0 (effective
        # order zero - see chrf_score), so `02_empty` passing perfectly would
        # otherwise report as the worst segment in the run.
        scored = [(sc, c["name"]) for sc, c, ident
                  in zip(per_seg, cases, seg_identical) if not ident]
        worst, worst_name = min(scored) if scored else (100.0, "-")
        print(f"\n[chrF] corpus {corpus:.2f}   worst NON-IDENTICAL segment "
              f"{worst:.2f} ({worst_name})   identical "
              f"{n_identical}/{len(seg_stats)}")
        sb = sacrebleu_crosscheck(seg_pairs)
        if sb is None:
            print("       (sacrebleu not installed - this is our own chrF2, "
                  "char_order=6, beta=2, whitespace removed. The cross-check "
                  "is optional by design: this gate must run with no "
                  "dependencies.)")
        else:
            d = abs(sb - corpus)
            verdict = ("ok" if d < 0.01 else
                       "<- IMPLEMENTATIONS DISAGREE. Trust sacrebleu and fix "
                       "chrf_score(); do not adjust the threshold.")
            print(f"       sacrebleu says {sb:.2f}, delta {d:.4f}  {verdict}")
            if d >= 0.01:
                n_fail += 1
        if args.allow_id_drift:
            if corpus < args.min_chrf:
                print(f"  FAIL corpus chrF {corpus:.2f} < {args.min_chrf:.1f}")
                n_fail += 1
            if n_identical == len(seg_stats):
                print("       every segment is byte-identical, so "
                      "--allow-id-drift bought nothing on this\n"
                      "       run. That is a good outcome and a reason to drop "
                      "the flag: assert the\n"
                      "       stronger claim while it is still true.")

    print()
    if n_fail:
        print(f"FAIL - {n_fail} check(s)")
        return 1
    if args.allow_id_drift:
        print("PASS - chrF held. NOT a claim of byte equality with HuggingFace.")
    else:
        print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
