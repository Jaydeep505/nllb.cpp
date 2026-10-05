#!/usr/bin/env python3
"""Does nllb-test-batch actually fire? Break the BATCHING source on purpose.

Each mutation is a plausible wrong implementation of one line of the batched
path in src/nllb.cpp, not a random edit. For each: patch the source, rebuild
nllb-test-batch with the project's own CMake (same flags as the real build),
run the gate, record which passes said FAIL. The source is restored afterwards
- in a `finally`, and from a backup file if the process is killed mid-run.

    python tests/corrupt_batch.py build MODEL.gguf [-t N] [--only a,b] [other nllb-test-batch args]

Run it at BOTH precisions and compare the tables. A gate whose matrix changes
with the weight type means something different at each one (step 11's lesson).

Two mutations are EXPECTED to be inert and are asserted to stay inert (trap 43):
a finished row's output is discarded and its ne[3] slice is computed
independently of every other row, so how it is fed cannot reach a running row.
If either ever fires, something has coupled the rows.

Harness rules from the four harness bugs in claude/handoff.md: sections are
found by "[" at the START of a line; a build failure is fatal, not a "missed";
a crash or an unexpected exit code counts as firing; and the baseline must be
green or nothing below it means anything.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("src/nllb.cpp")

# name -> (old, new, passes expected to fire (at least), expect_inert)
MUTATIONS = {
    # The mask tensor is built and uploaded, but masks NOTHING - the quiet form
    # of "forgot the mask". (Passing nullptr instead drops the tensor from the
    # graph and the upload then asserts: loud, and not what a real bug looks like.)
    "no-cross-mask": (
        "for (int k = len[b]; k < Ts; ++k) m[(size_t) b * Ts + k] = -INFINITY;",
        "for (int k = len[b]; k < Ts; ++k) m[(size_t) b * Ts + k] = 0.0f;",
        {"mask"}, False),
    "no-encoder-mask": (
        "for (int k = len[b]; k < Ts; ++k) row[k] = -INFINITY;",
        "for (int k = len[b]; k < Ts; ++k) row[k] = 0.0f;",
        {"mask"}, False),
    "cross-mask-off-by-one": (
        "for (int k = len[b]; k < Ts; ++k) m[(size_t) b * Ts + k] = -INFINITY;",
        "for (int k = len[b] + 1; k < Ts; ++k) m[(size_t) b * Ts + k] = -INFINITY;",
        {"mask"}, False),
    "encoder-mask-off-by-one": (
        "for (int k = len[b]; k < Ts; ++k) row[k] = -INFINITY;",
        "for (int k = len[b] + 1; k < Ts; ++k) row[k] = -INFINITY;",
        {"mask"}, False),
    "mask-from-row-zero": (
        "for (int k = len[b]; k < Ts; ++k) m[(size_t) b * Ts + k] = -INFINITY;",
        "for (int k = len[0]; k < Ts; ++k) m[(size_t) b * Ts + k] = -INFINITY;",
        {"perm"}, False),
    "kv-write-batch-stride": (
        "kc->nb[1], kc->nb[2], kc->nb[3], (size_t) n_past * kc->nb[1]);",
        "kc->nb[1], kc->nb[2], kc->nb[2], (size_t) n_past * kc->nb[1]);",
        {"equal"}, False),
    "argmax-from-row-zero": (
        "const float * p = dec.logits.data() + (size_t) b * dec.n_vocab;",
        "const float * p = dec.logits.data();",
        {"pad"}, False),
    "never-finish": (
        "            if (s.token == eos) {\n                r.stop = NLLB_STOP_EOS;",
        "            if (false && s.token == eos) {\n                r.stop = NLLB_STOP_EOS;",
        {"eos"}, False),
    "pos-ignores-pad": (
        "pos[b] = (new_ids[b] == hp.pad_id) ? hp.pad_id : n_past + 1 + hp.pad_id;",
        "pos[b] = n_past + 1 + hp.pad_id;",
        set(), True),
    "feed-last-not-pad": (
        "for (int b = 0; b < B; ++b) feed[b] = done[b] ? hp.pad_id : out.seq[b].ids.back();",
        "for (int b = 0; b < B; ++b) feed[b] = out.seq[b].ids.back();",
        set(), True),
}

PASSES = ["plan", "b1", "equal", "pad", "perm", "mask", "eos"]


def to_file_newlines(s: str, crlf: bool) -> str:
    return s.replace("\n", "\r\n") if crlf else s


def build(build_dir: str) -> None:
    r = subprocess.run(["cmake", "--build", build_dir, "--target", "nllb-test-batch", "-j"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit("build failed - a mutation that does not compile is a harness bug, not a catch")


def run_gate(exe: Path, model: str, extra: list) -> tuple:
    r = subprocess.run([str(exe), model] + extra, capture_output=True, text=True, timeout=1800)
    fired = set()
    for line in r.stdout.splitlines():
        m = re.match(r"^\[(\w+)\]\s+FAIL", line)   # "[" at the START of a line only
        if m:
            fired.add(m.group(1))
    crashed = r.returncode not in (0, 1)
    return fired, crashed, r.returncode, r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("model")
    ap.add_argument("--only", help="comma-separated mutation names")
    # Anything not recognised here (e.g. -t 4, --rows 8) is passed to
    # nllb-test-batch, in whatever position it was given.
    a, extra = ap.parse_known_args()
    a.extra = extra

    if not SRC.exists():
        raise SystemExit("run from the repo root")
    exe = Path(a.build_dir) / "bin" / "nllb-test-batch"
    original = SRC.read_bytes()
    crlf = b"\r\n" in original
    backup = SRC.with_suffix(".cpp.corrupt_batch_backup")
    if backup.exists():
        raise SystemExit(f"{backup} exists: a previous run was killed. Restore it with\n"
                         f"  mv {backup} {SRC}\nand run again.")

    names = list(MUTATIONS) if not a.only else a.only.split(",")
    text = original.decode("utf-8")

    # Every mutation must find its line exactly once in the CURRENT source.
    for n in names:
        old = to_file_newlines(MUTATIONS[n][0], crlf)
        if text.count(old) != 1:
            raise SystemExit(f"mutation {n}: expected 1 match in {SRC}, found {text.count(old)} - "
                             "the source moved; update the mutation")

    rows = []
    shutil.copyfile(SRC, backup)
    try:
        build(a.build_dir)
        fired, crashed, rc, out = run_gate(exe, a.model, a.extra)
        if fired or crashed:
            sys.stdout.write(out)
            raise SystemExit("baseline is not green; fix that first")
        print(f"baseline: PASS ({a.model})\n")

        for n in names:
            old, new, expect, inert = MUTATIONS[n]
            SRC.write_bytes(text.replace(to_file_newlines(old, crlf), to_file_newlines(new, crlf)).encode("utf-8"))
            build(a.build_dir)
            fired, crashed, rc, out = run_gate(exe, a.model, a.extra)
            if crashed:
                fired = fired | {f"crash(rc={rc})"}
            if inert:
                ok = not fired
            else:
                ok = bool(fired) and expect <= fired
            rows.append((n, fired, expect, inert, ok))
            print(f"  {n:24s} {'inert (expected)' if inert and not fired else ' '.join(sorted(fired)) or 'MISSED':40s} {'ok' if ok else 'UNEXPECTED'}")
    finally:
        SRC.write_bytes(original)
        backup.unlink(missing_ok=True)
        build(a.build_dir)   # leave the real binary behind, not the last mutant

    print()
    header = f"{'mutation':24s} " + " ".join(f"{p:5s}" for p in PASSES)
    print(header)
    print("-" * len(header))
    for n, fired, expect, inert, ok in rows:
        cells = " ".join(f"{('FAIL' if p in fired else ''):5s}" for p in PASSES)
        tag = "  (inert, expected)" if inert else ""
        print(f"{n:24s} {cells}{tag}{'' if ok else '   <- UNEXPECTED'}")
    bad = [r for r in rows if not r[4]]
    print()
    if bad:
        print(f"FAIL - {len(bad)} of {len(rows)} mutation(s) did not behave as expected")
        return 1
    print(f"PASS - {len(rows)} mutations behaved as expected "
          f"({sum(1 for r in rows if not r[3])} caught, {sum(1 for r in rows if r[3])} inert by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
