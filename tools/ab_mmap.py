#!/usr/bin/env python3
"""
ab_mmap.py - decide mmap vs read WITHOUT the ordering artefact.

    python tools/ab_mmap.py models/nllb-600M-f16.gguf --reps 7

Replaces tools/ab_mmap.sh, which was WRONG: it shelled out to `python3 -c` for
each timestamp, so it compared time.perf_counter() across two separate
processes - whose epochs need not be comparable - and paid interpreter startup
on both sides. It reported NEGATIVE durations and whole translations finishing
in 2 ms. Time a subprocess from inside one process, or not at all.

WHY INTERLEAVE. Two sequential runs cannot answer this: whichever goes second
inherits a page cache the first one warmed, so the answer flips with the order.
Measured on an M1 Air, same file, minutes apart - mmap-then-read said mmap
loses by 1.4 s, read-then-mmap said mmap wins by 0.4 s. So run A B A B ...,
drop the first pair, and take MEDIANS: one scheduler hiccup moves a mean and
not a median. Same shape as the interleaved arena comparison in
tests/bench_cached.cpp, and for the same reason (trap 17).

WHAT IS TIMED is total wall clock of the whole invocation - load, generate,
exit. `load:` alone flatters mmap, because mmap's cost reappears inside
generate as page faults. Only the total is honest.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_one(cmd: list[str], stdin_bytes: bytes) -> float:
    """Wall-clock ms for one full invocation, timed from this process."""
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=stdin_bytes,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        raise SystemExit(f"\n{' '.join(cmd)}\nexited {p.returncode} - "
                         f"fix that before trusting any timing")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--input", type=Path, default=Path("input.txt"))
    ap.add_argument("--bin", default="./build/bin/nllb-cli")
    ap.add_argument("--src", default="deu_Latn")
    ap.add_argument("--tgt", default="eng_Latn")
    ap.add_argument("-t", "--threads", type=int, default=8)
    a = ap.parse_args()

    if not a.model.exists(): raise SystemExit(f"no such model: {a.model}")
    if not a.input.exists(): raise SystemExit(f"no such input: {a.input}")
    text = a.input.read_bytes()

    base = [a.bin, "-m", str(a.model), "--src", a.src, "--tgt", a.tgt,
            "-t", str(a.threads)]
    variants = [("mmap", base), ("read", base + ["--no-mmap"])]

    print(f"model:   {a.model}  ({a.model.stat().st_size / 1e9:.3f} GB)")
    print(f"reps:    {a.reps} (first pair discarded)   threads: {a.threads}")
    print("interleaved A/B, total wall clock per invocation in ms\n")
    print(f"{'rep':<5}{'mmap':>12}{'read':>12}")

    got: dict[str, list[float]] = {"mmap": [], "read": []}
    for i in range(1, a.reps + 1):
        row = {name: run_one(cmd, text) for name, cmd in variants}
        tag = "   (warm-up, discarded)" if i == 1 else ""
        print(f"{i:<5}{row['mmap']:12.1f}{row['read']:12.1f}{tag}")
        if i > 1:
            for k, v in row.items():
                got[k].append(v)

    if not got["mmap"]:
        raise SystemExit("not enough reps - pass --reps 3 or more")

    med = {k: statistics.median(v) for k, v in got.items()}
    spread = {k: max(v) - min(v) for k, v in got.items()}
    print(f"\nmedian   mmap {med['mmap']:8.1f} ms   read {med['read']:8.1f} ms")
    print(f"spread   mmap {spread['mmap']:8.1f} ms   read {spread['read']:8.1f} ms")

    delta = med["read"] - med["mmap"]
    worst = max(spread.values())
    if abs(delta) < worst:
        print(f"\nINSIDE THE NOISE. The medians differ by {abs(delta):.1f} ms and the\n"
              f"worst spread is {worst:.1f} ms, so this run does not distinguish the\n"
              f"two. Report that, not the ratio.")
    else:
        win = "mmap" if delta > 0 else "read"
        print(f"\n{win} wins by {abs(delta):.1f} ms ({med['read'] / med['mmap']:.2f}x),\n"
              f"which exceeds the {worst:.1f} ms spread.")

    print("\nBoth columns are WARM: every rep reads a file the previous rep left in\n"
          "the page cache, which is the steady state a user lives in. The cold case\n"
          "cannot be interleaved - purging is global - so collect it separately with\n"
          "`sudo purge` before each single run, and treat it as weaker evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
