#!/usr/bin/env python3
"""Break the ggml-backend execution path on purpose, to find out whether the gates notice.

Every graph runs through ggml_backend_sched on gallocr, which has its own
failure modes: a tensor the host reads back but never marked as an output (gallocr hands
its memory to something later), an input written through ->data instead of
through the backend, a causal mask that is now DATA rather than an op. Each
mutation below is one of those, written the way it would plausibly be written.

    python tests/corrupt_backend.py build models/small-f32.gguf ref-small/
    NLLB_BACKEND=cuda0 python tests/corrupt_backend.py build ...   # a device

Compiles each mutated src/nllb.cpp into /tmp (never touching your tree) with
tests/test-cache.cpp and tests/dump-states.cpp, then runs two instruments:

  cache  nllb-test-cache: cached vs un-cached, [full] [step] [chunk] [arena]
  dump   nllb-dump-states --diff against the UNMUTATED source built the same way

`expect` says what a correct gate must do: `fire` means nllb-test-cache must
go red; `inert` means nothing may move, and is asserted so that a change in
behaviour is noticed.

Two mutations are REPORTED, not asserted, and that is a finding rather than a
hole. Writing an input through ->data and leaving an input unflagged both work
on the CPU by accident - its buffers are host memory - and they very probably
work on an M1 too: ggml_backend_sched places every graph input in the CPU
backend's buffer and copies it across, and Apple silicon's Metal buffers are
shared memory anyway. The machines where they break are discrete GPUs, which
nothing in this project runs on. No behavioural gate available here can see
the host-pointer class, so it is enforced by reading the source instead: the
ci.yml step that refuses any `->data` in src/ and cli/.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CACHED_HS = "        for (ggml_tensor * t : hs) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }"

# (name, what it models, expectation on the CPU, [(before, after, occurrence)])
# occurrence: None = the text must be unique; k = replace the k-th (0-based).
MUTATIONS = [
    ("no-output-states",
     "forget ggml_set_output on the cached step's hidden states - gallocr "
     "then reuses their memory for later tensors",
     "fire",
     [(CACHED_HS, "        for (ggml_tensor * t : hs) { ggml_build_forward_expand(gf, t); }", 2)]),

    ("no-output-final",
     "forget ggml_set_output on the cached step's final state (dec_out). "
     "INERT, and asserted to stay so: it is the last node before the lm_head "
     "and nothing allocated after it can reuse its memory. If this ever fires, "
     "the graph order changed - and every output that is only safe by position "
     "should be re-checked.",
     "inert",
     [("    ggml_set_name(cur, \"dec_out\");\n    ggml_set_output(cur);\n    hs.push_back(cur);\n\n    ggml_tensor * logits",
       "    ggml_set_name(cur, \"dec_out\");\n    hs.push_back(cur);\n\n    ggml_tensor * logits", None)]),

    ("mask-too-large",
     "causal mask offset one too LARGE: invisible at one token per step",
     "fire",
     [("for (int i = n_past + j + 1; i < n_kv; ++i) {",
       "for (int i = n_past + j + 2; i < n_kv; ++i) {", None)]),

    ("mask-too-small",
     "causal mask offset one too SMALL: a token cannot see itself",
     "fire",
     [("for (int i = n_past + j + 1; i < n_kv; ++i) {",
       "for (int i = n_past + j; i < n_kv; ++i) {", None)]),

    ("kq-before-write",
     "expand the real kq node before the cache write",
     "fire",
     [("        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kp, k_dst));\n"
       "        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vp, v_dst));\n", "", None),
      ("        kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f, 0.0f);\n"
       "        ggml_tensor * kqv = ggml_mul_mat(ctx, Vv, kq);              // (HD, Tn, H)\n",
       "        kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f, 0.0f);\n"
       "        ggml_build_forward_expand(gf, kq);\n"
       "        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kp, k_dst));\n"
       "        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vp, v_dst));\n"
       "        ggml_tensor * kqv = ggml_mul_mat(ctx, Vv, kq);              // (HD, Tn, H)\n", None)]),

    ("mask-through-data",
     "write the cached step's mask through ->data instead of "
     "ggml_backend_tensor_set",
     "report",
     [("        ggml_backend_tensor_set(kq_mask, mask.data(),    0, mask.size() * sizeof(float));\n"
       "        ok = nllb_compute_graph(model, sched, gf, n_threads, \"cached decoder\", err);",
       "        if (kq_mask->data) std::memcpy(kq_mask->data, mask.data(), mask.size() * sizeof(float));\n"
       "        ok = nllb_compute_graph(model, sched, gf, n_threads, \"cached decoder\", err);", None)]),

    ("mask-not-input",
     "forget ggml_set_input on the cached step's mask",
     "report",
     [("    ggml_set_name(kq_mask, \"kq_mask\");\n    ggml_set_input(kq_mask);\n    std::vector<float> mask;\n    nllb_causal_mask(mask, n_kv, Tn, n_past);",
       "    ggml_set_name(kq_mask, \"kq_mask\");\n    std::vector<float> mask;\n    nllb_causal_mask(mask, n_kv, Tn, n_past);", None)]),
]


def mutate(src: str, edits) -> str:
    for before, after, occ in edits:
        n = src.count(before)
        if occ is None:
            if n != 1:
                sys.exit(f"mutation anchor found {n} times, expected 1:\n{before[:200]}")
            src = src.replace(before, after)
        else:
            if n <= occ:
                sys.exit(f"mutation anchor found {n} times, need occurrence {occ}:\n{before[:200]}")
            idx = -1
            for _ in range(occ + 1):
                idx = src.index(before, idx + 1)
            src = src[:idx] + after + src[idx + len(before):]
    return src


def build(tmp: Path, build_dir: Path, src_text: str, tag: str) -> tuple[Path, Path]:
    lib = None
    for p in build_dir.rglob("libggml-base.*"):
        lib = p.parent
        break
    if lib is None:
        sys.exit(f"no libggml-base under {build_dir}; configure and build first")
    d = tmp / tag
    d.mkdir()
    (d / "nllb.cpp").write_text(src_text)
    shutil.copy(ROOT / "include/nllb.h", d / "nllb.h")
    common = ["g++", "-std=c++17", "-O2", "-w", f"-I{d}",
              f"-I{ROOT / 'ggml/include'}"]
    libs = [f"-L{lib}", "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{lib}"]
    obj = d / "nllb.o"
    r = subprocess.run(common + ["-c", str(d / "nllb.cpp"), "-o", str(obj)],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        # A mutation that does not compile is a broken HARNESS, not a caught bug.
        sys.exit(f"compile of {tag} failed - fix the mutation, do not count it")
    outs = []
    for name, src in (("cache", "tests/test-cache.cpp"), ("dump", "tests/dump-states.cpp")):
        exe = d / name
        r = subprocess.run(common + ["-o", str(exe), str(ROOT / src), str(obj)] + libs,
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:])
            sys.exit(f"link of {tag}/{name} failed")
        outs.append(exe)
    return outs[0], outs[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("gguf")
    ap.add_argument("ref")
    ap.add_argument("-t", type=int, default=2)
    ap.add_argument("--only")
    args = ap.parse_args()

    backend = os.environ.get("NLLB_BACKEND", "cpu")
    src = (ROOT / "src/nllb.cpp").read_text()

    with tempfile.TemporaryDirectory(prefix="corrupt_backend_") as t:
        tmp = Path(t)
        base_cache, base_dump = build(tmp, args.build_dir, src, "baseline")
        base_bin = tmp / "baseline.bin"
        r = subprocess.run([str(base_dump), args.gguf, args.ref, str(base_bin), "-t", str(args.t)],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2000:])
            sys.exit("the UNMUTATED build failed to dump - nothing below would mean anything")
        r = subprocess.run([str(base_cache), args.gguf, args.ref, "-t", str(args.t)],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            sys.exit("the UNMUTATED build fails the cache gate - fix that first")

        print(f"backend: {backend}   baseline: cache gate PASS, dump written\n")
        print(f"{'mutation':22s} {'cache gate':30s} {'dump diff':22s} expected       verdict")
        print("-" * 104)
        n_bad = 0
        for name, what, expect, edits in MUTATIONS:
            if args.only and name != args.only:
                continue
            cache, dump = build(tmp, args.build_dir, mutate(src, edits), name)

            rc = subprocess.run([str(cache), args.gguf, args.ref, "-t", str(args.t)],
                                capture_output=True, text=True)
            cache_fired = rc.returncode != 0
            # Which passes fired, by the per-case column text the gate prints.
            fired = []
            for line in rc.stdout.splitlines():
                m = re.match(r"^\s+\S+\s+\[(\w+)\]", line)
                if m and m.group(1) not in fired:
                    fired.append(m.group(1))
            if "full=DIFFERS" in rc.stdout and "full" not in fired:
                fired.insert(0, "full")
            if rc.returncode not in (0, 1):
                fired.append(f"rc={rc.returncode}")   # a crash is a firing, and says so
            cache_txt = ("FAIL " + ",".join(fired)) if cache_fired else "pass"

            mbin = tmp / f"{name}.bin"
            rd = subprocess.run([str(dump), args.gguf, args.ref, str(mbin), "-t", str(args.t)],
                                capture_output=True, text=True)
            if rd.returncode != 0:
                dump_txt, dump_fired = f"dump rc={rd.returncode}", True
            else:
                rdiff = subprocess.run([str(dump), "--diff", str(base_bin), str(mbin)],
                                       capture_output=True, text=True)
                last = rdiff.stdout.strip().splitlines()[-1] if rdiff.stdout.strip() else ""
                dump_fired = rdiff.returncode != 0
                m = re.search(r"(\d+) of (\d+) records differ", last)
                dump_txt = (f"{m.group(1)}/{m.group(2)} differ" if m
                            else ("identical" if last.startswith("IDENTICAL") else last[:22]))
            any_fired = cache_fired or dump_fired

            # The in-tree GATE is what must fire. The dump diff needs a second
            # build to compare against, so it is evidence, not the assertion.
            if expect == "fire":
                ok = cache_fired
            elif expect == "inert":
                ok = not any_fired
            else:   # "report": printed, never asserted - see the docstring
                ok = True
            n_bad += 0 if ok else 1
            print(f"{name:22s} {cache_txt[:30]:30s} {dump_txt[:22]:22s} {expect:14s} "
                  f"{'ok' if ok else 'UNEXPECTED'}")
        print()
        if n_bad:
            print(f"FAIL - {n_bad} mutation(s) did not behave as expected")
            return 1
        print("PASS - every mutation behaved as expected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
