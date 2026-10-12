#!/usr/bin/env bash
# Step 10, section 7.0 - the MEASUREMENT, on the real checkpoint.
#
# Answers one question before any C++ gets written: what does Q8_0 do to the
# per-decision headroom that step 7 left at 1.50x?
#
# Needs: an f32 and an f16 GGUF already converted (step 1 / step 7), ref/ with
# manifest_gen.txt (step 4), and a built build/bin. No torch, no downloads.
#
#   ./scripts/step10-measure.sh                 # 8 threads, models/, ref/
#   NT=4 MODELS=/data/m ./scripts/step10-measure.sh
set -euo pipefail

NT="${NT:-8}"
MODELS="${MODELS:-models}"
REF="${REF:-ref}"
BIN="${BIN:-build/bin}"
PY="${PY:-python}"

F32="$MODELS/nllb-600M-f32.gguf"
F16="$MODELS/nllb-600M-f16.gguf"
Q8E="$MODELS/nllb-600M-q8emb.gguf"     # token_embd Q8_0, everything else f16
Q8A="$MODELS/nllb-600M-q8all.gguf"     # every 2-D weight Q8_0

for f in "$F32" "$F16"; do
    [ -f "$f" ] || { echo "missing $f - convert it first"; exit 2; }
done
[ -f "$REF/manifest_gen.txt" ] || { echo "missing $REF/manifest_gen.txt"; exit 2; }

hr() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. build
# Two mixes. Both start from the f16 file so the ONLY variable against the
# step 7 baseline is which tensors became Q8_0.
hr "1/5  building the quantised files"
[ -f "$Q8E" ] || $PY models/convert-nllb-to-gguf.py "$F16" "$Q8E" \
    --quantize 'token_embd.weight' --type q8_0
[ -f "$Q8A" ] || $PY models/convert-nllb-to-gguf.py "$F16" "$Q8A" \
    --quantize '*' --min-rank 2 --type q8_0
ls -la "$F32" "$F16" "$Q8E" "$Q8A"

# ---------------------------------------------------------------- 2. tokenizer
# Requantising REWRITES the file, and since step 8 the tokenizer lives in it.
# Trap 32: a dropped charsmap or merge hash makes nllb_vocab_load refuse the
# file, so if this goes red nothing below is worth reading.
#
# --list-langs is the cheap version of that assertion and the right one to run
# first: it opens the GGUF no_alloc=true and reads metadata only, so it costs
# the same on a 1.24 GB file as on a 6.7 MB one, and it fails loudly on exactly
# the keys the rewrite could have lost.
hr "2/5  the tokenizer survived the rewrite"
n_langs=$("$BIN/nllb-cli" -m "$Q8A" --list-langs 2>/dev/null | grep -c . || true)
if [ "${NO_VOCAB:-0}" = "1" ]; then
    # tests/make_shape_model.py writes geometry and no vocabulary, so this pass
    # cannot run there. Set NO_VOCAB=1 to dry-run the rest of the script in the
    # sandbox; never set it against a real checkpoint.
    echo "  NO_VOCAB=1: skipped (--list-langs returned $n_langs). This run"
    echo "  proves NOTHING about trap 32."
elif [ "$n_langs" -lt 200 ]; then
    echo "FAIL: --list-langs returned $n_langs codes, expected 202."
    echo "      The requantised file lost tokenizer metadata. Trap 32."
    exit 1
fi
echo "  --list-langs: $n_langs FLORES codes out of the requantised file   ok"

# The full gate too, but only when its goldens are actually present. It treats
# a missing corpus as a FAILURE by design (the step-4 SKIP lesson), which is
# right for the tokenizer's own step and wrong as a gate on THIS one.
if [ -f "$REF/tokenizer_corpus.txt" ] && [ -f "$REF/tokenizer_sweep.txt" ]; then
    "$BIN/nllb-test-tokenizer" "$Q8A" "$REF"
else
    echo "  (skipping nllb-test-tokenizer: no tokenizer_corpus.txt / _sweep.txt"
    echo "   in $REF. Dump them with scripts/dump_tokenizer_reference.py to get"
    echo "   the 237798-case check as well.)"
fi

# ---------------------------------------------------------------- 3. headroom
# THE MEASUREMENT. f32 is the reference in all three so the numbers form a
# ladder. Read the RATIOS between the three tightest-headroom lines, not the
# absolute values.
#
# THE f16 RUN KEEPS THE OLD ASSERTION. Exact id equality held there in step 7
# and it should still hold; a run that permits drift where drift does not happen
# is a run asserting less than the truth, which is how a regression gets in.
hr "3/6  headroom ladder - f16 baseline (exact ids ASSERTED)"
"$BIN/nllb-test-precision" "$F32" "$F16" "$REF" -t "$NT"

# THE TWO Q8 RUNS ASK A DIFFERENT QUESTION and say so on the command line.
# --allow-id-drift swaps the assertion from exact ids to [topk]: the f32 argmax
# must stay inside the quantised top 4. That is trap 30's replacement, and it is
# a REAL assertion - these two lines have no `|| true`, so a red here is a
# failure again rather than a shrug. What the run reports and does not gate is
# the id drift itself, which is the measurement.
hr "3/6  headroom ladder - Q8_0 on token_embd only (assertion: [topk])"
"$BIN/nllb-test-precision" "$F32" "$Q8E" "$REF" -t "$NT" --allow-id-drift
hr "3/6  headroom ladder - Q8_0 on every 2-D weight (assertion: [topk])"
"$BIN/nllb-test-precision" "$F32" "$Q8A" "$REF" -t "$NT" --allow-id-drift

# Note what changed since the first version of this script: it used to run all
# three with `|| true` and call the red "the measurement". That was right while
# there was no other assertion to make. There is one now, so the red is back to
# meaning what red means. Do not put `|| true` back - see trap 19.

# ---------------------------------------------------------------- 3b. KV
# The other precision knob, and the one only the RUN can set. Same file both
# sides; the weights are the constant and the cache is the variable.
hr "3b/6  what an f16 KV cache costs (same file, --kv f32:f16)"
"$BIN/nllb-test-precision" "$F16" "$F16" "$REF" -t "$NT" --kv f32:f16 \
    --allow-id-drift

# ---------------------------------------------------------------- 4. cache
# full = exact must hold at ANY weight type: same sequence length, both sides
# read the same bytes, so nothing can regroup. If this fails it is a real bug.
hr "4/6  full = exact must still hold at Q8_0"
"$BIN/nllb-test-cache" "$Q8A" "$REF" -t "$NT"

# And with the f16 KV pass added. [full] is still asserted bit-identical at f32
# KV; [kv16] is the same single cached call with K/V at f16 and is MEASURED,
# never gated. Red at [full] is a cache bug, a number at [kv16] is f16. Trap 37.
hr "4b/6  the f16 KV cache, measured against the bit-identical f32 one"
"$BIN/nllb-test-cache" "$F16" "$REF" -t "$NT" --kv-f16

# ---------------------------------------------------------------- 5. speed
hr "5/6  where the step goes, at each precision"
for m in "$F16" "$Q8E" "$Q8A"; do
    printf '\n--- %s ---\n' "$m"
    "$BIN/nllb-bench" "$m" -t "$NT" --src 24 --steps 64 | tail -12
done
printf '\n--- %s, KV f16 ---\n' "$F16"
"$BIN/nllb-bench" "$F16" -t "$NT" --src 24 --steps 64 --kv-f16 | tail -12

# ---------------------------------------------------------------- 6. product
# The gate that speaks in translations rather than in floats. Byte equality with
# HuggingFace at f16 (still the assertion there), chrF at Q8_0.
#
# Needs ref/translations.txt from scripts/dump_translation_reference.py, which is
# the only dump in this project that needs torch. Skipped rather than failed if
# it is absent, because THIS script is not the tokenizer step and a missing
# golden here means "you have not dumped it", not "the CLI is broken".
if [ -f "$REF/translations.txt" ] && [ "${NO_VOCAB:-0}" != "1" ]; then
    hr "6/6  text -> text: byte equality at f16, chrF at Q8_0"
    $PY tests/verify_translation.py "$BIN/nllb-cli" "$F16" "$REF" -t "$NT"
    $PY tests/verify_translation.py "$BIN/nllb-cli" "$Q8E" "$REF" -t "$NT" \
        --allow-id-drift
    $PY tests/verify_translation.py "$BIN/nllb-cli" "$Q8A" "$REF" -t "$NT" \
        --allow-id-drift
else
    hr "6/6  text -> text  SKIPPED"
    echo "  no $REF/translations.txt (or NO_VOCAB=1). Dump it with"
    echo "  scripts/dump_translation_reference.py to get the chrF numbers, which"
    echo "  are the only ones in this script a user would recognise."
fi

cat <<'EOF'

-------------------------------------------------------------------------
WHAT TO RECORD, in this order:

  1. `tightest headroom` from each of the three runs in step 3.  The RATIO
     f16 : q8emb : q8all is the transferable number.  Sandbox measured
     38.5x : 11.8x : 1.46x, i.e. Q8_0 costs 3.3x on token_embd alone and
     26.4x on everything.  Your f16 figure should be ~1.50x, so the
     predictions to falsify are ~0.45x and ~0.06x.

  2. The [topk] block from each of the two Q8 runs: the rank histogram, the
     worst rank, and how many decisions moved the argmax.  This is the
     number that replaces exact id equality, so it is the one a quantised
     default gets shipped on.  Read it NEXT TO the headroom, not instead of
     it: every rank 0 with headroom below 1 means the decisions were coin
     tosses that landed the same way, which is a weaker claim than it looks.

  3. Whether `ids=same` held.  Now reported rather than gated on the Q8 runs.

  4. `full=exact` and `0 differing floats` from step 4.  Anything else is a
     bug, not a precision fact.  And from 4b, the `[kv16]` relative number
     next to it - that one is measured and never gated.

  5. `cached step` median and the lm_head's share from step 5, at all four
     configurations including KV f16.  The sandbox is not bandwidth-bound
     and read 1.21x / 1.47x against 1.36x / 1.88x predicted; the M1 should
     land much closer to the predictions.  Expect the KV f16 line to move
     the step by single-digit percent at most: at 24 source tokens the
     cache is a small share of the bytes read and the lm_head is 54-65%.

  6. The `[chrF]` line from each run in step 6.  corpus chrF and how many
     segments were byte-identical.  This is the only number in the script a
     user of the translator would recognise.

WHAT THIS SCRIPT STILL DOES NOT DO: any K-quant.  Build one with
  ./build/bin/nllb-quantize models/nllb-600M-f16.gguf \
      models/nllb-600M-q5km.gguf q5_k_m
and put it through the same three gates.  Prefer converting from the f32
file when you have it - quantising the f16 file has already lost the
mantissa bits the quantiser would have used to choose its scales.
-------------------------------------------------------------------------
EOF
