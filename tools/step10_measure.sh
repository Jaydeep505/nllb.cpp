#!/usr/bin/env bash
# Step 10, section 7.0 - the MEASUREMENT, on the real checkpoint.
#
# Answers one question before any C++ gets written: what does Q8_0 do to the
# per-decision headroom that step 7 left at 1.50x?
#
# Needs: an f32 and an f16 GGUF already converted (step 1 / step 7), ref/ with
# manifest_gen.txt (step 4), and a built build/bin. No torch, no downloads.
#
#   ./tools/step10_measure.sh                 # 8 threads, models/, ref/
#   NT=4 MODELS=/data/m ./tools/step10_measure.sh
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
[ -f "$Q8E" ] || $PY tools/requantize_gguf.py "$F16" "$Q8E" \
    --quantize 'token_embd.weight' --type q8_0
[ -f "$Q8A" ] || $PY tools/requantize_gguf.py "$F16" "$Q8A" \
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
    echo "   in $REF. Dump them with tools/dump_tokenizer_reference.py to get"
    echo "   the 237798-case check as well.)"
fi

# ---------------------------------------------------------------- 3. headroom
# THE MEASUREMENT. f32 is the reference in all three so the numbers form a
# ladder. Read the RATIOS between the three tightest-headroom lines, not the
# absolute values.
hr "3/5  headroom ladder - f16 baseline"
"$BIN/nllb-test-precision" "$F32" "$F16" "$REF" -t "$NT" || true
hr "3/5  headroom ladder - Q8_0 on token_embd only"
"$BIN/nllb-test-precision" "$F32" "$Q8E" "$REF" -t "$NT" || true
hr "3/5  headroom ladder - Q8_0 on every 2-D weight"
"$BIN/nllb-test-precision" "$F32" "$Q8A" "$REF" -t "$NT" || true

# `|| true` above is deliberate and is the whole point of section 7.0: at Q8_0
# the ids are EXPECTED to move, and this gate's assertion is exact id equality,
# so a non-zero exit here is data, not a defect. Do not "fix" it by loosening
# the gate - see trap 19 and section 7.3.

# ---------------------------------------------------------------- 4. cache
# full = exact must hold at ANY weight type: same sequence length, both sides
# read the same bytes, so nothing can regroup. If this fails it is a real bug.
hr "4/5  full = exact must still hold at Q8_0"
"$BIN/nllb-test-cache" "$Q8A" "$REF" -t "$NT"

# ---------------------------------------------------------------- 5. speed
hr "5/5  where the step goes, at each precision"
for m in "$F16" "$Q8E" "$Q8A"; do
    printf '\n--- %s ---\n' "$m"
    "$BIN/nllb-bench" "$m" -t "$NT" --src 24 --steps 64 | tail -12
done

cat <<'EOF'

-------------------------------------------------------------------------
WHAT TO RECORD, in this order:

  1. `tightest headroom` from each of the three runs in step 3.  The RATIO
     f16 : q8emb : q8all is the transferable number.  Sandbox measured
     38.5x : 11.8x : 1.46x, i.e. Q8_0 costs 3.3x on token_embd alone and
     26.4x on everything.  Your f16 figure should be ~1.50x, so the
     predictions to falsify are ~0.45x and ~0.06x.

  2. Whether `ids=same` held.  Below 1.0x headroom the ids can still match -
     that is agreeing by luck, and it is exactly what section 7.3's new
     assertion has to replace.

  3. `full=exact` and `0 differing floats` from step 4.  Anything else is a
     bug, not a precision fact.

  4. `cached step` median and the lm_head's share from step 5.  The sandbox
     is not bandwidth-bound and read 1.21x / 1.47x against 1.36x / 1.88x
     predicted; the M1 should land much closer to the predictions.
-------------------------------------------------------------------------
EOF
