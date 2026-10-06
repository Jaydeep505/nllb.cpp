#!/usr/bin/env bash
#
# MAINTAINER ONLY. Build the GGUFs that models/download-nllb-model.sh serves,
# gate them, checksum them, write the model card, and (with --upload) push
# them to the Hugging Face repo.
#
#   ./tools/publish_models.sh models/nllb-600M                       # build + gate into dist/
#   ./tools/publish_models.sh models/nllb-600M --ref ref/            # + the text gate
#   ./tools/publish_models.sh models/nllb-600M --ref ref/ --upload   # + push
#
# Needs: a Release build (nllb-cli, nllb-quantize), the converter's python deps
# INCLUDING `regex` (without it the converter writes a file with no grapheme
# tables and says so in one line nobody reads - handoff, step 9), and for
# --upload the `hf` CLI logged in (`pip install -U huggingface_hub && hf auth login`).
#
# What ships, and why only these three: step 10's ladder. f16 is the only file
# with headroom > 1; q8emb is the only quantised mix that passed chrF; f32 is
# the reference every gate measures against. q8all and q5_k_m were REJECTED on
# measured quality and are deliberately not published.

set -euo pipefail

SNAP=""
REPO="${NLLB_HF_REPO:-JaydeepGupta/nllb-200-distilled-600M-GGUF}"
OUT="dist"
REF=""
UPLOAD=0
BIN="build/bin"
NT="${NT:-8}"
SOURCE_REV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)       REPO="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --ref)        REF="$2"; shift 2 ;;
        --bin)        BIN="$2"; shift 2 ;;
        --source-rev) SOURCE_REV="$2"; shift 2 ;;
        --upload)     UPLOAD=1; shift ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        -*)           echo "unknown flag $1" >&2; exit 2 ;;
        *)            SNAP="$1"; shift ;;
    esac
done
[[ -n "$SNAP" && -f "$SNAP/config.json" ]] || { echo "usage: $0 <hf-snapshot-dir> [--ref ref/] [--upload]" >&2; exit 2; }

for b in nllb-cli nllb-quantize; do
    [[ -x "$BIN/$b" ]] || { echo "missing $BIN/$b - build Release first" >&2; exit 1; }
done
python3 -c "import regex" 2>/dev/null || {
    echo "python 'regex' is not importable: the converter would write a GGUF with no" >&2
    echo "grapheme tables. pip install regex, then re-run." >&2
    exit 1
}

if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi

# The checkpoint revision the files were built from. snapshot_download(local_dir=)
# leaves one .metadata per file whose first line is the commit sha.
if [[ -z "$SOURCE_REV" ]]; then
    meta="$(ls "$SNAP"/.cache/huggingface/download/*.metadata 2>/dev/null | head -1 || true)"
    [[ -n "$meta" ]] && SOURCE_REV="$(head -1 "$meta")"
fi
SOURCE_REV="${SOURCE_REV:-unknown - pass --source-rev}"
NLLB_REV="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git status --porcelain -- src cli tools 2>/dev/null || true)" ]]; then
    NLLB_REV="${NLLB_REV}-dirty"
    echo "WARNING: src/ cli/ or tools/ has uncommitted changes - the card will say -dirty" >&2
fi

mkdir -p "$OUT"
F32="$OUT/nllb-600M-f32.gguf"
F16="$OUT/nllb-600M-f16.gguf"
Q8E="$OUT/nllb-600M-q8emb.gguf"

echo "== convert"
python3 tools/convert_nllb_to_gguf.py "$SNAP" --outfile "$F32" --outtype f32 | tee "$OUT/convert-f32.log"
python3 tools/convert_nllb_to_gguf.py "$SNAP" --outfile "$F16" --outtype f16 | tee "$OUT/convert-f16.log"
if grep -q "no grapheme tables" "$OUT"/convert-*.log; then
    echo "converter wrote a file without grapheme tables - refusing to publish" >&2; exit 1
fi

echo "== quantise (from f32, never from f16 - see QUANTIZE.md)"
"$BIN/nllb-quantize" "$F32" "$Q8E" q8emb

echo "== smoke: tokenizer survived, and every file translates"
for f in "$F32" "$F16" "$Q8E"; do
    n="$("$BIN/nllb-cli" -m "$f" --list-langs | wc -l | tr -d ' ')"
    [[ "$n" == 202 ]] || { echo "$f: --list-langs printed $n codes, expected 202" >&2; exit 1; }
    t="$("$BIN/nllb-cli" -m "$f" --backend cpu -t "$NT" --src eng_Latn --tgt deu_Latn "Hello, world!")"
    [[ -n "$t" ]] || { echo "$f: empty translation" >&2; exit 1; }
    printf '  %-26s 202 langs   "%s"\n' "$(basename "$f")" "$t"
done

if [[ -n "$REF" ]]; then
    echo "== the text gate (CPU, the reference backend)"
    python3 tests/verify_translation.py "$BIN/nllb-cli" "$F32" "$REF" -t "$NT" --backend cpu
    python3 tests/verify_translation.py "$BIN/nllb-cli" "$F16" "$REF" -t "$NT" --backend cpu
    python3 tests/verify_translation.py "$BIN/nllb-cli" "$Q8E" "$REF" -t "$NT" --backend cpu --allow-id-drift
    GATED="yes - tests/verify_translation.py against ref/ (f32, f16 exact; q8emb chrF)"
else
    echo "WARNING: no --ref; publishing on the smoke test alone" >&2
    GATED="smoke test only (--list-langs = 202, one translation)"
fi

echo "== checksums"
( cd "$OUT" && $SHA nllb-600M-f16.gguf nllb-600M-q8emb.gguf nllb-600M-f32.gguf > SHA256SUMS )
cat "$OUT/SHA256SUMS"

human() { awk -v b="$(wc -c < "$1" | tr -d ' ')" 'BEGIN { printf "%.2f GB", b/1e9 }'; }
sum_of() { awk -v f="$1" '$2 == f {print $1}' "$OUT/SHA256SUMS"; }

echo "== model card"
python3 - "$OUT/README.md" <<PY
import sys
card = open("tools/model-card.md").read()
for k, v in {
    "@REPO@": "$REPO",
    "@SOURCE_REV@": "$SOURCE_REV",
    "@NLLB_REV@": "$NLLB_REV",
    "@GATED@": "$GATED",
    "@DATE@": "$(date -u +%Y-%m-%d)",
    "@F16_SIZE@": "$(human "$F16")",   "@F16_SHA@": "$(sum_of nllb-600M-f16.gguf)",
    "@Q8E_SIZE@": "$(human "$Q8E")",   "@Q8E_SHA@": "$(sum_of nllb-600M-q8emb.gguf)",
    "@F32_SIZE@": "$(human "$F32")",   "@F32_SHA@": "$(sum_of nllb-600M-f32.gguf)",
}.items():
    card = card.replace(k, v)
assert "@" + "F16" not in card, "unfilled placeholder in model card"
open(sys.argv[1], "w").write(card)
PY
echo "  wrote $OUT/README.md"

if [[ "$UPLOAD" == 1 ]]; then
    command -v hf >/dev/null || { echo "need the hf CLI: pip install -U huggingface_hub" >&2; exit 1; }
    echo "== upload to $REPO"
    # `hf upload` creates a missing repo itself; this only makes the intent
    # explicit. `repos` in huggingface_hub >= 1.0, `repo` before it.
    hf repos create "$REPO" --repo-type model --exist-ok >/dev/null 2>&1 \
        || hf repo create "$REPO" --repo-type model --exist-ok >/dev/null 2>&1 || true
    hf upload "$REPO" "$OUT" . \
        --exclude "*.log" --exclude "*.part" \
        --commit-message "nllb.cpp ${NLLB_REV} from facebook/nllb-200-distilled-600M@${SOURCE_REV}"
    echo
    echo "done. Pin the commit sha it printed in MODELS.md / NLLB_HF_REV for reproducible downloads."
else
    echo
    echo "built and gated in $OUT/. Re-run with --upload to publish to $REPO."
fi
