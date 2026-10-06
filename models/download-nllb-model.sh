#!/usr/bin/env bash
#
# Download a pre-converted nllb.cpp GGUF - no Python, no torch, no HF snapshot.
#
#   ./models/download-nllb-model.sh              # f16, the default
#   ./models/download-nllb-model.sh q8emb        # smaller / faster, opt-in
#   ./models/download-nllb-model.sh f16 ~/nllb   # into another directory
#   ./models/download-nllb-model.sh --list
#
# Every file is checked against the repo's SHA256SUMS before it is moved into
# place, so a truncated or tampered download never ends up named *.gguf.
#
# Environment:
#   NLLB_HF_REPO      model repo           (default below)
#   NLLB_HF_REV       branch, tag or sha   (default main - PIN IT for anything
#                     reproducible; a re-upload on main changes your bytes)
#   NLLB_BASE_URL     full base URL, overrides the two above (mirrors, CI)
#   HF_TOKEN          sent as a bearer token if set (private / gated repos)
#
# Exit codes: 0 ok, 1 download/verify failure, 2 usage error.

set -euo pipefail

REPO="${NLLB_HF_REPO:-JaydeepGupta/nllb-200-distilled-600M-GGUF}"
REV="${NLLB_HF_REV:-main}"
BASE_URL="${NLLB_BASE_URL:-https://huggingface.co/${REPO}/resolve/${REV}}"

# name | file | size | what it is. Keep in step with MODELS.md and the model card.
MODELS=(
    "f16|nllb-600M-f16.gguf|1.24 GB|the default. Greedy ids identical to HF on every golden"
    "q8emb|nllb-600M-q8emb.gguf|0.99 GB|token_embd at Q8_0, rest f16. 1.33x faster on M1 CPU, chrF 96.57"
    "f32|nllb-600M-f32.gguf|2.46 GB|reference precision. For the gates and as nllb-quantize input"
)

usage() {
    cat >&2 <<EOF
usage: $0 [MODEL] [DIR]
       $0 --list

  MODEL   one of: $(for m in "${MODELS[@]}"; do printf '%s ' "${m%%|*}"; done)(default f16)
  DIR     where to save it (default: the directory this script is in)

source: ${BASE_URL}
EOF
}

list_models() {
    printf '%-7s %-24s %-8s %s\n' name file size notes
    for m in "${MODELS[@]}"; do
        IFS='|' read -r n f s d <<<"$m"
        printf '%-7s %-24s %-8s %s\n' "$n" "$f" "$s" "$d"
    done
    echo
    echo "source: ${BASE_URL}"
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    -l|--list) list_models; exit 0 ;;
esac

NAME="${1:-f16}"
DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

FILE=""
for m in "${MODELS[@]}"; do
    IFS='|' read -r n f _ _ <<<"$m"
    if [[ "$n" == "$NAME" ]]; then FILE="$f"; fi
done
if [[ -z "$FILE" ]]; then
    echo "unknown model '$NAME'" >&2
    echo >&2
    list_models >&2
    exit 2
fi

# --- tools -----------------------------------------------------------------

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then          # macOS
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    echo "need sha256sum or shasum to verify the download" >&2
    exit 1
fi

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then AUTH=(-H "Authorization: Bearer ${HF_TOKEN}"); fi

# fetch URL OUT [resume]
if command -v curl >/dev/null 2>&1; then
    fetch() {
        local resume=()
        [[ "${3:-}" == resume ]] && resume=(-C -)
        curl -fL --retry 3 --retry-delay 2 ${resume[@]+"${resume[@]}"} \
             ${AUTH[@]+"${AUTH[@]}"} \
             "$([[ -t 2 && "${3:-}" == resume ]] && echo --progress-bar || echo -sS)" \
             -o "$2" "$1"
    }
elif command -v wget >/dev/null 2>&1; then
    fetch() {
        local resume=(-q)
        [[ "${3:-}" == resume ]] && resume=(-q --show-progress -c)
        local hdr=()
        [[ -n "${HF_TOKEN:-}" ]] && hdr=(--header "Authorization: Bearer ${HF_TOKEN}")
        wget "${resume[@]}" ${hdr[@]+"${hdr[@]}"} -O "$2" "$1"
    }
else
    echo "need curl or wget" >&2
    exit 1
fi

# --- checksum first: it is tiny, and without it nothing below can be trusted --

mkdir -p "$DIR"
SUMS="$(mktemp)"
trap 'rm -f "$SUMS"' EXIT

if ! fetch "${BASE_URL}/SHA256SUMS" "$SUMS"; then
    echo "could not fetch ${BASE_URL}/SHA256SUMS" >&2
    echo "  check NLLB_HF_REPO / NLLB_HF_REV, or your network" >&2
    exit 1
fi
EXPECTED="$(awk -v f="$FILE" '$2 == f || $2 == "*"f {print $1}' "$SUMS")"
if [[ ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
    echo "SHA256SUMS at ${BASE_URL} has no entry for ${FILE}" >&2
    exit 1
fi

OUT="${DIR}/${FILE}"
PART="${OUT}.part"

if [[ -f "$OUT" ]]; then
    echo "${OUT} exists - verifying..."
    if [[ "$(sha256 "$OUT")" == "$EXPECTED" ]]; then
        echo "ok, already up to date."
        exit 0
    fi
    # A complete file with the wrong bytes (an older upload, or damage) cannot
    # be repaired by resuming, so start over.
    echo "checksum differs from ${REV}; re-downloading." >&2
    rm -f "$OUT"
fi

verify_part() { [[ "$(sha256 "$PART")" == "$EXPECTED" ]]; }

echo "downloading ${FILE} from ${BASE_URL}"
ok=0
if [[ -s "$PART" ]]; then
    echo "resuming from $(wc -c < "$PART" | tr -d ' ') bytes in ${PART}"
    # Resume can fail on a server without range support, and a .part left by a
    # different upload resumes into garbage. Either way: one clean retry.
    if fetch "${BASE_URL}/${FILE}" "$PART" resume && verify_part; then
        ok=1
    else
        echo "resume did not produce a valid file; restarting from zero" >&2
        rm -f "$PART"
    fi
fi
if [[ "$ok" == 0 ]]; then
    if ! fetch "${BASE_URL}/${FILE}" "$PART" resume; then
        echo "download failed; re-run to resume from ${PART}" >&2
        exit 1
    fi
    echo "verifying sha256..."
    if ! verify_part; then
        echo "CHECKSUM MISMATCH for ${FILE}" >&2
        echo "  expected ${EXPECTED}" >&2
        echo "  got      $(sha256 "$PART")" >&2
        rm -f "$PART"
        exit 1
    fi
fi
mv -f "$PART" "$OUT"

echo
echo "saved ${OUT}"
echo
echo "try it:"
echo "  ./build/bin/nllb-cli -m ${OUT} --src eng_Latn --tgt deu_Latn \"Hello, world!\""
