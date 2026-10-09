#!/usr/bin/env bash
# nllb.cpp full CI on the REAL 600M model - for your own machine, or a
# self-hosted GitHub Actions runner. Same idea as llama.cpp's ci/run.sh.
#
#   bash ci/run.sh <out-dir> <mnt-dir>
#
#   out-dir  logs + summary.md for this run (wiped each run)
#   mnt-dir  persistent cache: venv, HF snapshot, GGUFs, goldens (reused)
#
# Examples:
#   bash ci/run.sh ./tmp/results ./tmp/mnt                  # CPU only
#   NLLB_BUILD_METAL=1  bash ci/run.sh ./tmp/results ./tmp/mnt
#   NLLB_BUILD_CUDA=1   bash ci/run.sh ./tmp/results ./tmp/mnt
#   NLLB_BUILD_VULKAN=1 bash ci/run.sh ./tmp/results ./tmp/mnt
#   NLLB_LOW_PERF=1     bash ci/run.sh ./tmp/results ./tmp/mnt  # skip the slow gates
#
# Environment:
#   NLLB_BUILD_CUDA / NLLB_BUILD_METAL / NLLB_BUILD_VULKAN   GPU backend to build and test
#   NLLB_CUDA_ARCH      CMAKE_CUDA_ARCHITECTURES            [native]
#   NLLB_LOW_PERF=1     skip un-cached generate, tokenizer sweep and bench
#   NLLB_THREADS        CPU threads                         [min(nproc, 8)]
#   NLLB_MODEL_ID       HF model                            [facebook/nllb-200-distilled-600M]
#   NLLB_MODEL_REV      HF revision - pin to a commit sha   [main]
#   NLLB_TGT            target language for the goldens     [deu_Latn]
#   NLLB_MAX_NEW        generated tokens per golden         [24]
#   NLLB_CI_ONLY        run only these sections, e.g. "build venv lint cache"
#
# Exit code is non-zero if any section failed; summary.md lists each one.

# shellcheck disable=SC2329  # sec_* functions are invoked through run()
set -u

if [ $# -lt 2 ]; then
    sed -n '2,30p' "$0"; exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$1" "$2"
OUT="$(cd "$1" && pwd)"
MNT="$(cd "$2" && pwd)"
rm -f "$OUT"/*.log "$OUT"/summary.md

NPROC="$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 4)"
NT="${NLLB_THREADS:-$(( NPROC < 8 ? NPROC : 8 ))}"
MODEL_ID="${NLLB_MODEL_ID:-facebook/nllb-200-distilled-600M}"
MODEL_REV="${NLLB_MODEL_REV:-main}"
TGT="${NLLB_TGT:-deu_Latn}"
MAX_NEW="${NLLB_MAX_NEW:-24}"
LOW_PERF="${NLLB_LOW_PERF:-0}"

SNAP="$MNT/models/nllb-600M"
F32="$MNT/models/nllb-600M-f32.gguf"
F16="$MNT/models/nllb-600M-f16.gguf"
REF="$MNT/ref-$MODEL_REV-$TGT-$MAX_NEW"
BUILD="$ROOT/build-ci"
BIN="$BUILD/bin"

CMAKE_EXTRA=(-DNLLB_BUILD_SERVER_MOCK=ON)
GPU=""
if [ "${NLLB_BUILD_CUDA:-0}" = "1" ]; then
    CMAKE_EXTRA+=(-DNLLB_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${NLLB_CUDA_ARCH:-native}"); GPU=cuda
elif [ "${NLLB_BUILD_METAL:-0}" = "1" ]; then
    CMAKE_EXTRA+=(-DGGML_METAL=ON); GPU=metal
elif [ "${NLLB_BUILD_VULKAN:-0}" = "1" ]; then
    CMAKE_EXTRA+=(-DGGML_VULKAN=ON); GPU=gpu
fi

# ---------------------------------------------------------------- helpers

declare -a RESULTS=()
FAILED=0

# run <name> <function>: log to $OUT/<name>.log, record pass/fail, keep going.
run() {
    local name="$1"; shift
    if [ -n "${NLLB_CI_ONLY:-}" ] && [[ " $NLLB_CI_ONLY " != *" $name "* ]]; then
        skip "$name" "not in NLLB_CI_ONLY"; return 0
    fi
    local t0=$SECONDS rc
    printf '\033[1m==> %-22s\033[0m' "$name"
    # Not inside `if`: errexit is ignored in a condition, and the section
    # must stop at its first failing command.
    ( set -eo pipefail; cd "$ROOT"; "$@" ) > "$OUT/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        RESULTS+=("| $name | pass | $((SECONDS - t0)) s |"); echo " pass ($((SECONDS - t0)) s)"
    else
        RESULTS+=("| $name | **FAIL** (exit $rc) | $((SECONDS - t0)) s |"); echo " FAIL - see $OUT/$name.log"
        tail -n 15 "$OUT/$name.log" | sed 's/^/    /'
        FAILED=1
    fi
}

skip() { RESULTS+=("| $1 | skipped | $2 |"); printf '\033[1m==> %-22s\033[0m skipped (%s)\n' "$1" "$2"; }

# ---------------------------------------------------------------- sections

sec_build() {
    cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "${CMAKE_EXTRA[@]}"
    cmake --build "$BUILD" -j"$NPROC"
    "$BIN/nllb-cli" --list-backends
}

sec_venv() {
    [ -x "$MNT/venv/bin/python" ] || python3 -m venv "$MNT/venv"
    # shellcheck disable=SC1091
    source "$MNT/venv/bin/activate"
    python -m pip install -q --upgrade pip
    python -c "import torch" 2>/dev/null || \
        python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -q numpy safetensors gguf sentencepiece transformers huggingface_hub regex \
        -r tests/server/requirements.txt flake8
}

sec_lint() { flake8 . ; }

sec_download() {
    MODEL_ID="$MODEL_ID" MODEL_REV="$MODEL_REV" SNAP="$SNAP" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"], revision=os.environ["MODEL_REV"], local_dir=os.environ["SNAP"],
                  allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin", "*.txt"])
PY
    du -sh "$SNAP"
}

sec_convert() {
    # Always reconvert: the converter is under test too.
    python models/convert-nllb-to-gguf.py "$SNAP" --outfile "$F32" --outtype f32
    python models/convert-nllb-to-gguf.py "$SNAP" --outfile "$F16" --outtype f16
    "$BIN/nllb-cli" -m "$F16" --list-langs | grep -c . | tee /dev/stderr | grep -qx 202
}

sec_py_gates() {
    python tests/verify_encoder.py "$SNAP" "$F32"
    python tests/verify_decoder.py "$SNAP" "$F32"
}

sec_goldens() {
    # Cached per (revision, target, length): regenerating takes minutes and
    # only changes when one of those does.
    if [ -f "$REF/translations.txt" ]; then echo "reusing $REF"; return 0; fi
    rm -rf "$REF"
    python scripts/dump_reference.py "$SNAP" --out "$REF/" --lang eng_Latn --per-layer
    python scripts/dump_decoder_reference.py "$SNAP" --out "$REF/" \
        --tgt-lang "$TGT" --per-layer --greedy --max-new-tokens "$MAX_NEW"
    local fuzz=20000; [ "$LOW_PERF" = "1" ] && fuzz=2000
    python scripts/dump_tokenizer_reference.py "$SNAP" --out "$REF/" --src-lang eng_Latn --fuzz "$fuzz"
    python scripts/dump_translation_reference.py "$SNAP" --out "$REF/" \
        --src-lang eng_Latn --tgt-lang "$TGT" --max-new-tokens "$MAX_NEW"
}

sec_tokenizer() {
    local extra=(); [ "$LOW_PERF" = "1" ] && extra=(--no-sweep)
    "$BIN/nllb-test-tokenizer" "$F32" "$REF/" "${extra[@]}"
}

sec_encoder()  { "$BIN/nllb-test"         "$F32" "$REF/" -t "$NT"; }
sec_decoder()  { "$BIN/nllb-test-decoder" "$F32" "$REF/" -t "$NT"; }
sec_generate() { "$BIN/nllb-test-generate" "$F32" "$REF/" -t "$NT" --check-modes; }

sec_cache() {
    "$BIN/nllb-test-generate" "$F32" "$REF/" -t "$NT" --use-cache
    "$BIN/nllb-test-cache"    "$F32" "$REF/" -t "$NT"
}

sec_f16() {
    "$BIN/nllb-test-precision" "$F32" "$F16" "$REF/" -t "$NT"
    "$BIN/nllb-test-generate"  "$F16" "$REF/" -t "$NT" --use-cache
}

sec_batch() { "$BIN/nllb-test-batch" "$F16" -t "$NT" --rows 8; }

sec_gpu() {
    # Same file twice, kernels are the variable: exact ids + margin audit.
    "$BIN/nllb-test-precision" "$F16" "$F16" "$REF/" -t "$NT" --backend "cpu:$GPU"
    NLLB_BACKEND="$GPU" "$BIN/nllb-test-cache" "$F16" "$REF/" -t "$NT"
    "$BIN/nllb-test-batch" "$F16" -t "$NT" --backend "$GPU" --rows 8 -v
}

sec_cli() { python tests/verify_translation.py "$BIN/nllb-cli" "$F32" "$REF/" -t "$NT"; }

sec_server() {
    pytest tests/server -m "not real" -q    # mock: HTTP contract, queueing, batching
    NLLB_SERVER_BIN="$BIN/nllb-server" NLLB_SERVER_MODEL="$F16" NLLB_CLI_BIN="$BIN/nllb-cli" \
        pytest tests/server -m "not mock" -q
}

sec_bench() { "$BIN/nllb-bench" "$F16" -t "$NT" --steps 32; }

# ---------------------------------------------------------------- main

echo "nllb.cpp ci: out=$OUT mnt=$MNT threads=$NT gpu=${GPU:-none} low_perf=$LOW_PERF"

run build sec_build
run venv  sec_venv
# shellcheck disable=SC1091
[ -f "$MNT/venv/bin/activate" ] && source "$MNT/venv/bin/activate"
export NLLB_SERVER_BIN="$BIN/nllb-server-mock" NLLB_CLI_BIN="$BIN/nllb-cli"

run lint      sec_lint
run download  sec_download
run convert   sec_convert
run py-gates  sec_py_gates
run goldens   sec_goldens
run tokenizer sec_tokenizer
run encoder   sec_encoder
run decoder   sec_decoder
if [ "$LOW_PERF" = "1" ]; then skip generate "NLLB_LOW_PERF"; else run generate sec_generate; fi
run cache     sec_cache
run f16       sec_f16
run batch     sec_batch
if [ -n "$GPU" ]; then run "gpu-$GPU" sec_gpu; else skip gpu "no NLLB_BUILD_<GPU>"; fi
run cli       sec_cli
run server    sec_server
if [ "$LOW_PERF" = "1" ]; then skip bench "NLLB_LOW_PERF"; else run bench sec_bench; fi

{
    echo "# nllb.cpp ci - $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo
    echo "commit \`$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)\`," \
         "model \`$MODEL_ID@$MODEL_REV\`, gpu \`${GPU:-none}\`, threads $NT"
    echo
    echo "| section | result | time |"
    echo "|---|---|---|"
    printf '%s\n' "${RESULTS[@]}"
} > "$OUT/summary.md"

echo; cat "$OUT/summary.md"
exit $FAILED
