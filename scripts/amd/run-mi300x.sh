#!/usr/bin/env bash
# The ROCm/HIP acceptance run, in one command, for a rented AMD GPU.
#
# Written for an AMD Developer Cloud MI300X droplet (ROCm image), where every
# minute is billed: build, fixtures, gates, real model and timings in one go,
# everything teed into results/, and a SUMMARY.txt to read afterwards.
#
#   git clone --recursive https://github.com/Jaydeep505/nllb.cpp && cd nllb.cpp
#   ./scripts/amd/run-mi300x.sh                 # fixtures + real f16 model
#   REF=~/ref ./scripts/amd/run-mi300x.sh       # + the gates that need HF goldens
#
# REF is the ref/ directory the dump_*_reference.py scripts write (it is not in
# git). Copy it from the dev machine first:   scp -r ref root@<ip>:~/ref
# Without it the real-model cache/precision/translation gates are SKIPPED and
# the summary says so.
#
# Other knobs:  GPU_TARGETS=gfx942  BACKEND=rocm  T=<threads>  SKIP_REAL=1
#               SKIP_CORRUPT=1  OUT=results-rocm
#
# A step that fails does not stop the run - the point is to leave with as much
# evidence as the session can give. The exit code is the number of FAIL lines.

set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
GPU_TARGETS=${GPU_TARGETS:-gfx942}
BACKEND=${BACKEND:-rocm}
T=${T:-$(nproc)}
OUT=${OUT:-$ROOT/results-rocm}
REF=${REF:-}
MODEL=${MODEL:-$ROOT/models/nllb-600M-f16.gguf}
CPU=$ROOT/build
GPU=$ROOT/build-hip
mkdir -p "$OUT"
SUM=$OUT/SUMMARY.txt
: > "$SUM"
NFAIL=0

note() { echo "$*" | tee -a "$SUM"; }
# run <label> <command...>: tee the output, record PASS/FAIL by exit code.
run() {
    local label=$1; shift
    echo; echo "=== $label: $*"
    local t0=$SECONDS
    "$@" > "$OUT/$label.txt" 2>&1
    local rc=$?
    tail -n 25 "$OUT/$label.txt"
    if [ $rc -eq 0 ]; then
        note "$(printf '  %-34s %-6s %4ss' "$label" PASS $((SECONDS - t0)))"
    else
        note "$(printf '  %-34s %-6s %4ss  rc=%s  see %s.txt' "$label" FAIL $((SECONDS - t0)) $rc "$label")"
        NFAIL=$((NFAIL + 1))
    fi
    return $rc
}
# report <label> <command...>: like run, but a non-zero exit is not a FAIL.
report() {
    local label=$1; shift
    echo; echo "=== $label (report): $*"
    "$@" > "$OUT/$label.txt" 2>&1
    local rc=$?
    tail -n 25 "$OUT/$label.txt"
    note "$(printf '  %-34s %-6s        rc=%s' "$label" report $rc)"
}
skip() { note "$(printf '  %-34s %-6s        %s' "$1" SKIP "$2")"; }

# ---------------------------------------------------------------- 0 environment
note "nllb.cpp ROCm/HIP acceptance run   $(date -u +%Y-%m-%dT%H:%MZ)"
note "  repo   $(git rev-parse --short HEAD 2>/dev/null)   ggml $(git -C ggml rev-parse --short HEAD 2>/dev/null)"
note "  target $GPU_TARGETS   backend name '$BACKEND'   threads $T"
{
    echo "--- uname";      uname -a
    echo "--- cpu";        lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
    echo "--- mem";        free -g | head -2
    echo "--- hipconfig";  hipconfig --full 2>&1 | head -40
    echo "--- rocm-smi";   rocm-smi --showproductname --showmeminfo vram --showdriverversion 2>&1
    echo "--- rocminfo";   rocminfo 2>&1 | grep -E "Marketing Name|Name:|Compute Unit|Device Type" | head -40
    echo "--- cmake";      cmake --version | head -1
} > "$OUT/0-env.txt" 2>&1
note "  gpu    $(rocm-smi --showproductname 2>/dev/null | grep -m1 -iE "card (series|model)" | sed 's/.*: *//')"
note "  rocm   $(hipconfig --version 2>/dev/null)"
note ""

# ---------------------------------------------------------------- 1 build
# The CPU build is the reference every comparison below is made against.
run 1-build-cpu bash -c "cmake -B '$CPU' -DCMAKE_BUILD_TYPE=Release && cmake --build '$CPU' -j$T"
HIPCLANG="$(hipconfig -l 2>/dev/null)/clang"
run 1-build-hip bash -c "cmake -B '$GPU' -DNLLB_HIP=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER='$HIPCLANG' -DGPU_TARGETS='$GPU_TARGETS' && cmake --build '$GPU' -j$T"
if [ ! -x "$GPU/bin/nllb-cli" ]; then
    note "the HIP build produced no nllb-cli - nothing below can run. Read $OUT/1-build-hip.txt"
    exit 99
fi

report 1-list-backends "$GPU/bin/nllb-cli" --list-backends
if ! grep -qi "rocm" "$OUT/1-list-backends.txt"; then
    note "NO ROCm DEVICE REGISTERED. ggml's reason is in 1-list-backends.txt. Stopping:"
    note "every gate below would silently measure nothing."
    exit 98
fi

# ---------------------------------------------------------------- 2 fixtures
# Small random models with the real graph: seconds to build, and they decide
# whether a failure is the backend or the checkpoint.
# The ROCm cloud image ships python3 WITHOUT pip, and every fixture gate below
# needs numpy + gguf. Install pip from apt if it is missing, and stop here if
# the imports still fail - otherwise sections 2-4 all go red for one cause.
{
    python3 -m pip --version || { apt-get update -qq && apt-get install -y -qq python3-pip; }
    python3 -m pip install -q numpy gguf \
        || python3 -m pip install -q --break-system-packages numpy gguf
} > "$OUT/2-pip.txt" 2>&1
if ! python3 -c "import numpy, gguf" 2>> "$OUT/2-pip.txt"; then
    note "numpy/gguf not importable - the fixture gates (sections 2-4) cannot run. See 2-pip.txt"
    SKIP_FIXTURES=1
fi
if [ -z "${SKIP_FIXTURES:-}" ]; then
FX=$ROOT/models/small
SHAPE="--d-model 256 --enc-layers 4 --dec-layers 4 --heads 4 --ffn 1024 --vocab 4000 --n-ctx 64"
run 2-fixture-f32 python3 tests/make_shape_model.py "$FX-f32.gguf" $SHAPE --outtype f32
run 2-fixture-f16 python3 tests/make_shape_model.py "$FX-f16.gguf" $SHAPE --outtype f16
run 2-fixture-ref python3 tests/make_shape_ref.py ref-small/ --vocab 4000 --n-ctx 64 --max-new 24 --lengths 6 12 24 40
FXREF=$ROOT/ref-small/

# ---------------------------------------------------------------- 3 the flag must not move the CPU path
run 3-dump-cpu-build "$CPU/bin/nllb-dump-states" "$FX-f32.gguf" "$FXREF" "$OUT/cpu.bin" -t "$T"
run 3-dump-hip-build-on-cpu "$GPU/bin/nllb-dump-states" "$FX-f32.gguf" "$FXREF" "$OUT/hipcpu.bin" -t "$T" --backend cpu
run 3-cpu-path-identical bash -c "'$CPU/bin/nllb-dump-states' --diff '$OUT/cpu.bin' '$OUT/hipcpu.bin' | tee /dev/stderr | grep -q '^IDENTICAL'"

# ---------------------------------------------------------------- 4 device gates, fixtures
# The KV-cache ops alone, device vs CPU, no model. "half" rows are an f16
# kernel (expected on a GPU); a BAD row is a wrong kernel.
run 4-probe-kv-ops "$GPU/bin/nllb-probe-kv-ops" "$BACKEND"
for p in f32 f16; do
    # full = exact and "splits 1" (nothing fell back to the CPU) are the lines to read.
    run 4-cache-$p env NLLB_BACKEND="$BACKEND" NLLB_ARENA_TRACE=1 \
        "$GPU/bin/nllb-test-cache" "$FX-$p.gguf" "$FXREF" -t "$T"
    run 4-precision-cpu-gpu-$p "$GPU/bin/nllb-test-precision" "$FX-$p.gguf" "$FX-$p.gguf" "$FXREF" -t "$T" --backend "cpu:$BACKEND"
done
run 4-batch-f16 "$GPU/bin/nllb-test-batch" "$FX-f16.gguf" -t "$T" --backend "$BACKEND"
report 4-states-diff-cpu-vs-gpu bash -c "'$GPU/bin/nllb-dump-states' '$FX-f32.gguf' '$FXREF' '$OUT/gpu.bin' -t $T --backend '$BACKEND' \
    && '$CPU/bin/nllb-dump-states' --diff '$OUT/cpu.bin' '$OUT/gpu.bin'"

# The first backend here with memory of its own: mask-through-data and
# mask-not-input are marked `report` because host memory hides them. Read
# whether they FIRE on this device.
if [ -z "${SKIP_CORRUPT:-}" ]; then
    run 4-corrupt-backend env NLLB_BACKEND="$BACKEND" python3 tests/corrupt_backend.py "$GPU" "$FX-f16.gguf" "$FXREF" -t "$T"
else
    skip 4-corrupt-backend "SKIP_CORRUPT set"
fi
else
    skip 2-4-fixture-gates "numpy/gguf missing"
fi   # SKIP_FIXTURES

# ---------------------------------------------------------------- 5 the real model
if [ -n "${SKIP_REAL:-}" ]; then
    skip 5-real-model "SKIP_REAL set"
else
    [ -f "$MODEL" ] || run 5-download ./models/download-nllb-model.sh f16
    if [ ! -f "$MODEL" ]; then
        skip 5-real-model "no $MODEL"
    else
        run 5-hello-cpu "$CPU/bin/nllb-cli" -m "$MODEL" --tgt deu_Latn --backend cpu -t "$T" -v "Hello, world!"
        run 5-hello-gpu "$GPU/bin/nllb-cli" -m "$MODEL" --tgt deu_Latn --backend "$BACKEND" -t "$T" -v "Hello, world!"

        # Same sentences, CPU vs GPU, unbatched and batched: the outputs must be the same text.
        S=$ROOT/scripts/qdc/sentences.txt
        "$CPU/bin/nllb-cli" -m "$MODEL" --tgt deu_Latn --backend cpu        -t "$T" < "$S" > "$OUT/5-out-cpu.txt"     2> "$OUT/5-out-cpu.log"
        "$GPU/bin/nllb-cli" -m "$MODEL" --tgt deu_Latn --backend "$BACKEND" -t "$T" < "$S" > "$OUT/5-out-gpu.txt"     2> "$OUT/5-out-gpu.log"
        "$GPU/bin/nllb-cli" -m "$MODEL" --tgt deu_Latn --backend "$BACKEND" -t "$T" --batch 16 < "$S" > "$OUT/5-out-gpu-b16.txt" 2> "$OUT/5-out-gpu-b16.log"
        run 5-text-cpu-vs-gpu      diff "$OUT/5-out-cpu.txt" "$OUT/5-out-gpu.txt"
        run 5-text-cpu-vs-gpu-b16  diff "$OUT/5-out-cpu.txt" "$OUT/5-out-gpu-b16.txt"

        run 5-batch-real "$GPU/bin/nllb-test-batch" "$MODEL" -t "$T" --backend "$BACKEND"

        if [ -n "$REF" ] && [ -d "$REF" ]; then
            R=${REF%/}/
            run 5-cache-real env NLLB_BACKEND="$BACKEND" NLLB_ARENA_TRACE=1 "$GPU/bin/nllb-test-cache" "$MODEL" "$R" -t "$T"
            run 5-precision-cpu-gpu-real "$GPU/bin/nllb-test-precision" "$MODEL" "$MODEL" "$R" -t "$T" --backend "cpu:$BACKEND"
            if [ -f "$R/translations.txt" ]; then
                run 5-translation-vs-hf     python3 tests/verify_translation.py "$GPU/bin/nllb-cli" "$MODEL" "$R" -t "$T" --backend "$BACKEND"
                run 5-translation-vs-hf-b4  python3 tests/verify_translation.py "$GPU/bin/nllb-cli" "$MODEL" "$R" -t "$T" --backend "$BACKEND" --batch 4
            else
                skip 5-translation-vs-hf "no translations.txt in REF"
            fi
        else
            skip 5-cache-real              "REF not set - scp ref/ here and re-run with REF=..."
            skip 5-precision-cpu-gpu-real  "REF not set"
            skip 5-translation-vs-hf       "REF not set"
        fi

        # ------------------------------------------------------------ 6 timings
        # Read the encoder pass and the cached step SEPARATELY: on a discrete GPU
        # the one-token step is mostly launch + scheduler overhead, not arithmetic.
        report 6-bench-cpu "$CPU/bin/nllb-bench" "$MODEL" -t "$T" --src 24 --steps 64
        report 6-bench-gpu env NLLB_BACKEND="$BACKEND" NLLB_ARENA_TRACE=1 "$GPU/bin/nllb-bench" "$MODEL" -t "$T" --src 24 --steps 64
        # Where ms/token stops falling. Pick B from this table, not from the M1's.
        report 6-batch-sweep-gpu "$GPU/bin/nllb-test-batch" "$MODEL" --bench 1,2,4,8,9,16,32,64,128 --src 24 --steps 32 -t "$T" --backend "$BACKEND"
        report 6-batch-sweep-cpu "$CPU/bin/nllb-test-batch" "$MODEL" --bench 1,4,16,64 --src 24 --steps 32 -t "$T"
        rocm-smi --showmeminfo vram --showuse > "$OUT/6-rocm-smi-after.txt" 2>&1
    fi
fi

# ---------------------------------------------------------------- summary
note ""
note "$NFAIL FAIL line(s). Logs: $OUT/"
tar -czf "$ROOT/results-rocm.tar.gz" --exclude="*.bin" -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
echo
echo "results-rocm.tar.gz written. From the dev machine:"
echo "  scp root@<ip>:$ROOT/results-rocm.tar.gz ."
echo "THEN DESTROY THE DROPLET - a powered-off one is still billed."
exit $NFAIL
