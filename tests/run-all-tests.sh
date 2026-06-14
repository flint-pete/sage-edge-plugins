#!/usr/bin/env bash
# run-all-tests.sh — Run all Sage plugin unit tests
#
# Usage:
#   cd ~/AI-projects/Sage-agents
#   ./tests/run-all-tests.sh
#
# Requirements:
#   tests/.venv with pywaggle, numpy, opencv-python-headless, Pillow

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Sage Edge Plugin Test Suite"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# Activate venv
VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Run: python3 -m venv $VENV && $VENV/bin/pip install pywaggle numpy opencv-python-headless Pillow"
    exit 1
fi
source "$VENV/bin/activate"
echo "Using Python: $(which python3) ($(python3 --version))"
echo ""

# Generate sample images if missing
if [ ! -d "$SCRIPT_DIR/sample-images" ] || [ -z "$(ls -A "$SCRIPT_DIR/sample-images/" 2>/dev/null)" ]; then
    echo "Generating sample images ..."
    python3 "$SCRIPT_DIR/generate_test_images.py"
    echo ""
fi

# Track results
PASSED=0
FAILED=0
RESULTS=()

run_test() {
    local name="$1"
    local script="$2"
    echo "----------------------------------------------"
    echo "  Running: $name"
    echo "----------------------------------------------"
    if python3 "$script"; then
        PASSED=$((PASSED + 1))
        RESULTS+=("PASS: $name")
    else
        FAILED=$((FAILED + 1))
        RESULTS+=("FAIL: $name")
    fi
}

# Run each test
run_test "YOLO Object Counter"          "$SCRIPT_DIR/test_yolo.py"
run_test "BioCLIP Species Classifier"   "$SCRIPT_DIR/test_bioclip.py"
run_test "vLLM Edge Inference"          "$SCRIPT_DIR/test_vllm.py"

# Final summary
echo ""
echo "############################################"
echo "  FINAL SUMMARY"
echo "############################################"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Total: $((PASSED + FAILED))  Passed: $PASSED  Failed: $FAILED"
echo "############################################"
echo ""

# Output files location
echo "Test output written to: $SCRIPT_DIR/output/"
if [ -d "$SCRIPT_DIR/output" ]; then
    echo "  Subdirectories:"
    for d in "$SCRIPT_DIR/output"/*/; do
        [ -d "$d" ] || continue
        ndjson="$d/data.ndjson"
        uploads="$d/uploads"
        lines=0
        files=0
        [ -f "$ndjson" ] && lines=$(wc -l < "$ndjson")
        [ -d "$uploads" ] && files=$(find "$uploads" -type f | wc -l)
        echo "    $(basename "$d"): $lines measurements, $files uploads"
    done
fi
echo ""

exit $FAILED
