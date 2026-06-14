#!/usr/bin/env bash
# run-all-tests.sh — Run all Sage plugin unit tests
#
# Usage:
#   cd ~/AI-projects/Sage-agents
#   ./tests/run-all-tests.sh              # run all unit tests
#   ./tests/run-all-tests.sh yolo         # run only YOLO tests
#   ./tests/run-all-tests.sh bioclip vllm # run BioCLIP and vLLM tests
#
# Each plugin has self-contained tests in plugins/<name>/tests/.
# This script discovers and runs them using a shared venv at tests/.venv.
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

# Activate shared venv
VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Run: python3 -m venv $VENV && $VENV/bin/pip install pywaggle numpy opencv-python-headless Pillow"
    exit 1
fi
source "$VENV/bin/activate"
echo "Using Python: $(which python3) ($(python3 --version))"
echo ""

# Map of short names -> plugin directories
declare -A PLUGIN_MAP=(
    [yolo]="yolo-object-counter"
    [bioclip]="bioclip-species-classifier"
    [vllm]="vllm-edge-inference"
)

# Determine which plugins to test
if [ $# -gt 0 ]; then
    PLUGINS=()
    for arg in "$@"; do
        key="${arg,,}"  # lowercase
        if [[ -v "PLUGIN_MAP[$key]" ]]; then
            PLUGINS+=("${PLUGIN_MAP[$key]}")
        elif [ -d "$PROJECT_DIR/plugins/$arg/tests" ]; then
            PLUGINS+=("$arg")
        else
            echo "WARNING: Unknown plugin '$arg' — skipping"
        fi
    done
else
    # Run all plugins that have tests/
    PLUGINS=()
    for d in "$PROJECT_DIR"/plugins/*/tests/; do
        [ -d "$d" ] || continue
        plugin="$(basename "$(dirname "$d")")"
        PLUGINS+=("$plugin")
    done
fi

if [ ${#PLUGINS[@]} -eq 0 ]; then
    echo "ERROR: No plugins with tests/ found"
    exit 1
fi

echo "Plugins to test: ${PLUGINS[*]}"
echo ""

# Generate sample images for any plugin that lacks them
for plugin in "${PLUGINS[@]}"; do
    sample_dir="$PROJECT_DIR/plugins/$plugin/tests/sample-images"
    if [ ! -d "$sample_dir" ] || [ -z "$(ls -A "$sample_dir/" 2>/dev/null)" ]; then
        echo "Generating sample images for $plugin ..."
        python3 "$SCRIPT_DIR/generate_test_images.py" "$PROJECT_DIR/plugins/$plugin/tests/sample-images"
    fi
done

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

# Run unit tests for each plugin
for plugin in "${PLUGINS[@]}"; do
    test_dir="$PROJECT_DIR/plugins/$plugin/tests"
    # Find unit test files (not integration, not local)
    for test_file in "$test_dir"/test_*.py; do
        [ -f "$test_file" ] || continue
        basename="$(basename "$test_file")"
        # Skip integration and local tests (those need GPU/real models)
        # Skip test_harness.py (shared library, not a runnable test)
        if [[ "$basename" == *"_integration"* ]] || [[ "$basename" == *"_local"* ]] || [[ "$basename" == "test_harness.py" ]]; then
            continue
        fi
        run_test "$plugin / $basename" "$test_file"
    done
done

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
echo "Test output written to each plugin's tests/output/ directory:"
for plugin in "${PLUGINS[@]}"; do
    out_dir="$PROJECT_DIR/plugins/$plugin/tests/output"
    if [ -d "$out_dir" ]; then
        for d in "$out_dir"/*/; do
            [ -d "$d" ] || continue
            ndjson="$d/data.ndjson"
            uploads="$d/uploads"
            lines=0
            files=0
            [ -f "$ndjson" ] && lines=$(wc -l < "$ndjson")
            [ -d "$uploads" ] && files=$(find "$uploads" -type f | wc -l)
            echo "  $plugin/tests/output/$(basename "$d"): $lines measurements, $files uploads"
        done
    fi
done
echo ""

exit $FAILED
