#!/usr/bin/env bash
# run-tests.sh — Run vLLM Edge Inference unit tests
#
# Usage:
#   cd plugins/vllm-edge-inference
#   ./tests/run-tests.sh
#
# This script activates the shared venv at tests/.venv (project root)
# and runs the unit test for this plugin.

set -euo pipefail
TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(dirname "$TESTS_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$PLUGIN_DIR")")"

# Activate shared venv
VENV="$PROJECT_DIR/tests/.venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Create it from the project root:"
    echo "  python3 -m venv tests/.venv"
    echo "  tests/.venv/bin/pip install pywaggle numpy opencv-python-headless Pillow"
    exit 1
fi
source "$VENV/bin/activate"

# Generate sample images if missing
if [ ! -d "$TESTS_DIR/sample-images" ] || [ -z "$(ls -A "$TESTS_DIR/sample-images/" 2>/dev/null)" ]; then
    echo "Generating sample images ..."
    python3 "$PROJECT_DIR/tests/generate_test_images.py" "$TESTS_DIR/sample-images"
fi

echo "=============================================="
echo "  vLLM Edge Inference — Unit Tests"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python3 "$TESTS_DIR/test_vllm.py"
