#!/usr/bin/env bash
# clean.sh — Remove all generated/temporary files from the repo.
#
# Run before: git add, rsync to another machine, or archiving.
#
# Usage:
#   ./clean.sh          # preview what will be removed
#   ./clean.sh --force  # actually remove it

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

FORCE=false
if [ "${1:-}" = "--force" ] || [ "${1:-}" = "-f" ]; then
    FORCE=true
fi

# What to clean:
#   1. Test output directories (data.ndjson, uploads/, reports)
#   2. Downloaded model weights (.pt, .pth, .bin, .safetensors)
#   3. Python bytecode caches (__pycache__, .pyc)
#   4. macOS resource forks (._*, .DS_Store)
#   5. Pytest cache (.pytest_cache)

items=()

# Test outputs
while IFS= read -r d; do
    [ -d "$d" ] && items+=("$d")
done < <(find plugins/*/tests/output -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
# Also top-level output/
[ -d "output" ] && items+=("output")

# Model weights (auto-downloaded, not committed)
while IFS= read -r f; do
    items+=("$f")
done < <(find . -maxdepth 2 \( -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' \) \
    -not -path './.git/*' -not -path '*/tests/.venv/*' 2>/dev/null)

# Python caches
while IFS= read -r d; do
    items+=("$d")
done < <(find . -name '__pycache__' -not -path './.git/*' -not -path '*/tests/.venv/*' 2>/dev/null)

# .pyc files outside __pycache__
while IFS= read -r f; do
    items+=("$f")
done < <(find . -name '*.pyc' -not -path './.git/*' -not -path '*/tests/.venv/*' -not -path '*__pycache__*' 2>/dev/null)

# macOS junk
while IFS= read -r f; do
    items+=("$f")
done < <(find . \( -name '._*' -o -name '.DS_Store' \) -not -path './.git/*' -not -path '*/tests/.venv/*' 2>/dev/null)

# Pytest cache
while IFS= read -r d; do
    items+=("$d")
done < <(find . -name '.pytest_cache' -not -path './.git/*' -not -path '*/tests/.venv/*' 2>/dev/null)

if [ ${#items[@]} -eq 0 ]; then
    echo "Nothing to clean."
    exit 0
fi

echo "Items to remove (${#items[@]}):"
for item in "${items[@]}"; do
    if [ -d "$item" ]; then
        count=$(find "$item" -type f 2>/dev/null | wc -l)
        echo "  [dir]  $item  ($count files)"
    else
        size=$(du -sh "$item" 2>/dev/null | cut -f1)
        echo "  [file] $item  ($size)"
    fi
done

if [ "$FORCE" = true ]; then
    echo ""
    for item in "${items[@]}"; do
        rm -rf "$item"
    done
    echo "Cleaned ${#items[@]} items."
else
    echo ""
    echo "Dry run — nothing removed. Use './clean.sh --force' to delete."
fi
