#!/usr/bin/env bash
# Fetch the LightGBM ranking example dataset (LETOR-style graded relevance).
#
# Source: microsoft/LightGBM examples/lambdarank. Sparse LibSVM rows
# (`label idx:val ...`) with a sibling `.query` file giving per-query group
# sizes. Relevance is graded 0-4. ~3000 train / ~770 test rows.
#
# Data lands in rankit/data/lightgbm_rank/ which is gitignored.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/lightgbm_rank"
BASE="https://github.com/microsoft/LightGBM/raw/master/examples/lambdarank"

mkdir -p "$DEST"
for f in rank.train rank.train.query rank.test rank.test.query; do
  if [ -f "$DEST/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -sSL --fail -o "$DEST/$f" "$BASE/$f"
  fi
done
echo "done -> $DEST"
