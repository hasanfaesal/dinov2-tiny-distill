#!/usr/bin/env bash
# Download DataComp-12M metadata shards (resumable).
# Usage: scripts/download_data.sh [max_shards]
set -euo pipefail

DEST=${DATA_DIR:-data}/datacomp
mkdir -p "$DEST"
export HF_XET_HIGH_PERFORMANCE=1

INC=(--include "*.tar")
if [ $# -ge 1 ]; then
  INC=()
  for i in $(seq 0 $(($1 - 1))); do
    INC+=(--include "$(printf '%08d.tar' "$i")")
  done
fi

hf download mlfoundations/DataComp-12M --repo-type dataset --local-dir "$DEST" "${INC[@]}"
echo "shards: $(ls "$DEST"/*.tar | wc -l)  size: $(du -sh "$DEST" | cut -f1)"
