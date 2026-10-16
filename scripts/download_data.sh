#!/usr/bin/env bash
# Usage: download_data.sh [max_shards]   (default: all). Resumable.
set -euo pipefail
source /home/dev/venv/bin/activate
DEST=/home/dev/data/datacomp
mkdir -p "$DEST"
export HF_XET_HIGH_PERFORMANCE=1
INC=(--include "*.tar")
if [ $# -ge 1 ]; then
  INC=()
  for i in $(seq 0 $(($1 - 1))); do INC+=(--include "$(printf '%08d.tar' "$i")"); done
fi
hf download mlfoundations/DataComp-12M --repo-type dataset --local-dir "$DEST" "${INC[@]}"
echo "shards: $(ls "$DEST"/*.tar | wc -l)  size: $(du -sh "$DEST" | cut -f1)"
