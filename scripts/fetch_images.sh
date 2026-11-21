#!/usr/bin/env bash
# Download DataComp-12M metadata shards, extract URLs, and fetch images as WebDataset shards.
# Usage: scripts/fetch_images.sh [start=0] [count=170] [out=data/images]
# Env: PROCS (img2dataset processes, default 8)
set -euo pipefail

START=${1:-0}
N=${2:-170}
OUT=${3:-data/images}
TAG=$(basename "$OUT")
META_DIR="data/meta_$TAG"
URLS="data/urls_$TAG.parquet"

mkdir -p "$META_DIR" "$OUT"
if [ ! -f "$URLS" ]; then
  INC=()
  for i in $(seq "$START" $((START + N - 1))); do
    INC+=(--include "$(printf '%08d.tar' "$i")")
  done
  HF_XET_HIGH_PERFORMANCE=1 hf download mlfoundations/DataComp-12M --repo-type dataset --local-dir "$META_DIR" "${INC[@]}"
  python scripts/build_urls.py "$META_DIR" "${URLS}.tmp" && mv "${URLS}.tmp" "$URLS"
  rm -f "$META_DIR"/*.tar
fi

img2dataset --url_list "$URLS" --input_format parquet --url_col url --caption_col uid \
  --output_format webdataset --output_folder "$OUT" --image_size 256 --resize_mode keep_ratio \
  --encode_quality 90 --processes_count "${PROCS:-8}" --thread_count 128 --number_sample_per_shard 10000 \
  --resize_only_if_bigger True --enable_wandb False --timeout 6 --retries 0

echo "shards: $(ls "$OUT"/*.tar | wc -l)  size: $(du -sh "$OUT" | cut -f1)"
