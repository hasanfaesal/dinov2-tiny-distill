#!/usr/bin/env bash
# Run on the cpu node. Usage: fetch_images.sh [start=0] [count=170] [out=images]  (~12.5k urls per meta shard)
set -euo pipefail
START=${1:-0}
N=${2:-170}
OUT=${3:-images}
cd /home/dev
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q img2dataset "huggingface_hub" pyarrow
mkdir -p "meta_$OUT" "$OUT" logs
INC=(); for i in $(seq "$START" $((START + N - 1))); do INC+=(--include "$(printf '%08d.tar' "$i")"); done
HF_XET_HIGH_PERFORMANCE=1 hf download mlfoundations/DataComp-12M --repo-type dataset --local-dir "meta_$OUT" "${INC[@]}"
python repo/scripts/build_urls.py "meta_$OUT" "urls_$OUT.parquet"
rm -f "meta_$OUT"/*.tar  # urls extracted; metadata tars no longer needed
img2dataset --url_list "urls_$OUT.parquet" --input_format parquet --url_col url --caption_col uid \
  --output_format webdataset --output_folder "$OUT" --image_size 256 --resize_mode keep_ratio \
  --encode_quality 90 --processes_count 8 --thread_count 128 --number_sample_per_shard 10000 \
  --resize_only_if_bigger True --enable_wandb False --timeout 6 --retries 0
echo "shards: $(ls "$OUT"/*.tar | wc -l)  size: $(du -sh "$OUT" | cut -f1)"
