#!/usr/bin/env bash
# Run on the cpu node. Usage: fetch_images.sh [n_meta_shards=170]  (~12.5k urls each)
set -euo pipefail
N=${1:-170}
cd /home/dev
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q img2dataset "huggingface_hub" pyarrow
mkdir -p meta images logs
INC=(); for i in $(seq 0 $((N - 1))); do INC+=(--include "$(printf '%08d.tar' "$i")"); done
HF_XET_HIGH_PERFORMANCE=1 hf download mlfoundations/DataComp-12M --repo-type dataset --local-dir meta "${INC[@]}"
python repo/scripts/build_urls.py meta urls.parquet
img2dataset --url_list urls.parquet --input_format parquet --url_col url --caption_col uid \
  --output_format webdataset --output_folder images --image_size 256 --resize_mode keep_ratio \
  --encode_quality 90 --processes_count 8 --thread_count 128 --number_sample_per_shard 10000 \
  --resize_only_if_bigger True --enable_wandb False --timeout 6 --retries 0
echo "shards: $(ls images/*.tar | wc -l)  size: $(du -sh images | cut -f1)"
