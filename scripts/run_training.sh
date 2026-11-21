#!/usr/bin/env bash
# Launch training (automatically resumes from <ckpt_dir>/last.pt if present).
# Usage: scripts/run_training.sh [config.yaml] [log_file]
set -euo pipefail

CONFIG=${1:-configs/distill_config.yaml}
LOG=${2:-logs/train.log}
mkdir -p "$(dirname "$LOG")"
exec python -u src/train.py --config "$CONFIG" >> "$LOG" 2>&1
