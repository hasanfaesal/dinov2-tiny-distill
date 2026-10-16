#!/usr/bin/env bash
# Training launcher (run on H100 node via run_command detach:true, restart:"on-failure").
# Foreground so the platform tracks the process; resumes from last.pt automatically.
# Usage: run_training.sh [config.yaml] [log]
set -e
CONFIG=${1:-configs/distill_config.yaml}
LOG=${2:-/home/dev/logs/train.log}
cd /home/dev/repo && git pull -q
source /home/dev/venv/bin/activate
pip install -q prometheus_client pyarrow
mkdir -p /home/dev/logs
exec python -u src/train.py --config "$CONFIG" >> "$LOG" 2>&1
