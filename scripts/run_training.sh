#!/usr/bin/env bash
# Detached launcher. Run on the node from /home/dev/src (so `dataset` etc. import).
# Resumes automatically from /home/dev/checkpoints/last.pt. Pass --smoke N for a short test.
set -euo pipefail
mkdir -p /home/dev/logs /home/dev/checkpoints
cd /home/dev
nohup python src/train.py configs/distill_config.yaml "$@" >> /home/dev/logs/train.log 2>&1 &
echo "pid $! ; tail -f /home/dev/logs/train.log"
