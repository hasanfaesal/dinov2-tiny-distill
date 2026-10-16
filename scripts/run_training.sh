#!/usr/bin/env bash
# Detached training launcher (run on H100 node). Resumes from last.pt automatically.
cd /home/dev/repo && git pull -q
source /home/dev/venv/bin/activate
pip install -q prometheus_client
mkdir -p /home/dev/logs
nohup python src/train.py >> /home/dev/logs/train.log 2>&1 &
echo "launched pid $!"
