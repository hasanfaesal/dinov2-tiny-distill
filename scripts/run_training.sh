#!/usr/bin/env bash
# Training launcher (run on H100 node via run_command detach:true, restart:"on-failure").
# Foreground so the platform tracks the process; resumes from last.pt automatically.
set -e
cd /home/dev/repo && git pull -q
source /home/dev/venv/bin/activate
pip install -q prometheus_client
mkdir -p /home/dev/logs
exec python -u src/train.py >> /home/dev/logs/train.log 2>&1
