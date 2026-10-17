#!/usr/bin/env bash
# night_pipeline.sh - runs as the H100 boot command (restart:"on-failure", 720min per attempt).
# Flow: stage images2..images6 (mesh -> export fallback) -> verify sha256 -> preflight ->
#       run-1 eval -> smoke -> train -> post-eval -> self-stop.
# Idempotent: per-step markers in /home/dev/night_state let a restarted attempt resume.
set -euo pipefail

LOG=/home/dev/logs/night_pipeline.log
STATE=/home/dev/night_state
REPO=/home/dev/repo
DATA=/home/dev/data
MAN=$REPO/manifests
mkdir -p /home/dev/logs "$STATE"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PIPELINE ERROR rc=$? at line $LINENO"' ERR
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

cd "$REPO"
say "night pipeline start rev=$(git rev-parse --short HEAD)"
nvidia-smi -L | head -1 || true
df -h /home/dev | tail -1
git pull -q || say "WARN: git pull failed, using current revision"

[ -f "$MAN/manifest.sizes" ] || { say "FATAL: manifest.sizes missing"; exit 10; }
[ -f "$MAN/manifest.sha256" ] || { say "FATAL: manifest.sha256 missing"; exit 10; }

# ---- staging -------------------------------------------------------------
need=0
for chunk in images2 images3 images4 images5 images6; do
  [ -f "$STATE/staged_$chunk" ] && continue
  b=$(awk -v c="$chunk/" '$2 ~ "^"c {s+=$1} END{print s+0}' "$MAN/manifest.sizes")
  need=$((need + b))
done
need_gb=$(( need / 1000000000 + 3 ))
avail_gb=$(df -BG --output=avail /home/dev | tail -1 | tr -dc '0-9')
say "staging: need ~${need_gb}G, have ${avail_gb}G free"
[ "$avail_gb" -ge "$need_gb" ] || { say "FATAL: not enough disk"; exit 11; }

mesh_srv() {
  local c
  for c in "dinov2-fetch-cpu.givemeanode.internal:8099" "100.64.0.1:8099"; do
    curl -fsS -m 8 "http://$c/health.txt" >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  return 1
}

todo_list() {
  awk -v c="$1/" '$2 ~ "^"c {print $2" "$1}' "$MAN/manifest.sizes" |
  while read -r f sz; do
    [ "$(stat -c %s "$DATA/$f" 2>/dev/null || echo 0)" = "$sz" ] || echo "$f"
  done
}

verify_chunk() {
  local chunk=$1 miss
  miss=$(todo_list "$chunk" | wc -l)
  if [ "$miss" -ne 0 ]; then say "$chunk: $miss file(s) still missing/short"; return 1; fi
  say "$chunk: sizes ok, hashing"
  ( cd "$DATA" && grep " $chunk/" "$MAN/manifest.sha256" | sha256sum -c - >/dev/null ) || { say "$chunk: sha256 MISMATCH"; return 1; }
  say "$chunk: VERIFIED"
}

pull_mesh() {
  local chunk=$1 srv=$2 n
  todo_list "$chunk" > "$STATE/todo_$chunk.txt"
  n=$(wc -l < "$STATE/todo_$chunk.txt")
  [ "$n" -eq 0 ] && return 0
  say "$chunk: mesh pull $n file(s) from $srv"
  xargs -a "$STATE/todo_$chunk.txt" -P 12 -I{} curl -fSsS --connect-timeout 10 --retry 6 --retry-delay 3 -m 900 -o "$DATA/{}" "http://$srv/{}" || true
  verify_chunk "$chunk"
}

pull_exports() {
  local chunk=$1 tmp=/home/dev/export_tmp rows bid cc url
  [ -n "${NIGHT_KEY:-}" ] || return 1
  [ -f "$MAN/night_sources.enc" ] || return 1
  if [ ! -f "$STATE/night_sources.tsv" ]; then
    openssl enc -d -aes-256-cbc -pbkdf2 -in "$MAN/night_sources.enc" -out "$STATE/night_sources.tsv" -pass env:NIGHT_KEY || return 1
  fi
  rows=$(awk -F'\t' -v c="$chunk" '$2==c {print}' "$STATE/night_sources.tsv")
  [ -n "$rows" ] || return 1
  rm -rf "$tmp"; mkdir -p "$tmp"
  while IFS=$'\t' read -r bid cc url; do
    [ -n "$bid" ] || continue
    say "$chunk: export bundle $bid"
    curl -fSsS --connect-timeout 15 --retry 6 --retry-delay 5 -m 3600 -o "$tmp/$bid.tar" "$url" || { rm -rf "$tmp"; return 1; }
    tar -xf "$tmp/$bid.tar" -C "$tmp" || { rm -rf "$tmp"; return 1; }
    rm -f "$tmp/$bid.tar"
  done <<< "$rows"
  find "$tmp" -maxdepth 1 -type f -exec mv -f -t "$DATA/$chunk/" {} +
  rm -rf "$tmp"
  verify_chunk "$chunk"
}

stage_start=$(date +%s)
for chunk in images2 images3 images4 images5 images6; do
  if [ -f "$STATE/staged_$chunk" ]; then say "$chunk: staged (marker)"; continue; fi
  mkdir -p "$DATA/$chunk"
  ok=0
  for attempt in 1 2 3 4; do
    say "$chunk: attempt $attempt"
    if srv=$(mesh_srv); then pull_mesh "$chunk" "$srv" && { ok=1; break; }; fi
    pull_exports "$chunk" && { ok=1; break; }
    if [ $(( $(date +%s) - stage_start )) -gt 12600 ]; then say "FATAL: staging deadline (3.5h) exceeded"; exit 12; fi
    say "$chunk: no source delivered; sleep 300 then refresh repo"
    sleep 300
    git pull -q || true
  done
  [ "$ok" -eq 1 ] || { say "FATAL: could not stage $chunk"; exit 13; }
  touch "$STATE/staged_$chunk"
  df -h /home/dev | tail -1
done
say "staging complete"

# ---- env + preflight -----------------------------------------------------
cd "$REPO"
source /home/dev/venv/bin/activate
pip install -q prometheus_client pyarrow || true

if [ ! -f "$STATE/preflight_ok" ]; then
  say "preflight"
  python -u scripts/preflight.py || { say "FATAL: preflight failed"; exit 20; }
  touch "$STATE/preflight_ok"
fi

# ---- run-1 numbers (best-effort; never blocks training) ------------------
if [ ! -f "$STATE/run1_eval_ok" ]; then
  say "run-1 eval (best-effort)"
  {
    python -u - <<'PY'
import torch
for p in ("/home/dev/checkpoints/best.pt", "/home/dev/checkpoints/last.pt"):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        print(p, "epoch", ck.get("epoch"), "step", ck.get("step"), "best_loss", ck.get("best"))
    except Exception as e:
        print(p, "LOAD FAIL", repr(e))
PY
    python -u src/evaluate_knn.py --teacher
    python -u src/evaluate_knn.py /home/dev/checkpoints/best.pt /home/dev/checkpoints/last.pt
  } 2>&1 | tee -a /home/dev/logs/run1_eval.log && touch "$STATE/run1_eval_ok" || say "WARN: run-1 eval failed (continuing)"
fi

# ---- smoke ---------------------------------------------------------------
if [ ! -f "$STATE/smoke_ok" ]; then
  say "smoke (2 epochs, isolated ckpt dir)"
  python -u src/train.py --smoke --config configs/distill_full.yaml >> /home/dev/logs/smoke_full.log 2>&1 || { say "FATAL: smoke failed"; exit 21; }
  touch "$STATE/smoke_ok"
fi

# ---- train ---------------------------------------------------------------
say "launching training (foreground; auto-resumes from last.pt)"
bash scripts/run_training.sh configs/distill_full.yaml /home/dev/logs/train_full.log || { rc=$?; say "train exited rc=$rc"; exit 30; }

# ---- post eval + self-stop ----------------------------------------------
say "training complete; post eval"
python -u src/evaluate_knn.py /home/dev/checkpoints_full/best_knn.pt /home/dev/checkpoints_full/best.pt /home/dev/checkpoints_full/last.pt 2>&1 | tee -a /home/dev/logs/train_full.log || say "WARN: post eval failed"
sleep 3
say "self-stop (billing ends now; disk parks intact)"
curl -sS -m 15 -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop || say "WARN: self-stop failed"
say "ALL DONE"
