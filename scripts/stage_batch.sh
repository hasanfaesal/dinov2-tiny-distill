#!/usr/bin/env bash
# On cpu node: hardlink up to 4.5 GiB of finished, not-yet-staged shards from SRC into /home/dev/stage
# (copy_file caps one artifact at 5 GiB). Records staged shards in /home/dev/staged_SRC.txt.
# Usage: stage_batch.sh [src=images]
SRC=${1:-images}
LOG=staged_$SRC.txt
[ "$SRC" = images ] && LOG=staged.txt
cd /home/dev || exit 1
rm -rf stage && mkdir stage && touch "$LOG"
total=0
for s in "$SRC"/*_stats.json; do
  b=${s%_stats.json}
  grep -qx "$b" "$LOG" && continue
  sz=$(stat -c %s "$b.tar")
  [ $((total + sz)) -gt 4831838208 ] && break
  ln "$b.tar" "$s" stage/ && echo "$b" >> "$LOG"
  total=$((total + sz))
done
echo "staged $(ls stage/*.tar 2>/dev/null | wc -l) shards, $((total / 1048576)) MiB; total staged $(wc -l < "$LOG")"
