#!/usr/bin/env bash
# On cpu node: hardlink up to 4.5 GiB of finished, not-yet-staged shards into /home/dev/stage
# (copy_file caps one artifact at 5 GiB). Records staged shards in /home/dev/staged.txt.
cd /home/dev && rm -rf stage && mkdir stage && touch staged.txt
total=0
for s in images/*_stats.json; do
  b=${s%_stats.json}
  grep -qx "$b" staged.txt && continue
  sz=$(stat -c %s "$b.tar")
  [ $((total + sz)) -gt 4831838208 ] && break
  ln "$b.tar" "$s" stage/ && echo "$b" >> staged.txt
  total=$((total + sz))
done
echo "staged $(ls stage/*.tar 2>/dev/null | wc -l) shards, $((total / 1048576)) MiB; total staged $(wc -l < staged.txt)"
