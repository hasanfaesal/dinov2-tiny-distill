"""Extract uid/url from DataComp metadata tars into one parquet. Usage: build_urls.py SHARD_DIR OUT.parquet"""
import glob
import sys
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq

uids, urls = [], []
for path in sorted(glob.glob(f"{sys.argv[1]}/*.tar")):
    with tarfile.open(path) as tf:
        for m in tf:
            if m.name.endswith(".url.txt"):
                uids.append(m.name[: -len(".url.txt")])
                urls.append(tf.extractfile(m).read().decode().strip())
pq.write_table(pa.table({"uid": uids, "url": urls}), sys.argv[2])
print(len(urls), "urls")
