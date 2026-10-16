"""Phase 2 I/O gate: loader throughput in img/s (>= 2500 target)."""
import sys
import time

sys.path.insert(0, "/home/dev/repo/src")
from dataset import make_loader

dl = make_loader("/home/dev/data/images/*.tar", num_workers=24)
it = iter(dl)
for _ in range(5):  # warmup
    next(it)
n, t = 0, time.time()
for _ in range(400):  # >> 24 workers x prefetch 4 = 96 buffered batches
    v1, v2 = next(it)
    assert v1.shape == v2.shape == (256, 3, 224, 224), v1.shape
    n += v1.shape[0]
rate = n / (time.time() - t)
print(f"{rate:.0f} img/s ->", "PASS" if rate >= 2500 else "FAIL")
