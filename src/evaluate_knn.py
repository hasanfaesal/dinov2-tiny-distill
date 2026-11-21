"""k-NN (k=20, cosine, temperature-weighted vote) on CIFAR-100 from backbone CLS features.

Usage:
  python src/evaluate_knn.py CKPT [CKPT...] [--cifar-root data/cifar100]
  python src/evaluate_knn.py --teacher [--cifar-root data/cifar100]
"""
import argparse
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import CIFAR100

sys.path.insert(0, os.path.dirname(__file__))
from dataset import normalize


def load_cifar(root=None):
    """Load CIFAR-100 as cached uint8 tensors; falls back to torchvision download."""
    root = root or os.environ.get("CIFAR_ROOT", "data/cifar100")
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, "cifar100_uint8.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=True)

    train_pq = os.path.join(root, "train-00000-of-00001.parquet")
    if os.path.exists(train_pq):
        import pyarrow.parquet as pq

        out = []
        for split in ("train", "test"):
            t = pq.read_table(os.path.join(root, f"{split}-00000-of-00001.parquet")).to_pydict()
            x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
            out.append((torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(t["fine_label"])))
    else:
        out = []
        for is_train in (True, False):
            ds = CIFAR100(root=root, train=is_train, download=True)
            x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
            out.append((x, torch.tensor(ds.targets, dtype=torch.long)))

    torch.save(out, cache)
    return out


@torch.no_grad()
def features(fn, x_uint8, size, bs=1000):
    out = []
    for i in range(0, len(x_uint8), bs):
        x = normalize(x_uint8[i : i + bs].cuda(non_blocking=True))
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
        out.append(F.normalize(fn(x).float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def knn_top1(train_f, train_y, test_f, test_y, k=20, T=0.07, num_classes=100):
    train_y, test_y, correct = train_y.cuda(), test_y.cuda(), 0
    for i in range(0, len(test_f), 1000):
        w, idx = (test_f[i : i + 1000] @ train_f.T).topk(k, dim=1)
        votes = torch.zeros(len(idx), num_classes, device=w.device).scatter_add_(1, train_y[idx], (w / T).exp())
        correct += (votes.argmax(1) == test_y[i : i + 1000]).sum().item()
    return correct / len(test_f)


def knn_eval(fn, cifar, size):
    (xtr, ytr), (xte, yte) = cifar
    return knn_top1(features(fn, xtr, size), ytr, features(fn, xte, size), yte)


def student_fn(student):
    def fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return student.backbone.forward_features(x)[:, 0]

    return fn


def teacher_fn(teacher):
    return lambda x: teacher.forward_features(x.to(torch.bfloat16))["x_norm_clstoken"]


if __name__ == "__main__":
    from models import Student, make_teacher

    ap = argparse.ArgumentParser(description="Evaluate CIFAR-100 k-NN (k=20) accuracy.")
    ap.add_argument("checkpoints", nargs="*", help="Student checkpoint (.pt) paths")
    ap.add_argument("--teacher", action="store_true", help="Evaluate frozen DINOv2 teacher")
    ap.add_argument("--cifar-root", default=None, help="CIFAR-100 cache/download directory")
    args = ap.parse_args()

    cifar = load_cifar(args.cifar_root)
    if args.teacher:
        print(f"teacher knn top1 {knn_eval(teacher_fn(make_teacher().cuda()), cifar, 196):.4f}")
    for path in args.checkpoints:
        s = Student().cuda().eval()
        ck = torch.load(path, map_location="cuda", weights_only=True)
        s.load_state_dict(ck["student"] if "student" in ck else ck)
        print(f"{path} knn top1 {knn_eval(student_fn(s), cifar, 224):.4f}")
