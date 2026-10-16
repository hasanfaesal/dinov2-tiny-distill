"""k-NN (k=20, cosine, temperature-weighted vote) on CIFAR-100 from backbone CLS features.
Usage: evaluate_knn.py CKPT [CKPT...]   |   evaluate_knn.py --teacher"""
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import normalize


def load_cifar(root="/home/dev/data/cifar100_hf/cifar100"):
    """HF mirror (uoft-cs/cifar100; the toronto host is ~100 KB/s from the node). Cached as uint8 tensors."""
    cache = f"{root}/cifar100_uint8.pt"
    if os.path.exists(cache):
        return torch.load(cache)
    out = []
    for split in ("train", "test"):
        t = pq.read_table(f"{root}/{split}-00000-of-00001.parquet").to_pydict()
        x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        out.append((torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(t["fine_label"])))
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

    cifar = load_cifar()
    if sys.argv[1] == "--teacher":
        print("teacher knn top1", knn_eval(teacher_fn(make_teacher().cuda()), cifar, 196))
    for path in sys.argv[1:]:
        if path == "--teacher":
            continue
        s = Student().cuda().eval()
        s.load_state_dict(torch.load(path, map_location="cuda")["student"])
        print(path, "knn top1", knn_eval(student_fn(s), cifar, 224))
