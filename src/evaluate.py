"""Evaluation suite: downstream classification, feature geometry, and dense VOC patch probing.

Benchmarks:
  1. cifar100   - k-NN (k=20, T=0.07) & 15-epoch SGD linear probe
  2. imagenette - k-NN (k=20, T=0.07) & 15-epoch SGD linear probe
  3. geometry   - Linear CKA vs teacher, Effective Rank, pairwise cosine anisotropy
  4. voc        - Pascal VOC 2012 14x14 patch linear probe (mIoU & pixel accuracy)

Usage:
  python src/evaluate.py \\
    --models run2=hf-hub:hasanfaesal/vit-tiny-patch16-224-dinov2-distill \\
    --teacher --benchmarks cifar100 geometry --out-dir eval_out
"""
import argparse
import json
import os
import sys
import tarfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as tv_datasets
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
from dataset import MEAN, STD, normalize
from evaluate_knn import knn_top1, load_cifar
from models import make_teacher
from visualize_pca import load_student_backbone


def load_imagenette(data_dir):
    root = os.path.join(data_dir, "imagenette", "imagenette2-320")
    if not os.path.exists(root):
        os.makedirs(os.path.join(data_dir, "imagenette"), exist_ok=True)
        tgz = os.path.join(data_dir, "imagenette", "imagenette2-320.tgz")
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        print(f"Downloading Imagenette from {url} ...")
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(os.path.join(data_dir, "imagenette"), filter="data")
    return (
        tv_datasets.ImageFolder(os.path.join(root, "train")),
        tv_datasets.ImageFolder(os.path.join(root, "val")),
    )


@torch.no_grad()
def extract_cifar_cls(model, x_uint8, is_teacher=False, batch_size=256):
    size = (196, 196) if is_teacher else (224, 224)
    feats = []
    for i in range(0, len(x_uint8), batch_size):
        x = normalize(x_uint8[i : i + batch_size].cuda(non_blocking=True))
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
        if is_teacher:
            out = model.forward_features(x.to(torch.bfloat16))["x_norm_clstoken"]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_features(x)[:, 0]
        feats.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(feats)


@torch.no_grad()
def extract_imagefolder_cls(model, dataset, is_teacher=False, batch_size=128):
    size = (196, 196) if is_teacher else (224, 224)

    def collate(batch):
        imgs, targets = zip(*batch)
        tensors = [TF.normalize(TF.to_tensor(TF.resize(TF.center_crop(im, min(im.size)), size)), MEAN, STD) for im in imgs]
        return torch.stack(tensors), torch.tensor(targets, dtype=torch.long)

    feats, labels = [], []
    for imgs, targets in DataLoader(dataset, batch_size=batch_size, num_workers=2, collate_fn=collate):
        imgs = imgs.cuda(non_blocking=True)
        if is_teacher:
            out = model.forward_features(imgs.to(torch.bfloat16))["x_norm_clstoken"]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_features(imgs)[:, 0]
        feats.append(F.normalize(out.float(), dim=-1).cpu())
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(train_f, train_y, test_f, test_y, num_classes, epochs=15, lr=0.1, weight_decay=1e-4):
    probe = nn.Linear(train_f.shape[1], num_classes).cuda()
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(train_f, train_y), batch_size=256, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for bx, by in loader:
            loss = F.cross_entropy(probe(bx.cuda()), by.cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(test_f.cuda()).argmax(dim=-1).cpu()
    return float((preds == test_y).float().mean().item() * 100.0)


def linear_cka(x, y):
    """Linear Centered Kernel Alignment between [N, D1] and [N, D2] matrices."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    dot_xy = (x.T @ y).pow(2).sum()
    dot_xx = (x.T @ x).pow(2).sum()
    dot_yy = (y.T @ y).pow(2).sum()
    return float((dot_xy / (torch.sqrt(dot_xx * dot_yy) + 1e-12)).item())


def effective_rank(x):
    """Effective rank exp(H(singular values)) and normalized ER / D."""
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    erank = float(np.exp(-(p * torch.log(p)).sum().item()))
    return erank, erank / x.shape[1]


def pairwise_anisotropy(x, n_samples=1000):
    """Mean off-diagonal cosine similarity over random pairs (lower = more isotropic)."""
    idx = torch.randperm(len(x))[: min(n_samples, len(x))]
    sub = F.normalize(x[idx], dim=-1)
    sim = sub @ sub.T
    mask = ~torch.eye(len(sub), dtype=torch.bool)
    return float(sim[mask].mean().item())


def evaluate_voc_probe(models, voc_root, max_train=700, max_val=400, epochs=5):
    """14x14 patch-token linear probe on Pascal VOC 2012 segmentation (21 classes)."""
    download = not os.path.exists(os.path.join(voc_root, "VOCdevkit", "VOC2012"))
    train_voc = tv_datasets.VOCSegmentation(root=voc_root, year="2012", image_set="train", download=download)
    val_voc = tv_datasets.VOCSegmentation(root=voc_root, year="2012", image_set="val", download=False)
    num_classes = 21

    @torch.no_grad()
    def extract_patches(model, is_teacher, ds, limit):
        size = (196, 196) if is_teacher else (224, 224)
        feats, masks = [], []
        for i in range(min(len(ds), limit)):
            img, mask = ds[i]
            x = TF.normalize(TF.to_tensor(TF.resize(img, size)), MEAN, STD).unsqueeze(0).cuda()
            m14 = TF.resize(mask, (14, 14), interpolation=TF.InterpolationMode.NEAREST)
            if is_teacher:
                tok = model.forward_features(x.to(torch.bfloat16))["x_norm_patchtokens"][0].float()
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok = model.forward_features(x)[0, 1:].float()
            feats.append(tok.cpu())
            masks.append(torch.from_numpy(np.array(m14, dtype=np.int64)).view(-1))
        return torch.cat(feats), torch.cat(masks)

    results = {}
    for name, (model, is_teacher) in models.items():
        x_tr, y_tr = extract_patches(model, is_teacher, train_voc, max_train)
        x_val, y_val = extract_patches(model, is_teacher, val_voc, max_val)
        v_tr = (y_tr != 255) & (y_tr < num_classes)
        v_val = (y_val != 255) & (y_val < num_classes)

        probe = nn.Linear(x_tr.shape[1], num_classes).cuda()
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(x_tr[v_tr], y_tr[v_tr]), batch_size=1024, shuffle=True)
        for _ in range(epochs):
            probe.train()
            for bx, by in loader:
                loss = F.cross_entropy(probe(bx.cuda()), by.cuda())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(x_val[v_val].cuda()).argmax(dim=-1).cpu()
        y_gt = y_val[v_val]
        ious = []
        for c in range(num_classes):
            inter = ((preds == c) & (y_gt == c)).sum().item()
            union = ((preds == c) | (y_gt == c)).sum().item()
            if union > 0:
                ious.append(inter / union)
        miou = float(np.mean(ious) * 100.0) if ious else 0.0
        acc = float((preds == y_gt).float().mean().item() * 100.0)
        results[name] = {"mIoU": miou, "pixel_acc": acc}
        print(f"  VOC {name}: mIoU={miou:.2f}% pixel_acc={acc:.2f}%")
    return results


def parse_models(specs):
    models = {}
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = os.path.basename(spec), spec
        models[name] = (load_student_backbone(path, "cuda"), False)
    return models


def main():
    ap = argparse.ArgumentParser(description="Evaluate distilled ViT-Tiny checkpoints vs DINOv2 teacher.")
    ap.add_argument("--models", nargs="+", required=True, help="Model specs as name=path (or hf-hub:...)")
    ap.add_argument("--teacher", action="store_true", help="Include frozen dinov2_vits14_reg teacher")
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=["cifar100", "imagenette", "geometry", "voc"],
        choices=["cifar100", "imagenette", "geometry", "voc"],
    )
    ap.add_argument("--data-dir", default="data", help="Root directory for benchmark datasets")
    ap.add_argument("--out-dir", default="eval_out", help="Directory to write metrics.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    models = parse_models(args.models)
    if args.teacher:
        models["teacher"] = (make_teacher().cuda(), True)

    metrics = {}
    c100_test_feats = {}

    if "cifar100" in args.benchmarks or "geometry" in args.benchmarks:
        print("\n--- CIFAR-100 ---")
        (x_tr, y_tr), (x_te, y_te) = load_cifar(os.path.join(args.data_dir, "cifar100"))
        c100_res = {"knn_top1": {}, "linear_probe_top1": {}}
        for name, (m, is_t) in models.items():
            f_te = extract_cifar_cls(m, x_te, is_teacher=is_t)
            c100_test_feats[name] = f_te
            if "cifar100" in args.benchmarks:
                f_tr = extract_cifar_cls(m, x_tr, is_teacher=is_t)
                knn = knn_top1(f_tr.cuda(), y_tr, f_te.cuda(), y_te, k=20, num_classes=100) * 100.0
                lp = train_linear_probe(f_tr, y_tr, f_te, y_te, num_classes=100)
                c100_res["knn_top1"][name] = knn
                c100_res["linear_probe_top1"][name] = lp
                print(f"  {name}: k-NN={knn:.2f}% linear_probe={lp:.2f}%")
        if "cifar100" in args.benchmarks:
            metrics["cifar100"] = c100_res

    if "geometry" in args.benchmarks:
        print("\n--- Representation Geometry & Alignment (CIFAR-100 test) ---")
        geom = {"effective_rank": {}, "effective_rank_normalized": {}, "anisotropy": {}, "linear_cka": {}}
        names = list(c100_test_feats.keys())
        for name in names:
            er, er_norm = effective_rank(c100_test_feats[name])
            aniso = pairwise_anisotropy(c100_test_feats[name])
            geom["effective_rank"][name] = er
            geom["effective_rank_normalized"][name] = er_norm
            geom["anisotropy"][name] = aniso
            print(f"  {name}: ER={er:.1f} ({er_norm:.3f}) anisotropy={aniso:.4f}")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                cka = linear_cka(c100_test_feats[a], c100_test_feats[b])
                geom["linear_cka"][f"{a}_vs_{b}"] = cka
                print(f"  Linear CKA ({a} vs {b}): {cka:.4f}")
        metrics["geometry"] = geom

    if "imagenette" in args.benchmarks:
        print("\n--- Imagenette ---")
        tr_ds, val_ds = load_imagenette(args.data_dir)
        inet_res = {"knn_top1": {}, "linear_probe_top1": {}}
        for name, (m, is_t) in models.items():
            f_tr, y_tr = extract_imagefolder_cls(m, tr_ds, is_teacher=is_t)
            f_te, y_te = extract_imagefolder_cls(m, val_ds, is_teacher=is_t)
            knn = knn_top1(f_tr.cuda(), y_tr, f_te.cuda(), y_te, k=20, num_classes=10) * 100.0
            lp = train_linear_probe(f_tr, y_tr, f_te, y_te, num_classes=10)
            inet_res["knn_top1"][name] = knn
            inet_res["linear_probe_top1"][name] = lp
            print(f"  {name}: k-NN={knn:.2f}% linear_probe={lp:.2f}%")
        metrics["imagenette"] = inet_res

    if "voc" in args.benchmarks:
        print("\n--- Pascal VOC 2012 Patch Linear Probe ---")
        metrics["voc_segmentation"] = evaluate_voc_probe(models, os.path.join(args.data_dir, "voc"))

    out_path = os.path.join(args.out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
