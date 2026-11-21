"""Generate side-by-side 3-component patch-token PCA maps for DINOv2 teacher and ViT-Tiny student.

Usage:
  python src/visualize_pca.py --checkpoint checkpoints/run2/best_knn.pt --images img1.jpg img2.jpg
  python src/visualize_pca.py --checkpoint hf-hub:hasanfaesal/vit-tiny-patch16-224-dinov2-distill --images img1.jpg
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from dataset import normalize
from models import Student, make_teacher


def get_pca_map(patch_tokens, h=14, w=14):
    """Project (N, D) patch tokens onto top-3 PCA components normalized to [0, 1] RGB."""
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(patch_tokens.cpu().float().numpy())
    for c in range(3):
        comp = feats_pca[:, c]
        low, high = np.percentile(comp, 1), np.percentile(comp, 99)
        comp = np.clip(comp, low, high)
        feats_pca[:, c] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
    return feats_pca.reshape(h, w, 3)


def load_student_backbone(checkpoint, device):
    if checkpoint.startswith("hf-hub:"):
        return timm.create_model(checkpoint, pretrained=True).to(device).eval()
    backbone = Student().backbone.to(device).eval()
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(checkpoint, device=device)
    else:
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        sd = ck["student"] if "student" in ck else ck
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    backbone.load_state_dict(sd, strict=False)
    return backbone


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Render side-by-side Teacher vs Student patch PCA maps.")
    ap.add_argument("--checkpoint", default=None, help="Student checkpoint (.pt, .safetensors, or hf-hub:...)")
    ap.add_argument("--images", nargs="*", default=[], help="Input image paths (defaults to CIFAR-100 samples)")
    ap.add_argument("--cifar-root", default="data/cifar100", help="Fallback CIFAR-100 directory when --images is omitted")
    ap.add_argument("--out-dir", default="visualizations", help="Output directory for PNGs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    teacher = make_teacher().to(device).eval()
    untrained = Student().backbone.to(device).eval()
    trained = load_student_backbone(args.checkpoint, device) if args.checkpoint else None

    samples = []
    if args.images:
        for path in args.images:
            arr = np.asarray(Image.open(path).convert("RGB"))
            samples.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    else:
        from evaluate_knn import load_cifar

        (xtr, _), _ = load_cifar(args.cifar_root)
        samples = [xtr[i] for i in (10, 42, 105, 250, 512, 1024)]

    amp_ctx = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
    for idx, img_uint8 in enumerate(samples):
        img_t = img_uint8.to(device).unsqueeze(0)
        x_teacher = F.interpolate(normalize(img_t), size=(196, 196), mode="bicubic", align_corners=False)
        x_student = F.interpolate(normalize(img_t), size=(224, 224), mode="bicubic", align_corners=False)

        t_patches = teacher.forward_features(x_teacher.to(torch.bfloat16))["x_norm_patchtokens"][0]
        t_pca = get_pca_map(t_patches)

        with amp_ctx:
            s_un_pca = get_pca_map(untrained.forward_features(x_student)[0, 1:])
            s_tr_pca = get_pca_map(trained.forward_features(x_student)[0, 1:]) if trained is not None else None

        cols = 3 if s_tr_pca is None else 4
        fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
        axes[0].imshow(img_uint8.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Input Image")
        axes[1].imshow(t_pca, interpolation="nearest")
        axes[1].set_title("DINOv2 Teacher (ViT-S/14)\n14x14 Patch PCA")
        axes[2].imshow(s_un_pca, interpolation="nearest")
        axes[2].set_title("ViT-Tiny (Untrained)\nRandom Features")
        if s_tr_pca is not None:
            axes[3].imshow(s_tr_pca, interpolation="nearest")
            axes[3].set_title("ViT-Tiny (Distilled)\n14x14 Patch PCA")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_file = os.path.join(args.out_dir, f"pca_comparison_sample_{idx}.png")
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
