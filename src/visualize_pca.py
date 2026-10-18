"""Generate side-by-side patch token PCA visualizations for DINOv2 teacher and student.
Outputs high-res PNG comparison for the blog post.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from models import Student, make_teacher
from dataset import normalize

def get_pca_map(patch_tokens, h, w):
    """
    patch_tokens: (N, D) where N = h * w
    Returns: (h, w, 3) normalized RGB array in [0, 1]
    """
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(patch_tokens.cpu().float().numpy())  # (N, 3)
    
    # Normalize each component to [0, 1]
    for c in range(3):
        comp = feats_pca[:, c]
        low, high = np.percentile(comp, 1), np.percentile(comp, 99)
        comp = np.clip(comp, low, high)
        feats_pca[:, c] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
        
    return feats_pca.reshape(h, w, 3)

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load teacher
    print("Loading teacher (dinov2_vits14_reg)...")
    teacher = make_teacher().to(device).eval()
    
    # 2. Baseline untrained student
    print("Initializing baseline untrained student (vit_tiny_patch16_224)...")
    student_untrained = Student().to(device).eval()
    
    # 3. Check for trained student checkpoint
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    student_trained = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading trained student from {ckpt_path}...")
        student_trained = Student().to(device).eval()
        ck = torch.load(ckpt_path, map_location=device)
        student_trained.load_state_dict(ck.get("student", ck))
    
    # 4. Prepare sample images (from CIFAR-100 or dummy if none)
    os.makedirs("/home/dev/visualizations", exist_ok=True)
    
    cifar_path = "/home/dev/data/cifar100_hf/cifar100/cifar100_uint8.pt"
    samples = []
    if os.path.exists(cifar_path):
        cifar = torch.load(cifar_path)
        (xtr, _), _ = cifar
        # Pick 6 diverse samples
        indices = [10, 42, 105, 250, 512, 1024]
        for idx in indices:
            samples.append(xtr[idx])
    else:
        print("CIFAR uint8 cache not found, generating synthetic test pattern")
        samples.append(torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8))
        
    print(f"Processing {len(samples)} sample images...")
    
    for idx, img_uint8 in enumerate(samples):
        # Image shape: (3, H, W)
        img_t = img_uint8.to(device)
        # Normalize and prepare sizes:
        # Teacher expects 196x196 -> 14x14 patches (14px patch)
        # Student expects 224x224 -> 14x14 patches (16px patch)
        x_teacher = normalize(img_t.unsqueeze(0))
        x_teacher = F.interpolate(x_teacher, size=(196, 196), mode="bicubic", align_corners=False)
        
        x_student = normalize(img_t.unsqueeze(0))
        x_student = F.interpolate(x_student, size=(224, 224), mode="bicubic", align_corners=False)
        
        # Extract patch tokens
        # Teacher output dict: x_norm_patchtokens shape: (B, 196, 384)
        t_out = teacher.forward_features(x_teacher.to(torch.bfloat16))
        t_patches = t_out["x_norm_patchtokens"][0] # (196, 384)
        t_pca = get_pca_map(t_patches, 14, 14)
        
        # Untrained student:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_un_feat = student_untrained.backbone.forward_features(x_student)
            s_un_patches = s_un_feat[:, 1:] # drop CLS token -> (1, 196, 192)
            s_un_pca = get_pca_map(s_un_patches[0], 14, 14)
            
        cols = 3 if student_trained is None else 4
        fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
        
        # Original Image
        orig_np = img_uint8.permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(orig_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        # Teacher PCA
        axes[1].imshow(t_pca, interpolation="nearest")
        axes[1].set_title("DINOv2 Teacher (vits14)\n14x14 Patch PCA")
        axes[1].axis("off")
        
        # Untrained Student PCA
        axes[2].imshow(s_un_pca, interpolation="nearest")
        axes[2].set_title("ViT-Tiny (Untrained)\nRandom Features")
        axes[2].axis("off")
        
        if student_trained is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_tr_feat = student_trained.backbone.forward_features(x_student)
                s_tr_patches = s_tr_feat[:, 1:]
                s_tr_pca = get_pca_map(s_tr_patches[0], 14, 14)
            axes[3].imshow(s_tr_pca, interpolation="nearest")
            axes[3].set_title(f"ViT-Tiny (Distilled)\nRun 2 / {os.path.basename(ckpt_path)}")
            axes[3].axis("off")
            
        plt.tight_layout()
        out_file = f"/home/dev/visualizations/pca_comparison_sample_{idx}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
