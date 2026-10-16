import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_teacher():
    t = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(torch.bfloat16).eval()
    for p in t.parameters():
        p.requires_grad = False
    return t


@torch.no_grad()
def teacher_features(teacher, x224):
    """x224: [B,3,224,224] -> cls [B,384], patches [B,196,384] at 196x196 input."""
    x = F.interpolate(x224.to(torch.bfloat16), size=196, mode="bicubic", antialias=True, align_corners=False)
    out = teacher.forward_features(x)
    return out["x_norm_clstoken"].float(), out["x_norm_patchtokens"].float()


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        self.cls_head = nn.Sequential(nn.Linear(192, 1024), nn.GELU(), nn.Linear(1024, 384))
        self.patch_head = nn.Linear(192, 384)

    def forward(self, x):
        f = self.backbone.forward_features(x)  # [B,197,192]
        return self.cls_head(f[:, 0]), self.patch_head(f[:, 1:])
