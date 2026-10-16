"""Frozen DINOv2 teacher, ViT-Tiny student + projection heads."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(torch.bfloat16).eval()
        for p in self.m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):  # x: [B,3,224,224] -> cls [B,384], patches [B,196,384]
        x = F.interpolate(x, size=196, mode="bicubic", align_corners=False).to(torch.bfloat16)
        o = self.m.forward_features(x)
        return o["x_norm_clstoken"].float(), o["x_norm_patchtokens"].float()


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        self.cls_head = nn.Sequential(nn.Linear(192, 1024), nn.GELU(), nn.Linear(1024, 384))
        self.patch_head = nn.Linear(192, 384)

    def forward(self, x):  # -> cls [B,384], patches [B,196,384]
        f = self.backbone.forward_features(x)  # [B,197,192] (no reg tokens in tiny)
        return self.cls_head(f[:, 0]), self.patch_head(f[:, 1:])
