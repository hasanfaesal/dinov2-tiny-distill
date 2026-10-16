"""Phase 1 preflight: BF16, SDPA, teacher/student 14x14 grids, torch.compile."""
import torch
import timm

assert torch.cuda.is_available(), "no CUDA"
print("gpu:", torch.cuda.get_device_name(0), "torch:", torch.__version__, "cuda:", torch.version.cuda)
assert torch.cuda.is_bf16_supported(), "no bf16"
assert hasattr(torch.nn.functional, "scaled_dot_product_attention")
dev = "cuda"

# Teacher: 196x196 -> 14x14 patches
teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(dev, torch.bfloat16).eval()
for p in teacher.parameters():
    p.requires_grad = False
with torch.no_grad():
    out = teacher.forward_features(torch.randn(2, 3, 196, 196, device=dev, dtype=torch.bfloat16))
cls_t, patch_t = out["x_norm_clstoken"], out["x_norm_patchtokens"]
assert cls_t.shape == (2, 384) and patch_t.shape == (2, 196, 384), (cls_t.shape, patch_t.shape)
print("teacher ok", cls_t.shape, patch_t.shape)

# Student: 224x224 -> 14x14 patches
student = timm.create_model("vit_tiny_patch16_224", pretrained=False).to(dev, torch.bfloat16)
feats = student.forward_features(torch.randn(2, 3, 224, 224, device=dev, dtype=torch.bfloat16))
assert feats.shape == (2, 197, 192), feats.shape  # 1 cls + 196 patches
print("student ok", feats.shape)

# Fixed SDPA fused attention available (flash) on bf16
q = torch.randn(2, 6, 197, 32, device=dev, dtype=torch.bfloat16)
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
print("flash sdpa ok")

# torch.compile reduce-overhead, fwd+bwd
compiled = torch.compile(student, mode="reduce-overhead")
x = torch.randn(64, 3, 224, 224, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    compiled.forward_features(x).float().mean().backward()
torch.cuda.synchronize()
print("compile ok")
print("PREFLIGHT PASSED")
