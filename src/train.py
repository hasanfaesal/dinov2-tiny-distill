"""Distill frozen DINOv2 ViT-S/14-reg into ViT-Tiny. Usage: python train.py [config] [--smoke N]"""
import math
import os
import sys
import time

import torch
import yaml
from prometheus_client import Gauge, start_http_server

from dataset import make_loader
from losses import distill_loss
from models import Student, Teacher

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1][0] != "-" else "configs/distill_config.yaml"))
smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0  # run N steps then exit

os.makedirs(cfg["ckpt_dir"], exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
teacher, student = Teacher().to(dev), Student().to(dev)  # student fp32 master weights, bf16 autocast
student.backbone = torch.compile(student.backbone, mode="reduce-overhead")
params = [p for p in student.parameters()]
decay = [p for p in params if p.ndim > 1]
no_decay = [p for p in params if p.ndim <= 1]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0}],
    lr=cfg["lr"], betas=(0.9, 0.999), fused=True,
)

spe, total = cfg["steps_per_epoch"], cfg["epochs"] * cfg["steps_per_epoch"]
warm = cfg["warmup_epochs"] * spe


def lr_at(step):
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    return 0.5 * cfg["lr"] * (1 + math.cos(math.pi * (step - warm) / (total - warm)))


G = {k: Gauge(k, k) for k in [
    "loss_total", "loss_cls", "loss_patch", "lr", "grad_norm_pre", "grad_norm_post", "clip_hit_frac",
    "param_norm", "update_to_weight", "imgs_per_sec", "step_time_s", "data_wait_s",
    "cos_mean", "cos_std", "feat_std_mean", "epoch", "step",
]}
start_http_server(cfg["metrics_port"])

resume = os.path.join(cfg["ckpt_dir"], "last.pt")
step, best = 0, float("inf")
if os.path.exists(resume):
    ck = torch.load(resume, map_location=dev)
    student.load_state_dict(ck["student"]); opt.load_state_dict(ck["opt"])
    step, best = ck["step"], ck["best"]
    print(f"resumed at step {step}")


def save(path):
    torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, path)


loader = make_loader(cfg["shards"], cfg["batch_size"], cfg["num_workers"])
it = iter(loader)
clip, ema, t_prev = cfg["grad_clip"], None, time.time()
while step < total:
    t0 = time.time()
    try:
        v1, v2 = next(it)
    except StopIteration:  # shard pass finished early; restart
        it = iter(loader); continue
    v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
    wait = time.time() - t0
    x = torch.cat([v1, v2])
    for g in opt.param_groups:
        g["lr"] = lr_at(step)

    t_cls, t_patch = teacher(x)
    with torch.autocast("cuda", torch.bfloat16):
        s_cls, s_patch = student(x)
    loss, l_cls, l_patch = distill_loss(s_cls.float(), s_patch.float(), t_cls, t_patch, cfg["patch_w"])

    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, clip)
    assert torch.isfinite(gn), f"non-finite grad at step {step}"
    log = step % cfg["log_every"] == 0
    if log:
        w_before = torch.cat([p.detach().flatten()[:1000] for p in decay]).clone()
    opt.step()
    step += 1

    if log:
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(s_patch.float(), t_patch, dim=-1)
            dt = time.time() - t_prev; t_prev = time.time()
            w_after = torch.cat([p.detach().flatten()[:1000] for p in decay])
            vals = dict(
                loss_total=loss.item(), loss_cls=l_cls.item(), loss_patch=l_patch.item(), lr=lr_at(step),
                grad_norm_pre=gn.item(), grad_norm_post=min(gn.item(), clip), clip_hit_frac=float(gn > clip),
                param_norm=torch.sqrt(sum((p.detach() ** 2).sum() for p in params)).item(),
                update_to_weight=((w_after - w_before).norm() / w_after.norm()).item(),
                imgs_per_sec=cfg["log_every"] * x.shape[0] / dt, step_time_s=dt / cfg["log_every"],
                data_wait_s=wait, cos_mean=cos.mean().item(), cos_std=cos.std().item(),
                feat_std_mean=s_patch.float().std(dim=(0, 1)).mean().item(),  # ~0 => collapse
                epoch=step / spe, step=step,
            )
        for k, v in vals.items():
            G[k].set(v)
        print(" ".join(f"{k}={v:.4g}" for k, v in vals.items()), flush=True)

    if step % spe == 0:
        ep = step // spe
        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        save(resume)
        if ema < best:
            best = ema; save(os.path.join(cfg["ckpt_dir"], "best.pt"))
        if ep % cfg["ckpt_every_epochs"] == 0:
            save(os.path.join(cfg["ckpt_dir"], f"epoch_{ep:03d}.pt"))
    if smoke and step >= smoke:
        print("SMOKE OK"); break
