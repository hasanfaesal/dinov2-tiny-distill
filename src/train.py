"""Distill frozen DINOv2 ViT-S/14-reg into ViT-Tiny/16. Usage: train.py [--smoke]"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dataset import make_loader, normalize
from losses import distill_loss
from models import Student, make_teacher, teacher_features

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "../configs/distill_config.yaml"))
ap.add_argument("--smoke", action="store_true", help="short run on a few shards, own ckpt dir")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))

shards = sorted(glob.glob(cfg["data_glob"]))
n_images = sum(
    json.load(open(s.replace(".tar", "_stats.json")))["successes"]
    for s in shards
    if os.path.exists(s.replace(".tar", "_stats.json"))
)
bs = cfg["batch_size"]
steps_per_epoch = n_images // bs
if args.smoke:
    cfg["data_glob"] = cfg["data_glob"].replace("*.tar", "0000[0-7].tar")
    cfg["ckpt_dir"] += "_smoke"
    cfg["num_workers"] = 8
    cfg["log_every"] = 5
    steps_per_epoch, cfg["epochs"], cfg["warmup_epochs"], cfg["ckpt_every_epochs"] = 40, 2, 1, 1
total_steps = steps_per_epoch * cfg["epochs"]
warmup = steps_per_epoch * cfg["warmup_epochs"]
print(f"images={n_images} steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
teacher = make_teacher().to(dev)
student = Student().to(dev)  # fp32 master weights, bf16 autocast
opt = torch.optim.AdamW(
    [
        {"params": [p for p in student.parameters() if p.ndim > 1], "weight_decay": cfg["weight_decay"]},
        {"params": [p for p in student.parameters() if p.ndim <= 1], "weight_decay": 0.0},
    ],
    lr=cfg["lr"], betas=(0.9, 0.999),
)
cstudent = torch.compile(student, mode="reduce-overhead")


def lr_at(step):
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    return 0.5 * cfg["lr"] * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))


os.makedirs(cfg["ckpt_dir"], exist_ok=True)
last = os.path.join(cfg["ckpt_dir"], "last.pt")
step, start_epoch, best = 0, 0, float("inf")
if os.path.exists(last):
    ck = torch.load(last, map_location=dev)
    student.load_state_dict(ck["student"])
    opt.load_state_dict(ck["opt"])
    step, start_epoch, best = ck["step"], ck["epoch"], ck["best"]
    print(f"resumed from epoch {start_epoch} step {step}", flush=True)

try:  # live metrics for the mission page
    from prometheus_client import Gauge, start_http_server

    start_http_server(cfg["metrics_port"])
    G = {k: Gauge(k, k) for k in ["loss", "loss_cls", "loss_patch", "lr", "grad_norm", "img_per_sec", "step_time", "data_wait"]}
except ImportError:
    G = None

loader = make_loader(cfg["data_glob"], bs, cfg["num_workers"])


def save(path, epoch):
    torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch, "best": best}, path)


for epoch in range(start_epoch, cfg["epochs"]):
    ep_loss, ep_n, t_prev = 0.0, 0, time.time()
    it = iter(loader)
    for _ in range(steps_per_epoch):
        t0 = time.time()
        try:
            v1, v2 = next(it)
        except StopIteration:  # n_images estimate slightly above what loader yields
            break
        x = normalize(torch.cat([v1, v2]).to(dev, non_blocking=True))
        wait = time.time() - t0
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        t_cls, t_patch = teacher_features(teacher, x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_cls, s_patch = cstudent(x)
        loss, l_cls, l_patch = distill_loss(s_cls.float(), s_patch.float(), t_cls, t_patch, cfg["patch_weight"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1
        if step % cfg["log_every"] == 0:
            lv = loss.item()
            assert math.isfinite(lv) and math.isfinite(gn.item()), f"non-finite at step {step}"
            dt = (time.time() - t_prev) / cfg["log_every"]
            t_prev = time.time()
            ep_loss += lv
            ep_n += 1
            print(f"ep {epoch} step {step}/{total_steps} loss {lv:.4f} cls {l_cls.item():.4f} patch {l_patch.item():.4f} "
                  f"lr {lr_at(step):.2e} gn {gn.item():.2f} {2 * bs / dt:.0f} img/s wait {wait:.3f}", flush=True)
            if G:
                for k, v in zip(G, [lv, l_cls.item(), l_patch.item(), lr_at(step), gn.item(), 2 * bs / dt, dt, wait]):
                    G[k].set(v)
    avg = ep_loss / max(ep_n, 1)
    print(f"=== epoch {epoch} done avg_loss {avg:.4f}", flush=True)
    if avg < best:
        best = avg
        save(os.path.join(cfg["ckpt_dir"], "best.pt"), epoch + 1)
    if (epoch + 1) % cfg["ckpt_every_epochs"] == 0:
        save(os.path.join(cfg["ckpt_dir"], f"epoch{epoch + 1}.pt"), epoch + 1)
    save(last, epoch + 1)
print("TRAINING DONE", flush=True)
