"""WebDataset loader: each image -> 2 global 224x224 views (fixed shape)."""
import glob
import random

import torch
import webdataset as wds
from PIL import ImageFilter
from torchvision.transforms import v2 as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def pil_blur(img):  # PIL's C blur; torchvision's tensor GaussianBlur was ~37 ms/img on one core
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def global_view():
    """uint8 [3,224,224]; float conversion + normalize happen on GPU (normalize())."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2)], p=0.8),
        T.RandomApply([pil_blur], p=0.5),
        T.RandomSolarize(threshold=128, p=0.2),
        T.PILToTensor(),
    ])


def normalize(x):
    """uint8 [B,3,H,W] on GPU -> normalized float."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255 - mean) / std


def make_loader(shard_glob, batch_size=256, num_workers=8, shuffle_buf=2000):
    shards = sorted(glob.glob(shard_glob))
    assert shards, f"no shards match {shard_glob}"
    v1, v2 = global_view(), global_view()

    def views(sample):
        img = sample["jpg"].convert("RGB")
        return v1(img), v2(img)

    ds = (
        wds.WebDataset(shards, shardshuffle=100, handler=wds.warn_and_continue, empty_check=False)
        .shuffle(shuffle_buf)
        .decode("pil", handler=wds.warn_and_continue)
        .map(views, handler=wds.warn_and_continue)
        .batched(batch_size, partial=False)
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=True, prefetch_factor=4,
        persistent_workers=True,
    )  # yields [B,3,224,224] pair: (view1, view2)
