# ViT-Tiny/16 Distilled from DINOv2 on DataComp-12M

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-vit--tiny--patch16--224--dinov2--distill-yellow.svg)](https://huggingface.co/hasanfaesal/vit-tiny-patch16-224-dinov2-distill)
[![Teacher: DINOv2 ViT-S/14](https://img.shields.io/badge/Teacher-DINOv2_ViT--S%2F14-orange.svg)](https://github.com/facebookresearch/dinov2)

Self-supervised feature distillation of a frozen DINOv2 ViT-S/14 (`dinov2_vits14_reg`, 21.8M parameters) teacher into a 5.7M-parameter ViT-Tiny/16 (`vit_tiny_patch16_224`) trained from random initialization on up to 5.94M images from DataComp-12M on a single NVIDIA H100.

Feeding the student 224×224 images (patch size 16) and resizing the same views to 196×196 for the teacher (patch size 14) gives both models an identical 14×14 grid of 196 patch tokens. Each student patch token pairs directly with one spatial counterpart in the teacher, so cosine regression on `[CLS]` and patch tokens runs without token interpolation, centering, or sharpening.

![Emergent self-attention maps of distilled ViT-Tiny](assets/vit_tiny_attention.png)
*Run 2 last-layer `[CLS]` self-attention, showing the best of 3 heads per probe image without segmentation or detection labels.*

## Quickstart

Pretrained backbones and full student checkpoints (including projection heads) are hosted on Hugging Face as `safetensors` at [`hasanfaesal/vit-tiny-patch16-224-dinov2-distill`](https://huggingface.co/hasanfaesal/vit-tiny-patch16-224-dinov2-distill):

```python
import timm
import torch
from timm.data import create_transform, resolve_data_config

model = timm.create_model(
    "hf-hub:hasanfaesal/vit-tiny-patch16-224-dinov2-distill",
    pretrained=True,
).eval()

cfg = resolve_data_config({}, model=model)  # uses ImageNet mean/std from config.json
transform = create_transform(**cfg)

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    feats = model.forward_features(x)  # [2, 197, 192]
    cls_token = feats[:, 0]            # [2, 192]
    patch_tokens = feats[:, 1:]        # [2, 196, 192]
```

## Empirical results

We trained two runs on a single NVIDIA H100 80GB SXM5: Run 1 on 1.22M images for 40 epochs (190,040 steps, 3.6 h at ~4,088 img/s) and Run 2 on 5.94M images for 30 epochs (696,000 steps, 12.4 h at ~3,984 img/s). Both are evaluated under identical protocols using L2-normalized `[CLS]` features, cosine $k$-NN ($k=20, T=0.07$), 15-epoch SGD linear probes, and a Pascal VOC 2012 14×14 patch linear probe.

| Benchmark | Metric | Run 1 (1.22M / 40 ep) | **Run 2 (5.94M / 30 ep)** | Teacher (`dinov2_vits14_reg`) |
|---|---|---:|---:|---:|
| CIFAR-100 | $k$-NN Top-1 ($k=20$) | 75.50% | **76.25%** | 84.12% |
| CIFAR-100 | Linear Probe Top-1 | 65.72% | **66.97%** | 79.48% |
| Imagenette | $k$-NN Top-1 ($k=20$) | 97.58% | **97.83%** | 99.41% |
| Imagenette | Linear Probe Top-1 | 95.69% | **95.90%** | 98.55% |
| Pascal VOC 2012 | Patch Linear Probe mIoU | **56.97%** | 56.41% | 68.23% |
| Pascal VOC 2012 | Patch Pixel Accuracy | **89.19%** | 89.13% | 92.12% |
| Alignment | Linear CKA vs. Teacher | 0.6977 | 0.6971 | 1.0000 |
| Geometry | Effective Rank ($\text{ER} / D$) | 153.6 (0.800) | 151.7 (0.790) | 335.3 (0.873) |
| Geometry | Pairwise Cosine Anisotropy ↓ | 0.2387 | **0.1606** | 0.1090 |

Run 2 wins all four classification benchmarks and cuts mean pairwise cosine similarity from 0.2387 to 0.1606, moving closer to the teacher's 0.1090. Both runs reach nearly the same representation manifold (Run 1 vs. Run 2 Linear CKA is 0.9010) and recover over 82% of the teacher's VOC patch mIoU with a quarter of the parameters, though Run 1 stays 0.56 mIoU ahead on VOC patch probing.

Results are from a single seed per run on CIFAR-100, Imagenette, and Pascal VOC 2012 rather than full ImageNet-1k. Note also that the student was trained with ImageNet normalization, whereas `timm`'s default config for `vit_tiny_patch16_224` uses 0.5/0.5 mean and std; `resolve_data_config` above reads the right values from `config.json`.

## Method

The teacher is `facebookresearch/dinov2` (`dinov2_vits14_reg`, 384-dim, 4 register tokens), kept frozen in `bfloat16` with inputs bicubic-resized to 196×196. The student is `timm`'s `vit_tiny_patch16_224` (192-dim, 12 blocks, 3 heads, 5.7M parameters), trained from random initialization under `bfloat16` autocast with `torch.compile(mode="reduce-overhead")`. Two projection heads map student tokens to the teacher's 384-dim space: `Linear(192, 1024) -> GELU -> Linear(1024, 384)` for `[CLS]` and `Linear(192, 384)` for patch tokens.

The training loss sums cosine distance on `[CLS]` and mean cosine distance across the 196 patch tokens:

$$\mathcal{L} = \underbrace{1 - \cos\big(\mathrm{cls\_head}(S_{\text{cls}}),\, T_{\text{cls}}\big)}_{\mathcal{L}_{\text{cls}}} + \lambda \cdot \underbrace{\frac{1}{196}\sum_{i=1}^{196}\Big(1 - \cos\big(\mathrm{patch\_head}(S^{(i)}),\, T^{(i)}\big)\Big)}_{\mathcal{L}_{\text{patch}}}, \quad \lambda = 1.0$$

Each training image produces two global 224×224 views (random resized crop `0.4–1.0`, horizontal flip, color jitter `p=0.8`, Gaussian blur `p=0.5`, solarization `p=0.2`). Optimization uses AdamW (`lr=1e-3`, `weight_decay=0.05`, linear warmup with cosine decay, gradient clipping at `3.0`) at batch size 256 images (512 views per step).

## Reproducing training and evaluation

Install dependencies:

```bash
pip install -r requirements.txt
```

Fetch DataComp-12M WebDataset shards via `img2dataset` (the Hugging Face repository hosts URL metadata):

```bash
pip install img2dataset
bash scripts/fetch_images.sh 0 170 data/images
```

Train Run 1, Run 2, or a 2-epoch smoke test on 8 shards:

```bash
python src/train.py --config configs/distill_config.yaml
python src/train.py --config configs/distill_full.yaml
python src/train.py --config configs/distill_config.yaml --smoke
```

Run the full evaluation suite (CIFAR-100, Imagenette, CKA/ER/anisotropy, Pascal VOC) or a fast CIFAR-100 $k$-NN check:

```bash
python src/evaluate.py \
  --models run2=hf-hub:hasanfaesal/vit-tiny-patch16-224-dinov2-distill \
  --teacher \
  --benchmarks cifar100 imagenette geometry voc \
  --out-dir eval_out

python src/evaluate_knn.py checkpoints/run2/best_knn.pt --teacher
```

## References

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024 ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193))
- Gadre et al., *DataComp: In Search of the Next Generation of Multimodal Datasets*, NeurIPS 2023 ([arXiv:2304.14108](https://arxiv.org/abs/2304.14108))
