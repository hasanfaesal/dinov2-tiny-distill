import torch.nn.functional as F


def distill_loss(s_cls, s_patch, t_cls, t_patch, patch_w=1.0):
    l_cls = (1 - F.cosine_similarity(s_cls, t_cls, dim=-1)).mean()
    l_patch = (1 - F.cosine_similarity(s_patch, t_patch, dim=-1)).mean()  # mean over B and 196
    return l_cls + patch_w * l_patch, l_cls, l_patch
