import torch
import torch.nn as nn
import torch.optim as optim
import monai
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

def collate_with_meta(batch):
    # batch: list of (x, y, meta)
    xs, ys, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    metas = list(metas)  # keep as list[dict]
    return xs, ys, metas

