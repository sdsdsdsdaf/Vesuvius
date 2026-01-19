import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    Lambdad
)


def get_train_transform():
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"]),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.4,
            max_k=3,
            spatial_axes=(0, 1),
        ),

        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),

        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
        Lambdad(keys=["label"], func=lambda x: x.clamp(0, 2)),
    ])

def get_val_transform() -> Compose:
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"]),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.float32),
        Lambdad(keys=["label"], func=lambda x: x.clamp(0, 2)),
    ])
