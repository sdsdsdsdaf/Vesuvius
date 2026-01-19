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


def get_train_transform(filp_prob=0.5, rotate_prob=0.4, shift_intensity_prob=0.5):
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=filp_prob),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=filp_prob),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=filp_prob),

        RandRotate90d(
            keys=["image", "label"],
            prob=rotate_prob,
            max_k=3,
            spatial_axes=(1, 2),
        ),

        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=shift_intensity_prob),

        Lambdad(keys=["label"], func=lambda x: x.clamp(0, 2)),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        EnsureTyped(keys=["label"], dtype=torch.long, track_meta=False),
        
    ])

def get_val_transform() -> Compose:
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        Lambdad(keys=["label"], func=lambda x: x.clamp(0, 2)),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        EnsureTyped(keys=["label"], dtype=torch.long, track_meta=False),
        
    ])
