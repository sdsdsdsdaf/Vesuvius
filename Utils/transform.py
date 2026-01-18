import torch
import torch.nn as nn
import torch.optim as optim
import monai

from monai.transforms import (
    Compose,
    ScaleIntensityRange,
    RandSpatialCrop,
    RandFlip,
    RandRotate90,
    RandShiftIntensity,
)

def train_transformation(image, label):
    data = {"image": image, "label": label}
    pipeline = Compose([
        ScaleIntensityRange(
            keys=["image"],
            a_min = 0,
            a_max = 255,
            clip = True,
        ),
        
        RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlip(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90(
            keys=["image", "label"], 
            prob=0.4, 
            max_k=3, 
            spatial_axes=(0, 1)
        ),
        RandShiftIntensity(keys=["image"], offsets=0.10, prob=0.5),
    ])
    result = pipeline(data)
    return result["image"], result["label"]


def val_transformation(image, label):
    data = {"image": image, "label": label}
    pipeline = Compose([
        ScaleIntensityRange(
            keys=["image"],
            a_min = 0,
            a_max = 255,
            clip = True,
        ),
    ])
    result = pipeline(data)
    return result["image"], result["label"]