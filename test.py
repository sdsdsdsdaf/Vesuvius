from monai.networks.nets import UNet
import torch
from torchsummary import summary

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

input_t = torch.randn((1,1,160,160,160))
summary(input_t)