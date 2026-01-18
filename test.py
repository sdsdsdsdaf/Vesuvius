import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xformers")
import torch
from xformers.ops import memory_efficient_attention


device = "cuda"
dtype = torch.float16

B, H, S, D = 2, 8, 512, 64

q = torch.randn(B, S, H, D, device=device, dtype=dtype)
k = torch.randn(B, S, H, D, device=device, dtype=dtype)
v = torch.randn(B, S, H, D, device=device, dtype=dtype)

out = memory_efficient_attention(q, k, v)

print("Output shape:", out.shape)
print("OK: xFormers attention ran successfully")

import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1024, 1024)
    def forward(self, x):
        return self.l(x).relu()

m = M().cuda().half()
x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)

mc = torch.compile(m)  # 기본 backend=inductor
y = mc(x)
print("ok, y:", y.shape)