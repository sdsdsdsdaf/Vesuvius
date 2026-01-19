
# %% [markdown]
# ## PyVista 등치면 확인 (Blob체크)

# %%
import glob
import os
import kagglehub
import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv
import tifffile as tiff
from topometrics import compute_leaderboard_score
import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from pprint import pprint
import time

from Utils.metric import metric

data_path = kagglehub.competition_download('vesuvius-challenge-surface-detection')
test_data_dir = os.path.join(data_path, "test_images")
train_data_dir = os.path.join(data_path, "train_images")
train_lable_dir = os.path.join(data_path, "train_labels")

train_data = sorted(glob.glob(os.path.join(train_data_dir, "*")))
sample = train_data[:10]
label = sorted(glob.glob(os.path.join(train_lable_dir, "*")))

arr = tiff.imread(sample[0])
x = torch.from_numpy(arr)  # (Z,H,W) or (H,W)
arr = tiff.imread(label[0])
y = torch.from_numpy(arr)

print("Train Data Size: ", len(train_data))
print("IMG SHAPE: ",x.shape, x.dtype ,x.min(), x.max())
print("LABEL SHAPE: ", y.shape, y.dtype, y.min(), y.max())

L = y.numpy()
V = x.numpy()

print("=============== Label Visualize ================")
# 2D 라벨이면 broadcast (참고용일 뿐)
if L.ndim == 2:
    L = np.repeat(L[None, ...], V.shape[0], axis=0)

step = 1

Ld = L[::step, ::step, ::step]

ink = (Ld == 1).astype(np.float32)
verts, faces, normals, values = marching_cubes(ink, level=0.5)

faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
mesh = pv.PolyData(verts, faces_pv)

p = pv.Plotter()
p.add_mesh(mesh, opacity=0.9)
p.show()

# ======================== #
#   Prediction Visualize   #
# ======================== #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=============== Prediction Visualize ================\n")
model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

ckpt = torch.load("weights/unet_vesuvius.pth", map_location=device)

# ckpt가 state_dict 자체거나, {"state_dict": ...} 형태일 수 있어서 둘 다 처리
sd = ckpt.get("state_dict", ckpt)

# torch.compile 저장본이면 키가 "_orig_mod."로 시작함 -> 제거
if any(k.startswith("_orig_mod.") for k in sd.keys()):
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

model.load_state_dict(sd)
print("Model Load Complete.\n")

print("Performing inference...")

t0 = time.perf_counter()
with torch.inference_mode():
     with torch.autocast("cuda", dtype=torch.float16):
        inputs = torch.tensor(V[None, None, ...], dtype=torch.float32).to(device)  # (1, 1, D, H, W)
        logits = sliding_window_inference(
            inputs,
            roi_size=(160, 160, 160),
            sw_batch_size=2,
            predictor=model,
            overlap=0.25,
            mode="gaussian",
        )
t1 = time.perf_counter()
logits = logits.squeeze().cpu().numpy()  # (D, H, W)
print("Inference Complete.\n")
print("Logits Shape:", logits.shape)
print(f"[TIME] Inference took {t1 - t0:.2f} sec {inputs.size()} volume")

# 예측 결과 (임의로 thresholding)
pred = (V > 0.5).astype(np.uint8)
pred = pred[::step, ::step, ::step]
print("Pred Shape:", pred.shape)

# 등치면 추출
Pd = pred[::step, ::step, ::step]

verts, faces, normals, values = marching_cubes(Pd, level=0.5)

faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
mesh = pv.PolyData(verts, faces_pv)

p = pv.Plotter()
p.add_mesh(mesh, opacity=0.9)
p.show()

print("Computing leaderboard score...")

score = metric(torch.tensor(pred), torch.tensor(Ld), mode="full", threshold=0.5)
rep = score["leaderboard_score"]


for key, value in score.items():
    if key != "leaderboard_score":
        print(f"{key}: {value}")
        
print()
print("Leaderboard score:", rep.score)                # scalar in [0,1]
print("Topo score:", rep.topo.toposcore)              # [0,1]
print("Surface Dice:", rep.surface_dice)              # [0,1]
print("VOI score:", rep.voi.voi_score)                # (0,1]
print("VOI split/merge:", rep.voi.voi_split, rep.voi.voi_merge)
print("Params used:")
pprint(rep.params)
