# %%
import monai
from monai.transforms import LoadImage
import kagglehub, os
import glob
import tifffile as tiff
import torch

# %% [markdown]
# # Path

# %%
data_path = kagglehub.competition_download('vesuvius-challenge-surface-detection')
test_data_dir = os.path.join(data_path, "test_images")
train_data_dir = os.path.join(data_path, "train_images")
train_lable_dir = os.path.join(data_path, "train_labels")

# %% [markdown]
# # Data Load

# %%
train_data = sorted(glob.glob(os.path.join(train_data_dir, "*")))
sample = train_data[:10]
label = sorted(glob.glob(os.path.join(train_lable_dir, "*")))

arr = tiff.imread(sample[0])
x = torch.from_numpy(arr)  # (Z,H,W) or (H,W)
arr = tiff.imread("1407735.tif")
y = torch.from_numpy(arr)
print("IMG SHAPE: ",x.shape, x.dtype ,x.min(), x.max())
print("LABEL SHAPE: ", y.shape, y.dtype, y.min(), y.max())




# %% [markdown]
# ## PyVista 등치면 확인 (Blob체크)

# %%
import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv

V = x.detach().cpu().numpy()
L = y.detach().cpu().numpy()

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



