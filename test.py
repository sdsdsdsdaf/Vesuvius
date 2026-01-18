import os
from topometrics import compute_leaderboard_score
import tifffile
import glob
import kagglehub
import time

data_path = kagglehub.competition_download('vesuvius-challenge-surface-detection')
print("Data Load Complete")
test_data_dir = os.path.join(data_path, "test_images")
train_data_dir = os.path.join(data_path, "train_images")
train_lable_dir = os.path.join(data_path, "train_labels")

train_data = sorted(glob.glob(os.path.join(train_data_dir, "*")))
sample = train_data[:10]
label = sorted(glob.glob(os.path.join(train_lable_dir, "*")))

print(label)

# pr, gt are 3D arrays with identical shape (Z, Y, X)
pr = tifffile.imread(label[0])  # Here we just use the ground truth as a dummy prediction   
gt = tifffile.imread(label[0])
print("Prediction shape:", pr.shape)
print("Ground truth shape:", gt.shape)

print("Computing leaderboard score...")
t0 = time.perf_counter()
rep = compute_leaderboard_score(
    predictions=pr,
    labels=gt,
    dims=(0,1,2),
    spacing=(1.0, 1.0, 1.0),          # (z, y, x)
    surface_tolerance=2.0,            # in spacing units
    voi_connectivity=26,
    voi_transform="one_over_one_plus",
    voi_alpha=0.3,
    combine_weights=(0.3, 0.35, 0.35),  # (Topo, SurfaceDice, VOI)
    fg_threshold=None,                # None => legacy "!= 0"; else uses "x > threshold"
    ignore_label=2,                   # voxels with this GT label are ignored
    ignore_mask=None,                 # or pass an explicit boolean mask
)
t1 = time.perf_counter()

print("Leaderboard score:", rep.score)                # scalar in [0,1]
print("Topo score:", rep.topo.toposcore)              # [0,1]
print("Surface Dice:", rep.surface_dice)              # [0,1]
print("VOI score:", rep.voi.voi_score)                # (0,1]
print("VOI split/merge:", rep.voi.voi_split, rep.voi.voi_merge)
print("Params used:", rep.params)
print(f"[TIME] Compute Score took {t1 - t0:.2f} sec [320, 320, 320] volume")