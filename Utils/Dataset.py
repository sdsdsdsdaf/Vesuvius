import os
from pathlib import Path
import random
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pickle as pkl

try:
    from Utils.utils import build_h5_group_from_train_images, ensure_dir
except:
    from utils import build_h5_group_from_train_images


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))

def _starts(L: int, P: int, S: int) -> List[int]:
    """Axis start positions with last coverage."""
    if P > L:
        return []  # cannot place patch
    starts = list(range(0, L - P + 1, S))
    last = L - P
    if len(starts) == 0:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return starts

class VesuviusH5PatchDataset3D(Dataset):
    """
    HDF5 group-based dataset supporting variable (Z,H,W) per sample.

    HDF5 structure:
      /samples/{id}/images : (Z,H,W)
      /samples/{id}/labels : (Z,H,W)

    patch_index item:
      (group_id_str, z0, y0, x0)

    returns:
      x: (dz, dy, dx) float32
      y: (dz, dy, dx) long
    """
    
    def __init__(
        self,
        h5_path: str,
        group_root: str = "samples",
        image_name: str = "images",
        label_name: str = "labels",
        patch_size: Tuple[int, int, int] = (160, 160, 160),  # (dz, dy, dx)
        stride: Tuple[int, int, int] = (160, 160, 160),      # (sz, sy, sx)
        mode: str = "lazy",     # "lazy" | "preload"
        train: bool = True,
        jitter: Tuple[int, int, int] = (0, 0, 0),            # (jz, jy, jx)
        transform=None,         # callable(x, y) -> (x, y)
        seed: int = 42,
        meta_return: bool = False,
        image_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
        skip_all_ignore: bool = True,
        ignore_value: int = 2,
        cache_dir="Cache",
        fold_idx=-1,
        allowed_sample_ids: Optional[list[int]] = None,
        allowed_scroll_ids: Optional[list[int]] = None,
    ):
        
        assert mode in ("lazy", "preload")
        self.h5_path = h5_path
        self.group_root = group_root
        self.image_name = image_name
        self.label_name = label_name

        self.dz, self.dy, self.dx = map(int, patch_size)
        self.sz, self.sy, self.sx = map(int, stride)

        self.mode = mode
        self.train = bool(train)
        self.jz, self.jy, self.jx = map(int, jitter)

        self.transform = transform
        self.seed = int(seed)
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype

        self._h5: Optional[h5py.File] = None
        self._rng = random.Random(self.seed)

        # preload caches (optional)
        self._x_cache: Dict[str, np.ndarray] = {}
        self._y_cache: Dict[str, np.ndarray] = {}
        self._scroll_cache: Dict[str, int] = {}
        self._id_cache: Dict[str, int] = {}
        
        self.skip_all_ignore = bool(skip_all_ignore)
        self.ignore_value = int(ignore_value)
            
        # read group ids and shapes, build patch index
        self.sample_ids, self.sample_shapes = self._scan_samples()
        self.patch_index = self._build_patch_index()
        
        self.meta_return = meta_return
        self.cache_dir = cache_dir
        self.fold_idx = fold_idx
        
        # Train Spilt
        self.allowed_sample_ids = set(map(int, allowed_sample_ids)) if allowed_sample_ids is not None else None
        self.allowed_scroll_ids = set(map(int, allowed_scroll_ids)) if allowed_scroll_ids is not None else None

        if self.mode == "preload":
            self._preload_all()
            
        print("\nDataSet Load Complete !")
            
            
    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5
    
    def _scan_samples(self) -> Tuple[List[str], Dict[str, Tuple[int, int, int]]]:

        h5 = self._get_h5()
        root = h5[self.group_root]
        sample_ids = sorted(root.keys())
        kept_ids: List[str] = []
        sample_shapes = {}
        
        for sid in tqdm(sample_ids, desc="Scanning samples", leave=False):
            g = root[sid]

            # --- filter by attrs (cheap) ---
            if self.allowed_sample_ids is not None:
                if int(g.attrs["id"]) not in self.allowed_sample_ids:
                    continue

            if self.allowed_scroll_ids is not None:
                if int(g.attrs["scroll_id"]) not in self.allowed_scroll_ids:
                    continue

            img_ds = g[self.image_name]
            sample_shapes[sid] = tuple(map(int, img_ds.shape))
            kept_ids.append(sid)

        return kept_ids, sample_shapes
    
    def _build_patch_index(self):
        out: List[Tuple[str, int, int, int]] = []
        
        
        stem = Path(self.h5_path).stem
        suffix = f"_FOLD{self.fold_idx}" if self.fold_idx >= 0 else ""
        file_name = f"{stem}_patch_index{suffix}.pkl"
        
        ensure_dir(self.cache_dir)
        cache_path = os.path.join(self.cache_dir, file_name)
        if os.path.exists(os.path.join(cache_path)):
            with open(file_name, 'rb') as f:
                out = pkl.load(f)
            return out
        
        # Open once for indexing
        with h5py.File(self.h5_path, "r") as f:
            root = f[self.group_root]

            for sid in tqdm(self.sample_ids, desc="Creating Patch index", leave=False):
                Z, H, W = self.sample_shapes[sid]
                zs = _starts(Z, self.dz, self.sz)
                ys = _starts(H, self.dy, self.sy)
                xs = _starts(W, self.dx, self.sx)

                if len(zs) == 0 or len(ys) == 0 or len(xs) == 0:
                    continue

                lbl_ds = root[sid][self.label_name]

                for z0 in zs:
                    z1 = z0 + self.dz
                    for y0 in ys:
                        y1 = y0 + self.dy
                        for x0 in xs:
                            x1 = x0 + self.dx

                            if self.skip_all_ignore:
                                y_patch = lbl_ds[z0:z1, y0:y1, x0:x1]
                                if not (y_patch != self.ignore_value).any():
                                    continue

                            out.append((sid, z0, y0, x0))

        with open(cache_path, 'wb') as f:
            pkl.dump(out, f)
        
        return out

    
    def _preload_all(self):
            # Warning: this can use a lot of RAM
            with h5py.File(self.h5_path, "r") as f:
                root = f[self.group_root]
                for sid in self.sample_ids:
                    g = root[sid]
                    self._x_cache[sid] = g[self.image_name][...]
                    self._y_cache[sid] = g[self.label_name][...]
                    self._scroll_cache[sid] = int(g.attrs["scroll_id"])
                    self._id_cache[sid] = int(g.attrs["id"])
                    
    
    def __len__(self):
        return len(self.patch_index)
    
    def __getitem__(self, idx: int):
        sid, z0, y0, x0 = self.patch_index[idx]
        Z, H, W = self.sample_shapes[sid]

        # jitter
        if self.train and (self.jz, self.jy, self.jx) != (0, 0, 0):
            dz_j = self._rng.randint(-self.jz, self.jz) if self.jz > 0 else 0
            dy_j = self._rng.randint(-self.jy, self.jy) if self.jy > 0 else 0
            dx_j = self._rng.randint(-self.jx, self.jx) if self.jx > 0 else 0
        else:
            dz_j = dy_j = dx_j = 0

        z1 = _clamp(z0 + dz_j, 0, Z - self.dz)
        y1 = _clamp(y0 + dy_j, 0, H - self.dy)
        x1 = _clamp(x0 + dx_j, 0, W - self.dx)

        # load slice
        if self.mode == "preload":
            x = self._x_cache[sid][z1:z1+self.dz, y1:y1+self.dy, x1:x1+self.dx]
            y = self._y_cache[sid][z1:z1+self.dz, y1:y1+self.dy, x1:x1+self.dx]
            scroll_id = int(self._scroll_cache[sid])
            sample_id = int(self._id_cache[sid])
        else:
            f = self._get_h5()
            g = f[self.group_root][sid]
            x = g[self.image_name][z1:z1+self.dz, y1:y1+self.dy, x1:x1+self.dx]
            y = g[self.label_name][z1:z1+self.dz, y1:y1+self.dy, x1:x1+self.dx]
            scroll_id = int(g.attrs["scroll_id"])
            sample_id = int(g.attrs["id"])

        x_t = torch.from_numpy(np.array(x, copy=False)).to(self.image_dtype)
        y_t = torch.from_numpy(np.array(y, copy=False)).to(self.label_dtype)

        if self.transform is not None:
            data = self.transform({"image": x_t, "label": y_t})
            x_t, y_t = data["image"], data["label"]
            
        if self.meta_return:
            return (x_t, y_t, {
                "sample_id": sample_id,
                "scroll_id": scroll_id,
                "patch_start": (z1, y1, x1),
            })

        return x_t, y_t
    
    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
    
if __name__ == "__main__":
    import kagglehub
    from pprint import pprint
    
    data_path = kagglehub.competition_download('vesuvius-challenge-surface-detection')
    test_data_dir = os.path.join(data_path, "test_images")
    train_data_dir = os.path.join(data_path, "train_images")
    train_lable_dir = os.path.join(data_path, "train_labels")
        
    build_h5_group_from_train_images(
        train_images_dir=train_data_dir,
        train_labels_dir=train_lable_dir,
        train_csv_path=os.path.join(data_path, "train.csv"),
        out_h5_path="vesuvius_train_zyx_zyx.h5",
    )
    
    # Example usage
    dataset = VesuviusH5PatchDataset3D(
        h5_path="vesuvius_train_zyx_zyx.h5",
        meta_return=True,
    )
    
    print(f"Dataset size: {len(dataset)} patches")
    x_patch, y_patch, meta = dataset[0]
    print(f"Patch shape: x={x_patch.shape}, y={y_patch.shape} meta={meta}")
    
    for data in tqdm(dataset):
        x_patch, y_patch, meta = data
        if torch.allclose(y_patch, torch.full_like(y_patch, 2)):
            print("All 2 Index: ")
            pprint(meta)
            exit(0)
            
    print("Test Passed")