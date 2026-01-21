import os
import random
from collections import defaultdict
from typing import Optional, Tuple
import numpy as np
from torch.utils.data import Sampler
from tqdm.auto import tqdm
import torch
import pickle as pkl

try:
    from Utils.Dataset import VesuviusH5PatchDataset3D
except:
    from Dataset import VesuviusH5PatchDataset3D

@torch.no_grad()
def build_index_cache(
    file_path: str,
    ds:VesuviusH5PatchDataset3D,
    pos_thr: int = 1,
    max_items: Optional[int] = None,
    use_cache: bool = True,
    *,
    ignore_value: Optional[int] = 2,   # 필요하면 ignore=2 제외하고 sum하고 싶을 때
) -> Tuple[list, np.ndarray]:
    """
    Build per-patch-index cache for:
      - scroll_ids[i]  where i indexes ds.patch_index
      - is_pos[i]      (sum(label_patch) >= pos_thr)

    IMPORTANT:
      - This uses ds.patch_index directly (i.e., sampler index space == dataset index space).
      - Reads label patch directly from HDF5 for speed and to avoid ds.transform/jitter effects.

    Returns:
      scroll_ids: list[int] length N
      is_pos: np.ndarray bool length N
    """

    # --- cache key: invalidate when dataset size/pos_thr changes ---
    N_full = len(ds.patch_index) if hasattr(ds, "patch_index") else len(ds)
    N = N_full if max_items is None else min(N_full, max_items)

    cache_payload = None
    if use_cache and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            cache_payload = pkl.load(f)

        # Backward compatible: allow old (scroll_ids, is_pos) tuple
        if isinstance(cache_payload, tuple) and len(cache_payload) == 2:
            scroll_ids, is_pos = cache_payload
            if len(scroll_ids) == N and len(is_pos) == N:
                return scroll_ids, is_pos

        # New format with meta
        if isinstance(cache_payload, dict):
            if (
                cache_payload.get("N") == N
                and cache_payload.get("pos_thr") == int(pos_thr)
                and cache_payload.get("h5_path") == getattr(ds, "h5_path", None)
            ):
                scroll_ids = cache_payload["scroll_ids"]
                is_pos = cache_payload["is_pos"]
                return scroll_ids, is_pos

    if not hasattr(ds, "patch_index"):
        raise AttributeError("ds must have ds.patch_index for this function.")

    # --- prepare output ---
    scroll_ids = [0] * N
    is_pos = np.zeros(N, dtype=np.bool_)

    # --- open h5 once ---
    # Expect ds._get_h5() exists (you had it in your Dataset earlier).
    if hasattr(ds, "_get_h5"):
        h5 = ds._get_h5()
    else:
        # fallback: try ds._h5 or open via h5py if you want
        raise AttributeError("ds must provide _get_h5() to read label patches directly.")

    group_root = getattr(ds, "group_root", "samples")
    label_name = getattr(ds, "label_name", "labels")

    dz = int(getattr(ds, "dz", 160))
    dy = int(getattr(ds, "dy", 160))
    dx = int(getattr(ds, "dx", 160))

    # ignore_value: default from ds if exists
    if ignore_value is None and hasattr(ds, "ignore_value"):
        ignore_value = int(ds.ignore_value)

    # --- iterate patch_index directly ---
    for i in tqdm(range(N), desc="Index cache (patch_index)", leave=False):
        item = ds.patch_index[i]

        # patch_index item can be:
        #   (group_id_str, z0, y0, x0)  or (sid, z0, y0, x0)
        gid, z0, y0, x0 = item
        z0 = int(z0); y0 = int(y0); x0 = int(x0)

        g = h5[group_root][str(gid)]
        # scroll_id is usually stored in group attributes/datasets; you used meta["scroll_id"] before
        # In your dataset code you cache scroll_id somewhere; commonly g["scroll_id"][()]
        if "scroll_id" in g:
            scroll_ids[i] = int(g["scroll_id"][()])
        else:
            # fallback: if stored as attribute
            scroll_ids[i] = int(g.attrs.get("scroll_id", -1))

        y_ds = g[label_name]  # (Z,H,W)
        y_patch = y_ds[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]

        # sum like your original y.sum()
        if ignore_value is not None:
            # optional: ignore label voxels from sum
            y_patch = y_patch[y_patch != ignore_value]

        # y_patch could be numpy scalar types already
        s = int(np.sum(y_patch))
        is_pos[i] = (s >= int(pos_thr))

    # --- save cache (include metadata so it doesn't silently mismatch) ---
    payload = {
        "N": N,
        "pos_thr": int(pos_thr),
        "h5_path": getattr(ds, "h5_path", None),
        "scroll_ids": scroll_ids,
        "is_pos": is_pos,
    }
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "wb") as f:
        pkl.dump(payload, f)

    return scroll_ids, is_pos

def get_batch_sampler(file_path:str, batch_size: int, pos_fraction: float = 0.5, pos_thr: int = 1, dataset:VesuviusH5PatchDataset3D = None,
                      shuffle: bool = True, drop_last: bool = True, max_items=None, seed: int = 42, use_cache:bool=True, **kwargs):
    """
    Create a MultiScrollBalancedBatchSampler for the given dataset.
    
    Example usage:
        ```python
        sampler = get_batch_sampler(
            batch_size=8,
            pos_fraction=0.5,
            pos_thr=1,
            shuffle=True,
            drop_last=True,
            seed=42,
            h5_path="data/train.h5",
            patch_size=(160,160,160),
            stride=(80,80,80),
            mode="lazy",
        )
        ```
    
    """
    if dataset is None:
        dataset = VesuviusH5PatchDataset3D(**kwargs)
        dataset.meta_return = True  # ensure meta is returned
        
    scroll_ids, is_pos = build_index_cache(file_path, dataset, max_items=max_items, pos_thr=pos_thr, use_cache=use_cache)
    
    sampler = MultiScrollBalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        pos_fraction=pos_fraction,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        scroll_ids=scroll_ids,
        is_pos=is_pos,
        max_items=max_items,
    )
    
    return sampler
    

class MultiScrollBalancedBatchSampler(Sampler):
    """
    - Mix different scrolls within a batch
    - Control positive/negative ratio within each batch

    Assumptions:
    - You can access per-index metadata WITHOUT calling __getitem__ (recommended).
      e.g., dataset.scroll_id_of_index[idx], dataset.is_pos_of_index[idx]
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        pos_fraction: float = 0.5,  # e.g. 0.5 means half positives per batch
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        scroll_ids:list[int] = None,
        is_pos:list[bool] = None,
        max_items: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_k = int(round(batch_size * pos_fraction))
        self.neg_k = batch_size - self.pos_k
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        if max_items is not None:
            raise NotImplementedError("max_items not supported in MultiScrollBalancedBatchSampler")
            
        # ---- Build pools: (scroll_id, is_pos) -> indices ----
        pools = defaultdict(list)

        # IMPORTANT:
        # dataset should have fast-access arrays:
        # dataset.scroll_ids[idx], dataset.is_pos[idx]
        scroll_ids = scroll_ids if scroll_ids is not None else []
        is_pos = is_pos if is_pos is not None else []
        
        for idx in range(len(dataset)):
            sid = scroll_ids[idx]
            pos = is_pos[idx]
            pools[(sid, pos)].append(idx)
        
        self.pools = pools
        self.scroll_ids = sorted({scroll_ids[i] for i in range(len(dataset))})

        # Pre-shuffle pools if needed
        if self.shuffle:
            for k in list(self.pools.keys()):
                self.rng.shuffle(self.pools[k])

        # Maintain per-pool cursor for round-robin sampling
        self.cursors = {k: 0 for k in self.pools.keys()}

        # Count batches conservatively (optional)
        self._length = None

    def __len__(self):
        # Conservative estimate: based on total available samples
        if self._length is None:
            n = len(self.dataset)
            self._length = n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        return self._length

    def _draw_from_pool(self, key, k):
        """
        Draw k indices from a pool with replacement if needed.
        """
        arr = self.pools.get(key, [])
        if len(arr) == 0:
            return []

        out = []
        cur = self.cursors.get(key, 0)
        for _ in range(k):
            if cur >= len(arr):
                # restart (replacement-like)
                cur = 0
                if self.shuffle:
                    self.rng.shuffle(arr)
            out.append(arr[cur])
            cur += 1
        self.cursors[key] = cur
        return out

    def __iter__(self):
        
        if self.shuffle:
            for k in self.pools.keys():
                self.rng.shuffle(self.pools[k])
        
        scrolls = list(self.scroll_ids)

        for _ in range(len(self)):  # fixed number of batches per epoch
            if self.shuffle:
                self.rng.shuffle(scrolls)

            batch = []

            # positives
            if self.pos_k > 0:
                i = 0
                while len(batch) < self.pos_k and i < len(scrolls) * 2:
                    sid = scrolls[i % len(scrolls)]
                    batch += self._draw_from_pool((sid, True), 1)
                    i += 1

            # negatives
            if self.neg_k > 0:
                i = 0
                while len(batch) < self.batch_size and i < len(scrolls) * 3:
                    sid = scrolls[i % len(scrolls)]
                    batch += self._draw_from_pool((sid, False), 1)
                    i += 1

            # backfill
            if len(batch) < self.batch_size:
                all_keys = list(self.pools.keys())
                if self.shuffle:
                    self.rng.shuffle(all_keys)
                for key in all_keys:
                    need = self.batch_size - len(batch)
                    if need <= 0:
                        break
                    batch += self._draw_from_pool(key, need)

            if len(batch) < self.batch_size:
                if self.drop_last:
                    continue
                else:
                    yield batch
                    continue

            yield batch
