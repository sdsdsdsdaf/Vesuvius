import os
import random
from collections import defaultdict
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
def build_index_cache(file_path, ds, pos_thr=1, max_items=None, use_cache=True):
    """
    Build per-index cache for:
      - scroll_ids[idx]
      - is_pos[idx] (label sum >= pos_thr)
    """
    sids = [None] * len(ds)
    scroll_ids = [None] * len(ds)
    is_pos = np.zeros(len(ds), dtype=np.bool_)
    
    if use_cache and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            scroll_ids, is_pos = pkl.load(f)
        
        return scroll_ids, is_pos

    # Ensure ds returns meta and does NOT apply random augmentation for this pass.
    # Best: ds.meta_return=True, ds.augment=False (or mode="cache")
    n = len(ds) if max_items is None else min(len(ds), max_items)

    for idx in tqdm(range(n), desc="Index cache"):
        x, y, meta = ds[idx]  # lazy load happens here

        scroll_ids[idx] = meta["scroll_id"]
        sids[idx] = idx

        # y can be torch or numpy; make it cheap
        if torch.is_tensor(y):
            s = int(y.sum().item())
        else:
            s = int(np.sum(y))
        is_pos[idx] = (s >= pos_thr)

    # If max_items used, trim
    if max_items is not None and n < len(ds):
        scroll_ids = scroll_ids[:n]
        is_pos = is_pos[:n]
    
    with open(file_path, 'wb') as f:
        pkl.dump((scroll_ids, is_pos), f)
    
    return scroll_ids, is_pos

def get_batch_sampler(file_path:str, batch_size: int, pos_fraction: float = 0.5, pos_thr: int = 1,
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
