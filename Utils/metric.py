from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict
import torch
from topometrics import TopoReport, VOIReport, compute_leaderboard_score, LeaderboardReport
import cc3d

import numpy as np
import torch
import cc3d

try:
    from Utils.Typing import MetricsResult
except:
    from Typing import MetricsResult
    

@torch.no_grad()
def fast_split_merge_proxy(
    pred, gt,
    threshold=0.5,
    connectivity=26,
    min_pred_voxels=64,
    min_gt_voxels=64,
    min_overlap_voxels=32,   # edge exists if overlap >= this
):
    """
    pred: torch.Tensor (D,H,W) or (B,D,H,W), prob or bool
    gt:   torch.Tensor {0:bg, 1:fg, 2:ignore}

    Returns:
      split_stats, merge_stats, proxy_scalar
    """
    if pred.dim() == 4:
        pred = pred[0]
        gt = gt[0]

    ignore = (gt == 2)
    gt_fg = (gt == 1)

    if pred.dtype != torch.bool:
        pred_bin = pred > threshold
    else:
        pred_bin = pred
    pred_bin = pred_bin & (~ignore)

    # --- Connected components on CPU (fast enough for periodic eval) ---
    gt_lab = cc3d.connected_components(
        (gt_fg & (~ignore)).cpu().numpy().astype(np.uint8),
        connectivity=connectivity,
    )
    pred_lab = cc3d.connected_components(
        pred_bin.cpu().numpy().astype(np.uint8),
        connectivity=connectivity,
    )

    gt_n = int(gt_lab.max())
    pred_n = int(pred_lab.max())

    if gt_n == 0 or pred_n == 0:
        # handle degenerate cases
        split_ratio = 0.0 if gt_n == 0 else 1.0
        merge_ratio = 0.0 if pred_n == 0 else 1.0
        proxy = 1.0 / (1.0 + split_ratio + merge_ratio)
        return (
            {"gt_components": gt_n, "avg_pred_per_gt": 0.0, "gt_split_ratio": split_ratio, "max_pred_per_gt": 0},
            {"pred_components": pred_n, "avg_gt_per_pred": 0.0, "pred_merge_ratio": merge_ratio, "max_gt_per_pred": 0},
            float(proxy),
        )

    gt_counts = np.bincount(gt_lab.reshape(-1))
    pred_counts = np.bincount(pred_lab.reshape(-1))
    gt_counts[0] = 0
    pred_counts[0] = 0

    # Precompute overlap pairs by scanning fg voxels once:
    # We'll build a dict: (gt_id, pred_id) -> overlap_count
    fg_mask = (gt_lab != 0) & (pred_lab != 0)
    gt_ids = gt_lab[fg_mask].astype(np.int32, copy=False)
    pr_ids = pred_lab[fg_mask].astype(np.int32, copy=False)

    # Combine into unique pair keys
    pair_keys = gt_ids.astype(np.int64) * (pred_n + 1) + pr_ids.astype(np.int64)
    uniq, counts = np.unique(pair_keys, return_counts=True)

    # Build adjacency:
    # gt -> set(pred), pred -> set(gt) with filters
    gt_to_pred = [set() for _ in range(gt_n + 1)]
    pred_to_gt = [set() for _ in range(pred_n + 1)]

    for key, ov in zip(uniq, counts):
        if ov < min_overlap_voxels:
            continue
        gid = int(key // (pred_n + 1))
        pid = int(key %  (pred_n + 1))
        if gid == 0 or pid == 0:
            continue
        if gt_counts[gid] < min_gt_voxels:
            continue
        if pred_counts[pid] < min_pred_voxels:
            continue

        gt_to_pred[gid].add(pid)
        pred_to_gt[pid].add(gid)

    # --- Split: per GT component, how many pred comps cover it? ---
    pred_per_gt = []
    for gid in range(1, gt_n + 1):
        if gt_counts[gid] < min_gt_voxels:
            continue
        pred_per_gt.append(len(gt_to_pred[gid]))
    pred_per_gt = np.array(pred_per_gt, dtype=np.int32)
    if pred_per_gt.size == 0:
        pred_per_gt = np.array([0], dtype=np.int32)

    # --- Merge: per Pred component, how many GT comps does it touch? ---
    gt_per_pred = []
    for pid in range(1, pred_n + 1):
        if pred_counts[pid] < min_pred_voxels:
            continue
        gt_per_pred.append(len(pred_to_gt[pid]))
    gt_per_pred = np.array(gt_per_pred, dtype=np.int32)
    if gt_per_pred.size == 0:
        gt_per_pred = np.array([0], dtype=np.int32)

    split_ratio = float((pred_per_gt > 1).mean())
    merge_ratio = float((gt_per_pred > 1).mean())

    split_stats = {
        "gt_components": int((gt_counts[1:] >= min_gt_voxels).sum()),
        "avg_pred_per_gt": float(pred_per_gt.mean()),
        "max_pred_per_gt": int(pred_per_gt.max()),
        "gt_split_ratio": split_ratio,
    }

    merge_stats = {
        "pred_components": int((pred_counts[1:] >= min_pred_voxels).sum()),
        "avg_gt_per_pred": float(gt_per_pred.mean()),
        "max_gt_per_pred": int(gt_per_pred.max()),
        "pred_merge_ratio": merge_ratio,
    }

    # ---- Single scalar proxy ----
    # Idea: penalize split/merge ratios and also how far avg degree is from 1.
    split_pen = max(0.0, split_stats["avg_pred_per_gt"] - 1.0)
    merge_pen = max(0.0, merge_stats["avg_gt_per_pred"] - 1.0)

    # You can tune weights; these defaults are reasonable for quick monitoring.
    penalty = 0.7 * split_ratio + 0.7 * merge_ratio + 0.3 * split_pen + 0.3 * merge_pen
    proxy_scalar = 1.0 / (1.0 + penalty)  # higher is better, in (0,1]

    return split_stats, merge_stats, float(proxy_scalar)


@torch.no_grad()
def fast_dice_fg1_torch(pred, gt, threshold=0.5, eps=1e-6):
    """
    pred: torch.Tensor, shape (D,H,W) or (B,D,H,W), float(prob) or bool
    gt:   torch.Tensor, same shape, int with {0:bg, 1:fg, 2:ignore}
    """
    # Ensure same device
    gt = gt.to(pred.device)

    ignore = (gt == 2)
    gt_fg = (gt == 1)

    if pred.dtype != torch.bool:
        pred_bin = pred > threshold
    else:
        pred_bin = pred

    valid = ~ignore
    pred_bin = pred_bin & valid
    gt_fg = gt_fg & valid

    tp = (pred_bin & gt_fg).sum(dtype=torch.float32)
    p  = pred_bin.sum(dtype=torch.float32)
    g  = gt_fg.sum(dtype=torch.float32)

    dice = (2.0 * tp + eps) / (p + g + eps)
    return float(dice.item())


@torch.no_grad()
def fast_prf_fg1_torch(pred, gt, threshold=0.5, eps=1e-6):
    """
    Returns precision, recall, f1 (floats)
    """
    gt = gt.to(pred.device)

    ignore = (gt == 2)
    gt_fg = (gt == 1)

    if pred.dtype != torch.bool:
        pred_bin = pred > threshold
    else:
        pred_bin = pred

    valid = ~ignore
    pred_bin = pred_bin & valid
    gt_fg = gt_fg & valid

    tp = (pred_bin & gt_fg).sum(dtype=torch.float32)
    fp = (pred_bin & (~gt_fg) & valid).sum(dtype=torch.float32)
    fn = ((~pred_bin) & gt_fg).sum(dtype=torch.float32)

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = (2 * precision * recall) / (precision + recall + eps)

    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
    }


def metric(pred:torch.Tensor, gt:torch.Tensor, mode="default", threshold=0.5, **kwargs) -> MetricsResult:
    """
    pred: torch.Tensor, shape (D,H,W), float(prob) or bool
    gt:   torch.Tensor, same shape, int with {0:bg, 1:fg, 2:ignore}
    
    Args:
        pred (torch.Tensor): Prediction tensor (D, H, W)
        gt (torch.Tensor): Ground truth tensor (D, H, W)
        mode (str): `"default"`, `"tear"`, `"full"`
        threshold (float): Threshold for binarization
        
    Returns:
        dict
        {
            "dice_fg1": float,
                Dice score for foreground class (label = 1).

            "precision_fg1": float,
                Precision for foreground class.

            "recall_fg1": float,
                Recall for foreground class.

            "f1_fg1": float,
                F1 score for foreground class.

            "leaderboard_score": float or None,
                Final leaderboard score (weighted combination of
                Surface Dice, Topology score, and VOI).
                None if leaderboard evaluation is disabled.

            "split_stats": dict or None,
                Detailed statistics about split errors
                (one GT component predicted as multiple components).

            "merge_stats": dict or None,
                Detailed statistics about merge errors
                (multiple GT components predicted as one component).

            "split_merge_proxy": float or None,
                Proxy metric summarizing split/merge behavior.
                Lower is generally better.
        }
    
    
    Examples
    --------
    ```python
    from Utils.metric import metric
    import torch 
    
    score = metric(pr, gt, mode="full", threshold=0.5)
    rep = score["leaderboard_score"]

    print("Leaderboard score:", rep.score)                # scalar in [0,1]
    print("Topo score:", rep.topo.toposcore)              # [0,1]
    print("Surface Dice:", rep.surface_dice)              # [0,1]
    print("VOI score:", rep.voi.voi_score)                # (0,1]
    print("VOI split/merge:", rep.voi.voi_split, rep.voi.voi_merge)
    print("Params used:", rep.params)
    ```
    """
        
    dice = fast_dice_fg1_torch(pred, gt, threshold=threshold)

    prf = fast_prf_fg1_torch(pred, gt, threshold=threshold)
    
    if mode in ["tear", "full"]:
        split_stats, merge_stats, proxy = fast_split_merge_proxy(
            pred, gt,
            threshold=threshold,
            connectivity=26,
            min_pred_voxels=64,
            min_gt_voxels=64,
            min_overlap_voxels=32,
        )
    
    if mode == "full":
        pred = (pred >= threshold).cpu().numpy().astype(np.uint8)
        gt = gt.cpu().numpy().astype(np.uint8)
        
        # TODO Kwargs로 옵션들 받기
        rep = compute_leaderboard_score(
        predictions=pred,
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
    
    return {
        "dice_fg1": dice,
        "precision_fg1": prf["precision"],
        "recall_fg1": prf["recall"],
        "f1_fg1": prf["f1"],
        "leaderboard_score": rep if mode == "full" else None,
        "split_stats": split_stats if mode == "tear" else None,
        "merge_stats": merge_stats if mode == "tear" else None,
        "split_merge_proxy": proxy if mode == "tear" else None,
    }
    
if __name__ == "__main__":
    pr = torch.rand((320,320,320))
    gt = torch.randint(0,3,(320,320,320))
    
    import time
    from pprint import pprint
    
    t0 = time.perf_counter()
    res = metric(pr, gt, mode="default", threshold=0.5)
    t1 = time.perf_counter()
    print("Metrics:", res)
    print(f"[TIME] Metric computation took Mode: deafault {t1 - t0:.2f} sec")
    
    t0 = time.perf_counter()
    res = metric(pr, gt, mode="tear", threshold=0.5)
    t1 = time.perf_counter()
    pprint("Metrics:", res)
    print(f"[TIME] Metric computation took Mode: tear {t1 - t0:.2f} sec")