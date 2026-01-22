from collections import defaultdict
from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Tuple, Optional
import cc3d
import wandb
import pandas as pd
import numpy as np
import tifffile as tiff
import optuna

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SegResNet
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import time
from datetime import timedelta




try:
    from Utils.Dataset import VesuviusH5PatchDataset3D
    from Utils.Sampler import get_batch_sampler
    from Utils.utils import build_h5_group_from_train_images, detect_nan_inf, save_model
    from Utils.Loss import MaskedDiceBCEIgnore2
    from Utils.utils import cleanup_memory, seed_everything, to5dim, load_model_weights, get_fold_hp, build_model, get_wandb_config
    from Utils.transform import get_train_transform, get_val_transform
    from Utils.Typing import *
    from Utils.model import TTAPredictor
    
except:
    from Dataset import VesuviusH5PatchDataset3D
    from Sampler import get_batch_sampler
    from utils import build_h5_group_from_train_images, detect_nan_inf, save_model
    from Loss import MaskedDiceBCEIgnore2
    from utils import cleanup_memory, seed_everything, to5dim, load_model_weights, get_fold_hp, build_model, get_wandb_config
    from transform import get_val_transform, get_train_transform
    from Typing import CVConfig, CVResult
    from model import TTAPredictor
    
    

BuildFn = Callable[[Optional[Any]], torch.nn.Module]
TrainFoldFn = Callable[
    [torch.nn.Module, pd.DataFrame, pd.DataFrame, int, str],
    Dict[str, Any]
]
PredictProbaFn = Callable[[torch.nn.Module, pd.DataFrame, Optional[int], Optional[str], CVConfig,], Any]
MetricFn = Callable[[Any, Any], Dict[str, Any]]
PostprocessFn = Callable[[torch.Tensor, Any], torch.Tensor]
ObjectiveFn = Callable[[Dict[str, Any]], float]
LossFn = Callable[[torch.Tensor, torch.Tensor], LossLogOutput]

def make_objective_fn(
    objective: str = "f1_fg1",
    weights: Optional[Dict[str, float]] = None,
    use_proxy: bool = True,
) -> Callable[[Dict[str, Any]], float]:
    """
    Create a scalar objective function from your `metric()` output dict.

    Args:
        objective:
            One of:
              - "dice_fg1"
              - "precision_fg1"
              - "recall_fg1"
              - "f1_fg1"
              - "split_merge_proxy"   (only available when mode="tear")
              - "f1_minus_proxy"      (f1 - alpha * proxy; weights must contain alpha)
              - "weighted_sum"        (sum_k w_k * metric_k; use `weights`)
        weights:
            Used when objective is "weighted_sum" or "f1_minus_proxy".
            Examples:
              - weighted_sum: {"f1_fg1": 1.0, "split_merge_proxy": -0.2}
              - f1_minus_proxy: {"alpha": 0.2}
        use_proxy:
            If True, tries to use split_merge_proxy when present; otherwise falls back gracefully.

    Returns:
        objective_fn(metrics_dict) -> float
    """
    weights = weights or {}

    def get(m: Dict[str, Any], key: str, default: float = 0.0) -> float:
        v = m.get(key, None)
        if v is None:
            return default
        return float(v)

    if objective in {"dice_fg1", "precision_fg1", "recall_fg1", "f1_fg1"}:
        return lambda m: get(m, objective)

    if objective == "split_merge_proxy":
        # Lower is better, so return negative for maximization.
        return lambda m: -get(m, "split_merge_proxy", default=1e9)

    if objective == "f1_minus_proxy":
        alpha = float(weights.get("alpha", 0.2))
        def fn(m: Dict[str, Any]) -> float:
            f1 = get(m, "f1_fg1")
            proxy = get(m, "split_merge_proxy", default=0.0) if use_proxy else 0.0
            # proxy: lower is better -> subtract alpha * proxy
            return f1 - alpha * proxy
        return fn

    if objective == "weighted_sum":
        # You can mix metrics; use negative weights for "lower is better" terms.
        # Example: {"f1_fg1": 1.0, "split_merge_proxy": -0.2}
        def fn(m: Dict[str, Any]) -> float:
            s = 0.0
            for k, w in weights.items():
                if k == "leaderboard_score":
                    # Not supported here (too slow); ignore if passed accidentally.
                    continue
                s += float(w) * get(m, k, default=0.0)
            return s
        return fn

    raise ValueError(f"Unknown objective='{objective}'.")

def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def safe_bool_mask(prob: np.ndarray, th: float) -> np.ndarray:
    """
    Always create a bool mask (avoid ~uint8 pitfall).
    """
    return (prob > th)

def lcc_binary(mask: np.ndarray, connectivity: int = 26) -> np.ndarray:
    """
    Keep only the largest connected component from a binary mask.
    mask: bool or {0,1} array, shape (D,H,W)
    returns: bool array
    """
    mask = mask.astype(bool)
    labels = cc3d.connected_components(mask, connectivity=connectivity)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return labels == keep

        
def collate_with_meta(batch):
    # batch: list of (x, y, meta)
    xs, ys, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    metas = list(metas)  # keep as list[dict]
    return xs, ys, metas

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    epoch: int,
    scaler: torch.GradScaler = None,
    use_wnb: bool = False,
    grad_clip: float = float('inf')
):
    model.train()
    total_loss = defaultdict(float)
    
    for idx, batch in enumerate(tqdm(dataloader, leave=False, desc="Training ")):
        batch: tuple[torch.Tensor, torch.Tensor, dict[str, int]] = batch
        
        inputs, targets, meta = batch
        inputs = inputs.to(device, non_blocking=True)
        inputs = to5dim(inputs)  # (B,1,D,H,W)
        targets = targets.to(device, non_blocking=True)
        targets = to5dim(targets)  # (B,1,D,H,W)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None, dtype=torch.float16):
            outputs = model(inputs)
            loss, log = loss_fn(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()

            # For clipping, unscale first
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if detect_nan_inf(loss=loss, model=model, logs=log):
                print(f" [Epoch {epoch}] Batch [{idx}] Detect NaN/Inf -> Skipping step")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()          
                continue

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if detect_nan_inf(loss=loss, model=model, logs=log):
                print(f" [Epoch {epoch}] Batch [{idx}] Detect NaN/Inf -> Skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            
        for k, v in log.items():
            if not isinstance(v, float):
                continue
            total_loss[k] += v * inputs.size(0)
            
        if use_wnb:
            global_step = epoch * len(dataloader) + idx
            train_log = {"step": global_step,
                         "train_step/loss": loss.item(),
                        "train_step/grad_norm": grad_norm.item(),}
            if global_step % 10 == 0:
                prob = torch.sigmoid(outputs)
                valid_mask = (targets != 2)
                
                
                clip_label = torch.clamp(targets.float(), min=0.0, max=1.0)
                fg = (clip_label == 1) & valid_mask
                bg = (clip_label == 0) & valid_mask
                
                train_log.update({
                    "prob_step/train_mean": torch.mean(prob[valid_mask]).item(),
                    "prob_step/train_std": torch.std(prob[valid_mask]).item(),
                    "prob_step/label_mean": torch.mean(clip_label[valid_mask]).item(),
                    "prob_step/lable_std": torch.std(clip_label[valid_mask]).item(),
                    "prob_step/fg_mean": prob[fg].mean().item() if fg.any() else 0.0,
                    "prob_step/bg_mean": prob[bg].mean().item() if bg.any() else 0.0,
                    "prob_step/fg_std": prob[fg].std().item() if fg.any() else 0.0,
                    "prob_step/bg_std": prob[bg].std().item() if bg.any() else 0.0,
                })
                
            if global_step % len(dataloader) == 3:
                prob = torch.sigmoid(outputs)
                valid_mask = (targets != 2)
                train_log["prob_step/prob_hist"] = wandb.Histogram(
                    prob[valid_mask].detach().float().cpu().numpy()
                )
                
            for k, v in log.items():
                if "enabled" in k or "type" in k:
                    continue
                train_log[f"train_step/{k}"] = v
            
            wandb.log(train_log)

    avg_loss = {k: v / len(dataloader.dataset) for k, v in total_loss.items()}
    
    return avg_loss
        
        
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    num_epochs: int,
    scaler: torch.GradScaler = None,
    use_wnb: bool = False,
    grad_clip: float = float('inf'),
    save_dir:str = None,
    cfg:CVConfig = None,
    val_df:pd.DataFrame = None,
    predict_proba_fn: PredictProbaFn = None,
    postprocess_fn: PostprocessFn = None,
    metric_fn: MetricFn | None = None,
    objective_fn: ObjectiveFn | None = None,
    objective_key:str|None = None,
    fold_idx:int = None,
):

    if save_dir is None:
        save_dir = "weights"
    ensure_dir(save_dir)
    
    best_score = float('-inf')
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, scaler, use_wnb, grad_clip
        )
        
        if use_wnb:
            log = {
                "epoch": epoch,
            }
            for k, v in train_loss.items():
                if "enabled" in k or "type" in k:
                    continue
                log[f"train_epoch/{k}"] = v
            wandb.log(log)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss['loss_total']:.4f}")
        
        
        #Val 자리
        if cfg is not None:
            score = evaluate(
                model, val_df, cfg, predict_proba_fn, postprocess_fn, loss_fn, metric_fn, 
                workdir=save_dir, fold_idx=fold_idx, objective_fn=objective_fn, objective_key=objective_key
            )
            fold_score = score['fold_score']
            
            if best_score <= fold_score:
                print(f"🔥 Best updated: {best_score:.6f} → {fold_score:.6f} Model Saved -> {os.path.join(save_dir, 'best1.pth')}")
                save_model(model, workdir=save_dir, weight_file_name="best1.pth")
                best_score = fold_score
                
            if use_wnb:
                log = {
                    "epoch": epoch,
                    "val/score": fold_score
                }
                log.update({f"val/{k}": v for k,v in score["metrics"].items()})
                wandb.log(log)
                
        save_model(model, workdir=save_dir, weight_file_name="last.pth")
        # Clean up memory after each epoch
        cleanup_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

def train_one_fold(
        model: nn.Module, 
        train_df: pd.DataFrame, 
        val_df:pd.DataFrame,
        fold_idx: int, 
        workdir: str, 
        use_wnb:bool = True,
        * ,
        cv_config: CVConfig,
        predict_proba_fn:PredictProbaFn,   # ✅
        postprocess_fn:PostprocessFn,       # ✅
        metric_fn:MetricFn,                # ✅
        objective_fn:ObjectiveFn=None,          # ✅
        objective_key:str=None,
    ) -> dict:
    """
    Train one CV fold.

    Responsibilities:
      - Build dataset/dataloader for this fold
      - Build optimizer/loss/scaler
      - Call your existing `train()` loop
      - Save fold checkpoint(s)
      - (Optional) report/prune via Optuna trial
    """
    device = next(model.parameters()).device
    ensure_dir(workdir)
    
    # Get Config
    cfg = get_fold_hp(cv_config, fold_idx=fold_idx)

    # --- Build fold-specific dataset/loader ---
    # IMPORTANT: You must implement these two based on how df maps to patches/ids.
    train_dataset = VesuviusH5PatchDataset3D(
        h5_path=cfg.h5_path,
        meta_return=True,
        transform=get_train_transform(),
        # TODO: filter by train_df sample ids if your dataset supports it
        allowed_sample_ids=train_df["id"].tolist(),
        jitter=cv_config.hp.jitter,
        fold_idx=fold_idx + 1
    )

    stem = Path(cfg.h5_path).stem
    suffix = f"_FOLD{fold_idx + 1}" if fold_idx is not None and fold_idx >= 0 else ""
    file_name = f"{stem}_sampler{suffix}.pkl"
    ensure_dir("Cache")
    cache_path = os.path.join("Cache", file_name)
    
    sampler = get_batch_sampler(
        dataset=train_dataset,
        file_path=os.path.join(cache_path),
        batch_size=cfg.batch_size,
        pos_fraction=cfg.pos_fraction,
        pos_thr=cfg.pos_thr,
        shuffle=True,
        drop_last=cfg.drop_last,
        seed=cfg.seed,
        h5_path=cfg.h5_path,
        patch_size=cfg.patch_size,
        # TODO: pass allowed sample ids / indices if your sampler supports it
        # allowed_sample_ids=train_df["sample_id"].tolist(),
    )
    
    print(f"Val Scroll id: {cv_config.fold_groups[fold_idx]}")
    print(f"Training Patch Num: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_with_meta,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Optim / loss / scaler ---
    lr = cfg.lr
    optimizer = cfg.optimizer_class
    optimizer = optimizer(model.parameters(), lr=lr)

    # Keep your loss as-is (example)
    pos_weight = cfg.pos_weight
    loss_fn = MaskedDiceBCEIgnore2(use_tv=True, pos_weight=torch.tensor(pos_weight, device=device)).to(device)

    scaler = torch.GradScaler() if torch.cuda.is_available() and cv_config.use_amp else None

    # --- Train (your existing loop) ---
    num_epochs = cfg.epochs

    train(
        model=model,
        train_loader=train_loader,
        val_loader=None,      # you currently don't use it
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs,
        scaler=scaler,
        use_wnb=use_wnb,        # usually off inside CV/Optuna
        grad_clip=cfg.grad_clip,
        save_dir=workdir,           # ✅ 중요
        cfg=cv_config,              # ✅ epoch마다 eval 원하면
        val_df=val_df,              # ✅ fold val_df 전달 필요
        predict_proba_fn=predict_proba_fn,   # ✅
        postprocess_fn=postprocess_fn,       # ✅
        metric_fn=metric_fn,                # ✅
        objective_fn=objective_fn,          # ✅
        objective_key=objective_key,           # ✅ (오타도 아래 참고)
        fold_idx=fold_idx,
    )

    # --- Save "last" checkpoint for this fold ---
    ckpt_path = os.path.join(workdir, "last.pth")

    # Optionally store ckpt path in Optuna DB
    return {"cv_config": cfg, "last_ckpt": ckpt_path}


def default_postprocess(prob: np.ndarray | torch.Tensor, val_item: Any, th: float = 0.5, keep_lcc: bool = True) -> torch.Tensor:
    """
    prob: (D,H,W) float in [0,1]
    returns: pred uint8 (D,H,W) with {0,1}
    """
    if isinstance(prob, torch.Tensor):
        prob = prob.detach().cpu().numpy()
    mask = safe_bool_mask(prob, th)
    if keep_lcc:
        mask = lcc_binary(mask, connectivity=26)
    return torch.tensor(mask)

@torch.no_grad()
def predict_proba_fn_tiff_swi_from_valdf(
    model: torch.nn.Module,
    val_df:pd.DataFrame,                     # df rows (patch rows ok)
    fold_idx: int,
    workdir: str,
    cfg: CVConfig,
    *,
    train_images_dir: str = "/home/user/.cache/kagglehub/competitions/vesuvius-challenge-surface-detection/train_images",
    train_labels_dir: str = "/home/user/.cache/kagglehub/competitions/vesuvius-challenge-surface-detection/train_labels",
) -> List[Dict[str, Any]]:
    
    infer_cfg = cfg.inference_cfg
    device = next(model.parameters()).device
    model.eval()
    predictor = model
    
    if infer_cfg.use_tta:
        predictor = TTAPredictor(model=model, device=device, use_rot90=infer_cfg.use_rotate_90)

    # 🔥 핵심: patch rows -> unique volume list
    vols = (
        val_df[["id", "scroll_id"]]
        .drop_duplicates()
        .sort_values(["scroll_id", "id"])
        .to_records(index=False)
    )

    outs: List[Dict[str, Any]] = []

    for sample_id, scroll_id in tqdm(vols, desc="Predicting...", leave=False):
        sample_id = int(sample_id)
        scroll_id = int(scroll_id)

        img_path = os.path.join(train_images_dir, f"{sample_id}.tif")
        lbl_path = os.path.join(train_labels_dir, f"{sample_id}.tif")

        x_np = tiff.imread(img_path)  # (Z,H,W)
        y_np = tiff.imread(lbl_path)  # (Z,H,W)

        if x_np.ndim != 3 or y_np.ndim != 3:
            raise ValueError(f"Expected 3D (Z,H,W). got x={x_np.shape}, y={y_np.shape}, sample_id={sample_id}")
        if x_np.shape != y_np.shape:
            raise ValueError(f"Shape mismatch x={x_np.shape}, y={y_np.shape}, sample_id={sample_id}")

        x = torch.from_numpy(np.asarray(x_np)).float()
        x = to5dim(x).to(device, non_blocking=True)  # (1,1,Z,H,W)

        if cfg.use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=cfg.roi_size,
                    sw_batch_size=infer_cfg.sw_batch_size,
                    predictor=predictor,
                    overlap=infer_cfg.overlap,
                    mode=infer_cfg.overlap_mode,
                )
        else:
            logits = sliding_window_inference(
                inputs=x,
                roi_size=cfg.roi_size,
                sw_batch_size=infer_cfg.sw_batch_size,
                predictor=predictor,
                overlap=infer_cfg.overlap,
                mode=infer_cfg.overlap_mode,
            )

        logits = torch.squeeze(logits).float()  # (Z,H,W)

        outs.append({
            "logits": logits,
            "gt": torch.tensor(y_np, dtype=torch.uint8).squeeze(),  # {0,1,2}
            "sample_id": sample_id,
            "scroll_id": scroll_id,
            "img_path": img_path,
            "lbl_path": lbl_path,
            "fold": fold_idx,
        })

    return outs

def _to_float(v: Any) -> Optional[float]:
    """Safely convert common numeric types to python float, else return None."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    if torch.is_tensor(v):
        # Expect scalar tensor
        return float(v.detach().float().cpu().item())
    return None


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_df: pd.DataFrame,                     # df rows (patch rows ok)
    cfg:CVConfig,                                      # CVConfig
    predict_proba_fn: Optional[PredictProbaFn]=predict_proba_fn_tiff_swi_from_valdf,          # e.g., predict_proba_fn_tiff_swi_from_valdf
    postprocess_fn: Optional[PostprocessFn]=default_postprocess,
    loss_fn=None,
    metric_fn: Optional[MetricFn] = None,
    *,
    train_images_dir: str = "/home/user/.cache/kagglehub/competitions/vesuvius-challenge-surface-detection/train_images",
    train_labels_dir: str = "/home/user/.cache/kagglehub/competitions/vesuvius-challenge-surface-detection/train_labels",
    fold_idx: int = 99,
    workdir: str = "weights",
    objective_fn: Optional[ObjectiveFn] = None,
    objective_key: str = "score",
) -> Dict[str, Any]:
    """
    Evaluate a model on a fold using only cfg-controlled behavior.

    Returns a dict including:
      - fold_score: scalar objective score for the fold
      - metrics: averaged metrics across items (float-only aggregation)
      - items_scores: per-item scalar objective scores (if list output)
      - items: per-item metrics dicts (if list output)
      - main: metrics dict (if dict output)
      - loss: averaged loss dict (if loss_fn provided and returns numeric values)
    """

    if metric_fn is None:
        raise ValueError("metric_fn must be provided.")

    # --- 1) Predict probabilities / logits ---
    proba_out = predict_proba_fn(
        model, val_df, fold_idx, workdir, cfg,
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
    )

    # --- 2) Single-item evaluator (shared) ---
    def eval_one(item: Dict[str, Any]) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        logits = item["logits"]              # (Z,H,W) torch, usually on GPU
        gt = item["gt"]                      # (Z,H,W) torch, currently uint8 on CPU

        # --- ensure device ---
        device = logits.device
        gt = gt.to(device, non_blocking=True)

        # --- loss (match train convention: logits + 5D) ---
        loss_dict: Dict[str, Any] = {}
        if loss_fn is not None:
            logits5 = to5dim(logits.float())          # (1,1,Z,H,W)
            gt5 = to5dim(gt).float()                  # (1,1,Z,H,W)  (labels {0,1,2})
            loss, log = loss_fn(logits5, gt5)         # ✅ tuple(loss, log)
            loss_dict = {"loss": _to_float(loss)}
            if isinstance(log, dict):
                loss_dict.update(log)
                
            if "loss_total" not in loss_dict:
                loss_dict["loss_total"] = _to_float(loss)

        # --- metrics path (use prob for postprocess) ---
        prob = torch.sigmoid(logits)

        pred = postprocess_fn(
            prob,
            item,
            th=cfg.postprocess_cfg.threshold,
            **cfg.postprocess_cfg.kwargs,
        )

        metrics = metric_fn(pred, gt)  # gt is on GPU now; if your metric expects CPU, move here.

        if objective_fn is not None:
            score = float(objective_fn(metrics))
        else:
            if objective_key not in metrics:
                raise KeyError(f"objective_key='{objective_key}' not found in metrics keys={list(metrics.keys())}")
            score = float(metrics[objective_key])

        return score, metrics, loss_dict

    # --- 3) Handle list vs dict outputs ---
    if isinstance(proba_out, list):
        per_item_scores: list[float] = []
        per_item_metrics: list[Dict[str, Any]] = []
        per_item_losses: list[Dict[str, Any]] = []

        # For fold-level averaged float metrics / loss
        sum_metrics = defaultdict(float)
        sum_losses = defaultdict(float)
        n_items = 0

        for it in tqdm(proba_out, desc="Scoring...", leave=False):
            s, m, ld = eval_one(it)
            per_item_scores.append(s)
            per_item_metrics.append(m)
            per_item_losses.append(ld)

            n_items += 1

            # Aggregate float-like metric values
            for k, v in m.items():
                fv = _to_float(v)
                if fv is not None:
                    sum_metrics[k] += fv

            # Aggregate float-like loss values
            for k, v in (ld or {}).items():
                fv = _to_float(v)
                if fv is not None:
                    sum_losses[k] += fv

        fold_score = float(np.mean(per_item_scores)) if per_item_scores else float("nan")

        avg_metrics = {}
        if n_items > 0:
            avg_metrics = {k: (v / n_items) for k, v in sum_metrics.items()}

        avg_losses = {}
        if n_items > 0 and loss_fn is not None:
            avg_losses = {k: (v / n_items) for k, v in sum_losses.items()}

        out: Dict[str, Any] = {
            "fold_score": fold_score,
            "metrics": avg_metrics,                 # fold-averaged float metrics
            "items_scores": per_item_scores,        # per-item objective scores
            "items": per_item_metrics,              # per-item metrics dicts
        }
        if loss_fn is not None:
            out["loss"] = avg_losses

        return out

    elif isinstance(proba_out, dict):
        fold_score, metrics_main, loss_main = eval_one(proba_out)

        out: Dict[str, Any] = {
            "fold_score": fold_score,
            "main": metrics_main,   # keep original metrics dict (not averaged)
        }
        if loss_fn is not None:
            out["loss"] = loss_main
        return out

    else:
        raise TypeError(f"predict_proba_fn returned unsupported type: {type(proba_out)}")

def run_scroll_group_cv(
    df:pd.DataFrame,
    cfg: CVConfig,
    build_fn: BuildFn = build_model,
    train_fold_fn: TrainFoldFn = train_one_fold,
    predict_proba_fn: PredictProbaFn = predict_proba_fn_tiff_swi_from_valdf,
    metric_fn: MetricFn = None,
    # how to pick folds from df
    scroll_col: str = "scroll_id",
    # postprocess / threshold
    postprocess_fn: Optional[PostprocessFn] = default_postprocess,
    # objective extraction from metric dict
    objective_key: Optional[str] = None,
    objective_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    load_best = True,
    use_wnb = False,
) -> CVResult:
    """
    Grouped CV by scroll_id-like groups.


    Notes:
    - This function does not assume how you store GT. Your predict/metric functions define that.
    - If using Optuna, pass trial; this function will write per-fold summaries and a final summary.
    
    Run scroll-grouped cross-validation (LOSO-like CV) with optional Optuna integration.

    This function performs group-aware cross-validation where each fold contains
    entire scroll(s) held out for validation, preventing scroll-level data leakage.
    It supports:
      - Uneven fold sizes (with weighted aggregation)
      - Per-fold training / validation
      - Optional Optuna trial reporting and pruning
      - Custom post-processing (e.g., thresholding, LCC)
      - Arbitrary metric computation with a scalar objective

    Args:
        df:
            Pandas DataFrame containing at least a column identifying scroll IDs.
            Each row represents a training/validation sample (e.g., patch, subvolume).

        cfg:
            CVConfig object defining:
              - fold_groups: list of sets of scroll_ids for each fold
              - out_dir: base directory to store fold summaries and CV summary
              - seed: random seed for reproducibility
              - use_weighted_average: whether to aggregate fold scores by val size
              - save_fold_summary: whether to write per-fold JSON summaries

        build_fn (Callable[trial] -> torch.nn.Module.):
            Builds and returns a new model instance for each fold.
            May use Optuna trial to sample hyperparameters.

        train_fold_fn (Callable[model, train_df, trial, fold_idx, workdir] -> dict.):
            Trains the model on the training split of a fold.
            Should internally handle epochs, optimizer, checkpointing, and
            [optionally] call trial.report() / trial.should_prune().

        predict_proba_fn (Callable[model, val_df, trial, fold_idx, workdir] -> dict or list[dict].):
            Runs inference on the validation split and returns probability outputs.
            Expected output format:
              - dict with keys {"probs", "gt"} for single validation item, or
              - list of such dicts for multiple validation items.
            "probs" should be a numpy array (e.g., shape (D,H,W)).

        metric_fn (Callable[pred_bin, gt] -> dict):
            Computes evaluation metrics given a binary prediction and ground truth.
            Can return complex structures (e.g., leaderboard reports), as long as
            a scalar objective can be extracted.

        trial (Optional Optuna trial).:
            If provided, this function supports pruning and trial-level reporting.
            If None, runs plain cross-validation without Optuna.

        scroll_col (str):
            Name of the column in `df` that contains scroll identifiers.

        postprocess_fn (Optional callable[prob, val_item, **kwargs] -> np.ndarray (binary).):
            
            Applies post-processing to probability maps before metric computation
            (e.g., thresholding, largest connected component).
            If None, a default (threshold + optional LCC) is used.

        postprocess_kwargs:
            Dictionary of keyword arguments forwarded to `postprocess_fn`
            (e.g., {"th": 0.5, "keep_lcc": True}).

        objective_key:
            Key to extract a scalar objective directly from the metric dict
            (mutually exclusive with `objective_fn`).

        objective_fn (Callable(metrics_dict) -> float.):
            Custom function to compute a scalar objective from the full metric output
            (e.g., weighted combination of TopoScore and Surface Dice).

    Returns:
        A dictionary summarizing cross-validation results with the following keys:

        - "fold_results":
            List of per-fold dictionaries. Each entry contains:
              - fold: fold index
              - val_size: number of validation samples in the fold
              - fold_score: scalar objective score for the fold
              - train_info: optional metadata returned by train_fold_fn
              - metrics: full metric outputs for the fold
              - val_scrolls: list of scroll_ids used for validation in this fold

        - "fold_scores":
            List of scalar objective scores, one per fold.

        - "weights":
            List of validation sample counts per fold (used for weighted averaging).

        - "cv_score":
            Final cross-validation score.
            If cfg.use_weighted_average is True, this is a weighted average of
            fold scores using validation sizes as weights.
            Otherwise, it is the unweighted mean.

        - "cv_score_unweighted":
            Simple mean of fold scores without weighting.

        - "fold_groups":
            List of scroll_id groups defining each fold (for reproducibility).
    
    Example
    ------
    --------
    Below is a concrete usage example compatible with your `metric()` implementation.

    1) Define 4-fold (LOSO-like) scroll grouping:

    >>> cfg = CVConfig(
    ...     fold_groups=[
    ...         {26002, 44430, 53997},  # fold 0 val
    ...         {26010},               # fold 1 val
    ...         {35360},               # fold 2 val
    ...         {34117},               # fold 3 val
    ...     ],
    ...     out_dir="runs/cv4_scroll",
    ...     seed=42,
    ...     use_weighted_average=True,
    ... )

    2) Define postprocess: threshold + LCC (example)

    >>> postprocess_kwargs = {"th": 0.5, "keep_lcc": True}

    3) Define objective extractor for your `metric(mode="full")` output:

    >>> def objective_from_metrics(m: dict) -> float:
    ...     # `leaderboard_score` is a LeaderboardReport when mode="full"
    ...     rep = m["leaderboard_score"]
    ...     return float(rep.score)

    4) Provide wrappers matching the CV runner's expected signatures.

    >>> def compute_metric(pred_bin: np.ndarray, gt) -> dict:
    ...     # Convert to torch tensors for your metric() function.
    ...     # pred_bin: (D,H,W) uint8/bool
    ...     pred_t = torch.from_numpy(pred_bin.astype(np.float32))  # float or bool OK
    ...     if isinstance(gt, np.ndarray):
    ...         gt_t = torch.from_numpy(gt.astype(np.int64))
    ...     else:
    ...         gt_t = gt
    ...     return metric(pred_t, gt_t, mode="full", threshold=0.5)

    >>> def predict_val_proba(model, val_df, trial, fold_idx, workdir) -> dict:
    ...     # Example: assume val_df corresponds to exactly one (volume, label) pair.
    ...     # Replace this with your real inference pipeline (SWI + TTA + sigmoid).
    ...     probs = run_inference_return_probs(model, val_df)  # np.ndarray (D,H,W) in [0,1]
    ...     gt = load_gt_volume(val_df)                        # torch.Tensor or np.ndarray (D,H,W), {0,1,2}
    ...     return {"probs": probs, "gt": gt, "meta": {"fold": fold_idx}}

    >>> def train_one_fold(model, train_df, trial, fold_idx, workdir) -> dict:
    ...     # Replace with your training loop.
    ...     # If Optuna: call trial.report(val_score, step=epoch) inside epochs.
    ...     info = train_loop(model, train_df, fold_idx=fold_idx, out_dir=workdir, trial=trial)
    ...     return info

    5) Run CV:

    >>> result = run_scroll_group_cv(
    ...     df=df,
    ...     cfg=cfg,
    ...     build_fn=build_model,
    ...     train_fold_fn=train_one_fold,
    ...     predict_proba_fn=predict_val_proba,
    ...     metric_fn=compute_metric,
    ...     trial=None,  # or optuna trial
    ...     scroll_col="scroll_id",
    ...     postprocess_kwargs=postprocess_kwargs,
    ...     objective_fn=objective_from_metrics,
    ... )

    >>> print("CV score:", result["cv_score"])
    """
    
    assert (objective_key is not None) or (objective_fn is not None), \
        "Provide objective_key or objective_fn to compute a scalar score."
    
     
    # Build scroll->fold mapping 
    scroll_to_fold = {sid: i for i, g in enumerate(cfg.fold_groups) for sid in g}
    df = df.copy()
    df["fold"] = df[scroll_col].map(scroll_to_fold)
    
    if df["fold"].isna().any():
        missing = sorted(df.loc[df["fold"].isna(), scroll_col].unique().tolist())
        raise ValueError(f"Some {scroll_col} values are not assigned to any fold group: {missing}")
    
    fold_results:List[Dict[str, Any]] = []
    fold_scores:List[float] = []
    fold_weights:List[int] = []
    
    base_out_path = cfg.out_dir
    ensure_dir(base_out_path)
    
    cfg.fold_hp_overrides = {i:{"seed": i+42} for i in range(0, len(cfg.fold_groups))}
    
    for fold_idx in range(0, len(cfg.fold_groups)):
        
        if use_wnb:
            run = wandb.init(
                project="vesuvius",
                name=f"fold-{fold_idx}",
                group="cv",          # 모든 fold를 하나의 group으로 묶기
                reinit=True,         # 중요!
                config=get_wandb_config(cfg, fold_idx)
            )
            wandb.define_metric("step")
            wandb.define_metric("epoch")
            wandb.define_metric("train_step/*", step_metric="step")
            wandb.define_metric("train_epoch/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

            wandb.define_metric("step")
            wandb.define_metric("epoch")
            wandb.define_metric("train_step/*", step_metric="step")
            wandb.define_metric("train_epoch/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            
        workdir = os.path.join(base_out_path, f"fold{fold_idx + 1}")
        seed_everything(fold_idx + 42)
        ensure_dir(workdir)
        
        train_df = df[df["fold"] != fold_idx]
        val_df = df[df["fold"] == fold_idx]
        w = int(len(val_df))
        
        fold_weights.append(w)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = build_fn(cfg.model_cfg) 
        model = model.to(device)
        if cfg.use_compile: model = torch.compile(model)
        
        # 2) Train fold
        train_info = train_fold_fn(
            model, 
            train_df,
            val_df,
            fold_idx,
            workdir, 
            use_wnb=use_wnb, 
            cv_config=cfg,
            predict_proba_fn=predict_proba_fn,
            postprocess_fn=postprocess_fn,
            metric_fn=metric_fn,
            objective_fn=objective_fn,
            objective_key=objective_key
            
        ) or {}
        
        # 3) Predict probabilities on val
        #    You decide what "val item" is: could be a whole volume, list of volumes, etc.
        
        # Load Model
        model_path = os.path.join(workdir, "last.pth")
        if load_best and os.path.exists(os.path.join(workdir, "best1.pth")):
            model_path = os.path.join(workdir, "best1.pth")       
            
        model = build_fn(cfg.model_cfg)
        model = model.to(device) 
        load_model_weights(model, model_path, device=device)
        if cfg.use_compile: model = torch.compile(model)
        
        eval_out = evaluate(
            model=model,
            val_df=val_df,
            cfg=cfg,
            predict_proba_fn=predict_proba_fn,
            postprocess_fn=postprocess_fn,
            loss_fn=None,                 # 필요하면 넣기
            metric_fn=metric_fn,
            fold_idx=fold_idx,
            workdir=workdir,
            objective_fn=objective_fn,           # evaluate에 넣어둔 경우
            objective_key=objective_key,         # evaluate에 넣어둔 경우
        )

        fold_score = float(eval_out["fold_score"])

        # metrics_out은 기존 코드처럼 "items + items_scores" 또는 단일 dict 형태로 맞춰주기
        if "metrics" in eval_out:
            # list 케이스: 기존 metrics_out 구조와 가장 비슷하게 맞추려면
            metrics_out = {
                "avg": eval_out["metrics"],                 # fold 평균 metric
                "items": eval_out.get("items", []),         # item별 metric
                "items_scores": eval_out.get("items_scores", []),  # item별 objective score
            }
        else:
            # dict 케이스
            metrics_out = eval_out.get("main", {})

        fold_scores.append(fold_score)
        fold_summary = {
            "fold": fold_idx,
            "val_size": w,
            "fold_score": fold_score,
            "train_info": train_info,
            "metrics": metrics_out,
            "val_scrolls": sorted(list(cfg.fold_groups[fold_idx])),
        }
        fold_results.append(fold_summary)
        
        if cfg.save_fold_summary:
            save_json(os.path.join(workdir, "fold_summary.json"), fold_summary)
            
        if use_wnb:
            wandb.finish()
            
    scores = np.array(fold_scores, dtype=np.float64)
    weights = np.array(fold_weights, dtype=np.float64)
    cv_score_unweighted = float(np.nanmean(scores))
    
    if cfg.use_weighted_average:
    # Weighted by number of validation samples (or any unit you're using in val_df)
        cv_score = float(np.nansum(scores * weights) / np.nansum(weights))
    else:
        cv_score = cv_score_unweighted
        
    final = {
        "fold_results": fold_results,
        "fold_scores": fold_scores,
        "weights": fold_weights,
        "cv_score": cv_score,
        "cv_score_unweighted": cv_score_unweighted,
        "fold_groups": [sorted(list(g)) for g in cfg.fold_groups],
    }
        
    save_json(os.path.join(base_out_path, "cv_summary.json"), final)
    return final

if __name__ == "__main__":
    import kagglehub
    import wandb

    wandb.init(
        project="vesuvius-surface-operate-test",
        name="exp_tv_lambda001",
        config={
            "lr": 5e-4,
            "batch_size": 2,
            "loss": "MaskedDiceBCETwitterignore2",
            "lambda_tear": 0.001,
            "tear": "tv",
        }
    )
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    wandb.define_metric("train_step/*", step_metric="step")
    wandb.define_metric("train_epoch/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    seed_everything()
    
    # simple test
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model:nn.Module = torch.compile(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = MaskedDiceBCEIgnore2(use_tv=True, pos_weight=torch.tensor(5.0)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Model, optimizer, and loss function initialized.")
    
    # Note: Actual training loop would require a DataLoader with data.
    
    
    data_path = kagglehub.competition_download('vesuvius-challenge-surface-detection')
    test_data_dir = os.path.join(data_path, "test_images")
    train_data_dir = os.path.join(data_path, "train_images")
    train_lable_dir = os.path.join(data_path, "train_labels")
    
    train_trasform = get_train_transform()
    val_transform = get_val_transform()
    
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
        transform=train_trasform
    )
    
    print(f"Dataset size: {len(dataset)} patches")
    x_patch, y_patch, meta = dataset[0]
    print(f"Patch shape: x={x_patch.shape}, y={y_patch.shape} meta={meta}")
    
    sampler = get_batch_sampler(
        file_path="vesuvius_train_zyx_zyx_sampler_cache.pkl",
        batch_size=6,
        pos_fraction=0.5,
        pos_thr=int(1e-3*(160**3)),
        shuffle=True,
        drop_last=True,
        seed=42,
        h5_path="vesuvius_train_zyx_zyx.h5",
        patch_size=(160,160,160),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_with_meta,
        num_workers=2,
        pin_memory=True,
    )
    
    print("DataLoader initialized with batch size 12.")
    
    print("=============== Starting training for 30 epochs... ===============")
    
    t0 = time.time()
    train(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=30,
        scaler=torch.GradScaler(),
        use_wnb=True,
        grad_clip=5.0
    )
    el = time.time() - t0
    print(f"\nTotal Train time: {timedelta(seconds=int(el))}")
    
    print("\n=============== Training completed. Saving model... ===============\n")
    weight_dir = "weights"
    weight_path = os.path.join(weight_dir, "unet_vesuvius.pth")
    os.makedirs(weight_dir, exist_ok=True)
    try:
        to_save = model._orig_mod.state_dict()
    except:
        to_save = model.state_dict()
    torch.save(to_save, weight_path)
    print(f"Model saved to {weight_path}")

    
    wandb.finish()