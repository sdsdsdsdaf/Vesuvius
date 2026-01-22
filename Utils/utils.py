import copy
import gc
import importlib
import json
import os
import re
import glob
import subprocess
from typing import Any, Dict
import h5py
import pandas as pd
import tifffile as tiff
from tqdm.auto import tqdm
import random
import numpy as np
import torch

from typing import Union
import torch
import torch.nn as nn
from pathlib import Path
from torchsummary import summary
from dataclasses import asdict, is_dataclass
import yaml
import torch
import monai


try:
    from Utils.Typing import CVConfig, FoldHP, ModelConfig, InferConfig, PostProcessConfig
except ImportError:
    from Typing import CVConfig, FoldHP, ModelConfig, InferConfig, PostProcessConfig

def build_model(cfg: ModelConfig, verbose=False):
    
    model = cfg.model_cls(**cfg.model_params)
    if verbose:
        try:
            summary(model.cuda() if torch.cuda.is_available() else model, (1,1,160,160,160))
        except:
            print(model)    
    return model

def save_model(model:nn.Module, workdir:str, weight_file_name:str = "last.pth"):
    ckpt_path = os.path.join(workdir, weight_file_name)
    try:
        to_save = model._orig_mod.state_dict()
    except Exception:
        to_save = model.state_dict()
    torch.save(to_save, ckpt_path)

def load_model_weights(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load model weights safely from a checkpoint file.

    Handles:
    - checkpoint being either a raw state_dict or a dict with "state_dict"
    - torch.compile saved models with "_orig_mod." prefix
    - device mapping via map_location

    Args:
        model: target nn.Module to load weights into
        ckpt_path: path to .pth/.pt checkpoint
        device: torch device or string for map_location
        strict: whether to enforce that the keys in state_dict match exactly

    Returns:
        info dict with loading details:
            {
                "missing_keys": [...],
                "unexpected_keys": [...],
                "num_loaded": int
            }
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract state_dict
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Remove torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # Load
    load_info = model.load_state_dict(sd, strict=strict)

    info = {
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
        "num_loaded": len(sd),
    }
    return info

def to5dim(tensor:torch.Tensor):
    
    """
    3D Tensor (D, H, W) -> (1, 1, D, H, W)
    
    4D Tensor (B, D, H, W) -> (B, 1, D, H, W)
    """
    
    out = tensor
    
    if tensor.ndim == 3:
        out = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        out = tensor.unsqueeze(1)
    elif tensor.ndim == 5:
        pass
    else:
        raise ValueError("Input Dim is in [3, 4, 5]")
        
    return out

def _extract_numeric_id_from_filename(path: str) -> str:
    """
    Extract first numeric token from filename stem.
    Example: '1407735.tif' or 'img_1407735_x.tif' -> '1407735'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"\d+", stem)
    if not m:
        raise ValueError(f"Cannot extract numeric id from filename: {path}")
    return m.group(0)


def build_h5_group_from_train_images(
    train_images_dir: str,
    train_labels_dir: str,
    out_h5_path: str,
    train_csv_path: str,
    group_root: str = "samples",
    image_name: str = "images",
    label_name: str = "labels",
    compression: str | None = "lzf",   # "lzf" | "gzip" | None
    z_chunk: int = 16,
    hw_chunk: int = 160,
):
    
    if os.path.exists(out_h5_path):
        print(f"Output HDF5 already exists: {out_h5_path}")
        return
    
    # id -> scroll_id mapping
    df = pd.read_csv(train_csv_path)
    if "id" not in df.columns or "scroll_id" not in df.columns:
        raise ValueError(f"train.csv must contain columns: id, scroll_id. Got: {list(df.columns)}")
    df["id_str"] = df["id"].astype(str)
    id_to_scroll = dict(zip(df["id_str"], df["scroll_id"].astype(int)))

    img_files = sorted(glob.glob(os.path.join(train_images_dir, "*.tif")))
    if len(img_files) == 0:
        raise FileNotFoundError(f"No tif files in {train_images_dir}")

    # Only use train_images contents (deprecated not present here)
    kept_img_files = []
    kept_lbl_files = []
    skipped_no_label = 0
    skipped_no_csv = 0

    for ip in img_files:
        sample_id = _extract_numeric_id_from_filename(ip)

        if sample_id not in id_to_scroll:
            skipped_no_csv += 1
            continue

        lp = os.path.join(train_labels_dir, os.path.basename(ip))
        if not os.path.exists(lp):
            skipped_no_label += 1
            continue

        kept_img_files.append(ip)
        kept_lbl_files.append(lp)

    os.makedirs(os.path.dirname(out_h5_path) or ".", exist_ok=True)

    with h5py.File(out_h5_path, "w") as f:
        root = f.create_group(group_root)
        root.attrs["count"] = len(kept_img_files)
        root.attrs["image_name"] = image_name
        root.attrs["label_name"] = label_name
        root.attrs["compression"] = str(compression)
        root.attrs["skipped_no_label"] = int(skipped_no_label)
        root.attrs["skipped_no_csv"] = int(skipped_no_csv)
        root.attrs["source_dir_images"] = os.path.abspath(train_images_dir)
        root.attrs["source_dir_labels"] = os.path.abspath(train_labels_dir)

        for i, (ip, lp) in enumerate(tqdm(zip(kept_img_files, kept_lbl_files), total=len(kept_img_files), desc="Building HDF5")):
            x = tiff.imread(ip)
            y = tiff.imread(lp)

            if x.ndim != 3 or y.ndim != 3:
                raise ValueError(f"Expected 3D (Z,H,W). Got x={x.shape}, y={y.shape} at {ip}")
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape} at {ip}")

            Z, H, W = x.shape
            sample_id = _extract_numeric_id_from_filename(ip)
            scroll_id = int(id_to_scroll[sample_id])

            g = root.create_group(f"{i:06d}")
            g.attrs["basename"] = os.path.basename(ip)
            g.attrs["id"] = int(sample_id) if sample_id.isdigit() else sample_id
            g.attrs["scroll_id"] = scroll_id
            g.attrs["Z"] = int(Z)
            g.attrs["H"] = int(H)
            g.attrs["W"] = int(W)

            chunk = (min(z_chunk, Z), min(hw_chunk, H), min(hw_chunk, W))
            g.create_dataset(image_name, data=x, dtype=x.dtype, chunks=chunk, compression=compression)
            g.create_dataset(label_name, data=y, dtype=y.dtype, chunks=chunk, compression=compression)

    print(
        f"Saved HDF5(group): {out_h5_path}\n"
        f"- kept (train_images only): {len(kept_img_files)}\n"
        f"- skipped (no label): {skipped_no_label}\n"
        f"- skipped (id not in train.csv): {skipped_no_csv}\n"
        f"Deprecated excluded automatically because we only used train_images_dir."
    )

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def seed_everything(seed: int = 42):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN (deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.x deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[Seed Fixed] seed={seed}")

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
def detect_nan_inf(
    loss: torch.Tensor,
    model: torch.nn.Module,
    logs: dict[str, Any] | None = None,
    check_grad: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Returns True if NaN or Inf is detected in loss or gradients.
    Returns False if everything is finite.

    Assumes forward + backward has already been called.
    """

    has_issue = False

    # ---------- 1) Check loss ----------
    if not torch.isfinite(loss):
        has_issue = True
        if verbose:
            print("[NaN/Inf DETECTED] loss =", loss.item())

    # ---------- 2) Check logs ----------
    if logs is not None:
        for k, v in logs.items():
            if isinstance(v, (float, int)):
                if not torch.isfinite(torch.tensor(v)):
                    has_issue = True
                    if verbose:
                        print(f"[NaN/Inf DETECTED] log[{k}] =", v)

    # ---------- 3) Check gradients ----------
    if check_grad:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad

            if not torch.isfinite(grad).all():
                has_issue = True

                if verbose:
                    nan_cnt = torch.isnan(grad).sum().item()
                    inf_cnt = torch.isinf(grad).sum().item()

                    print(f"[NaN/Inf DETECTED] grad in '{name}'")
                    print(f"  shape={tuple(grad.shape)}")
                    print(f"  nan={nan_cnt}, inf={inf_cnt}")
                    print(f"  min={grad.min().item()}, max={grad.max().item()}")

                # 보통 하나만 터져도 충분
                break

    return has_issue


from typing import Dict, Any, Callable, Optional

def get_fold_hp(cfg: CVConfig, fold_idx: int) -> FoldHP:
    hp = copy.deepcopy(cfg.hp)

    if cfg.fold_hp_overrides and fold_idx in cfg.fold_hp_overrides:
        for k, v in cfg.fold_hp_overrides[fold_idx].items():
            setattr(hp, k, v)

    return hp

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

def dc_to_dict(dc):
    """Recursively convert dataclass to JSON-serializable dict."""
    if is_dataclass(dc):
        out = {}
        for k, v in asdict(dc).items():
            out[k] = dc_to_dict(v)
        return out
    elif isinstance(dc, dict):
        return {k: dc_to_dict(v) for k, v in dc.items()}
    elif isinstance(dc, tuple):
        return list(dc)   # wandb-safe
    elif hasattr(dc, "__name__"):  # class (e.g. optimizer)
        return dc.__name__
    else:
        return dc
    
def get_wandb_config(cfg:CVConfig, fold:int=99):
    wandb_cfg = {
        "fold": fold,
        "model": dc_to_dict(cfg.model_cfg),
        "train": dc_to_dict(cfg.hp),
        "inference": dc_to_dict(cfg.inference_cfg),
        "postprocess": dc_to_dict(cfg.postprocess_cfg),
        "cv": {
            "fold_groups": [list(g) for g in cfg.fold_groups],
            "use_weighted_average": cfg.use_weighted_average,
            "use_amp": cfg.use_amp,
            "roi_size": list(cfg.roi_size),
        },
    }
    
    return wandb_cfg

def serialize(obj):
    """
    Convert dataclass (or nested structure) into YAML-serializable dict.
    """
    if is_dataclass(obj):
        return serialize(asdict(obj))
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [serialize(v) for v in obj]
    elif isinstance(obj, type):
        # class → "module.ClassName"
        return f"{obj.__module__}.{obj.__name__}"
    else:
        return obj
    
def save_config_to_yaml(cfg, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(
            serialize(cfg),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
        
def resolve_class(path: str):
    """
    "torch.optim.AdamW" → torch.optim.AdamW
    """
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)

def load_config_from_yaml(path: str) -> CVConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # resolve classes
    raw["model_cfg"]["model_cls"] = resolve_class(
        raw["model_cfg"]["model_cls"]
    )
    raw["hp"]["optimizer_class"] = resolve_class(
        raw["hp"]["optimizer_class"]
    )

    # reconstruct dataclasses
    raw["model_cfg"] = ModelConfig(**raw["model_cfg"])
    raw["hp"] = FoldHP(**raw["hp"])
    raw["inference_cfg"] = InferConfig(**raw["inference_cfg"])
    raw["postprocess_cfg"] = PostProcessConfig(**raw["postprocess_cfg"])

    return CVConfig(**raw)

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
        ).decode().strip()
    except Exception:
        return "unknown"