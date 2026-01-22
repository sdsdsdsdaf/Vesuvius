from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from topometrics import LeaderboardReport
import torch
from monai.networks.nets import UNet


@dataclass
class ModelConfig:
    model_cls = UNet
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2),
            "num_res_units": 2,
        }
    )

@dataclass
class FoldHP:
    """
    Hyperparameters controlling **training inside a single CV fold**.

    Fields
    ------
    batch_size : int
        Number of samples per training batch.
    pos_fraction : float
        Fraction of positive patches in each batch.
    pos_thr : int
        Voxel threshold to consider a patch as positive.
    drop_last : bool
        Whether to drop the last incomplete batch.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Pin CPU memory for faster GPU transfer.

    patch_size : Tuple[int, int, int]
        Patch size (D, H, W) used by sampler and dataset.

    lr : float
        Learning rate for optimizer.
    weight_decay : float
        Weight decay (L2 regularization).
    betas : Tuple[float, float]
        Adam optimizer beta coefficients.

    pos_weight : float
        Positive class weight for BCE-based losses.
    use_tv : bool
        Whether to include Total Variation regularization.

    epochs : int
        Number of training epochs per fold.
    grad_clip : float
        Maximum gradient norm for clipping.
    amp : bool
        Enable Automatic Mixed Precision (GradScaler).
    seed : int
        Random seed for fold-level reproducibility.
    optimizer_class: torch.nn.optim
        optimizer cls
    epoch: int
        Train Epochs

    Example
    -------
    >>> hp = FoldHP(
    ...     batch_size=4,
    ...     lr=1e-4,
    ...     epochs=40,
    ...     pos_weight=8.0,
    ... )
    >>> train_one_fold(hp)
    """

    batch_size: int = 6
    """Number of samples per training batch."""

    pos_fraction: float = 0.5
    """Fraction of positive patches in each batch."""

    pos_thr: int = 4096
    """Voxel threshold to consider a patch as positive."""

    drop_last: bool = True
    """Whether to drop the last incomplete batch."""

    num_workers: int = 2
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin CPU memory for faster GPU transfer."""

    patch_size: Tuple[int, int, int] = (160, 160, 160)
    """Patch size (D, H, W) used by sampler and dataset."""

    lr: float = 3e-4
    """Learning rate for optimizer."""

    weight_decay: float = 1e-2
    """Weight decay (L2 regularization)."""

    betas: Tuple[float, float] = (0.9, 0.999)
    """Adam optimizer beta coefficients."""

    pos_weight: float = 5.0
    """Positive class weight for BCE-based losses."""

    use_tv: bool = True
    """Whether to include Total Variation regularization."""

    epochs: int = 30
    """Number of training epochs per fold."""

    grad_clip: float = 5.0
    """Maximum gradient norm for clipping."""


    seed: int = 42
    """Random seed for fold-level reproducibility."""

    optimizer_class:torch.nn.Module = torch.optim.AdamW

    h5_path = "vesuvius_train_zyx_zyx.h5"
    """Path to the HDF5 dataset containing all training patches."""
    
    jitter:int = (0,0,0)
    
@dataclass
class InferConfig:
    sw_batch_size: int = 1
    overlap: float = 0.25
    overlap_mode: str = "gaussian"
    use_tta: bool = True
    use_rotate_90: bool = True

@dataclass
class PostProcessConfig:
    threshold: float = 0.5
    kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"keep_lcc": True}
    )
    

@dataclass
class CVConfig:
    """
    Configuration for **cross-validation orchestration**.

    This class defines:
    - How folds are constructed (scroll-aware grouping)
    - Where results are stored
    - How fold metrics are aggregated
    - Which training hyperparameters (`FoldHP`) are shared or overridden

    Fields
    ------
    fold_groups : List[set]
        List of sets of scroll_ids.
        Each set corresponds to one validation fold.
    out_dir : str
        Root directory for all CV outputs.
    use_weighted_average : bool
        Whether to weight fold metrics by fold size.
    save_fold_summary : bool
        Whether to save per-fold checkpoints and summaries.
    hp : FoldHP
        Base hyperparameters shared across all folds.
    fold_hp_overrides : Optional[Dict[int, Dict]]
        Optional per-fold hyperparameter overrides.

    Example
    -------
    >>> cfg = CVConfig(
    ...     fold_groups=[{1, 2}, {3}, {4, 5}],
    ...     hp=FoldHP(batch_size=6, lr=3e-4),
    ...     fold_hp_overrides={
    ...         1: {"batch_size": 4},
    ...         2: {"lr": 1e-4, "epochs": 50},
    ...     }
    ... )
    >>> run_cross_validation(cfg)
    """

    fold_groups: List[set] = field(default_factory=lambda: [
        {34117},                    # Group 1 (382)
        {35360},                    # Group 2 (176)
        {26010},                    # Group 3 (130)
        {26002, 44430, 53997},      # Group 4 (118)
    ])

    """List of sets of scroll_ids, one set per validation fold."""

    out_dir: str = "cv_runs"
    """Root directory to store CV results and checkpoints."""

    use_weighted_average: bool = True
    """Use fold-size-weighted averaging when aggregating metrics."""

    save_fold_summary: bool = True
    """Save per-fold summaries, metrics, and checkpoints."""

    hp: FoldHP = field(default_factory=FoldHP)
    """Base hyperparameters shared across all folds."""

    fold_hp_overrides: Optional[Dict[int, Dict]] = None
    """
    Optional per-fold hyperparameter overrides.

    Example:
        {
            0: {"batch_size": 4},
            2: {"lr": 1e-4, "epochs": 50},
        }
    """
    use_amp: bool = True
    """Enable Automatic Mixed Precision (GradScaler)."""
    
    use_compile:bool = True
    
    roi_size: Tuple[int,int,int] = (160,160,160)
    
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    inference_cfg: InferConfig = field(default_factory=InferConfig)
    postprocess_cfg: PostProcessConfig = field(default_factory=PostProcessConfig)



class CVResult(TypedDict):
    fold_results: list
    cv_score: float
    cv_score_unweighted: float
    weights: list
      

class MetricsResult(TypedDict):
    dice_fg1: float
    precision_fg1: float
    recall_fg1: float
    f1_fg1: float
    leaderboard_score: Optional[LeaderboardReport]
    split_stats: Optional[dict]
    merge_stats: Optional[dict]
    split_merge_proxy: Optional[float]    

class LossLogOutput(TypedDict):
    """Logging dictionary returned by loss function.
    
    loss_total: float\n
    loss_base: float\n
    loss_dice: float\n
    loss_bce: float\n
    loss_tear: float\n
    loss_hd: float\n
    loss_aux: float\n
    hd_enabled: int\n
    aux_enabled: int\n
    aux_type: str\n
    loss_tear_enabled: int
    
    """
    loss_total: float
    loss_base: float
    loss_dice: float
    loss_bce: float
    loss_tear: float
    loss_hd: float
    loss_aux: float
    hd_enabled: int
    aux_enabled: int
    aux_type: str
    loss_tear_enabled: int