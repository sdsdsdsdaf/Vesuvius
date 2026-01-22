import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import (
    MaskedLoss,
    MaskedDiceLoss,
    HausdorffDTLoss,
)
try:
    from Utils.Typing import LossLogOutput
except:
    from Typing import LossLogOutput 

def _ensure_b1dhw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError(f"Unexpected shape: {tuple(x.shape)}")

def tv_loss_3d_masked(prob: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    prob:  (B,1,D,H,W) in [0,1]
    valid: (B,1,D,H,W) bool or 0/1
    Returns: scalar TV averaged over valid neighbor pairs only.
    """
    prob = prob.float()
    v = valid.to(prob.dtype)

    dz = (prob[:, :, 1:] - prob[:, :, :-1]).abs()
    dy = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs()
    dx = (prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1]).abs()

    # neighbor-pair validity masks (both voxels must be valid)
    vz = v[:, :, 1:] * v[:, :, :-1]
    vy = v[:, :, :, 1:] * v[:, :, :, :-1]
    vx = v[:, :, :, :, 1:] * v[:, :, :, :, :-1]

    eps = 1e-6
    tvz = (dz * vz).sum() / (vz.sum().clamp_min(1.0) + eps)
    tvy = (dy * vy).sum() / (vy.sum().clamp_min(1.0) + eps)
    tvx = (dx * vx).sum() / (vx.sum().clamp_min(1.0) + eps)

    return tvz + tvy + tvx

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import (
    MaskedLoss,
    DiceCELoss,
    HausdorffDTLoss,
    LocalNormalizedCrossCorrelationLoss,
    GlobalMutualInformationLoss,
    SSIMLoss,
)

# ---------- helpers ----------
def _ensure_b1dhw(x: torch.Tensor) -> torch.Tensor:
    # Accept (B,D,H,W) or (B,1,D,H,W)
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError(f"Expected 4D/5D tensor, got shape={tuple(x.shape)}")

def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    # Simple fixed-radius Gaussian
    radius = max(1, int(3.0 * sigma + 0.5))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum().clamp_min(1e-12)
    return k  # (K,)

def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: (B,1,D,H,W)
    separable Gaussian blur with conv3d
    """
    if sigma <= 0:
        return x

    B, C, D, H, W = x.shape
    k1 = _gaussian_kernel_1d(sigma, x.device, x.dtype)  # (K,)
    K = k1.numel()

    # Depth
    kd = k1.view(1, 1, K, 1, 1)
    x = F.conv3d(x, kd, padding=(K // 2, 0, 0), groups=1)

    # Height
    kh = k1.view(1, 1, 1, K, 1)
    x = F.conv3d(x, kh, padding=(0, K // 2, 0), groups=1)

    # Width
    kw = k1.view(1, 1, 1, 1, K)
    x = F.conv3d(x, kw, padding=(0, 0, K // 2), groups=1)

    return x

# ---------- loss ----------
class MaskedDiceBCEIgnore2(nn.Module):
    """
    Labels:
      0 = bg
      1 = fg
      2 = ignore

    logits: (B,1,D,H,W) or (B,D,H,W)
    gt:     (B,1,D,H,W) or (B,D,H,W) with int {0,1,2}
    """

    def __init__(
        self,
        # Base weights
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth_nr: float = 0.00001,
        smooth_dr: float = 0.00001,
        # Optional Hausdorff
        use_hausdorff: bool = False,
        lambda_hd: float = 0.02,
        use_tv: bool = False,
        lambda_tv: float = 0.02,
        apply_hd_every: int = 1,
        # Choose ONE aux similarity loss: "none"|"lncc"|"mi"|"ssim"
        aux: str = "none",
        lambda_aux: float = 0.02,
        soft_gt_sigma: float = 1.0,
        # BCE pos_weight (foreground upweight)
        pos_weight: torch.Tensor | None = None,
        # DiceCELoss label smoothing (only affects CE part if enabled; we set lambda_ce=0)
        label_smoothing: float = 0.0,
    ) -> None:
        
        """
        Create criterion function for binary segmentation with ignore label (=2).

        This loss is designed for voxel-wise binary segmentation tasks such as
        Vesuvius Challenge surface detection, where:
        - 0 = background
        - 1 = foreground
        - 2 = ignore (excluded from loss and gradients)

        Base loss:
        The main segmentation objective is a weighted sum of:
            - Dice loss (overlap-based, shape-aware)
            - BCEWithLogits loss (voxel-wise classification)

        Parameters
        ----------
        dice_weight : float
            Weight of Dice loss term.
            Increasing this emphasizes overlap and global shape consistency.

        bce_weight : float
            Weight of BCEWithLogits loss term.
            Increasing this emphasizes voxel-level classification accuracy,
            which is especially useful when foreground voxels are sparse.

        smooth_nr : float
            Smoothing constant added to the Dice numerator.
            Prevents zero intersection issues and stabilizes gradients
            in sparse foreground or early training stages.

        smooth_dr : float
            Smoothing constant added to the Dice denominator.
            Prevents division by zero when both prediction and target
            contain no foreground voxels (all-background or all-ignore patches).

        Optional surface-aware regularization
        ------------------------------------
        use_hausdorff : bool
            Whether to include HausdorffDTLoss as an additional surface alignment term.
            When enabled, encourages predicted surfaces to be spatially close
            to ground-truth surfaces.
        use_tv: bool
            Whether to include TotalVariantLoss
        lamdba_tv : float
            Weight for TotalVariantLoss.
            Should typically be small (e.g. 0.01 ~ 0.05) since this loss
            can produce large gradients.
        lambda_hd : float
            Weight for HausdorffDTLoss.
            Should typically be small (e.g. 0.01 ~ 0.05) since this loss
            can produce large gradients.

        apply_hd_every : int
            Apply HausdorffDTLoss only every N forward calls.
            Useful to reduce computational cost and avoid over-constraining
            early training.

        Auxiliary similarity loss (choose ONE)
        -------------------------------------
        aux : str
            Type of auxiliary similarity loss to use.
            One of:
            - "none" : no auxiliary loss
            - "lncc" : Local Normalized Cross-Correlation
            - "mi"   : Global Mutual Information
            - "ssim" : Structural Similarity Index
            These losses are used as soft surface regularizers and are NOT
            intended to replace Dice/BCE.

        lambda_aux : float
            Weight of the auxiliary similarity loss.
            Typically very small (e.g. 0.005 ~ 0.02), since these losses
            act as regularizers rather than primary objectives.

        soft_gt_sigma : float
            Standard deviation for Gaussian smoothing applied to the
            ground-truth mask when computing auxiliary losses.
            Converts hard {0,1} labels into a continuous "soft surface" map
            suitable for LNCC / MI / SSIM comparison.

        BCE weighting and smoothing
        ---------------------------
        pos_weight : torch.Tensor or None
            Positive class weight for BCEWithLogits loss.
            Used to counter severe foreground/background imbalance.
            If None, no class reweighting is applied.

        label_smoothing : float
            Amount of label smoothing applied to targets for BCE loss.
            When > 0, foreground labels are slightly reduced and background
            labels slightly increased to improve training stability.
            Typically kept at 0.0 for segmentation unless explicitly needed.
            
        
        Example Use
        ---------------------------
        
        ```python
        from Utils.Loss import MaskedDiceBCEIgnore2
        
        loss_fn = MaskedDiceBCEIgnore2(
            dice_weight = 1.0,
            bce_weight = 1.0,
            smooth_nr = 0.00001,
            smooth_dr = 0.00001,
            # Optional Hausdorff
            use_hausdorff = False,
            lambda_hd = 0.02,
            apply_hd_every = 1,
            # Choose ONE aux similarity loss: "none"|"lncc"|"mi"|"ssim"
            aux = "none",
            lambda_aux = 0.02,
            soft_gt_sigma = 1.0,
            # BCE pos_weight (foreground upweight)
            pos_weight = None,
            # DiceCELoss label smoothing (only affects CE part if enabled; we set lambda_ce=0)
            label_smoothing = 0.0,
        )
        
        loss = loss_fn(logits, gt)
        """
        
        
        super().__init__()

        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)

        self.use_hausdorff = bool(use_hausdorff)
        self.lambda_hd = float(lambda_hd)
        self.apply_hd_every = int(apply_hd_every)
        
        self.use_tv = bool(use_tv)
        self.lambda_tv = float(lambda_tv)
        
        self._step = 0

        self.aux = aux.lower()
        self.lambda_aux = float(lambda_aux)
        self.soft_gt_sigma = float(soft_gt_sigma)

        if pos_weight is None:
            pos_weight = torch.tensor([1.0])
        self.register_buffer("pos_weight", pos_weight.float())
        
        # ---- Masked Dice (MONAI DiceCELoss with CE disabled) ----
        # DiceCELoss exists in MONAI 1.5.1 docs. :contentReference[oaicite:6]{index=6}
        
        # TODO 후에 Label Smoothing까지 구현
        
        self.dice_loss = MaskedDiceLoss(
            include_background=True,  # ignore bg channel (for multi-class); for sigmoid+1ch it's fine
            sigmoid=False,           # <- CE off (we do BCE separately for pos_weight control)
            reduction="mean",
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            soft_label=False,
        )
            
        

        # ---- Optional HausdorffDTLoss (masked) ----
        # HausdorffDTLoss exists and expects logits or probs; sigmoid=True applies sigmoid inside. :contentReference[oaicite:7]{index=7}
        # IMPORTANT: for 1-channel binary, include_background must be True, otherwise channel 0 would be dropped.
        self.hd_loss = MaskedLoss(
            HausdorffDTLoss(
                sigmoid=True,
                include_background=True,
                reduction="mean",
            )
        )
        
        
        # ---- Optional Aux similarity/statistical loss (masked) ----
        # LNCC/MI/SSIM are in the docs page. :contentReference[oaicite:8]{index=8}
        if self.aux == "none":
            self.aux_loss = None
        elif self.aux == "lncc":
            self.aux_loss = MaskedLoss(
                LocalNormalizedCrossCorrelationLoss(
                    spatial_dims=3,
                    kernel_size=3,
                    kernel_type="rectangular",
                    reduction="mean",
                )
            )
        elif self.aux == "mi":
            self.aux_loss = MaskedLoss(
                GlobalMutualInformationLoss(
                    kernel_type="gaussian",
                    num_bins=23,
                    sigma_ratio=0.5,
                    reduction="mean",
                )
            )
        elif self.aux == "ssim":
            self.aux_loss = MaskedLoss(
                SSIMLoss(
                    spatial_dims=3,
                    data_range=1.0,
                    reduction="mean",
                )
            )
        else:
            raise ValueError("aux must be one of: 'none'|'lncc'|'mi'|'ssim'")

    def forward(self, logits: torch.Tensor, gt: torch.Tensor) -> LossLogOutput:
        self._step += 1
        
        device = logits.device
        logits = _ensure_b1dhw(logits)
        gt = _ensure_b1dhw(gt)

        valid_mask = (gt != 2)              # (B,1,D,H,W) bool  :contentReference[oaicite:9]{index=9}
        y = (gt == 1).float()               # (B,1,D,H,W) in {0,1}
        prob = torch.sigmoid(logits)

        # ---- Base: masked Dice ----
        dice = self.dice_loss(prob, y, mask=valid_mask)

        # ---- Base: masked BCE (with pos_weight) ----
        # We do reduction='none' then mask+normalize by valid voxel count.
        bce_vox = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=self.pos_weight.to(device), reduction="none"
        )
        denom = valid_mask.sum().clamp_min(8).to(bce_vox.dtype)
        bce = (bce_vox * valid_mask.to(bce_vox.dtype)).sum() / denom

        base = self.dice_weight * dice + self.bce_weight * bce

        # ---- Optional: Hausdorff ----
        do_hd = (
            self.use_hausdorff
            and (self.lambda_hd > 0)
            and (self.apply_hd_every > 0)
            and (self._step % self.apply_hd_every == 0)
        )
        
        do_tv = self.use_tv
        hd = logits.sum() * 0.0
        tv = logits.sum() * 0.0
        if do_hd:
            hd = self.hd_loss(logits, y, mask=valid_mask)
        if do_tv:
            tv = tv_loss_3d_masked(prob, valid_mask)
        
        # ---- Optional: Aux (LNCC/MI/SSIM) against soft_gt ----
        aux_val = logits.sum() * 0.0
        do_aux = (self.aux_loss is not None) and (self.lambda_aux > 0)

        if do_aux:
            # soft_gt: blur the hard mask (only on valid region)
            y_m = y * valid_mask.to(y.dtype)
            soft_gt = _gaussian_blur_3d(y_m, sigma=self.soft_gt_sigma).clamp(0.0, 1.0)

            # compare prob vs soft_gt (both in [0,1]) with same valid mask
            prob_m = prob * valid_mask.to(prob.dtype)
            soft_m = soft_gt * valid_mask.to(soft_gt.dtype)

            aux_val = self.aux_loss(prob_m, soft_m, mask=valid_mask)

        total = base + (self.lambda_hd * hd) + (self.lambda_aux * aux_val) + (self.lambda_tv * tv)

        logs = {
            "loss_total": float(total.detach().item()),
            "loss_base": float(base.detach().item()),
            "loss_dice": float(dice.detach().item()),
            "loss_bce": float(bce.detach().item()),
            "loss_hd": float(hd.detach().item()) if do_hd else 0.0,
            "loss_tv": float(tv.detach().item() if do_tv else 0.0),
            "loss_aux": float(aux_val.detach().item()) if do_aux else 0.0,
            "tv_enabled": int(do_tv),
            "hd_enabled": int(do_hd),
            "aux_enabled": int(do_aux),
            "aux_type": self.aux,
        }
        return total, logs


if __name__ == "__main__":
    # simple test
    loss_fn = MaskedDiceBCEIgnore2(use_hausdorff=True, aux="ssim")

    B, D, H, W = 2, 160, 160, 160
    
    gt = torch.zeros((B,1,D,H,W), dtype=torch.long)

    # cube size (edge length)
    s = 20

    # 3 cube centers (z,y,x)
    centers = [
        (20, 20, 20),
        (40, 55, 30),
        (60, 30, 60),
    ]
    for cz, cy, cx in centers:
        z1, z2 = cz - s//2, cz + s//2
        y1, y2 = cy - s//2, cy + s//2
        x1, x2 = cx - s//2, cx + s//2
        gt[..., z1:z2, y1:y2, x1:x2] = 1

    gt01 = gt.float()
    logits = torch.where(gt==1, torch.full_like(gt, 10.0), torch.full_like(gt, -10.0)).float()

    loss, logs = loss_fn(logits, gt)
    print("Loss:", loss.item())
    for k, v in logs.items():
        print(f"  {k}: {v}")