import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss

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

def tv_loss_3d(prob: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is not None:
        prob = prob * valid
    dz = (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()
    dy = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().mean()
    dx = (prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1]).abs().mean()
    return dz + dy + dx

class MaskedDiceBCETwitterignore2(nn.Module) :
    """
    0=bg, 1=fg, 2=ignore
    logits: (B,1,D,H,W) or (B,D,H,W)
    gt:     (B,1,D,H,W) same spatial, int {0,1,2}
    
    """

    def __init__(
        self,
        tear: str = "none",            # "none" | "hausdorff" | "boundary" | "tv"
        lambda_tear: float = 0.02,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        eps: float = 1e-6,
        apply_tear_every: int = 1,
    ):
        
        """
        Example usage:
        ```python
        loss_fn = MaskedDiceBCETwitterignore2(tear="tv", lambda_tear=0.1, dice_weight=1.0, bce_weight=1.0)
        loss, logs = loss_fn(logits, gt)
        ```
        """
        
        super().__init__()
        self.tear = tear.lower()
        self.lambda_tear = float(lambda_tear)
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.eps = float(eps)
        self.apply_tear_every = int(apply_tear_every)
        self._step = 0

        self.hd = HausdorffDTLoss(sigmoid=True, reduction="mean")
        self.bd = None
        
    def forward(self, logits: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, LossLogOutput]:
        self._step += 1

        logits = _ensure_b1dhw(logits)
        gt = _ensure_b1dhw(gt)

        valid = (gt != 2)                 # bool
        y = (gt == 1).float()             # {0,1}

        prob = torch.sigmoid(logits)

        # ---- Masked BCE ----
        n = valid.sum().clamp_min(1)
        bce = F.binary_cross_entropy_with_logits(logits[valid], y[valid], reduction="sum") / n

        # ---- Masked Soft Dice (fg only) ----
        p = prob[valid]
        g = y[valid]
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)

        base = self.dice_weight * dice + self.bce_weight * bce

        # ---- Tear term (optional, applied sparsely) ----
        tear_loss = logits.sum() * 0.0
        do_tear = (self.tear != "none") and (self.lambda_tear > 0) and (self.apply_tear_every > 0) and (self._step % self.apply_tear_every == 0)

        if do_tear:
            valid_f = valid.float()
            logits_m = logits * valid_f
            y_m = y * valid_f

            if self.tear == "hausdorff":
                tear_loss = self.hd(logits_m, y_m)
            elif self.tear == "boundary":
                raise NotImplementedError("Boundary loss not implemented yet.")
            elif self.tear == "tv":
                tear_loss = tv_loss_3d(prob, valid=valid_f)
            else:
                raise ValueError(f"Unknown tear='{self.tear}'")
        
        total = base + self.lambda_tear * tear_loss
        
        logs = {
            "loss_base": float(base.detach().item()),
            "loss_dice": float(dice.detach().item()),
            "loss_bce": float(bce.detach().item()),
            "loss_tear": float(tear_loss.detach().item()) if do_tear else 0.0,
            "loss_total": float(total.detach().item()),
            "loss_tear_enabled": int(do_tear),
        }
        return total, logs

if __name__ == "__main__":
    # simple test
    loss_fn = MaskedDiceBCETwitterignore2(tear="tv", lambda_tear=0.1, dice_weight=1.0, bce_weight=1.0)

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