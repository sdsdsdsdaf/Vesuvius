from collections import defaultdict
from collections.abc import Callable
import os
import wandb

import torch
import torch.nn as nn
import monai
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
    from Utils.utils import build_h5_group_from_train_images, detect_nan_inf
    from Utils.Loss import MaskedDiceBCEIgnore2
    from Utils.utils import cleanup_memory, seed_everything, to5dim
    from Utils.transform import get_train_transform, get_val_transform
    
except:
    from Dataset import VesuviusH5PatchDataset3D
    from Sampler import get_batch_sampler
    from utils import build_h5_group_from_train_images, detect_nan_inf
    from Loss import MaskedDiceBCEIgnore2
    from utils import cleanup_memory, seed_everything, to5dim
    from transform import get_val_transform, get_train_transform

    
    
LossFn = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, dict[str, float]]
]

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
    
    for idx, batch in enumerate(tqdm(dataloader, leave=False)):
        batch: tuple[torch.Tensor, torch.Tensor, dict[str, int]] = batch
        
        inputs, targets, meta = batch
        inputs = inputs.to(device, non_blocking=True)
        inputs = to5dim(inputs)  # (B,1,D,H,W)
        targets = targets.to(device, non_blocking=True)
        targets = to5dim(targets)  # (B,1,D,H,W)
        
        
        optimizer.zero_grad()
        
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss, log = loss_fn(outputs, targets)

        # scaler.scale(loss).backward()
        # (선택) gradient clipping
        # scaler.unscale_(optimizer)
        
        loss.backward()
        # TODO 후에 encoder decoder 분리
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        if detect_nan_inf(loss=loss, model=model, logs=log):
            print(f" [Epoch {epoch}] Batch [{idx}] Dectect Nan Or Inf --> Skipping Batch")
            optimizer.zero_grad(set_to_none=True)
            optimizer.step()
            continue

        # scaler.step(optimizer)
        # scaler.update()
        
        optimizer.step()
        total_loss['total'] += loss.item() * inputs.size(0)
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
    grad_clip: float = float('inf')
):

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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")
        
        # Clean up memory after each epoch
        cleanup_memory()
        torch.cuda.synchronize()
        
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