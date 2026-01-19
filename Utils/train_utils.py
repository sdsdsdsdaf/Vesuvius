from collections import defaultdict
from collections.abc import Callable
import os
import wandb

import torch
import torch.nn as nn
import monai
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm


try:
    from Utils.Dataset import VesuviusH5PatchDataset3D
    from Utils.Sampler import get_batch_sampler
    from Utils.utils import build_h5_group_from_train_images, detect_nan_inf
    from Utils.Loss import MaskedDiceBCETwitterignore2
    from Utils.utils import cleanup_memory, seed_everything, to5dim
    from Utils.transform import get_train_transform, get_val_transform
    
except:
    from Dataset import VesuviusH5PatchDataset3D
    from Sampler import get_batch_sampler
    from utils import build_h5_group_from_train_images, detect_nan_inf
    from Loss import MaskedDiceBCETwitterignore2
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
    
    for idx, batch in enumerate(tqdm(dataloader)):
        batch: tuple[torch.Tensor, torch.Tensor, dict[str, int]] = batch
        
        inputs, targets, meta = batch
        inputs = inputs.to(device)
        inputs = to5dim(inputs)  # (B,1,D,H,W)
        targets = targets.to(device)
        targets = to5dim(inputs)  # (B,1,D,H,W)
        
        
        optimizer.zero_grad()
        
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None, dtype=torch.float16):
            outputs = model(inputs)
            loss, log = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        if detect_nan_inf(loss=loss, model=model, logs=log):
            print(f" [Epoch {epoch}] Batch [{idx}] Dectect Nan Or Inf --> Skipping Batch")
            continue
        
        # (선택) gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        total_loss['total'] += loss.item() * inputs.size(0)
        for k, v in log.items():
            total_loss[k] += v * inputs.size(0)
            
        if use_wnb:
            global_step = epoch * len(dataloader) + idx
            train_log = {
                "step": global_step,
                "train_step/loss": loss.item()
            }
            
            for k, v in log.items():
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
):

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, scaler, use_wnb
        )
        
        if use_wnb:
            log = {
                "epoch": epoch,
            }
            for k, v in train_loss.items():
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
            "lr": 1e-4,
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
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = MaskedDiceBCETwitterignore2(tear="tv")

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
        batch_size=8,
        pos_fraction=0.5,
        pos_thr=1,
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
        num_workers=4,
        pin_memory=True,
    )
    
    print("DataLoader initialized with batch size 12.")
    
    print("=============== Starting training for 10 epochs... ===============")
    train(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=10,
        scaler=torch.GradScaler(),
        use_wnb=True,
    )
    
    print("\n=============== Training completed. Saving model... ===============\n")
    weight_dir = "weights"
    weight_path = os.path.join(weight_dir, "unet_vesuvius.pth")
    os.makedirs(weight_dir, exist_ok=True)
    to_save = model._orig_mod.state_dict()
    torch.save(to_save, weight_path)
    print(f"Model saved to {weight_path}")
    
    wandb.finish()