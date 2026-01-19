import gc
import os
import re
import glob
import h5py
import pandas as pd
import tifffile as tiff
from tqdm.auto import tqdm
import random
import numpy as np
import torch


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