import os
import kagglehub
from Utils.Typing import *
from Utils.train_utils import run_scroll_group_cv, make_objective_fn
from Utils.metric import Metric
from Utils.utils import build_h5_group_from_train_images
import pandas as pd

if __name__ == "__main__":
    
    data_path = kagglehub.competition_download('vesuvius-challenge-surface-detection')
    test_data_dir = os.path.join(data_path, "test_images")
    train_data_dir = os.path.join(data_path, "train_images")
    train_lable_dir = os.path.join(data_path, "train_labels")
        
    build_h5_group_from_train_images(
        train_images_dir=train_data_dir,
        train_labels_dir=train_lable_dir,
        train_csv_path=os.path.join(data_path, "train.csv"),
        out_h5_path="vesuvius_train_zyx_zyx.h5",
    )
    
    cfg = CVConfig()
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    metric_fn = Metric(cfg.postprocess_cfg.threshold, mode="tear")
    objective_fn = make_objective_fn(objective="f1_minus_proxy", weights={"alpha": 0.2},)
    result = run_scroll_group_cv(df, cfg, metric_fn=metric_fn, objective_fn=objective_fn)