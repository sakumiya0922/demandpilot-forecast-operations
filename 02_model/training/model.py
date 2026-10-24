"""LightGBMで共通利用する学習条件と特徴量の変換。"""

import pandas as pd

from training.training_data import FEATURE_COLUMNS

PARAMETERS = dict(objective="regression_l1", n_estimators=100, learning_rate=0.05,
                  num_leaves=15, min_child_samples=20, random_state=42,
                  n_jobs=2, deterministic=True, force_col_wise=True, verbosity=-1)


def matrix(frame, codes):
    values=frame[FEATURE_COLUMNS].copy()
    values["stock_code"]=pd.Categorical(values.stock_code,categories=codes)
    if values.isna().any().any():
        raise ValueError("特徴量に欠損または未知の商品があります")
    return values
