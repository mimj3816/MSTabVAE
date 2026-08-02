"""Data loading and the conditional Dataset used by MSTabVAE.

The loader label-encodes categorical columns, records the feature metadata the
model/loss need (categorical indices & cardinalities, continuous indices), and
splits chronologically-or-not depending on ``shuffle``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


@dataclass
class DataBundle:
    X: pd.DataFrame                 # full, label-encoded features (target dropped)
    y: pd.Series
    splits: dict                    # {"train"/"valid"/"test": (X_df, y_series)}
    cat_idxs: list
    cat_dims: list
    continuous_idx: list
    label_encoders: dict
    target_col: str
    feature_names: list = field(default_factory=list)


def load_tabular(csv_path, target_col, sort_col=None, train_size=0.75,
                 shuffle=False, seed=42):
    """Load a CSV, label-encode object columns, and split into train/valid/test.

    valid/test each take half of the ``1 - train_size`` remainder.
    """
    df = pd.read_csv(csv_path)
    if sort_col is not None and sort_col in df.columns:
        df = df.sort_values(sort_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int", "float"]).columns.tolist()

    categorical_dims, label_encoders = {}, {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].values)
        categorical_dims[col] = len(le.classes_)
        label_encoders[col] = le

    features = list(X.columns)
    cat_idxs = [features.index(f) for f in categorical_features]
    cat_dims = [categorical_dims[f] for f in categorical_features]
    continuous_idx = [features.index(f) for f in numerical_features]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, shuffle=shuffle,
        random_state=seed if shuffle else None)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=shuffle,
        random_state=seed if shuffle else None)

    return DataBundle(
        X=X, y=y,
        splits={"train": (X_train, y_train), "valid": (X_valid, y_valid),
                "test": (X_test, y_test)},
        cat_idxs=cat_idxs, cat_dims=cat_dims, continuous_idx=continuous_idx,
        label_encoders=label_encoders, target_col=target_col,
        feature_names=features,
    )


class TableWithConditions(Dataset):
    """Wraps a feature tensor plus a dict of per-row condition tensors."""

    def __init__(self, X, cond_tensors):
        self.X = X
        self.cond_tensors = cond_tensors
        for v in cond_tensors.values():
            assert len(v) == X.size(0)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.cond_tensors.items()}


def collate_fn_cond(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    conds = [item[1] for item in batch]
    cond_batch = {k: torch.stack([c[k] for c in conds], dim=0) for k in conds[0]}
    return xs, cond_batch


def make_condition_dataset(X_df, y_series, target_col):
    """Build a ``TableWithConditions`` conditioned on the (categorical) target."""
    X_t = torch.tensor(X_df.values, dtype=torch.float32)
    cond = {target_col: torch.tensor(y_series.values, dtype=torch.long)}
    return TableWithConditions(X_t, cond)
