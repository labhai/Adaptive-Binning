from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from adaptive_binning.data.loaders import load_raw_dataset


@dataclass
class TabularData:
    X_train: torch.Tensor
    X_val: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    cat_features: list[int]
    num_features: list[int]
    cat_cardinalities: list[int]
    tasktype: str
    ydim: int
    y_std: float
    columns: list[str]


def _split_arrays(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int):
    if groups is not None:
        unique_groups = np.unique(groups)
        rng = np.random.RandomState(seed)
        unique_groups = unique_groups[rng.permutation(len(unique_groups))]
        n_test = max(1, int(round(len(unique_groups) * 0.20)))
        n_val = max(1, int(round(len(unique_groups) * 0.16)))
        test_groups = unique_groups[:n_test]
        val_groups = unique_groups[n_test : n_test + n_val]
        test_mask = np.isin(groups, test_groups)
        val_mask = np.isin(groups, val_groups)
        train_mask = ~(test_mask | val_mask)
        return X[train_mask], X[val_mask], X[test_mask], y[train_mask], y[val_mask], y[test_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _task_dims(tasktype: str, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> tuple[int, float]:
    if tasktype == "regression":
        return 1, float(np.std(y_train.astype(np.float32)) + 1e-10)
    labels = np.concatenate([y_train, y_val, y_test], axis=0).reshape(-1)
    if tasktype == "binclass":
        return 1, 1.0
    return int(labels.max() - labels.min() + 1), 1.0


def load_tabular_data(
    dataname: str,
    root: str = "data/raw",
    seed: int = 123456,
    cat_threshold: int | None = 20,
    device: str | torch.device = "cpu",
) -> TabularData:
    raw = load_raw_dataset(dataname, root)
    X = raw["X"].astype(np.float32)
    y = raw["y"]
    groups = raw.get("groups")
    X_train, X_val, X_test, y_train, y_val, y_test = _split_arrays(X, y, groups, seed)

    counts = np.array([len(np.unique(X_train[:, i])) for i in range(X_train.shape[1])])
    if cat_threshold is None:
        cat_features: list[int] = []
    else:
        cat_features = np.where(counts <= int(cat_threshold))[0].astype(int).tolist()
    num_features = [i for i in range(X_train.shape[1]) if i not in cat_features]
    cat_cardinalities = [int(counts[i]) for i in cat_features]

    for col in cat_features:
        uniques = sorted(np.unique(X_train[:, col]).tolist())
        mapping = {v: i for i, v in enumerate(uniques)}
        for arr in (X_train, X_val, X_test):
            arr[:, col] = np.array([mapping.get(v, 0) for v in arr[:, col]], dtype=np.float32)

    if num_features:
        mu = X_train[:, num_features].mean(axis=0, keepdims=True)
        std = X_train[:, num_features].std(axis=0, keepdims=True) + 1e-10
        X_train[:, num_features] = (X_train[:, num_features] - mu) / std
        X_val[:, num_features] = (X_val[:, num_features] - mu) / std
        X_test[:, num_features] = (X_test[:, num_features] - mu) / std

    ydim, y_std = _task_dims(raw["tasktype"], y_train, y_val, y_test)
    if raw["tasktype"] == "regression":
        y_mu = y_train.astype(np.float32).mean(axis=0, keepdims=True)
        y_sigma = y_train.astype(np.float32).std(axis=0, keepdims=True) + 1e-10
        y_train = (y_train.astype(np.float32) - y_mu) / y_sigma
        y_val = (y_val.astype(np.float32) - y_mu) / y_sigma
        y_test = (y_test.astype(np.float32) - y_mu) / y_sigma

    device = torch.device(device)
    return TabularData(
        X_train=torch.as_tensor(X_train, dtype=torch.float32, device=device),
        X_val=torch.as_tensor(X_val, dtype=torch.float32, device=device),
        X_test=torch.as_tensor(X_test, dtype=torch.float32, device=device),
        y_train=torch.as_tensor(y_train, device=device),
        y_val=torch.as_tensor(y_val, device=device),
        y_test=torch.as_tensor(y_test, device=device),
        cat_features=cat_features,
        num_features=num_features,
        cat_cardinalities=cat_cardinalities,
        tasktype=raw["tasktype"],
        ydim=ydim,
        y_std=y_std,
        columns=raw["columns"],
    )


def batch_indices(n: int, batch_size: int, device: torch.device, shuffle: bool = True) -> Iterator[torch.Tensor]:
    if shuffle:
        indices = torch.randperm(n, device=device)
    else:
        indices = torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]
