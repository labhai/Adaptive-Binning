from __future__ import annotations

import numpy as np
import torch


class AdaptiveDiscretizer:
    def __init__(
        self,
        num_features: list[int],
        cat_features: list[int],
        initial_bins: int = 2,
        max_bins: int = 200,
    ) -> None:
        self.num_features = [int(x) for x in num_features]
        self.cat_features = [int(x) for x in cat_features]
        self.initial_bins = int(initial_bins)
        self.max_bins = int(max_bins)
        self.edges: dict[int, list[float]] = {}

    def fit(self, X_train: torch.Tensor) -> None:
        values = X_train.detach().cpu().numpy()
        for fid in self.num_features:
            col = values[:, fid]
            uniq = np.unique(col)
            if uniq.size <= 1 or self.initial_bins <= 1:
                cutpoints = np.array([], dtype=np.float64)
            elif uniq.size <= self.initial_bins:
                cutpoints = uniq[1:].astype(np.float64)
            else:
                qs = np.arange(1, self.initial_bins) * (100.0 / self.initial_bins)
                cutpoints = np.unique(np.percentile(col, qs).astype(np.float64))
            self.edges[fid] = np.concatenate(([-np.inf], cutpoints, [np.inf])).tolist()

    def bin_count(self, fid: int) -> int:
        return max(1, len(self.edges[int(fid)]) - 1)

    def current_num_bins(self) -> list[int]:
        return [self.bin_count(fid) for fid in self.num_features]

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        out = torch.zeros(X.shape, dtype=torch.long, device=device)
        if self.cat_features:
            out[:, self.cat_features] = X[:, self.cat_features].round().long()

        X_np = X.detach().cpu().numpy()
        for fid in self.num_features:
            edges = np.asarray(self.edges[fid], dtype=np.float64)
            cutpoints = edges[1:-1]
            if cutpoints.size == 0:
                vals = np.zeros(X_np.shape[0], dtype=np.int64)
            else:
                vals = np.digitize(X_np[:, fid], bins=cutpoints, right=False).astype(np.int64)
            out[:, fid] = torch.as_tensor(vals, dtype=torch.long, device=device)
        return out

    def update_edges(self, new_edges: dict[int, list[float]]) -> None:
        for fid, edges in new_edges.items():
            fid = int(fid)
            sorted_edges = sorted(set(float(x) for x in edges))
            if len(sorted_edges) - 1 <= self.max_bins:
                self.edges[fid] = sorted_edges
