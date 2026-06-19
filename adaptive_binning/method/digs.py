from __future__ import annotations

import numpy as np
import torch

from adaptive_binning.data.preprocessing import batch_indices
from adaptive_binning.method.discretizer import AdaptiveDiscretizer


def _dispersion(unit_emb: np.ndarray, eps: float = 1e-6) -> float:
    if unit_emb.shape[0] <= 1:
        return 0.0
    mean_u = unit_emb.mean(axis=0)
    hom = float(np.dot(mean_u, mean_u))
    hom = min(1.0, max(0.0, hom))
    return float(abs(np.log(eps + hom)))


def _extract_unit_embeddings(model, X: torch.Tensor, batch_size: int) -> np.ndarray:
    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for idx in batch_indices(X.shape[0], batch_size, X.device, shuffle=False):
            z = model.encode(X[idx])
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            chunks.append(z.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0).numpy()


def calculate_digs_splits(
    model,
    X_train: torch.Tensor,
    discretizer: AdaptiveDiscretizer,
    triggered_features: list[int],
    threshold: float,
    batch_size: int,
    min_bin_size: int = 32,
    eps: float = 1e-6,
) -> tuple[dict[int, list[float]], int]:
    if not triggered_features:
        return {}, 0

    X_np = X_train.detach().cpu().numpy()
    unit_z = _extract_unit_embeddings(model, X_train, batch_size=batch_size)
    new_edges: dict[int, list[float]] = {}
    split_count = 0

    for fid in triggered_features:
        fid = int(fid)
        old_edges = discretizer.edges[fid]
        if len(old_edges) - 1 >= discretizer.max_bins:
            continue

        feat_values = X_np[:, fid]
        candidate_edges: list[float] = []

        for left_edge, right_edge in zip(old_edges[:-1], old_edges[1:]):
            candidate_edges.append(float(left_edge))
            parent_mask = (feat_values >= left_edge) & (feat_values < right_edge)
            parent_values = feat_values[parent_mask]
            if parent_values.shape[0] < 2 * min_bin_size:
                continue

            split_point = float(np.median(parent_values))
            if not (split_point > left_edge and split_point < right_edge):
                continue

            left_mask_local = parent_values < split_point
            right_mask_local = ~left_mask_local
            if left_mask_local.sum() < min_bin_size or right_mask_local.sum() < min_bin_size:
                continue

            parent_z = unit_z[parent_mask]
            left_global = parent_mask.copy()
            left_global[parent_mask] = left_mask_local
            right_global = parent_mask.copy()
            right_global[parent_mask] = right_mask_local

            n_parent = float(parent_values.shape[0])
            w_left = float(left_mask_local.sum()) / n_parent
            w_right = 1.0 - w_left

            parent_var = float(np.var(parent_values, dtype=np.float64))
            left_var = float(np.var(parent_values[left_mask_local], dtype=np.float64))
            right_var = float(np.var(parent_values[right_mask_local], dtype=np.float64))
            gain_var = parent_var - (w_left * left_var + w_right * right_var)

            parent_disp = _dispersion(parent_z, eps=eps)
            left_disp = _dispersion(unit_z[left_global], eps=eps)
            right_disp = _dispersion(unit_z[right_global], eps=eps)
            gain_disp = parent_disp - (w_left * left_disp + w_right * right_disp)

            score = gain_var * gain_disp
            if gain_var > 0.0 and gain_disp > 0.0 and score > float(threshold):
                candidate_edges.append(split_point)
                split_count += 1

        candidate_edges.append(float(old_edges[-1]))
        if len(candidate_edges) > len(old_edges):
            new_edges[fid] = sorted(set(candidate_edges))

    return new_edges, split_count
