from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_SORD_CACHE: dict[tuple[str, int, str], torch.Tensor] = {}


def _sord_table(k: int, device: torch.device, distance: str = "squared") -> torch.Tensor:
    key = (str(device), int(k), str(distance))
    if key in _SORD_CACHE:
        return _SORD_CACHE[key]
    idx = torch.arange(k, device=device, dtype=torch.float32)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    if distance == "squared":
        dist = diff.abs().pow(2)
    elif distance == "abs":
        dist = diff.abs()
    else:
        raise ValueError(f"Unknown SORD distance: {distance}")
    table = F.softmax(-dist, dim=1)
    _SORD_CACHE[key] = table
    return table


def _scale_index(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return torch.zeros_like(x)
    mu = (k - 1) / 2.0
    sigma = math.sqrt((k * k - 1) / 12.0)
    return (x - mu) / (sigma + 1e-12)


def hord_loss(
    outputs: dict,
    targets: torch.Tensor,
    cat_features: list[int],
    num_features: list[int],
    num_bins: list[int],
    weights: dict[str, float],
    sord_distance: str = "squared",
) -> tuple[torch.Tensor, dict]:
    device = targets.device
    loss_cat = torch.tensor(0.0, device=device)
    loss_num = torch.tensor(0.0, device=device)
    cat_losses = []
    num_losses = []
    plateau_metric = []

    for j, fid in enumerate(cat_features):
        logits = outputs["cat_logits"][j]
        y = targets[:, fid].long()
        cat_losses.append(F.cross_entropy(logits, y))

    if cat_losses:
        loss_cat = torch.stack(cat_losses).mean()

    w_sord = float(weights.get("sord", 10.0))
    w_mse = float(weights.get("mse", 0.1))
    w_var = float(weights.get("var", 0.001))

    for j, fid in enumerate(num_features):
        k = int(num_bins[j])
        logits = outputs["num_logits"][j][:, :k]
        y = targets[:, fid].long().clamp(0, k - 1)
        probs = F.softmax(logits, dim=1)
        idx = torch.arange(k, device=device, dtype=probs.dtype).view(1, k)
        expected = (probs * idx).sum(dim=1)
        target_f = y.to(dtype=expected.dtype)

        table = _sord_table(k, device=device, distance=sord_distance)
        soft_targets = table.index_select(0, y)
        sord = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        mse = F.mse_loss(_scale_index(expected, k), _scale_index(target_f, k))
        var = (probs * (idx - expected.unsqueeze(1)).pow(2)).sum(dim=1).mean()
        feat_loss = w_sord * sord + w_mse * mse + w_var * var
        num_losses.append(feat_loss)
        plateau_metric.append((w_sord * sord + w_mse * mse + w_var * var).detach())

    if num_losses:
        loss_num = torch.stack(num_losses).mean()

    n_cat = len(cat_features)
    n_num = len(num_features)
    if n_cat and n_num:
        total = (n_cat / (n_cat + n_num)) * loss_cat + (n_num / (n_cat + n_num)) * loss_num
    elif n_cat:
        total = loss_cat
    elif n_num:
        total = loss_num
    else:
        total = torch.tensor(0.0, device=device)

    metric_tensor = (
        torch.stack(plateau_metric).detach()
        if plateau_metric
        else torch.empty(0, device=device)
    )
    return total, {
        "loss_cat": loss_cat.detach(),
        "loss_num": loss_num.detach(),
        "plateau_metric": metric_tensor,
    }
