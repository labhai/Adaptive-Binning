from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score


ORDINAL_MULTICLASS = {"EOL", "MH"}


def score_predictions(
    dataname: str,
    tasktype: str,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_std: float | torch.Tensor | None = None,
) -> float:
    dataname = dataname.upper()
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    if tasktype == "binclass":
        probs = torch.sigmoid(y_pred.view(-1)).numpy()
        target = y_true.view(-1).numpy()
        try:
            return float(roc_auc_score(target, probs))
        except ValueError:
            return 0.0

    if tasktype == "multiclass":
        pred = y_pred.argmax(dim=1).numpy()
        target = y_true.view(-1).long().numpy()
        if dataname in ORDINAL_MULTICLASS:
            return float(cohen_kappa_score(target, pred, weights="quadratic"))
        return float(np.mean(pred == target))

    pred = y_pred.view(-1)
    target = y_true.view(-1)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    if y_std is not None:
        if isinstance(y_std, torch.Tensor):
            y_std = float(y_std.detach().cpu().view(-1)[0].item())
        rmse *= float(y_std)
    return float(rmse)


def higher_is_better(tasktype: str) -> bool:
    return tasktype != "regression"
