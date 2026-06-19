from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from adaptive_binning.data.preprocessing import batch_indices, load_tabular_data
from adaptive_binning.data.registry import dataset_config
from adaptive_binning.metrics import higher_is_better, score_predictions
from adaptive_binning.models import AdaptiveBinningModel
from adaptive_binning.models.autoencoder import DownstreamModel
from adaptive_binning.seed import seed_everything


def _build_pretrained_model(data, cfg: dict, ds_cfg: dict, checkpoint: dict) -> AdaptiveBinningModel:
    model = AdaptiveBinningModel(
        input_dim=len(data.cat_features) + len(data.num_features),
        cat_features=data.cat_features,
        num_features=data.num_features,
        cat_cardinalities=data.cat_cardinalities,
        width=int(cfg.get("mlpwidth", ds_cfg["mlpwidth"])),
        depth=int(cfg.get("mlpdepth", ds_cfg["mlpdepth"])),
        max_bins=int(cfg["max_bins"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def _loss(tasktype: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tasktype == "binclass":
        return F.binary_cross_entropy_with_logits(pred.view(-1), target.view(-1).float())
    if tasktype == "multiclass":
        return F.cross_entropy(pred, target.view(-1).long())
    return F.mse_loss(pred.view(-1), target.view(-1).float())


def _predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for idx in batch_indices(X.shape[0], batch_size, X.device, shuffle=False):
            outputs.append(model(X[idx]).detach())
    return torch.cat(outputs, dim=0)


def _run_phase(
    model: DownstreamModel,
    data,
    dataname: str,
    phase: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> float:
    seed_everything(seed)
    if phase == "linear":
        for p in model.encoder.parameters():
            p.requires_grad = False
        trainable = list(model.head.parameters())
    else:
        for p in model.parameters():
            p.requires_grad = True
        trainable = list(model.parameters())

    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-5 if phase == "finetune" else 0.0)
    best_state = None
    best_score = -float("inf") if higher_is_better(data.tasktype) else float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        if phase == "linear":
            model.encoder.eval()
        for idx in batch_indices(data.X_train.shape[0], batch_size, data.X_train.device, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            pred = model(data.X_train[idx])
            loss = _loss(data.tasktype, pred, data.y_train[idx])
            loss.backward()
            optimizer.step()

        pred_val = _predict(model, data.X_val, batch_size)
        val_score = score_predictions(dataname, data.tasktype, data.y_val, pred_val, data.y_std)
        better = val_score > best_score if higher_is_better(data.tasktype) else val_score < best_score
        if better:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{phase}] epoch={epoch:04d} val_score={val_score:.6f}", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(data.X_train.device) for k, v in best_state.items()})
    pred_test = _predict(model, data.X_test, batch_size)
    return score_predictions(dataname, data.tasktype, data.y_test, pred_test, data.y_std)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cfg: dict,
    output_dir: str | Path,
    eval_seed: int,
    run_finetune: bool = False,
) -> dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.get("device", "cpu"), weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.get("device", "cpu"))
    dataname = checkpoint["dataset"]
    seed_everything(int(eval_seed))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ds_cfg = dataset_config(dataname)
    data = load_tabular_data(
        dataname,
        root=cfg.get("data_root", "data/raw"),
        seed=int(checkpoint["cfg"].get("seed", 0)),
        cat_threshold=int(cfg.get("cat_threshold", 20)),
        device=device,
    )
    batch_size = int(cfg.get("batch_size", ds_cfg["batch_size"]))
    pre_model = _build_pretrained_model(data, checkpoint["cfg"], ds_cfg, checkpoint).to(device)

    def selector(x):
        return pre_model._select_input(x)

    results = {}
    for phase, lr in [("linear", float(cfg.get("linear_lr", 1e-2)))]:
        downstream = DownstreamModel(pre_model.encoder, selector, int(checkpoint["cfg"].get("mlpwidth", ds_cfg["mlpwidth"])), data.ydim).to(device)
        results["linear"] = _run_phase(
            downstream,
            data,
            dataname,
            phase=phase,
            epochs=int(cfg.get("eval_epochs", 100)),
            batch_size=batch_size,
            lr=lr,
            seed=int(eval_seed),
        )

    if run_finetune:
        pre_model = _build_pretrained_model(data, checkpoint["cfg"], ds_cfg, checkpoint).to(device)

        def ft_selector(x):
            return pre_model._select_input(x)

        downstream = DownstreamModel(pre_model.encoder, ft_selector, int(checkpoint["cfg"].get("mlpwidth", ds_cfg["mlpwidth"])), data.ydim).to(device)
        results["finetune"] = _run_phase(
            downstream,
            data,
            dataname,
            phase="finetune",
            epochs=int(cfg.get("eval_epochs", 100)),
            batch_size=batch_size,
            lr=float(cfg.get("finetune_lr", 1e-3)),
            seed=int(eval_seed),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "evaluation_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "score"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.8f}"])
    return results
