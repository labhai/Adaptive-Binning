from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW

from adaptive_binning.data.preprocessing import batch_indices, load_tabular_data
from adaptive_binning.data.registry import dataset_config
from adaptive_binning.method.digs import calculate_digs_splits
from adaptive_binning.method.discretizer import AdaptiveDiscretizer
from adaptive_binning.method.hord import hord_loss
from adaptive_binning.models import AdaptiveBinningModel
from adaptive_binning.seed import seed_everything


def _build_model(data, cfg: dict, ds_cfg: dict) -> AdaptiveBinningModel:
    return AdaptiveBinningModel(
        input_dim=len(data.cat_features) + len(data.num_features),
        cat_features=data.cat_features,
        num_features=data.num_features,
        cat_cardinalities=data.cat_cardinalities,
        width=int(cfg.get("mlpwidth", ds_cfg["mlpwidth"])),
        depth=int(cfg.get("mlpdepth", ds_cfg["mlpdepth"])),
        max_bins=int(cfg["max_bins"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def pretrain(dataname: str, cfg: dict, output_dir: str | Path) -> Path:
    dataname = str(dataname)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ds_cfg = dataset_config(dataname)

    data = load_tabular_data(
        dataname,
        root=cfg.get("data_root", "data/raw"),
        seed=seed,
        cat_threshold=int(cfg.get("cat_threshold", 20)),
        device=device,
    )

    batch_size = int(cfg.get("batch_size", ds_cfg["batch_size"]))
    discretizer = AdaptiveDiscretizer(
        data.num_features,
        data.cat_features,
        initial_bins=int(cfg.get("initial_bins", 2)),
        max_bins=int(cfg.get("max_bins", 200)),
    )
    discretizer.fit(data.X_train)
    ybin_train = discretizer.transform(data.X_train)
    ybin_val = discretizer.transform(data.X_val)

    model = _build_model(data, cfg, ds_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg.get("pretrain_lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-5)))

    patience = int(cfg.get("patience", 5))
    plateau_delta = float(cfg.get("plateau_delta", 1e-4))
    threshold = float(cfg.get("digs_threshold", 1e-4))
    loss_weights = dict(cfg.get("loss_weights", {"sord": 10.0, "mse": 0.1, "var": 0.001}))
    best_metric = torch.full((len(data.num_features),), float("inf"), device=device)
    patience_count = torch.zeros((len(data.num_features),), dtype=torch.long, device=device)
    history = []

    epochs = int(cfg.get("pretrain_epochs", 1000))
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        metric_sum = torch.zeros((len(data.num_features),), device=device)
        n_steps = 0

        for idx in batch_indices(data.X_train.shape[0], batch_size, device, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data.X_train[idx])
            loss, detail = hord_loss(
                outputs,
                ybin_train[idx],
                data.cat_features,
                data.num_features,
                discretizer.current_num_bins(),
                loss_weights,
                sord_distance=str(cfg.get("sord_distance", "squared")),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 100.0)))
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            if detail["plateau_metric"].numel() > 0:
                metric_sum += detail["plateau_metric"].to(device)
            n_steps += 1

        avg_loss = epoch_loss / max(1, n_steps)
        split_count = 0
        avg_metric = metric_sum / max(1, n_steps)

        triggered = []
        if avg_metric.numel() > 0:
            improved = avg_metric < (best_metric - plateau_delta)
            best_metric = torch.where(improved, avg_metric, best_metric)
            patience_count = torch.where(improved, torch.zeros_like(patience_count), patience_count + 1)
            for pos, fid in enumerate(data.num_features):
                if patience_count[pos].item() >= patience and discretizer.bin_count(fid) < discretizer.max_bins:
                    triggered.append(int(fid))

        if triggered:
            new_edges, split_count = calculate_digs_splits(
                model=model,
                X_train=data.X_train,
                discretizer=discretizer,
                triggered_features=triggered,
                threshold=threshold,
                batch_size=batch_size,
                min_bin_size=int(cfg.get("min_bin_size", 32)),
            )
            if split_count > 0:
                discretizer.update_edges(new_edges)
                ybin_train = discretizer.transform(data.X_train)
                ybin_val = discretizer.transform(data.X_val)
            for fid in triggered:
                pos = data.num_features.index(fid)
                best_metric[pos] = float("inf")
                patience_count[pos] = 0

        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "avg_num_bins": float(sum(discretizer.current_num_bins()) / max(1, len(data.num_features))),
                "max_num_bins": int(max(discretizer.current_num_bins()) if data.num_features else 0),
                "splits": int(split_count),
            }
        )
        print(
            f"[pretrain] epoch={epoch:04d} loss={avg_loss:.6f} "
            f"avg_bins={history[-1]['avg_num_bins']:.2f} splits={split_count}",
            flush=True,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "adaptive_binning_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "edges": discretizer.edges,
            "cfg": cfg,
            "dataset": dataname,
            "cat_features": data.cat_features,
            "num_features": data.num_features,
            "cat_cardinalities": data.cat_cardinalities,
            "ydim": data.ydim,
            "tasktype": data.tasktype,
            "y_std": data.y_std,
            "history": history,
        },
        checkpoint_path,
    )
    with open(output_dir / "pretrain_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return checkpoint_path
