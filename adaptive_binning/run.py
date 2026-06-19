from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from adaptive_binning.config import load_yaml
from adaptive_binning.training.evaluate import evaluate_checkpoint
from adaptive_binning.training.pretrain import pretrain


def _resolve_device(value: str | None) -> str:
    if value in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def _set_if_present(cfg: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        cfg[key] = value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Adaptive Binning pretraining and evaluation.")
    parser.add_argument("--config", default="configs/adaptive_binning.yaml", help="Path to YAML config.")
    parser.add_argument("--stage", choices=["all", "pretrain", "evaluate"], default="all")
    parser.add_argument("--dataset", default=None, help="Dataset name or paper abbreviation, e.g., HF or HFC.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --stage evaluate.")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and summaries.")
    parser.add_argument("--data-root", default=None, help="Directory containing raw dataset folders.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--seed", type=int, default=None, help="Pretraining/data split seed.")
    parser.add_argument("--eval-seed", type=int, default=None, help="Downstream evaluation seed.")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--eval-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mlpwidth", type=int, default=None)
    parser.add_argument("--mlpdepth", type=int, default=None)
    parser.add_argument("--initial-bins", type=int, default=None)
    parser.add_argument("--max-bins", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-bin-size", type=int, default=None)
    parser.add_argument("--run-finetune", action="store_true", help="Also run end-to-end fine-tuning.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)

    _set_if_present(cfg, "dataset", args.dataset)
    _set_if_present(cfg, "data_root", args.data_root)
    _set_if_present(cfg, "device", args.device)
    _set_if_present(cfg, "seed", args.seed)
    _set_if_present(cfg, "eval_seed", args.eval_seed)
    _set_if_present(cfg, "pretrain_epochs", args.pretrain_epochs)
    _set_if_present(cfg, "eval_epochs", args.eval_epochs)
    _set_if_present(cfg, "batch_size", args.batch_size)
    _set_if_present(cfg, "mlpwidth", args.mlpwidth)
    _set_if_present(cfg, "mlpdepth", args.mlpdepth)
    _set_if_present(cfg, "initial_bins", args.initial_bins)
    _set_if_present(cfg, "max_bins", args.max_bins)
    _set_if_present(cfg, "patience", args.patience)
    _set_if_present(cfg, "min_bin_size", args.min_bin_size)
    cfg["device"] = _resolve_device(cfg.get("device"))

    dataset = str(cfg.get("dataset", "HF"))
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.dataset is None and cfg.get("output_dir") is not None:
        output_dir = Path(cfg["output_dir"])
    else:
        output_dir = Path(f"outputs/{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "adaptive_binning_checkpoint.pt"

    if args.stage in {"all", "pretrain"}:
        checkpoint_path = pretrain(dataset, cfg, output_dir)

    if args.stage in {"all", "evaluate"}:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        results = evaluate_checkpoint(
            checkpoint_path,
            cfg,
            output_dir,
            eval_seed=int(cfg.get("eval_seed", 0)),
            run_finetune=bool(args.run_finetune or cfg.get("run_finetune", False)),
        )
        for phase, score in results.items():
            print(f"[result] dataset={dataset} phase={phase} score={score:.8f}", flush=True)


if __name__ == "__main__":
    main()
