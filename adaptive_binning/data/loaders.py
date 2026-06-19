from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_binning.data.registry import canonical_name, dataset_config


def _encode_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            values = sorted(df[col].dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(values)}
            df[col] = df[col].map(mapping)
    return df


def _require_files(root: Path, dataname: str) -> Path:
    cfg = dataset_config(dataname)
    folder = root / canonical_name(dataname)
    missing = [f for f in cfg["files"] if not (folder / f).exists()]
    if missing:
        msg = (
            f"Missing raw files for {dataname}: {missing}\n"
            f"Expected under: {folder}\n"
            f"See README.md for download instructions."
        )
        raise FileNotFoundError(msg)
    return folder


def load_raw_dataset(dataname: str, root: str | Path = "data/raw") -> dict:
    dataname = canonical_name(dataname)
    root = Path(root)
    folder = _require_files(root, dataname)
    return globals()[f"_load_{dataname}"](folder)


def _load_ILPD(folder: Path) -> dict:
    col_names = [
        "Age",
        "Gender",
        "TB",
        "DB",
        "Alkphos",
        "Sgpt",
        "Sgot",
        "TP",
        "ALB",
        "AG_Ratio",
        "Selector",
    ]
    df = pd.read_csv(folder / "Indian Liver Patient Dataset (ILPD).csv", header=None, names=col_names)
    df = df.dropna(subset=["AG_Ratio"])
    df = _encode_non_numeric_columns(df)
    y = (df["Selector"] - df["Selector"].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df.drop(columns=["Selector"]).to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "binclass", "columns": col_names[:-1], "target": "Selector"}


def _load_HF(folder: Path) -> dict:
    df = pd.read_csv(folder / "heart_failure.csv")
    df = _encode_non_numeric_columns(df)
    target = "death_event"
    y = (df[target] - df[target].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "binclass", "columns": df.columns.drop(target).tolist(), "target": target}


def _load_Cardio(folder: Path) -> dict:
    df = pd.read_excel(folder / "CTG.xls", sheet_name="Data", header=1)
    df = df.dropna(subset=["CLASS"])
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    features = [
        "LB",
        "AC",
        "FM",
        "UC",
        "DL",
        "DS",
        "DP",
        "ASTV",
        "MSTV",
        "ALTV",
        "MLTV",
        "Width",
        "Min",
        "Max",
        "Nmax",
        "Nzeros",
        "Mode",
        "Mean",
        "Median",
        "Variance",
        "Tendency",
    ]
    df = _encode_non_numeric_columns(df[features + ["CLASS"]])
    y = (df["CLASS"] - df["CLASS"].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df[features].to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "multiclass", "columns": features, "target": "CLASS"}


def _load_ESR(folder: Path) -> dict:
    df = pd.read_csv(folder / "Epileptic Seizure Recognition.csv")
    first_col = df.columns[0]
    df = df.drop(columns=[first_col])
    df = _encode_non_numeric_columns(df)
    y = (df["y"] - df["y"].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df.drop(columns=["y"]).to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "multiclass", "columns": df.columns.drop("y").tolist(), "target": "y"}


def _load_EOL(folder: Path) -> dict:
    target = "NObeyesdad"
    df = pd.read_csv(folder / "ObesityDataSet_raw_and_data_sinthetic.csv")
    df = _encode_non_numeric_columns(df)
    y = (df[target] - df[target].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "multiclass", "columns": df.columns.drop(target).tolist(), "target": target}


def _load_MH(folder: Path) -> dict:
    target = "RiskLevel"
    df = pd.read_csv(folder / "Maternal Health Risk Data Set.csv", encoding="utf-8-sig")
    df = _encode_non_numeric_columns(df)
    y = (df[target] - df[target].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "multiclass", "columns": df.columns.drop(target).tolist(), "target": target}


def _load_PT(folder: Path) -> dict:
    df = pd.read_csv(folder / "parkinsons_updrs.data")
    subject_ids = df["subject#"].to_numpy(dtype=np.int64)
    drop_cols = ["subject#", "motor_UPDRS", "total_UPDRS"]
    df = _encode_non_numeric_columns(df)
    y = df["total_UPDRS"].astype(np.float32).to_numpy().reshape(-1, 1)
    X = df.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    return {
        "X": X,
        "y": y,
        "tasktype": "regression",
        "columns": df.columns.drop(drop_cols).tolist(),
        "target": "total_UPDRS",
        "groups": subject_ids,
    }


def _load_BF(folder: Path) -> dict:
    df = pd.read_csv(folder / "bodyfat.csv")
    df = _encode_non_numeric_columns(df)
    drop_cols = ["BodyFat", "Density"]
    y = df["BodyFat"].astype(np.float32).to_numpy().reshape(-1, 1)
    X = df.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    return {"X": X, "y": y, "tasktype": "regression", "columns": df.columns.drop(drop_cols).tolist(), "target": "BodyFat"}
