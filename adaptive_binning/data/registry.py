from __future__ import annotations

DATASETS = {
    "ILPD": {
        "full_name": "Indian Liver Patient Dataset",
        "tasktype": "binclass",
        "batch_size": 64,
        "mlpwidth": 512,
        "mlpdepth": 1,
        "files": ["Indian Liver Patient Dataset (ILPD).csv"],
        "redistributed": True,
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
        "doi": "10.24432/C5D02C",
    },
    "HF": {
        "paper_abbr": "HFC",
        "full_name": "Heart Failure Clinical Records",
        "tasktype": "binclass",
        "batch_size": 64,
        "mlpwidth": 512,
        "mlpdepth": 5,
        "files": ["heart_failure.csv"],
        "redistributed": True,
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records",
        "doi": "10.24432/C5Z89R",
    },
    "Cardio": {
        "paper_abbr": "CTG",
        "full_name": "Cardiotocography",
        "tasktype": "multiclass",
        "batch_size": 128,
        "mlpwidth": 256,
        "mlpdepth": 2,
        "files": ["CTG.xls"],
        "redistributed": True,
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/193/cardiotocography",
        "doi": "10.24432/C51S4N",
    },
    "ESR": {
        "full_name": "Epileptic Seizure Recognition",
        "tasktype": "multiclass",
        "batch_size": 256,
        "mlpwidth": 512,
        "mlpdepth": 4,
        "files": ["Epileptic Seizure Recognition.csv"],
        "redistributed": False,
        "license": "See source",
        "source": "https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition",
        "doi": "10.24432/C5G308",
    },
    "EOL": {
        "full_name": "Estimation of Obesity Levels",
        "tasktype": "multiclass",
        "batch_size": 128,
        "mlpwidth": 128,
        "mlpdepth": 2,
        "files": ["ObesityDataSet_raw_and_data_sinthetic.csv"],
        "redistributed": True,
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition",
        "doi": "10.24432/C5H31Z",
    },
    "MH": {
        "paper_abbr": "MHR",
        "full_name": "Maternal Health Risk",
        "tasktype": "multiclass",
        "batch_size": 64,
        "mlpwidth": 1024,
        "mlpdepth": 4,
        "files": ["Maternal Health Risk Data Set.csv"],
        "redistributed": True,
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/863/maternal+health+risk",
        "doi": "10.24432/C5DP5D",
    },
    "PT": {
        "full_name": "Parkinsons Telemonitoring",
        "tasktype": "regression",
        "batch_size": 128,
        "mlpwidth": 1024,
        "mlpdepth": 2,
        "files": ["parkinsons_updrs.data", "parkinsons_updrs.names"],
        "redistributed": True,
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring",
        "doi": "10.24432/C5ZS3N",
    },
    "BF": {
        "paper_abbr": "BFP",
        "full_name": "Body Fat Prediction",
        "tasktype": "regression",
        "batch_size": 64,
        "mlpwidth": 512,
        "mlpdepth": 5,
        "files": ["bodyfat.csv"],
        "redistributed": False,
        "license": "See source",
        "source": "https://hbiostat.org/data/",
        "doi": "",
    },
}


ALIASES = {
    "HFC": "HF",
    "CTG": "Cardio",
    "MHR": "MH",
    "BFP": "BF",
}


def canonical_name(name: str) -> str:
    name = str(name)
    return ALIASES.get(name, name)


def dataset_config(name: str) -> dict:
    name = canonical_name(name)
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Available: {', '.join(DATASETS)}")
    cfg = dict(DATASETS[name])
    cfg["name"] = name
    return cfg
