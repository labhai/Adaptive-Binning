<h1 align="center">When, Where, and How: Adaptive Binning for Tabular Self-Supervised Learning</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=kqOWf4MAAAAJ"><strong>Daehwan Kim</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://scholar.google.com/citations?user=O-oZnIwAAAAJ"><strong>Haejun Chung</strong></a><sup>&dagger;</sup>
  &nbsp;&middot;&nbsp;
  <a href="https://scholar.google.com/citations?user=1rBh9xkAAAAJ"><strong>Ikbeom Jang</strong></a><sup>&dagger;</sup>
  <br>
  <sub><sub><sup>&dagger;</sup> Corresponding author</sub></sub>
</p>

<p align="center">
  <img alt="MICCAI 2026" src="https://img.shields.io/badge/MICCAI-2026-2f6f9f">
  <a href="https://arxiv.org/abs/2606.19827"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.19827-b31b1b"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776ab">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
</p>

<p align="center">
  <img src="./main_figure.png" width="900">
</p>

<p align="center">
  Accepted to <em>MICCAI 2026</em>.
</p>

📄 **Paper (arXiv):** [https://arxiv.org/abs/2606.19827](https://arxiv.org/abs/2606.19827)

The MICCAI 2026 proceedings version will be updated when available.

## Abstract

*Medical tabular data are ubiquitous in clinical research, but deep learning for tables remains underexplored because reliable labels often require costly expert adjudication, even though structured clinical variables are routinely available in tabular form. Self-supervised learning can leverage these unlabeled tables, and recent binning-based pretexts offer a promising inductive bias, but existing objectives fix a single global quantile discretization and apply feature-agnostic supervision. We propose Adaptive Binning, a training-adaptive discretization pretext for tabular SSL that couples discretization to learning through a feature-wise coarse-to-fine curriculum. Motivated by the spectral bias of neural networks and the principles of curriculum learning, our method progressively refines discretization per feature upon plateau detection and selects representation-aware splits to jointly improve value-space concentration and representation-space coherence. A heterogeneity-aware objective unifies categorical reconstruction with ordinal supervision for numerical features, and experiments on public medical tabular datasets under unified evaluation protocols show consistent gains for linear probing and fine-tuning without dataset-specific discretization tuning. We further introduce a medical tabular SSL benchmark with standardized protocols to support reproducible progress in this underexplored domain.*

## 📁 Repository Structure

| Component | Role |
| --- | --- |
| `adaptive_binning/run.py` | CLI entry point for pretraining, linear probing, and optional fine-tuning. |
| `adaptive_binning/data/` | Dataset registry, loaders, preprocessing, feature typing, and splits. |
| `adaptive_binning/method/` | Adaptive Binning core: feature-wise adaptation, FPT-style plateau triggering, DIGS split selection, and HORD supervision. |
| `adaptive_binning/models/` | MLP encoder/decoder and downstream prediction head. |
| `adaptive_binning/training/` | Training loops for SSL pretraining and downstream evaluation. |
| `configs/adaptive_binning.yaml` | Default configuration shared across datasets. |
| `data/raw/` | Bundled public data files and placeholders for manually downloaded datasets. |

## ⚙️ Installation

```bash
git clone https://github.com/labhai/Adaptive-Binning.git
cd Adaptive-Binning

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🗂️ Data

Redistributable files are bundled under `data/raw`. `ESR` and `BF` are provided as folders only and require manual download from the original source.

| Key | Paper label | Dataset | Task | Repository status | Data file(s) |
| --- | --- | --- | --- | --- | --- |
| `ILPD` | ILPD | [Indian Liver Patient Dataset][ilpd] | Binary classification | ✓ Bundled | `Indian Liver Patient Dataset (ILPD).csv` |
| `HF` | HFC | [Heart Failure Clinical Records][hf] | Binary classification | ✓ Bundled | `heart_failure.csv` |
| `Cardio` | CTG | [Cardiotocography][cardio] | Nominal multiclass classification | ✓ Bundled | `CTG.xls` |
| `EOL` | EOL | [Estimation of Obesity Levels][eol] | Ordinal multiclass classification | ✓ Bundled | `ObesityDataSet_raw_and_data_sinthetic.csv` |
| `MH` | MHR | [Maternal Health Risk][mh] | Ordinal multiclass classification | ✓ Bundled | `Maternal Health Risk Data Set.csv` |
| `PT` | PT | [Parkinsons Telemonitoring][pt] | Regression | ✓ Bundled | `parkinsons_updrs.data`, `parkinsons_updrs.names` |
| `ESR` | ESR | [Epileptic Seizure Recognition][esr] | Nominal multiclass classification | Download manually | `Epileptic Seizure Recognition.csv` |
| `BF` | BFP | [Body Fat Prediction][bf] | Regression | Download manually | `bodyfat.csv` |

Expected `data/raw` layout:

```text
data/raw/
├── ILPD/
│   └── Indian Liver Patient Dataset (ILPD).csv
├── HF/
│   └── heart_failure.csv
├── Cardio/
│   └── CTG.xls
├── EOL/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
├── MH/
│   └── Maternal Health Risk Data Set.csv
├── PT/
│   ├── parkinsons_updrs.data
│   └── parkinsons_updrs.names
├── ESR/
│   └── Epileptic Seizure Recognition.csv
└── BF/
    └── bodyfat.csv
```

[ilpd]: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
[hf]: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
[cardio]: https://archive.ics.uci.edu/dataset/193/cardiotocography
[eol]: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
[mh]: https://archive.ics.uci.edu/dataset/863/maternal+health+risk
[pt]: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
[esr]: https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
[bf]: https://hbiostat.org/data/

## 🚀 Usage

**Default**

```bash
python -m adaptive_binning.run \
  --config configs/adaptive_binning.yaml \
  --dataset HF \
  --output-dir outputs/HF
```

**Quick CPU check**

```bash
python -m adaptive_binning.run \
  --config configs/adaptive_binning.yaml \
  --dataset HF \
  --device cpu \
  --pretrain-epochs 2 \
  --eval-epochs 2 \
  --output-dir outputs/HF_quick
```

**Linear probing + fine-tuning**

```bash
python -m adaptive_binning.run \
  --config configs/adaptive_binning.yaml \
  --dataset HF \
  --run-finetune \
  --output-dir outputs/HF_finetune
```

## 🔁 Seed Control

| Argument | Config key | Controls |
| --- | --- | --- |
| `--seed` | `seed` | Train/validation/test split, model initialization, discretizer fitting, and SSL pretraining order. |
| `--eval-seed` | `eval_seed` | Downstream head initialization and downstream training order. |

```bash
python -m adaptive_binning.run \
  --config configs/adaptive_binning.yaml \
  --dataset HF \
  --seed 123456 \
  --eval-seed 0 \
  --output-dir outputs/HF_seed123456_eval0
```

## 🧾 Configuration

Main config: `configs/adaptive_binning.yaml`

| Config key | Paper notation | Default | Role |
| --- | --- | --- | --- |
| `pretrain_epochs` | pretraining epochs | `1000` | SSL pretraining duration. |
| `eval_epochs` | linear probing / fine-tuning epochs | `100` | Downstream evaluation duration. |
| `pretrain_lr` | pretraining lr | `1e-4` | Learning rate for SSL pretraining. |
| `linear_lr` | linear probing lr | `1e-2` | Learning rate for the frozen-encoder linear head. |
| `finetune_lr` | fine-tuning lr | `1e-3` | Learning rate for end-to-end fine-tuning. |
| `initial_bins` | `T_init` | `2` | Initial numerical bin count. |
| `max_bins` | `T_max` | `200` | Maximum numerical bin count. |
| `patience` | FPT patience | `5` | Number of non-improving epochs before feature-wise refinement. |
| `digs_threshold` | DIGS threshold `τ` | `1e-4` | Minimum split score for representation-aware bin refinement. |
| `loss_weights.sord` | `w_SORD` | `10.0` | HORD soft ordinal reconstruction weight. |
| `loss_weights.mse` | `w_MSE` | `0.1` | HORD mean regularization weight. |
| `loss_weights.var` | `w_Var` | `0.001` | HORD variance regularization weight. |

Dataset-specific batch size, MLP width, and MLP depth: `adaptive_binning/data/registry.py`

```bash
python -m adaptive_binning.run \
  --dataset Cardio \
  --device cuda:0 \
  --pretrain-epochs 1000 \
  --eval-epochs 100 \
  --output-dir outputs/Cardio
```

Dataset shortcuts: `HFC` → `HF`, `CTG` → `Cardio`, `MHR` → `MH`, `BFP` → `BF`.

## ➕ Adding a New Dataset

1. Place raw files under `data/raw/<DatasetKey>/`.

```text
data/raw/MyDataset/
└── my_dataset.csv
```

2. Add an entry to `DATASETS` in `adaptive_binning/data/registry.py`.

```python
"MyDataset": {
    "full_name": "My Dataset",
    "tasktype": "multiclass",
    "batch_size": 128,
    "mlpwidth": 512,
    "mlpdepth": 2,
    "files": ["my_dataset.csv"],
    "redistributed": False,
    "license": "See source",
    "source": "https://example.org/my-dataset",
    "doi": "",
},
```

3. Add a loader named `_load_<DatasetKey>` in `adaptive_binning/data/loaders.py`.

```python
def _load_MyDataset(folder: Path) -> dict:
    target = "label"
    df = pd.read_csv(folder / "my_dataset.csv")
    df = _encode_non_numeric_columns(df)
    y = (df[target] - df[target].min()).astype(np.int64).to_numpy().reshape(-1, 1)
    X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    return {
        "X": X,
        "y": y,
        "tasktype": "multiclass",
        "columns": df.columns.drop(target).tolist(),
        "target": target,
    }
```

Loader contract: `X`, `y`, `tasktype`, `columns`, `target`; optional `groups` for grouped splits. `tasktype` must be `binclass`, `multiclass`, or `regression`.

```bash
python -m adaptive_binning.run \
  --config configs/adaptive_binning.yaml \
  --dataset MyDataset \
  --output-dir outputs/MyDataset
```

## 📚 BibTeX

```bibtex
@misc{kim2026whenwherehowadaptive,
      title={When, Where, and How: Adaptive Binning for Tabular Self-Supervised Learning},
      author={Daehwan Kim and Haejun Chung and Ikbeom Jang},
      year={2026},
      eprint={2606.19827},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.19827},
}
```

## 📬 Contact

For questions about the paper or code, please contact **Daehwan Kim**.

📧 [officialhwan@hanyang.ac.kr](mailto:officialhwan@hanyang.ac.kr)
