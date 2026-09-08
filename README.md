# MSTabVAE

**MSTabVAE** is a **TabNet-based Conditional VAE** for generating synthetic
tabular data, aimed at imbalanced / fraud-detection settings. A TabNet encoder produces per-step decision embeddings; each is passed
through its own label-conditioned VAE bottleneck and reconstructed by a TabNet
decoder. Class-conditional rows can then be sampled directly — useful for
augmenting rare classes.

<img width="5137" height="909" alt="mstabvae_fin2" src="https://github.com/user-attachments/assets/87dc8b11-b7fa-46e0-b78e-18b6750bb321" />



## Repository layout

```
MSTabVAE/
├── src/mstabvae/
│   ├── model.py            # TabNetCVAE, condition embedding, per-step VAE
│   ├── data.py             # table loading, label-encoding, conditional Dataset
│   ├── losses.py           # Gaussian-NLL + cross-entropy recon + KLD
│   ├── trainer.py          # training loop
│   ├── generate.py         # class-conditional synthetic-data sampling
│   └── evaluation/
│       └── tabnet_eval.py    # downstream ML utility (Recall@FPR, AUC-ROC/PR)
├── scripts/
│   ├── train.py            # train from a YAML config
│   ├── generate.py         # sample a synthetic CSV from a checkpoint
│   └── evaluate.py         # utility (± synthetic augmentation)
├── configs/default.yaml    # example config (UCI default of credit card clients)
├── pyproject.toml / requirements.txt
```

## Installation

```bash
pip install -e .            # installs the `mstabvae` package + dependencies
# or: pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch, and [`pytorch-tabnet`](https://github.com/dreamquark-ai/tabnet).

## Data

The example config targets the UCI **default of credit card clients** dataset
(30,000 rows, 24 features, 22.1% positives). It reads a pre-saved split so the
train/valid/test partition is fixed:

```
data/default/
├── X_train.csv  y_train.csv     # 22,500 rows
├── X_valid.csv  y_valid.csv     #  3,750 rows
└── X_test.csv   y_test.csv      #  3,750 rows
```

Point `data.splits_dir` in `configs/default.yaml` at that directory. The target
column is `default payment next month`.

Feature handling is derived from the data, not configured:

- **Categorical** = object columns plus any numeric column with fewer than 10
  distinct values. They are label-encoded, and the encoders are fit on the full
  table (train + valid + test).
- **Continuous** = everything else. `int_cols` names the continuous columns that
  are integer valued so generation can round them back; `auto` takes every
  integer-dtype column.

To use another dataset, write a new config. A single CSV is split in-process
with `csv_path` + `train_size`/`shuffle` (and `sort_col` for a chronological
split) instead of `splits_dir`; valid and test each take half of the remainder
and are never shuffled.

## Usage

```bash
# 1) Train
python scripts/train.py --config configs/default.yaml

# 2) Generate a synthetic CSV from the trained checkpoint
python scripts/generate.py --config configs/default.yaml

# 3) Evaluate: ML utility (± synthetic augmentation)
python scripts/evaluate.py --config configs/default.yaml \
    --synthetic artifacts/mstabvae_default_synthetic.csv
```

`configs/default.yaml` is the default for `--config`, so the flag can be
dropped. `--seed` (default 0) seeds python/numpy/torch for a reproducible run.

Step 2 generates **minority-class rows only**, as many as it takes to balance
the training split (`n_majority - n_minority`). Step 3 trains a TabNet
classifier twice — on the real training split, then on that split concatenated
with the synthetic rows and shuffled — and reports both on the real test
split.

<!-- Or use the library directly:

```python
from mstabvae import load_tabular, make_condition_dataset, TabNetCVAE, train_cvae

target = "default payment next month"
bundle = load_tabular(splits_dir="data/default", target_col=target, int_cols="auto")
train_ds = make_condition_dataset(*bundle.splits["train"], bundle.target_col)
valid_ds = make_condition_dataset(*bundle.splits["valid"], bundle.target_col)
model = TabNetCVAE(input_dim=bundle.X.shape[1], n_d=8, n_steps=3, z_dim=4,
                   condition_info={target: {"type": "categorical", "num_classes": 2}},
                   num_continuous=len(bundle.continuous_idx), cat_dims=bundle.cat_dims)
``` -->

## Evaluation metrics

- **Recall @ target FPR** (default 1%): recall achievable while keeping the
  false-positive rate below the threshold — the operating point that matters for
  fraud screening.
- **AUC-ROC / AUC-PR**: threshold-free ranking quality (PR is the informative
  one under heavy class imbalance).

Both runs use the same classifier seed, so the difference between them is the
effect of the synthetic data alone.

<!-- ## Design notes

- **Categorical reconstruction** uses one `Linear(n_d, dim)` head per
  categorical column, applied to every step's `d'_i` and averaged before the
  cross-entropy. Continuous columns use a learned per-column Gaussian sigma.
  Inputs are label-encoded, so the model operates on numeric columns throughout.
- **Generation** decodes `x̂` and rounds: `int_cols` back to integers, and
  categorical columns to the nearest valid code before inverse label-encoding.
- **Baselines** (CTGAN, TVAE, TabDDPM, SMOTE) used in the original study are
  *not* included here — this repo packages the proposed method and its
  evaluation only. -->

## Attribution

Built on [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet)
(`TabNetEncoder` / `TabNetDecoder`). Please cite TabNet and the dataset you use
where appropriate.
