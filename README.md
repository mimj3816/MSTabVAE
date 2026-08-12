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
│   ├── data.py             # CSV loading, label-encoding, conditional Dataset
│   ├── losses.py           # Gaussian-NLL + cross-entropy recon + KLD
│   ├── trainer.py          # training loop
│   ├── generate.py         # class-conditional synthetic-data sampling
│   └── evaluation/
│       ├── lightgbm_eval.py  # downstream ML utility (Recall@FPR, AUC-ROC/PR)
│       └── dcr.py            # privacy: Distance to Closest Record
├── scripts/
│   ├── train.py            # train from a YAML config
│   ├── generate.py         # sample a synthetic CSV from a checkpoint
│   └── evaluate.py         # utility (± synthetic augmentation) + DCR
├── configs/baf.yaml        # example config (Bank Account Fraud dataset)
├── pyproject.toml / requirements.txt
```

## Installation

```bash
pip install -e .            # installs the `mstabvae` package + dependencies
# or: pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch, and [`pytorch-tabnet`](https://github.com/dreamquark-ai/tabnet).

## Data

Data is **not** shipped (see `.gitignore`). The example config targets the
**Bank Account Fraud (BAF)** dataset (NeurIPS 2022 / Kaggle). Download `Base.csv`
and place it at the path in `configs/baf.yaml` (`data/Base.csv` by default):

- The target column is `fraud_bool` (binary, highly imbalanced).
- Object/string columns are auto-detected and label-encoded.
- The split is chronological by `month` (`shuffle: false`).

To use another dataset, edit `configs/baf.yaml` (`csv_path`, `target_col`,
`sort_col`).

## Usage

```bash
# 1) Train
python scripts/train.py --config configs/baf.yaml

# 2) Generate a synthetic CSV from the trained checkpoint
python scripts/generate.py --config configs/baf.yaml

# 3) Evaluate: ML utility (± synthetic augmentation) and DCR privacy
python scripts/evaluate.py --config configs/baf.yaml \
    --synthetic artifacts/mstabvae_baf_synthetic.csv
```

Or use the library directly:

```python
from mstabvae import load_tabular, make_condition_dataset, TabNetCVAE, train_cvae

bundle = load_tabular("data/Base.csv", target_col="fraud_bool", sort_col="month")
train_ds = make_condition_dataset(*bundle.splits["train"], bundle.target_col)
valid_ds = make_condition_dataset(*bundle.splits["valid"], bundle.target_col)
model = TabNetCVAE(input_dim=bundle.X.shape[1], n_d=8, n_steps=3, z_dim=4,
                   condition_info={"fraud_bool": {"type": "categorical", "num_classes": 2}},
                   num_continuous=len(bundle.continuous_idx))
```

## Evaluation metrics

- **Recall @ target FPR** (default 1%): recall achievable while keeping the
  false-positive rate below the threshold — the operating point that matters for
  fraud screening.
- **AUC-ROC / AUC-PR**: threshold-free ranking quality (PR is the informative
  one under heavy class imbalance).
- **DCR (Distance to Closest Record)**: privacy check. Fraction of synthetic
  rows closer to the real *train* set than to a held-out *test* set; **closer to
  0.5 is better** (no train-set memorisation).

<!-- ## Design notes

- **Categorical reconstruction** uses a logits *slice* (`x̂[:, idx:idx+dim]`) and
  `argmax` at generation time; continuous columns use a learned per-column
  Gaussian sigma. Inputs are label-encoded, so the model operates on numeric
  columns throughout.
- **Baselines** (CTGAN, TVAE, TabDDPM, SMOTE) used in the original study are
  *not* included here — this repo packages the proposed method and its
  evaluation only. -->

## Attribution

Built on [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet)
(`TabNetEncoder` / `TabNetDecoder`). Please cite TabNet and the BAF dataset where
appropriate.
