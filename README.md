# MSTabVAE: Multi-Step Latent Conditional Variational Autoencoder for Tabular Data Synthesis

**MSTabVAE** is a **TabNet-based Conditional VAE** for generating synthetic
tabular data, aimed at imbalanced / fraud-detection settings. A TabNet encoder produces per-step decision embeddings; each is passed
through its own label-conditioned VAE bottleneck and reconstructed by a TabNet
decoder. Class-conditional rows can then be sampled directly — useful for
augmenting rare classes.

```
        x ──► TabNet Encoder ──► [d_1 , d_2 , ... , d_S]        (S = n_steps)
                                    │                  each step, conditioned on label y:
                                    ▼
                         d_i ─► (mu_i, logvar_i) ─► z_i ─► d'_i
                                    │
                                    ▼
                    [d'_1 , ... , d'_S] ──► TabNet Decoder ──► x̂
```

## Repository layout

```
MSTabVAE/
├── src/mstabvae/
│   ├── model.py            # TabNetCVAE, condition embedding, per-step VAE
│   ├── data.py             # CSV loading, label-encoding, conditional Dataset
│   ├── losses.py           # Gaussian-NLL + cross-entropy recon + KLD
│   ├── trainer.py          # training loop (KL annealing + early stopping)
│   ├── generate.py         # class-conditional synthetic-data sampling
│   └── evaluation/
│       └── tabnet_eval.py    # downstream ML utility (Recall@FPR, AUC-ROC/PR)
├── scripts/
│   ├── train.py            # train from a YAML config
│   ├── generate.py         # sample a synthetic CSV from a checkpoint
│   └── evaluate.py         # utility (± synthetic augmentation)
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

The example config targets the
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

# 3) Evaluate: ML utility (± synthetic augmentation)
python scripts/evaluate.py --config configs/baf.yaml \
    --synthetic artifacts/mstabvae_baf_synthetic.csv
```

<!-- Or use the library directly:

```python
from mstabvae import load_tabular, make_condition_dataset, TabNetCVAE, train_cvae

bundle = load_tabular("data/Base.csv", target_col="fraud_bool", sort_col="month")
train_ds = make_condition_dataset(*bundle.splits["train"], bundle.target_col)
valid_ds = make_condition_dataset(*bundle.splits["valid"], bundle.target_col)
model = TabNetCVAE(input_dim=bundle.X.shape[1], n_d=8, n_steps=3, z_dim=4,
                   condition_info={"fraud_bool": {"type": "categorical", "num_classes": 2}},
                   num_continuous=len(bundle.continuous_idx))
``` -->

## Evaluation metrics

- **Recall @ target FPR** (default 1%): recall achievable while keeping the
  false-positive rate below the threshold — the operating point that matters for
  fraud screening.
- **AUC-ROC / AUC-PR**: threshold-free ranking quality (PR is the informative
  one under heavy class imbalance).

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
