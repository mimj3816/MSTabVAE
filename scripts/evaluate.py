"""Evaluate a real / synthetic dataset.

- ML utility: train a TabNet classifier on real (and optionally real +
  synthetic) and report Recall@FPR / AUC-ROC / AUC-PR on the *real* test split.

Usage:
    python scripts/evaluate.py --config configs/baf.yaml \
        --synthetic artifacts/mstabvae_baf_synthetic.csv
"""

import argparse

import pandas as pd

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "src"))

from _config import load_config
from mstabvae import load_tabular
from mstabvae.evaluation import train_and_evaluate_tabnet


def _encode_like(df, bundle):
    """Label-encode a synthetic df's categoricals with the real encoders."""
    out = df.copy()
    for col, le in bundle.label_encoders.items():
        out[col] = le.transform(out[col].astype(str))
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate MSTabVAE (ML utility)")
    ap.add_argument("--config", default="configs/baf.yaml")
    ap.add_argument("--synthetic", default=None, help="synthetic CSV to evaluate")
    ap.add_argument("--target_fpr", type=float, default=0.01)
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--device", default="cuda", help="TabNet device: cuda | cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    bundle = load_tabular(
        dcfg["csv_path"], target_col=dcfg["target_col"], sort_col=dcfg.get("sort_col"),
        train_size=dcfg["train_size"], shuffle=dcfg["shuffle"], seed=dcfg["seed"])
    tcol = bundle.target_col
    X_tr, y_tr = bundle.splits["train"]
    X_va, y_va = bundle.splits["valid"]
    X_te, y_te = bundle.splits["test"]

    print("\n===== ML utility: TabNet trained on REAL =====")
    train_and_evaluate_tabnet(X_tr, X_va, X_te, y_tr, y_va, y_te,
                              cat_idxs=bundle.cat_idxs, cat_dims=bundle.cat_dims,
                              target_fpr=args.target_fpr, device_name=args.device,
                              plot=not args.no_plot)

    if args.synthetic:
        syn = pd.read_csv(args.synthetic)
        syn_enc = _encode_like(syn, bundle)
        Xs, ys = syn_enc[bundle.feature_names], syn_enc[tcol]

        print("\n===== ML utility: TabNet trained on REAL + SYNTHETIC =====")
        # Augment with the synthetic *minority* class only, then shuffle: real
        # train is dominated by the majority class, so this lands near 1:1.
        minority = int(y_tr.value_counts().idxmin())
        Xs, ys = Xs[ys == minority], ys[ys == minority]
        X_aug = pd.concat([X_tr, Xs], ignore_index=True)
        y_aug = pd.concat([y_tr, ys], ignore_index=True)
        X_aug = X_aug.sample(frac=1, random_state=dcfg["seed"])
        y_aug = y_aug.loc[X_aug.index]
        print(f"[augment] +{len(Xs)} synthetic class-{minority} rows "
              f"-> {y_aug.value_counts().to_dict()}")
        train_and_evaluate_tabnet(X_aug, X_va, X_te, y_aug, y_va, y_te,
                                  cat_idxs=bundle.cat_idxs, cat_dims=bundle.cat_dims,
                                  target_fpr=args.target_fpr, device_name=args.device,
                                  plot=not args.no_plot)


if __name__ == "__main__":
    main()
