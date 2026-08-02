"""Evaluate a real / synthetic dataset.

- ML utility: train LightGBM on real (and optionally real + synthetic) and
  report Recall@FPR / AUC-ROC / AUC-PR on the *real* test split.
- Privacy: DCR score between the synthetic set and the real train/test splits.

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
from mstabvae.evaluation import compute_dcr, train_and_evaluate_lightgbm


def _encode_like(df, bundle):
    """Label-encode a synthetic df's categoricals with the real encoders."""
    out = df.copy()
    for col, le in bundle.label_encoders.items():
        out[col] = le.transform(out[col].astype(str))
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate MSTabVAE (utility + privacy)")
    ap.add_argument("--config", default="configs/baf.yaml")
    ap.add_argument("--synthetic", default=None, help="synthetic CSV to evaluate")
    ap.add_argument("--target_fpr", type=float, default=0.01)
    ap.add_argument("--no_plot", action="store_true")
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

    print("\n===== ML utility: LightGBM trained on REAL =====")
    train_and_evaluate_lightgbm(X_tr, X_va, X_te, y_tr, y_va, y_te,
                                target_fpr=args.target_fpr, plot=not args.no_plot)

    if args.synthetic:
        syn = pd.read_csv(args.synthetic)
        syn_enc = _encode_like(syn, bundle)
        Xs, ys = syn_enc[bundle.feature_names], syn_enc[tcol]

        print("\n===== ML utility: LightGBM trained on REAL + SYNTHETIC =====")
        X_aug = pd.concat([X_tr, Xs], ignore_index=True)
        y_aug = pd.concat([y_tr, ys], ignore_index=True)
        train_and_evaluate_lightgbm(X_aug, X_va, X_te, y_aug, y_va, y_te,
                                    target_fpr=args.target_fpr, plot=not args.no_plot)

        print("\n===== Privacy: DCR (closer to 0.5 is better) =====")
        num_cols = [bundle.feature_names[i] for i in bundle.continuous_idx]
        cat_cols = [bundle.feature_names[i] for i in bundle.cat_idxs]
        score, _, _ = compute_dcr(
            real_df=X_tr, syn_df=Xs, test_df=X_te,
            num_cols=num_cols, cat_cols=cat_cols)
        print(f"DCR score = {score:.4f}")


if __name__ == "__main__":
    main()
