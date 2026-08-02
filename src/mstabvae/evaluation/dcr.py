"""Privacy evaluation via Distance to Closest Record (DCR).

For each synthetic row we measure the L1 distance (numeric range-normalised,
categoricals label-encoded) to its closest record in the real *train* set and in
a held-out *test* set. The DCR score is the fraction of synthetic rows that are
closer to train than to test; a value near 0.5 indicates the synthetic data is
not memorising the training set.

Refactored from the original ``eval_dcr.py`` into an importable function.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def _encode(num_cols, cat_cols, real, syn, test):
    """Range-normalise numerics and shared-label-encode categoricals -> arrays."""
    num_ranges = np.array(
        [(real[c].max() - real[c].min()) or 1.0 for c in num_cols], dtype=np.float32)

    def num_arr(df):
        return df[num_cols].to_numpy(dtype=np.float32) / num_ranges

    cat_real, cat_syn, cat_test = {}, {}, {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([real[c], syn[c], test[c]]).astype(str))
        cat_real[c] = le.transform(real[c].astype(str))
        cat_syn[c] = le.transform(syn[c].astype(str))
        cat_test[c] = le.transform(test[c].astype(str))

    def cat_arr(d):
        return (np.stack([d[c] for c in cat_cols], axis=1).astype(np.float32)
                if cat_cols else np.empty((len(num_arr(real)), 0), dtype=np.float32))

    real_np = np.concatenate([num_arr(real), cat_arr(cat_real) if cat_cols else
                              np.empty((len(real), 0), np.float32)], axis=1)
    syn_np = np.concatenate([num_arr(syn), cat_arr(cat_syn) if cat_cols else
                             np.empty((len(syn), 0), np.float32)], axis=1)
    test_np = np.concatenate([num_arr(test), cat_arr(cat_test) if cat_cols else
                              np.empty((len(test), 0), np.float32)], axis=1)
    return real_np, syn_np, test_np


def compute_dcr(real_df, syn_df, test_df, num_cols, cat_cols,
                device="cpu", batch_size=100):
    """Return ``(score, dcr_to_real, dcr_to_test)``.

    ``score`` = fraction of synthetic rows closer to train than to test
    (closer to 0.5 is better).
    """
    real_np, syn_np, test_np = _encode(num_cols, cat_cols, real_df, syn_df, test_df)
    real_t = torch.tensor(real_np, device=device)
    syn_t = torch.tensor(syn_np, device=device)
    test_t = torch.tensor(test_np, device=device)

    dcrs_real, dcrs_test = [], []
    n = syn_t.shape[0]
    for i in range((n // batch_size) + 1):
        batch = syn_t[i * batch_size:(i + 1) * batch_size]
        if batch.shape[0] == 0:
            continue
        dcrs_real.append((batch[:, None] - real_t).abs().sum(dim=2).min(dim=1).values)
        dcrs_test.append((batch[:, None] - test_t).abs().sum(dim=2).min(dim=1).values)

    dcr_real = torch.cat(dcrs_real).cpu().numpy()
    dcr_test = torch.cat(dcrs_test).cpu().numpy()
    score = float((dcr_real < dcr_test).sum() / len(dcr_real))
    return score, dcr_real, dcr_test


def plot_dcr(dcr_real, dcr_test, title="DCR", savepath=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.hist(dcr_real, bins=50, alpha=0.6, label="Train (Real set)", density=True)
    plt.hist(dcr_test, bins=50, alpha=0.6, label="Holdout (Test set)", density=True)
    plt.xlabel("DCR (Distance to Closest Record)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180)
    plt.show()
