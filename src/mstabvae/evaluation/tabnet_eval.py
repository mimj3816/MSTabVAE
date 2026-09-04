"""Downstream ML-utility evaluation with a TabNet classifier.

Trains a TabNetClassifier on (real or synthetic-augmented) data and reports
fraud-detection style metrics on the real test set: Recall@target-FPR, AUC-ROC,
AUC-PR, plus the ROC / PR curves.
"""

import matplotlib.pyplot as plt
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def train_and_evaluate_tabnet(X_train, X_valid, X_test, y_train, y_valid, y_test,
                              cat_idxs=None, cat_dims=None, target_fpr=0.01,
                              device_name="cuda", seed=42, plot=True):
    """Train a TabNetClassifier and evaluate on the test set.

    Returns ``(model, recall_at_fpr, cm, metrics_dict)``.
    """
    model = TabNetClassifier(seed=seed, cat_idxs=cat_idxs or [],
                             cat_dims=cat_dims or [], device_name=device_name)
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int),
              eval_set=[(np.asarray(X_valid, dtype=float),
                         np.asarray(y_valid, dtype=int))],
              eval_metric=["auc"])

    y_prob = model.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    recall_at_fpr = tpr[fpr <= target_fpr].max() if np.any(fpr <= target_fpr) else 0.0

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "recall_at_fpr": float(recall_at_fpr),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),
    }
    print(cm)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if plot:
        plot_roc_pr(y_test, y_prob, metrics["auc_roc"], metrics["auc_pr"])
    return model, recall_at_fpr, cm, metrics


def plot_roc_pr(y_test, y_prob, auc_roc=None, auc_pr=None, savepath=None):
    """(Re)draw ROC and PR curves from labels + predicted probabilities."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    if auc_roc is None:
        auc_roc = roc_auc_score(y_test, y_prob)
    if auc_pr is None:
        auc_pr = average_precision_score(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(fpr, tpr, lw=1.5, label=f"ROC (AUC = {auc_roc:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.7)

    axes[1].plot(rec, prec, color="darkorange", lw=1.5, label=f"PR Curve (AP = {auc_pr:.4f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(loc="lower left"); axes[1].grid(alpha=0.7)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180)
    plt.show()
    return fig
