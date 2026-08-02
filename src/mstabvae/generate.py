"""Class-conditional synthetic-data generation with a trained MSTabVAE.

For each condition (target class) we sample ``z ~ N(0, I)`` per TabNet step and
decode. Continuous columns are taken directly; categorical columns are recovered
by ``argmax`` over their reconstructed-logits slice (the same slice convention
used by the training loss) and then inverse label-encoded back to their original
category strings.
"""

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def generate_synthetic(model, data_bundle, device, n_per_class):
    """Return a synthetic DataFrame with the dataset's original columns.

    ``n_per_class`` maps an (encoded) target class -> number of rows to draw.
    """
    model.eval()
    feat = data_bundle.feature_names
    target_col = data_bundle.target_col
    cat_span = dict(zip(data_bundle.cat_idxs, data_bundle.cat_dims))

    frames = []
    for cls, n in n_per_class.items():
        if n <= 0:
            continue
        cond = {target_col: torch.full((n,), int(cls), dtype=torch.long, device=device)}
        x_recon = model.generate_from_condition(cond, n_samples=n).cpu().numpy()

        df = pd.DataFrame({name: x_recon[:, j] for j, name in enumerate(feat)})
        for idx, dim in cat_span.items():
            name = feat[idx]
            labels = x_recon[:, idx:idx + dim].argmax(axis=1)
            le = data_bundle.label_encoders[name]
            labels = np.clip(labels, 0, len(le.classes_) - 1)
            df[name] = le.inverse_transform(labels)
        df[target_col] = int(cls)
        frames.append(df)

    synthetic = pd.concat(frames, ignore_index=True)
    return synthetic.sample(frac=1.0, random_state=42).reset_index(drop=True)


def class_counts_from_bundle(data_bundle, balance=False, total=None):
    """Derive ``n_per_class`` from the training split.

    - default: reproduce the real training class distribution.
    - ``balance=True``: equal number of rows per class (``total`` split evenly,
      else the majority-class count for every class).
    """
    y_train = data_bundle.splits["train"][1]
    counts = y_train.value_counts().to_dict()
    if not balance:
        return {int(k): int(v) for k, v in counts.items()}
    per = (total // len(counts)) if total else max(counts.values())
    return {int(k): int(per) for k in counts}
