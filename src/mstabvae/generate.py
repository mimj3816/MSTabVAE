"""Class-conditional synthetic-data generation with a trained MSTabVAE.

For a fixed condition we sample ``z ~ N(0, I)`` per TabNet step and decode.
The decoder emits one value per column, so categorical columns are recovered by
rounding that value to the nearest valid code and inverse label-encoding it;
integer-valued continuous columns are rounded back to integers.
"""

import pandas as pd
import torch


@torch.no_grad()
def generate_synthetic(model, data_bundle, device, n_per_class, batch_size=4096):
    """Return a synthetic DataFrame with the dataset's original columns.

    ``n_per_class`` maps an (encoded) target class -> number of rows to draw.
    Rows are drawn ``batch_size`` at a time, which fixes the order the latents
    are sampled in (matches the reference implementation).
    """
    model.eval()
    feat = data_bundle.feature_names
    target_col = data_bundle.target_col

    frames = []
    for cls, n in n_per_class.items():
        if n <= 0:
            continue
        # Every batch -- the last one included -- is a full ``k`` rows and the
        # surplus is cut off, as in the reference (keeps the GPU kernels, and
        # so the float results, identical).
        k = min(batch_size, n)
        cond = {target_col: torch.full((k,), int(cls), dtype=torch.long, device=device)}
        chunks = [model.generate_from_condition(cond, n_samples=k).cpu()
                  for _ in range(0, n, k)]
        x_gen = torch.cat(chunks)[:n].numpy()
        df = pd.DataFrame(x_gen, columns=feat)

        for col in data_bundle.int_cols:
            df[col] = df[col].round().astype(int)

        for col, le in data_bundle.label_encoders.items():
            n_classes = data_bundle.categorical_dims[col]
            df[col] = le.inverse_transform(
                df[col].round().astype(int).clip(0, n_classes - 1))

        df[target_col] = int(cls)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def class_counts_from_bundle(data_bundle):
    """Rows to draw so the training split becomes class-balanced.

    Only the minority class is generated, in the quantity that closes the gap
    to the majority class.
    """
    counts = data_bundle.splits["train"][1].value_counts()
    return {int(counts.idxmin()): int(counts.max() - counts.min())}
