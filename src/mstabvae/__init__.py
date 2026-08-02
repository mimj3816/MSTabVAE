"""MSTabVAE: a TabNet-based Conditional VAE for synthetic tabular data.

Public API
----------
- ``TabNetCVAE``            : the model.
- ``load_tabular``          : CSV -> label-encoded splits + feature metadata.
- ``make_condition_dataset``: build the conditional Dataset.
- ``train_cvae``            : training loop.
- ``generate_synthetic``    : class-conditional sampling to a DataFrame.
- ``evaluation``            : LightGBM utility + DCR privacy metrics.
"""

from .model import TabNetCVAE
from .data import load_tabular, make_condition_dataset, TableWithConditions, collate_fn_cond
from .losses import cvae_loss
from .trainer import train_cvae
from .generate import generate_synthetic, class_counts_from_bundle

__all__ = [
    "TabNetCVAE", "load_tabular", "make_condition_dataset", "TableWithConditions",
    "collate_fn_cond", "cvae_loss", "train_cvae", "generate_synthetic",
    "class_counts_from_bundle",
]
