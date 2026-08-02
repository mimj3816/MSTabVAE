"""Train MSTabVAE.

Usage:
    python scripts/train.py --config configs/baf.yaml
"""

import argparse

import torch

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "src"))

from _config import ensure_parent, load_config
from mstabvae import TabNetCVAE, load_tabular, make_condition_dataset, train_cvae


def main():
    ap = argparse.ArgumentParser(description="Train MSTabVAE")
    ap.add_argument("--config", default="configs/baf.yaml")
    ap.add_argument("--device", default=None, help="override train.device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    device = args.device or tcfg["device"]
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

    bundle = load_tabular(
        dcfg["csv_path"], target_col=dcfg["target_col"],
        sort_col=dcfg.get("sort_col"), train_size=dcfg["train_size"],
        shuffle=dcfg["shuffle"], seed=dcfg["seed"])
    print(f"[data] rows={len(bundle.X)}  features={len(bundle.feature_names)}  "
          f"cat={len(bundle.cat_idxs)}  cont={len(bundle.continuous_idx)}")

    train_ds = make_condition_dataset(*bundle.splits["train"], bundle.target_col)
    valid_ds = make_condition_dataset(*bundle.splits["valid"], bundle.target_col)

    n_classes = int(bundle.y.nunique())
    model = TabNetCVAE(
        input_dim=bundle.X.shape[1], n_d=mcfg["n_d"], n_steps=mcfg["n_steps"],
        z_dim=mcfg["z_dim"], cond_embed_dim=mcfg["cond_embed_dim"],
        condition_info={bundle.target_col: {"type": "categorical", "num_classes": n_classes}},
        num_continuous=len(bundle.continuous_idx),
    ).to(device)

    # Optionally reuse a frozen, pretrained TabNet encoder.
    pre = tcfg.get("pretrained_encoder")
    if pre:
        from pytorch_tabnet.tab_model import TabNetClassifier
        clf = TabNetClassifier()
        clf.load_model(pre)
        model.encoder = clf.network.tabnet.encoder.to(device)
        model.encoder.group_attention_matrix = model.encoder.group_attention_matrix.to(device)
        model.freeze_encoder()
        print(f"[encoder] loaded & frozen from {pre}")

    ckpt = ensure_parent(tcfg["ckpt_path"])
    model, _ = train_cvae(
        model, train_ds, valid_ds, bundle, device,
        num_epochs=tcfg["num_epochs"], batch_size=tcfg["batch_size"], lr=tcfg["lr"],
        max_beta=tcfg["max_beta"], kl_anneal_epochs=tcfg["kl_anneal_epochs"],
        patience=tcfg["patience"], ckpt_path=ckpt)
    print(f"[done] best checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
