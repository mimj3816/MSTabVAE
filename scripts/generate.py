"""Generate synthetic data from a trained MSTabVAE checkpoint.

Usage:
    python scripts/generate.py --config configs/baf.yaml
"""

import argparse

import torch

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "src"))

from _config import ensure_parent, load_config
from mstabvae import (TabNetCVAE, class_counts_from_bundle, generate_synthetic,
                      load_tabular, set_seed)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic data with MSTabVAE")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0, help="run seed (latent sampling)")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    dcfg, mcfg, tcfg, gcfg = cfg["data"], cfg["model"], cfg["train"], cfg["generate"]
    device = args.device or tcfg["device"]
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

    bundle = load_tabular(
        dcfg.get("csv_path"), target_col=dcfg["target_col"], sort_col=dcfg.get("sort_col"),
        train_size=dcfg.get("train_size", 0.75), shuffle=dcfg.get("shuffle", False),
        seed=dcfg["seed"],
        int_cols=dcfg.get("int_cols"), header=dcfg.get("header", 0),
        splits_dir=dcfg.get("splits_dir"))

    n_classes = int(bundle.y.nunique())
    model = TabNetCVAE(
        input_dim=bundle.X.shape[1], n_d=mcfg["n_d"], n_steps=mcfg["n_steps"],
        z_dim=mcfg["z_dim"], cond_embed_dim=mcfg["cond_embed_dim"],
        condition_info={bundle.target_col: {"type": "categorical", "num_classes": n_classes}},
        num_continuous=len(bundle.continuous_idx), cat_dims=bundle.cat_dims,
    ).to(device)
    model.load_state_dict(torch.load(tcfg["ckpt_path"], map_location=device))

    n_per_class = class_counts_from_bundle(bundle)
    print(f"[generate] n_per_class={n_per_class}")

    synthetic = generate_synthetic(model, bundle, device, n_per_class)
    out = ensure_parent(gcfg["out_path"])
    synthetic.to_csv(out, index=False)
    print(f"[done] {len(synthetic)} rows -> {out}")


if __name__ == "__main__":
    main()
