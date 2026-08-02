"""Training loop for MSTabVAE (KL annealing + early stopping)."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import collate_fn_cond
from .losses import cvae_loss


def train_cvae(model, train_ds, valid_ds, data_bundle, device,
               num_epochs=100, batch_size=1024, lr=1e-3,
               max_beta=1.0, kl_anneal_epochs=100, patience=5,
               ckpt_path="mstabvae.pth"):
    """Fit ``model`` and checkpoint the best (lowest validation loss) weights."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_cond)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn_cond)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    cont, cat_i, cat_d = (data_bundle.continuous_idx,
                          data_bundle.cat_idxs, data_bundle.cat_dims)

    best_loss, counter = float("inf"), 0
    history = []
    for epoch in range(num_epochs):
        beta = min(max_beta, epoch / max(kl_anneal_epochs, 1))

        model.train()
        tr_loss = 0.0
        for x_batch, cond_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x_batch = x_batch.to(device)
            cond_batch = {k: v.to(device) for k, v in cond_batch.items()}
            x_recon, mu_list, logvar_list, _, _ = model(x_batch, cond_batch)
            recon, kld = cvae_loss(x_recon, x_batch, mu_list, logvar_list,
                                   cont, cat_i, cat_d, model.sigma_list)
            loss = recon + beta * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x_batch.size(0)
        tr_loss /= len(train_ds)

        val_loss = _evaluate(model, valid_loader, valid_ds, cont, cat_i, cat_d,
                             beta, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "beta": beta})
        print(f"Epoch {epoch}: train={tr_loss:.4f}  val={val_loss:.4f}  beta={beta:.3f}")

        if val_loss < best_loss:
            best_loss, counter = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val={best_loss:.4f}).")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history


@torch.no_grad()
def _evaluate(model, loader, ds, cont, cat_i, cat_d, beta, device):
    model.eval()
    total = 0.0
    for x_batch, cond_batch in loader:
        x_batch = x_batch.to(device)
        cond_batch = {k: v.to(device) for k, v in cond_batch.items()}
        x_recon, mu_list, logvar_list, _, _ = model(x_batch, cond_batch)
        recon, kld = cvae_loss(x_recon, x_batch, mu_list, logvar_list,
                               cont, cat_i, cat_d, model.sigma_list)
        total += (recon + beta * kld).item() * x_batch.size(0)
    return total / len(ds)
