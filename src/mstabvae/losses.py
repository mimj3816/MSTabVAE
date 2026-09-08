"""MSTabVAE training objective: Gaussian NLL (continuous) + cross-entropy
(categorical) reconstruction, plus a KL divergence term on the per-step latents.
"""

import math

import torch
import torch.nn.functional as F


def cvae_loss(x_recon, x, mu_list, logvar_list,
              continuous_idx, cat_idxs, cat_dims, sigma_list, cat_logits):
    """Return ``(recon_loss, kld)`` averaged over the batch.

    - continuous columns: Gaussian negative log-likelihood with a learned,
      per-column sigma (``softplus``-parameterised).
    - categorical columns: cross-entropy over the per-column head logits.
    - KLD: standard VAE term summed over all steps, per-sample.
    """
    B = x.size(0)
    recon_loss = 0.0

    for i, idx in enumerate(continuous_idx):
        std = F.softplus(sigma_list[i]) + 1e-3
        var = std ** 2
        recon_loss += (((x[:, idx] - x_recon[:, idx]) ** 2) / (2 * var)
                       + 0.5 * torch.log(2 * math.pi * var)).sum()

    for i, idx in enumerate(cat_idxs):
        recon_loss += F.cross_entropy(cat_logits[i], x[:, idx].long(),
                                      reduction="sum")

    mu_all = torch.stack(mu_list)
    logvar_all = torch.stack(logvar_list)
    kld = -0.5 * torch.sum(1 + logvar_all - mu_all.pow(2) - logvar_all.exp()) / B
    return recon_loss / B, kld
