"""MSTabVAE model: a TabNet-based Conditional VAE for tabular synthesis.

Architecture
------------
1. A (optionally pretrained) TabNet ``encoder`` maps a row ``x`` to a list of
   per-step decision embeddings ``d_1 .. d_S`` (``S = n_steps``).
2. Each ``d_i`` is pushed through its own VAE bottleneck, conditioned on the
   label embedding: ``d_i -> (mu_i, logvar_i) -> z_i -> d'_i``.
3. A TabNet ``decoder`` reconstructs the row from ``d'_1 .. d'_S``.

Conditional generation samples ``z_i ~ N(0, I)`` per step together with a
condition vector and decodes, so class-conditional rows can be drawn directly.
"""

import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNetEncoder, TabNetDecoder


class ConditionEmbedding(nn.Module):
    """Embeds a dict of condition columns (categorical / continuous) and
    concatenates them into a single condition vector."""

    def __init__(self, condition_info, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedders = nn.ModuleDict()
        for col, info in condition_info.items():
            if info["type"] == "categorical":
                self.embedders[col] = nn.Embedding(info["num_classes"], embed_dim)
            elif info["type"] == "continuous":
                self.embedders[col] = nn.Linear(1, embed_dim)
            else:
                raise ValueError(f"Unknown condition type {info['type']}")

    def forward(self, cond_dict):
        cond_embeds = []
        for col, val in cond_dict.items():
            embedder = self.embedders[col]
            if isinstance(embedder, nn.Embedding):
                cond_embeds.append(embedder(val))
            else:
                cond_embeds.append(embedder(val.unsqueeze(-1).float()))
        return torch.cat(cond_embeds, dim=-1)


class DToZEncoder(nn.Module):
    """VAE encoder for one TabNet step: (d_i, cond) -> (mu_i, logvar_i)."""

    def __init__(self, d_dim, z_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.mu_mlp = nn.Sequential(
            nn.Linear(d_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.logvar_mlp = nn.Sequential(
            nn.Linear(d_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, d, cond_vec):
        h = torch.cat([d, cond_vec], dim=-1)
        return self.mu_mlp(h), self.logvar_mlp(h)


class ZToDDecoder(nn.Module):
    """VAE decoder for one TabNet step: (z_i, cond) -> d'_i."""

    def __init__(self, z_dim, d_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_dim), nn.ReLU(),
        )

    def forward(self, z, cond_vec):
        return self.mlp(torch.cat([z, cond_vec], dim=-1))


class TabNetCVAE(nn.Module):
    def __init__(self, input_dim, n_d, n_steps, z_dim, condition_info,
                 num_continuous=0, cond_embed_dim=8, cat_dims=None):
        super().__init__()
        self.n_steps = n_steps
        self.z_dim = z_dim

        self.encoder = TabNetEncoder(
            input_dim=input_dim, output_dim=n_d, n_a=n_d, n_d=n_d,
            n_steps=n_steps, gamma=1.5, n_independent=2, n_shared=2,
            virtual_batch_size=128, momentum=0.02, mask_type="sparsemax",
        )
        # pytorch-tabnet stores this as a plain tensor attribute, so
        # nn.Module.to(device) would leave it on the CPU; re-register as buffer.
        gam = self.encoder.group_attention_matrix
        del self.encoder.group_attention_matrix
        self.encoder.register_buffer("group_attention_matrix", gam)
        self.decoder = TabNetDecoder(
            input_dim=input_dim, n_d=n_d, n_steps=n_steps,
            n_independent=2, n_shared=2, virtual_batch_size=128, momentum=0.02,
        )
        self.cond_embedding = ConditionEmbedding(condition_info, embed_dim=cond_embed_dim)
        cond_dim = len(condition_info) * cond_embed_dim
        self.d_to_z_encoders = nn.ModuleList(
            [DToZEncoder(n_d, z_dim, cond_dim) for _ in range(n_steps)])
        self.z_to_d = nn.ModuleList(
            [ZToDDecoder(z_dim, n_d, cond_dim) for _ in range(n_steps)])
        # One classification head per categorical column, applied to each step's
        # d'_i; the per-step logits are averaged before the loss sees them.
        self.cat_dims = cat_dims or []
        self.cat_heads = nn.ModuleList([nn.Linear(n_d, dim) for dim in self.cat_dims])
        self.sigma_list = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(num_continuous)])

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x, cond_dict):
        steps_output, _ = self.encoder(x)
        cond_vec = self.cond_embedding(cond_dict)
        mu_list, logvar_list, d_prime_list, step_cat_logits = [], [], [], []
        for i, d in enumerate(steps_output):
            mu, logvar = self.d_to_z_encoders[i](d, cond_vec)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            d_prime = self.z_to_d[i](z, cond_vec)
            mu_list.append(mu)
            logvar_list.append(logvar)
            d_prime_list.append(d_prime)
            step_cat_logits.append([head(d_prime) for head in self.cat_heads])
        cat_logits = [torch.stack([step[j] for step in step_cat_logits]).mean(dim=0)
                      for j in range(len(self.cat_heads))]
        x_recon = self.decoder(d_prime_list)
        return x_recon, mu_list, logvar_list, cat_logits

    @torch.no_grad()
    def generate_from_condition(self, cond_dict, n_samples=100):
        """Draw ``n_samples`` rows for a fixed condition (z ~ N(0, I) per step)."""
        cond_vec = self.cond_embedding(cond_dict)
        steps_outputs = []
        for step in range(self.n_steps):
            z = torch.randn(n_samples, self.z_dim, device=cond_vec.device)
            steps_outputs.append(self.z_to_d[step](z, cond_vec))
        return self.decoder(steps_outputs)
