import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDAE(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        corruption_ratio: float,
    ) -> None:
        super(CDAE, self).__init__()

        self.num_users = n_users
        self.num_items = n_items
        self.num_hidden_units = embedding_dim
        self.corruption_ratio = corruption_ratio

        # CDAE consists of user embedding, encoder, decoder
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.encoder = nn.Linear(n_items, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, n_items)
        self.corrupt = nn.Dropout(p=corruption_ratio)

    def forward(self, users: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        # Apply corruption
        matrix = self.corrupt(matrix)
        encoder = F.tanh(self.encoder(matrix) + self.user_embedding(users))
        return self.decoder(encoder)


class MLP(nn.Module):
    def __init__(self, mlp_dims):
        super().__init__()

        model = []
        for in_dim, out_dim in zip(mlp_dims[:-1], mlp_dims[1:]):
            model.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                ]
            )
        model.pop()
        self.mlp = nn.Sequential(*model)

    def forward(self, x):
        return self.mlp(x)


class MultiVAE(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, dropout=0.5, anneal=0.2):
        super().__init__()
        self.encoder = MLP(encoder_dims)
        self.decoder = MLP(decoder_dims)
        self.dropout = nn.Dropout(dropout)
        self.anneal = anneal

    def forward(self, items):
        mu_z, log_std_z = self.encoder(self.dropout(items)).chunk(2, dim=1)
        z = torch.randn_like(mu_z) * log_std_z.exp() + mu_z
        # mu_x, log_std_x = self.decoder(z).chunk(2, dim=1)
        items_recon = self.decoder(z)

        return items_recon, mu_z, log_std_z

    def loss(self, items):
        items_recon, mu_z, log_std_z = self(items)
        recon_loss = F.binary_cross_entropy_with_logits(items_recon, items)
        kl_loss = -log_std_z - 0.5 + (torch.exp(2 * log_std_z) + mu_z**2) * 0.5
        kl_loss = kl_loss.sum(1).mean()

        return recon_loss + self.anneal * kl_loss
