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


class MultiDAE(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, dropout=0.5):
        super().__init__()
        self.mlp = MLP(encoder_dims + decoder_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, items):
        items_recon = self.mlp(self.dropout(items))

        return items_recon

    def loss(self, items):
        items_recon = self(items)

        return F.binary_cross_entropy_with_logits(items_recon, items)

    def negative_sampling_loss(self, items, n_negatives):
        negative_samples = items.squeeze().sum(dim=0).repeat((items.shape[0], 1))
        negative_samples[items.squeeze().nonzero(as_tuple=True)] = -1
        negative_samples = negative_samples.argsort(descending=True, dim=1)[:, :500]

        row_positives, train_positives = items.squeeze().nonzero().chunk(2, dim=1)
        row_negatives = row_positives.squeeze().repeat(n_negatives)
        col_negatives = torch.randint(
            0, negative_samples.shape[1], (row_negatives.shape[0],)
        )

        train_negatives = negative_samples[row_negatives, col_negatives].view(
            -1, n_negatives
        )
        inputs = torch.hstack([train_positives, train_negatives])
        labels = torch.zeros_like(inputs)
        labels[:, 0] = 1

        scores = self(items)
        scores = scores[row_positives.squeeze()].squeeze()
        scores = torch.take_along_dim(scores, inputs, dim=1)

        return F.binary_cross_entropy_with_logits(scores, labels.float())


class MultiVAE(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, dropout=0.5):
        super().__init__()
        self.encoder = MLP(encoder_dims)
        self.decoder = MLP(decoder_dims)
        self.dropout = nn.Dropout(dropout)

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

    def negative_sampling_loss(self, items, n_negatives):
        negative_samples = items.squeeze().sum(dim=0).repeat((items.shape[0], 1))
        negative_samples[items.squeeze().nonzero(as_tuple=True)] = -1
        negative_samples = negative_samples.argsort(descending=True, dim=1)[:, :500]

        row_positives, train_positives = items.squeeze().nonzero().chunk(2, dim=1)
        row_negatives = row_positives.squeeze().repeat(n_negatives)
        col_negatives = torch.randint(
            0, negative_samples.shape[1], (row_negatives.shape[0],)
        )

        train_negatives = negative_samples[row_negatives, col_negatives].view(
            -1, n_negatives
        )
        inputs = torch.hstack([train_positives, train_negatives])
        labels = torch.zeros_like(inputs)
        labels[:, 0] = 1

        scores, mu_z, log_std_z = self(items)
        scores = scores[row_positives.squeeze()].squeeze()
        scores = torch.take_along_dim(scores, inputs, dim=1)
        recon_loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        kl_loss = -log_std_z - 0.5 + (torch.exp(2 * log_std_z) + mu_z**2) * 0.5
        kl_loss = kl_loss.sum(1).mean()

        return recon_loss + kl_loss
