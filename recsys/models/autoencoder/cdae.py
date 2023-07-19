import torch
from torch import nn as nn
from torch.nn import functional as F


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
