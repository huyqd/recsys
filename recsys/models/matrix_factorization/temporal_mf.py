import torch
import torch.nn.functional as F
from torch import nn


class TemporalMF(nn.Module):
    def __init__(
        self, n_users, n_items, n_occupations, max_timestamp_rank, embedding_dim
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_occupations = n_occupations
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.occupation_embedding = nn.Embedding(
            num_embeddings=n_occupations, embedding_dim=embedding_dim
        )

        self.user_temporal_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.temporal_embedding = nn.Embedding(
            num_embeddings=max_timestamp_rank, embedding_dim=embedding_dim
        )

        self.user_bias = nn.Parameter(torch.randn(n_users), requires_grad=True)
        self.item_bias = nn.Parameter(torch.randn(n_items), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, inputs):
        users, items, occupations, timestamp_rank = (
            inputs["user_code"],
            inputs["item_code"],
            inputs["user_occupation"],
            inputs["item_timestamp_rank"],
        )

        if items is None:
            items = torch.arange(self.n_items)
            outputs = (
                self.item_bias
                + self.user_bias[users].view(-1, 1)
                + self.bias
                + (
                    self.user_embedding(users).squeeze(1)
                    + self.occupation_embedding(occupations)
                ).matmul(self.item_embedding(items).T)
                + self.user_temporal_embedding(users).mul(
                    self.temporal_embedding(timestamp_rank)
                )
            )
        else:
            outputs = (
                self.item_bias[items]
                + self.user_bias[users].view(-1, 1)
                + self.bias
                + (
                    (
                        self.user_embedding(users)
                        + self.occupation_embedding(occupations)
                    )
                    .unsqueeze(1)
                    .mul(self.item_embedding(items))
                    .sum(dim=-1)
                )
                + self.user_temporal_embedding(users)
                .mul(self.temporal_embedding(timestamp_rank))
                .sum(dim=-1, keepdim=True)
            )

        return outputs

    def loss(self, inputs):
        labels = inputs["label"]
        logits = self(inputs)

        return F.binary_cross_entropy_with_logits(logits, labels)
