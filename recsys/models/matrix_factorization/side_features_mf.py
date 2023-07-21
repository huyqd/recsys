import torch
from torch import nn
import torch.nn.functional as F


class SideFeaturesMF(nn.Module):
    def __init__(self, n_users, n_items, n_occupations, embedding_dim):
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
        self.user_bias = nn.Parameter(torch.randn(n_users), requires_grad=True)
        self.item_bias = nn.Parameter(torch.randn(n_items), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, users, occupations, items=None):
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
            )
        else:
            outputs = (
                self.item_bias[items]
                + self.user_bias[users].view(-1, 1)
                + self.bias
                + (self.user_embedding(users) + self.occupation_embedding(occupations))
                .unsqueeze(1)
                .mul(self.item_embedding(items))
                .sum(dim=-1)
            )

        return outputs

    def loss(self, users, items, occupations, labels):
        outputs = self(users, items, occupations)

        return F.binary_cross_entropy_with_logits(outputs, labels)
