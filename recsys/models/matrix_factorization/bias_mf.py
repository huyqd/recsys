import torch
from torch import nn
import torch.nn.functional as F


class BiasMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.user_bias = nn.Parameter(torch.randn(n_users), requires_grad=True)
        self.item_bias = nn.Parameter(torch.randn(n_items), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, users, items=None):
        if items is None:
            items = torch.arange(self.n_items)
            outputs = (
                self.item_bias
                + self.user_bias[users].view(-1, 1)
                + self.bias
                + self.user_embedding(users)
                .squeeze(1)
                .matmul(self.item_embedding(items).T)
            )
        else:
            outputs = (
                self.item_bias[items]
                + self.user_bias[users].view(-1, 1)
                + self.bias
                + self.user_embedding(users)
                .unsqueeze(1)
                .mul(self.item_embedding(items))
                .sum(dim=-1)
            )

        return outputs

    def loss(self, inputs):
        users, items, labels = (
            inputs["user_code"],
            inputs["item_code"],
            inputs["label"],
        )
        outputs = self(users, items)

        return F.binary_cross_entropy_with_logits(outputs, labels)
