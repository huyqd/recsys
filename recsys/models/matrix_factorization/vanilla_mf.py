import torch
from torch import nn
import torch.nn.functional as F


class VanillaMF(nn.Module):
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

    def forward(self, inputs):
        users, items = inputs["user_code"], inputs["item_code"]
        if items is None:
            items = torch.arange(self.n_items)
            logits = (
                self.user_embedding(users)
                .squeeze(1)
                .matmul(self.item_embedding(items).T)
            )
        else:
            logits = (
                self.user_embedding(users).unsqueeze(1).mul(self.item_embedding(items))
            ).sum(dim=-1)

        return logits

    def loss(self, inputs):
        labels = inputs["label"]
        logits = self(inputs)

        return F.binary_cross_entropy_with_logits(logits, labels)
