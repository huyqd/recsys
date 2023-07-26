import torch
from torch import nn

from recsys.models.matrix_factorization.vanilla_mf import VanillaMF


class SideFeaturesMF(VanillaMF):
    def __init__(self, n_users, n_items, n_occupations, embedding_dim):
        super().__init__(n_users, n_items, embedding_dim)
        self.user_bias = nn.Parameter(torch.randn(n_users), requires_grad=True)
        self.item_bias = nn.Parameter(torch.randn(n_items), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

        self.occupation_embedding = nn.Embedding(
            num_embeddings=n_occupations, embedding_dim=embedding_dim
        )

    def forward(self, inputs):
        users, items, occupations = (
            inputs["user_code"],
            inputs["item_code"],
            inputs["user_occupation"],
        )
        bias_term = (
            self.bias + self.user_bias[users].view(-1, 1) + self.item_bias[items]
        )
        uo_embeddings = self.user_embedding(users) + self.occupation_embedding(
            occupations
        )

        if items is None:
            items = torch.arange(self.n_items)
            matrix_factorization_term = uo_embeddings.squeeze(1).matmul(
                self.item_embedding(items).T
            )
        else:
            matrix_factorization_term = (
                uo_embeddings.unsqueeze(1).mul(self.item_embedding(items)).sum(dim=-1)
            )
        logits = bias_term + matrix_factorization_term

        return logits
