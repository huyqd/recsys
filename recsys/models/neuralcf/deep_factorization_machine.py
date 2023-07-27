import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        for hidden_dims in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dims),
                    nn.BatchNorm1d(hidden_dims),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            input_dim = hidden_dims
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DeepFactorizationMachine(nn.Module):
    def __init__(
        self, n_users, n_items, n_occupations, max_timestamp_rank, embedding_dim
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_occupations = n_occupations
        self.max_timestamp_rank = max_timestamp_rank
        self.embedding_dim = embedding_dim
        self.n_features = n_users + n_items + n_occupations + max_timestamp_rank

        self.feature_embedding = nn.Embedding(
            num_embeddings=self.n_features, embedding_dim=embedding_dim
        )

        self.bias_embedding = nn.Embedding(
            num_embeddings=self.n_features, embedding_dim=1
        )

        self.deep_mlp = MLP(
            input_dim=embedding_dim * 4,
            hidden_dims=[embedding_dim * 4, embedding_dim * 2, embedding_dim],
            dropout=0.1,
        )

    def forward(self, inputs):
        users, items, occupations, timestamp_rank = (
            inputs["user_code"],
            inputs["item_code"],
            inputs["user_occupation"],
            inputs["item_timestamp_rank"],
        )
        batch_size = users.shape[0]
        n_items = items.shape[1]
        user_idx = users + 0
        item_idx = items + self.n_users
        occupation_idx = occupations + self.n_users + self.n_items
        timestamp_rank_idx = (
            timestamp_rank + self.n_users + self.n_items + self.n_occupations
        )

        item_bias_embeddings = self.bias_embedding(item_idx)
        ex_item_bias_embeddings = self.bias_embedding(
            torch.vstack([user_idx, occupation_idx, timestamp_rank_idx]).T
        )
        bias_term = ex_item_bias_embeddings.sum(dim=1).add(
            item_bias_embeddings.squeeze(dim=-1)
        )

        item_feature_embeddings = self.feature_embedding(item_idx)
        ex_item_feature_embeddings = self.feature_embedding(
            torch.vstack([user_idx, occupation_idx, timestamp_rank_idx]).T
        )
        square_of_sum = (
            ex_item_feature_embeddings.sum(dim=1, keepdim=True)
            .add(item_feature_embeddings)
            .pow(2)
        )
        sum_of_square = (
            ex_item_feature_embeddings.pow(2)
            .sum(dim=1, keepdim=True)
            .add(item_feature_embeddings.pow(2))
        )

        fm_term = 0.5 * (square_of_sum - sum_of_square).sum(dim=-1)

        deep_embeddings = torch.cat(
            [
                item_feature_embeddings,
                ex_item_feature_embeddings.view(batch_size, 1, -1).repeat(
                    (1, n_items, 1)
                ),
            ],
            dim=-1,
        ).view(batch_size * n_items, -1)
        deep_term = self.deep_mlp(deep_embeddings).view(batch_size, n_items)

        logits = bias_term + fm_term + deep_term

        return logits

    def loss(self, inputs):
        labels = inputs["label"]
        logits = self(inputs)

        return F.binary_cross_entropy_with_logits(logits, labels)
