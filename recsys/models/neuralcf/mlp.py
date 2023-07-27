import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, mlp_dims=None, dropout=0.1):
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
        mlp = []
        self.mlp_dims = mlp_dims or (
            embedding_dim * 2,
            embedding_dim,
            embedding_dim // 2,
            1,
        )
        for in_dim, out_dim in zip(self.mlp_dims[:-1], self.mlp_dims[1:]):
            mlp.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        mlp = mlp[:-2]  # remove relu and dropout for the last layer
        self.mlp = nn.Sequential(*mlp)

        # self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs, pointwise=False):
        users, items = inputs["user_code"], inputs["item_code"]
        if pointwise:
            return self._forward_pointwise(users, items)
        else:
            return self._forward(users, items)

    def loss(self, inputs, pointwise=False):
        labels = inputs["label"]
        logits = self.forward(inputs, pointwise)

        return F.binary_cross_entropy_with_logits(logits, labels)

    def _forward(self, users, items=None):
        items = items if items is not None else torch.arange(self.n_items)
        item_embedding = self.item_embedding(items)
        user_embedding = (
            self.user_embedding(users)
            .unsqueeze(1)
            .repeat((1, item_embedding.shape[1], 1))
        )
        output = self.mlp(
            torch.cat([user_embedding, item_embedding], dim=2).view(
                -1, self.embedding_dim * 2
            )
        )

        return output.view(users.shape[0], -1)

    def _forward_pointwise(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        output = torch.hstack([user_embeddings, item_embeddings])

        return self.mlp(output)
