import torch
from torch import nn
import torch.nn.functional as F


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, mlp_dims=None, dropout=0.1):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF
        self.user_embedding_gmf = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding_gmf = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.linear_gmf = nn.Linear(embedding_dim, embedding_dim // 2)

        # MLP
        self.user_embedding_mlp = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding_mlp = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        mlp = []
        self.mlp_dims = mlp_dims or (
            embedding_dim * 2,
            embedding_dim,
            embedding_dim // 2,
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
        self.linear_mlp = nn.Sequential(*mlp)

        self.linear_final = nn.Linear(embedding_dim, 1)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)

        for layer in self.linear_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.xavier_uniform_(self.linear_gmf.weight)
        nn.init.xavier_uniform_(self.linear_final.weight)

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
        # GMF
        items = items if items is not None else torch.arange(self.n_items)
        output_gmf = self.linear_gmf(
            self.user_embedding_gmf(users)
            .unsqueeze(1)
            .mul(self.item_embedding_gmf(items))
            .view(-1, self.embedding_dim)
        )

        # MLP
        item_embedding_mlp = self.item_embedding_mlp(items)
        user_embedding_mlp = (
            self.user_embedding_mlp(users)
            .unsqueeze(1)
            .repeat((1, item_embedding_mlp.shape[1], 1))
        )
        output_mlp = self.linear_mlp(
            torch.cat([user_embedding_mlp, item_embedding_mlp], dim=2).view(
                -1, self.embedding_dim * 2
            )
        )

        output = self.linear_final(torch.cat([output_gmf, output_mlp], dim=1))

        return output.view(users.shape[0], -1)

    def _forward_pointwise(self, users, items):
        # GMF
        user_embeddings_gmf = self.user_embedding_gmf(users)
        item_embeddings_gmf = self.item_embedding_gmf(items)
        embeddings_gmf = user_embeddings_gmf.mul(item_embeddings_gmf)
        output_gmf = self.linear_gmf(embeddings_gmf)

        # MLP
        user_embeddings_mlp = self.user_embedding_mlp(users)
        item_embeddings_mlp = self.item_embedding_mlp(items)
        embeddings_mlp = torch.cat([user_embeddings_mlp, item_embeddings_mlp], dim=1)
        output_mlp = self.linear_mlp(embeddings_mlp)

        # Output
        output = torch.cat([output_mlp, output_gmf], dim=1)
        output = self.linear_final(output)

        return output
