import torch
from torch import nn


class GMF(nn.Module):
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
        self.linear = nn.Linear(embedding_dim, 1)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.linear.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, users, items, pointwise=False):
        if pointwise:
            return self._forward_pointwise(users, items)
        else:
            return self._forward(users, items)

    def _forward(self, users, items=None):
        items = items if items is not None else torch.arange(self.n_items)
        out = (
            self.user_embedding(users)
            .unsqueeze(1)
            .mul(self.item_embedding(items))
            .view(-1, self.embedding_dim)
        )
        output = self.linear(out)

        return output.view(users.shape[0], -1)

    def _forward_pointwise(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        element_product = torch.mul(user_embeddings, item_embeddings)
        logits = self.linear(element_product)

        return logits


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
            mlp.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(p=dropout)])
        mlp = mlp[:-2]  # remove relu and dropout for the last layer
        self.mlp = nn.Sequential(*mlp)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, users, items, pointwise=False):
        if pointwise:
            return self._forward_pointwise(users, items)
        else:
            return self._forward(users, items)

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


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding_mlp = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding_mlp = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        self.user_embedding_gmf = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding_gmf = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        self.linear_gmf = nn.Linear(embedding_dim, int(embedding_dim / 2))

        linear_mlp_dims = (
            embedding_dim * 2,
            embedding_dim,
            embedding_dim,
            int(embedding_dim / 2),
        )
        self.linear_mlp_layers = nn.ModuleList()
        for idx, (in_dim, out_dim) in enumerate(
            zip(linear_mlp_dims[:-1], linear_mlp_dims[1:])
        ):
            self.linear_mlp_layers.append(nn.Linear(in_dim, out_dim))
            if idx != (
                len(linear_mlp_dims) - 2
            ):  # No activation and dropout for last layers
                self.linear_mlp_layers.append(nn.ReLU())
                self.linear_mlp_layers.append(nn.Dropout(p=dropout))

        self.linear_final = nn.Linear(embedding_dim, 1)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)

        for layer in self.linear_mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.xavier_uniform_(self.linear_gmf.weight)
        nn.init.xavier_uniform_(self.linear_final.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, users, items, pointwise=False):
        if pointwise:
            return self._forward_pointwise(users, items)
        else:
            return self._forward(users, items)

    def _forward(self, users, items):
        output_gmf = self.gmf(users, items, return_hidden=True)
        output_mlp = self.mlp(users, items, return_hidden=True)
        output = torch.cat([output_gmf, output_mlp], dim=1)

        return self.linear(output).view(users.shape[0], -1)

    def _forward_pointwise(self, users, items):
        user_embeddings_mlp = self.user_embedding_mlp(users)
        item_embeddings_mlp = self.item_embedding_mlp(items)
        embeddings_mlp = torch.cat([user_embeddings_mlp, item_embeddings_mlp], dim=1)

        user_embeddings_gmf = self.user_embedding_gmf(users)
        item_embeddings_gmf = self.item_embedding_gmf(items)
        embeddings_gmf = user_embeddings_gmf.mul(item_embeddings_gmf)

        output_gmf = self.linear_gmf(embeddings_gmf)
        output_mlp = embeddings_mlp
        for layer in self.linear_mlp_layers:
            output_mlp = layer(output_mlp)

        output = torch.cat([output_mlp, output_gmf], dim=1)
        output = self.linear_final(output)

        return output.squeeze()
