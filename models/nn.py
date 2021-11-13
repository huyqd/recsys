import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(embedding_dim, 1)

        self.init_weight()

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = user_embeddings.mul(item_embeddings)
        output = self.linear(embeddings)

        return output.squeeze()

    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.linear.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        linear_dims = (embedding_dim * 2, embedding_dim, int(embedding_dim / 2), 1)
        self.linear_layers = nn.ModuleList()
        for idx, (in_dim, out_dim) in enumerate(zip(linear_dims[:-1], linear_dims[1:])):
            self.linear_layers.append(nn.Linear(in_dim, out_dim))
            if idx != (len(linear_dims) - 2):  # No activation and dropout for last layers
                self.linear_layers.append(nn.ReLU())
                self.linear_layers.append(nn.Dropout(p=dropout))

        self.init_weight()

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        output = torch.cat([user_embeddings, item_embeddings], axis=1)
        for layer in self.linear_layers:
            output = layer(output)

        return output.squeeze()

    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

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

        linear_mlp_dims = (embedding_dim * 2, embedding_dim, embedding_dim, int(embedding_dim / 2))
        self.linear_mlp_layers = nn.ModuleList()
        for idx, (in_dim, out_dim) in enumerate(zip(linear_mlp_dims[:-1], linear_mlp_dims[1:])):
            self.linear_mlp_layers.append(nn.Linear(in_dim, out_dim))
            if idx != (len(linear_mlp_dims) - 2):  # No activation and dropout for last layers
                self.linear_mlp_layers.append(nn.ReLU())
                self.linear_mlp_layers.append(nn.Dropout(p=dropout))

        self.linear_final = nn.Linear(embedding_dim, 1)

        self.init_weight()

    def forward(self, users, items):
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
