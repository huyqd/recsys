import torch
from torch import nn
from torch.nn import functional as F


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
