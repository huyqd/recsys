from torch import nn


class VanillaMF(nn.Module):
    """A matrix factorization model trained using SGD and negative sampling."""

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

    def forward(self, users):
        return (
            self.user_embedding(users).squeeze(1).matmul(self.item_embedding.weight.T)
        )
