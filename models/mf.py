import implicit
import scipy.sparse as sparse
import torch
from torch import nn


class AlsMF(nn.Module):
    def __init__(self, embedding_dim, regularization=0.1, iterations=50):
        super(AlsMF, self).__init__()
        model = implicit.als.AlternatingLeastSquares(
            factors=embedding_dim,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False,
        )

        self.model = model

    def fit(self, dm):
        train = dm.train_sparse.to_dense()
        train = sparse.csr_matrix(train)
        self.model.fit(train.T)

    def forward(self, dm):
        test_items = dm.test_items
        test_scores = []
        for u in range(dm.n_users):
            items = test_items[u]
            user_features, item_features = self.model.user_factors[u], self.model.item_factors[items]
            item_scores = user_features.dot(item_features.T)
            item_scores = dict(zip(items.tolist(), item_scores.tolist()))
            test_scores.append(item_scores)

        return test_scores


class VanillaMF(nn.Module):
    """A matrix factorization model trained using SGD and negative sampling."""

    def __init__(self, n_users, n_items, embedding_dim):
        super(VanillaMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

    def forward(self, users, items):
        return (self.user_embedding(users).mul(self.item_embedding(items))).sum(dim=-1)


class BiasMF(nn.Module):
    """A matrix factorization model trained using SGD and negative sampling."""

    def __init__(self, n_users, n_items, embedding_dim):
        super(BiasMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.user_bias = nn.Parameter(torch.zeros((n_users)))
        self.item_bias = nn.Parameter(torch.zeros((n_items)))
        self.bias = nn.Parameter(torch.Tensor([0]))

    def forward(self, users, items):
        return (
                self.bias +
                self.user_bias[users] +
                self.item_bias[items] +
                (self.user_embedding(users).mul(self.item_embedding(items))).sum(dim=-1)
        )
