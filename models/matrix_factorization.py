import implicit
import scipy.sparse as sparse


class AlsMF:
    def __init__(self, embedding_dim, regularization=0.1, iterations=50):
        model = implicit.als.AlternatingLeastSquares(
            factors=embedding_dim,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False,
        )

        self.model = model

    def fit(self, ds):
        train = ds.train.to_dense()
        train = sparse.csr_matrix(train)
        self.model.fit(train.T)

    def predict(self, ds):
        test_items = ds.test_items
        test_scores = []
        for u in range(ds.n_users):
            items = test_items[u]
            user_features, item_features = self.model.user_factors[u], self.model.item_factors[items]
            item_scores = user_features.dot(item_features.T)
            item_scores = dict(zip(items.tolist(), item_scores.tolist()))
            test_scores.append(item_scores)

        return test_scores
