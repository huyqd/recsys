import torch


class Popularity:

    def __init__(self):
        return

    def fit(self, ds):
        pass

    def predict(self, ds):
        score = ds.train.to_dense().sum(dim=0)
        test_items = ds.test_items

        test_scores = []
        for u in range(ds.n_users):
            items = test_items[u]
            item_scores = score[items]
            item_scores = dict(zip(items.tolist(), item_scores.tolist()))
            test_scores.append(item_scores)

        return test_scores
