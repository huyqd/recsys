import torch


class Popularity:

    def __init__(self):
        return

    def predict(self, ds):
        score = ds.train.to_dense().sum(dim=0)
        test_pos = ds.test_pos
        test_neg = ds.test_neg

        test_scores = []
        for u in range(ds.n_users):
            items = torch.cat((test_pos[u, 1].view(1), test_neg[u]))
            item_scores = score[items]
            test_scores.append(item_scores)

        return torch.vstack(test_scores)
