import implicit
import scipy.sparse as sparse
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from metrics import get_eval_metrics


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

    def fit(self, ds):
        train = ds.train_ds.train.to_dense()
        train = sparse.csr_matrix(train)
        self.model.fit(train.T)

    def forward(self, ds):
        test_items = ds.test_ds.test_items
        test_scores = []
        for u in range(ds.test_ds.n_users):
            items = test_items[u]
            user_features, item_features = self.model.user_factors[u], self.model.item_factors[items]
            item_scores = user_features.dot(item_features.T)
            item_scores = dict(zip(items.tolist(), item_scores.tolist()))
            test_scores.append(item_scores)

        return test_scores


class TorchMF(nn.Module):
    """A matrix factorization model trained using SGD and negative sampling."""

    def __init__(self, n_users, n_items, embedding_dim):
        super(TorchMF, self).__init__()
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


class LightningMF(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        pos, score = batch
        users, pos_items = pos[:, 0], pos[:, 1]

        n_neg_items = 5
        neg_items = torch.multinomial(score, n_neg_items)
        items = torch.cat((pos_items.view(-1, 1), neg_items), dim=1)

        labels = torch.zeros(items.shape)
        labels[:, 0] += 1
        users = users.view(-1, 1).repeat(1, items.shape[1])

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits.detach(),
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        pos, items, labels = batch
        users = pos[:, 0].view(-1, 1).repeat(1, items.shape[1])

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        item_true = pos[:, 1].view(-1, 1)
        item_scores = [dict(zip(item.tolist(), score.tolist())) for item, score in zip(items, logits)]
        ncdg, apak, hr = get_eval_metrics(item_scores, item_true)
        metrics = {
            'ncdg': round(ncdg, 2),
            'apak': round(apak, 2),
            'hr': round(hr, 2),
        }
        self.log("Val Metrics", metrics, prog_bar=True, on_epoch=True)

        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
        }

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return optimizer

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)


"""
TODO:
- data loaders: train & test
- negative sampling
- forward to get scores
- BCE Log Loss
- SGD
"""


def predict(self, pairs, batch_size, verbose):
    """Computes predictions for a given set of user-item pairs.
    Args:
      pairs: A pair of lists (users, items) of the same length.
      batch_size: unused.
      verbose: unused.
    Returns:
      predictions: A list of the same length as users and items, such that
      predictions[i] is the models prediction for (users[i], items[i]).
    """
    del batch_size, verbose
    num_examples = len(pairs[0])
    assert num_examples == len(pairs[1])
    predictions = np.empty(num_examples)
    for i in range(num_examples):
        predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
    return predictions


def fit(self, positive_pairs, learning_rate, num_negatives):
    """Trains the model for one epoch.
    Args:
      positive_pairs: an array of shape [n, 2], each row representing a positive
        user-item pair.
      learning_rate: the learning rate to use.
      num_negatives: the number of negative items to sample for each positive.
    Returns:
      The logistic loss averaged across examples.
    """
    # Convert to implicit format and sample negatives.
    user_item_label_matrix = self._convert_ratings_to_implicit_data(
        positive_pairs, num_negatives)
    np.random.shuffle(user_item_label_matrix)

    # Iterate over all examples and perform one SGD step.
    num_examples = user_item_label_matrix.shape[0]
    reg = self.reg
    lr = learning_rate
    sum_of_loss = 0.0
    for i in range(num_examples):
        (user, item, rating) = user_item_label_matrix[i, :]
        user_emb = self.user_embedding[user]
        item_emb = self.item_embedding[item]
        prediction = self._predict_one(user, item)

        if prediction > 0:
            one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
            sigmoid = 1.0 / one_plus_exp_minus_pred
            this_loss = (np.log(one_plus_exp_minus_pred) +
                         (1.0 - rating) * prediction)
        else:
            exp_pred = np.exp(prediction)
            sigmoid = exp_pred / (1.0 + exp_pred)
            this_loss = -rating * prediction + np.log(1.0 + exp_pred)

        grad = rating - sigmoid

        self.user_embedding[user, :] += lr * (grad * item_emb - reg * user_emb)
        self.item_embedding[item, :] += lr * (grad * user_emb - reg * item_emb)
        self.user_bias[user] += lr * (grad - reg * self.user_bias[user])
        self.item_bias[item] += lr * (grad - reg * self.item_bias[item])
        self.bias += lr * (grad - reg * self.bias)

        sum_of_loss += this_loss

    # Return the mean logistic loss.
    return sum_of_loss / num_examples


def _convert_ratings_to_implicit_data(self, positive_pairs, num_negatives):
    """Converts a list of positive pairs into a two class dataset.
    Args:
      positive_pairs: an array of shape [n, 2], each row representing a positive
        user-item pair.
      num_negatives: the number of negative items to sample for each positive.
    Returns:
      An array of shape [n*(1 + num_negatives), 3], where each row is a tuple
      (user, item, label). The examples are obtained as follows:
      To each (user, item) pair in positive_pairs correspond:
      * one positive example (user, item, 1)
      * num_negatives negative examples (user, item', 0) where item' is sampled
        uniformly at random.
    """
    num_items = self.item_embedding.shape[0]
    num_pos_examples = positive_pairs.shape[0]
    training_matrix = np.empty([num_pos_examples * (1 + num_negatives), 3],
                               dtype=np.int32)
    index = 0
    for pos_index in range(num_pos_examples):
        u = positive_pairs[pos_index, 0]
        i = positive_pairs[pos_index, 1]

        # Treat the rating as a positive training instance
        training_matrix[index] = [u, i, 1]
        index += 1

        # Add N negatives by sampling random items.
        # This code does not enforce that the sampled negatives are not present in
        # the training data. It is possible that the sampling procedure adds a
        # negative that is already in the set of positives. It is also possible
        # that an item is sampled twice. Both cases should be fine.
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            training_matrix[index] = [u, j, 0]
            index += 1
    return training_matrix
