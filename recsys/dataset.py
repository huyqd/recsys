import numpy as np
import pandas as pd
import scipy.sparse as scs
import torch

from recsys.utils import col, path


def _ratings_time_rank():
    """Read in ratings file and rank items according to timestamp group by users"""
    ratings = pd.read_csv(
        path.ml1m_ratings,
        sep="::",
        header=None,
        names=[
            col.user_id,
            col.movie_id,
            col.rating,
            col.timestamp,
        ],
        dtype={
            col.user_id: np.int32,
            col.movie_id: "category",
            col.rating: np.float32,
            col.timestamp: np.int64,
        },
        engine="python",
        encoding="ISO-8859-1",
    )

    ratings[col.user_code] = ratings[col.user_id].sub(1)
    ratings[col.movie_code] = ratings[col.movie_id].cat.codes

    ratings = ratings.assign(
        rank=ratings.groupby(col.user_code)[col.timestamp]
        .rank(method="first", ascending=False)
        .sub(1)
    )

    return ratings


def _users():
    users = pd.read_csv(
        path.ml1m_users,
        sep="::",
        header=None,
        names=[
            col.user_id,
            col.gender,
            col.age,
            col.occupation,
            col.zipcode,
        ],
        dtype={
            col.user_id: np.int32,
            col.gender: "category",
            col.age: "category",
            col.occupation: np.int32,
            col.zipcode: str,
        },
        engine="python",
        encoding="ISO-8859-1",
    )

    users[col.user_id] = users[col.user_id].sub(1)
    users[col.age] = users[col.age].cat.codes
    users[col.gender] = users[col.gender].cat.codes

    return users


def split_data_loo(n_test_codes=100):
    """Split data into train and test sets, with loo (leave one out, i.e. latest one) logic"""
    ratings = _ratings_time_rank()
    users = _users()
    train_loo, test_loo = ratings.query("rank > 0"), ratings.query("rank == 0")

    # Add more test codes
    user_movie_matrix = scs.csr_matrix(
        (ratings[col.rating], (ratings[col.user_code], ratings[col.movie_code]))
    )
    negative_samples = user_movie_matrix.sum(axis=0).repeat(
        user_movie_matrix.shape[0], axis=0
    )
    negative_samples[user_movie_matrix.nonzero()] = -1
    negative_samples = np.asarray(np.argsort(negative_samples, axis=1)[:, ::-1])[
        :, :500
    ]
    negative_codes = np.take_along_axis(
        negative_samples,
        np.random.randint(
            0, negative_samples.shape[1], (negative_samples.shape[0], n_test_codes - 1)
        ),
        1,
    )

    # Save npz to disk
    user_codes = test_loo[[col.user_code]].to_numpy()
    test_labels = test_loo[[col.movie_code]].to_numpy()
    test_codes = np.hstack([test_labels, negative_codes])
    rating_matrix = scs.csr_matrix(
        (train_loo[col.rating], (train_loo[col.user_code], train_loo[col.movie_code]))
    )
    implicit_matrix = rating_matrix.copy()
    implicit_matrix[implicit_matrix.nonzero()] = 1
    user_infos = users[[col.user_id, col.gender, col.age, col.occupation]].to_numpy()
    npz = {
        "rating_matrix": scs.hstack([user_codes, rating_matrix]),
        "implicit_matrix": scs.hstack([user_codes, implicit_matrix]),
        "labels": test_labels,
        "test_codes": np.hstack([user_codes, test_codes]),
        "negative_samples": negative_samples,
        "user_infos": user_infos,
    }
    np.savez(path.ml1m_npz, data=npz)


def load_implicit_data():
    data = np.load(path.ml1m_npz, allow_pickle=True)["data"].item()

    return data


def train_dataloader(
    train_inputs: scs.csr_matrix,
    batch_size: int = 64,
    device: str = "cuda",
    add_user_codes: bool = True,
    negative_samples: np.ndarray = None,
    n_negatives: int = None,
) -> torch.utils.data.DataLoader:
    """
    Return train dataloader given inputs
    If negative_samples and n_negatives are provided, return train_dataloader with negative sampling
    """
    # Negative sampling train data
    if negative_samples is not None:
        assert (
            n_negatives is not None
        ), "Must provide n_negatives if negative samples is not None"
        row_positives, col_positives = train_inputs.nonzero()
        row_negatives = row_positives.repeat(n_negatives)
        col_negatives = np.random.randint(
            0, negative_samples.shape[1], row_negatives.shape[0]
        )

        train_negatives = negative_samples[row_negatives, col_negatives].reshape(
            -1, n_negatives
        )
        # Add user indices/codes if add_user_codes is True
        if add_user_codes:
            train_positives = np.vstack([row_positives, col_positives]).T
        else:
            train_positives = col_positives.reshape(-1, 1)
        train_data = np.hstack(
            [
                train_positives,
                train_negatives,
            ]
        )
    # Just use sparse input as train data
    else:
        train_data = train_inputs.toarray()
        if add_user_codes:
            train_data = np.hstack(
                [
                    np.arange(train_data.shape[0]).reshape(-1, 1),
                    train_data,
                ]
            )

    dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    return dl


class ImplicitData:
    def __init__(
        self,
        train_inputs: scs.csr_matrix,
        test_inputs: np.ndarray,
        train_batch_size: int = 64,
        test_batch_size: int = 1024,
        add_user_codes: bool = True,
        device: str = "cuda",
        negative_samples: np.ndarray = None,
        n_negatives: int = None,
    ):
        self.train_inputs = train_inputs
        self.test_inputs = test_inputs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        self.add_user_codes = add_user_codes
        self.negative_samples = negative_samples
        self.n_negatives = n_negatives

        self._train_dataloader = None
        self._test_dataloader = None

    @property
    def train_dataloader(self):
        if self.negative_samples is not None or self._train_dataloader is None:
            self._train_dataloader = train_dataloader(
                self.train_inputs,
                self.train_batch_size,
                self.device,
                self.add_user_codes,
                self.negative_samples,
                self.n_negatives,
            )

        return self._train_dataloader

    @property
    def test_dataloader(self):
        if self._test_dataloader is None:
            self._test_dataloader = torch.utils.data.DataLoader(
                self.test_inputs,
                batch_size=self.test_batch_size,
                generator=torch.Generator(device=self.device),
            )

        return self._test_dataloader


if __name__ == "__main__":
    split_data_loo()
