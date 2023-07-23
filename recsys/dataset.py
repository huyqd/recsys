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
        timestamp_rank=ratings.groupby(col.user_code)[col.timestamp]
        .rank(method="first", ascending=True)
        .sub(1),
        reverse_timestamp_rank=ratings.groupby(col.user_code)[col.timestamp]
        .rank(method="first", ascending=False)
        .sub(1),
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
    train_loo = ratings.query(f"{col.reverse_timestamp_rank} > 0")
    test_loo = ratings.query(f"{col.reverse_timestamp_rank} == 0")

    # Add more test codes
    np.random.seed(47)
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
    test_true = test_loo[[col.movie_code]].to_numpy()
    test_codes = np.hstack([test_true, negative_codes])
    test_timestamp_rank = test_loo[col.timestamp_rank].to_numpy().astype(int)
    rating_matrix = scs.csr_matrix(
        (train_loo[col.rating], (train_loo[col.user_code], train_loo[col.movie_code]))
    )
    implicit_matrix = rating_matrix.copy()
    implicit_matrix[implicit_matrix.nonzero()] = 1
    timestamp_matrix = scs.csr_matrix(
        (
            train_loo[col.timestamp_rank],
            (train_loo[col.user_code], train_loo[col.movie_code]),
        )
    )
    user_infos = users[[col.user_id, col.gender, col.age, col.occupation]].to_numpy()
    npz = {
        # "rating_matrix": scs.hstack([user_codes, rating_matrix]),
        "rating_matrix": rating_matrix,
        # "implicit_matrix": scs.hstack([user_codes, implicit_matrix]),
        "implicit_matrix": implicit_matrix,
        # "test_codes": np.hstack([user_codes, test_codes]),
        "timestamp_matrix": timestamp_matrix,
        "test_codes": test_codes,
        "test_true": test_true,
        "test_timestamp_rank": test_timestamp_rank,
        "negative_samples": negative_samples,
        "user_infos": user_infos,
    }
    np.savez(path.ml1m_npz, data=npz)


def load_ml1m_data():
    data = np.load(path.ml1m_npz, allow_pickle=True)["data"].item()

    return data


def create_negative_sampled_train_data(train_inputs, negative_samples, n_negatives):
    row_positives, col_positives = train_inputs.nonzero()
    row_negatives = row_positives.repeat(n_negatives)
    col_negatives = np.random.randint(
        0, negative_samples.shape[1], row_negatives.shape[0]
    )

    train_negatives = negative_samples[row_negatives, col_negatives].reshape(
        -1, n_negatives
    )

    train_positives = np.vstack([row_positives, col_positives]).T
    train_data = np.hstack(
        [
            train_positives,
            train_negatives,
        ]
    )

    train_labels = np.zeros((train_data.shape[0], n_negatives + 1), dtype=float)
    train_labels[:, 0] = 1

    return train_data, train_labels


class Ml1mDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.length = self.data.pop("length")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        items = {key: val[idx] for key, val in self.data.items()}

        return items


class ImplicitData:
    def __init__(self, data: dict, device: str = "cpu"):
        self.implicit_matrix = data["implicit_matrix"]
        self.timestamp_matrix = data["timestamp_matrix"]
        self.test_codes = data["test_codes"]
        self.test_true = data["test_true"]
        self.test_timestamp_rank = data["test_timestamp_rank"]
        self.negative_samples = data["negative_samples"]
        self.user_infos = data["user_infos"]
        self.n_users, self.n_items = self.implicit_matrix.shape
        self.n_occupations = np.unique(self.user_infos[:, -1]).shape[0]
        self.max_timestamp_rank = int(self.timestamp_matrix.data.max() + 2)
        self.device = device

    def create_negative_sampled_train_dataloader(
        self,
        n_negatives=4,
        batch_size=512,
    ):
        train_data, train_labels = create_negative_sampled_train_data(
            self.implicit_matrix,
            self.negative_samples,
            n_negatives,
        )
        user_occupations = self.user_infos[train_data[:, 0], -1]
        item_timestamp_rank = (
            np.asarray(
                self.timestamp_matrix[
                    train_data[:, 0],
                    train_data[:, 1],
                ]
            )
            .squeeze()
            .astype(int)
        )
        length = train_data.shape[0]
        assert (
            length
            == user_occupations.shape[0]
            == item_timestamp_rank.shape[0]
            == train_labels.shape[0]
        )
        data_dict = {
            "user_code": train_data[:, 0],
            "item_code": train_data[:, 1:],
            "user_occupation": user_occupations,
            "item_timestamp_rank": item_timestamp_rank,
            "label": train_labels,
            "length": length,
        }
        dataset = Ml1mDataset(data_dict)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device=self.device),
        )

    def create_test_dataloader(self, batch_size=1024):
        user_codes = np.arange(self.n_users)
        user_occupations = self.user_infos[user_codes, -1]
        data_dict = {
            "user_code": user_codes,
            "item_code": self.test_codes,
            "user_occupation": user_occupations,
            "item_timestamp_rank": self.test_timestamp_rank,
            "length": self.test_codes.shape[0],
        }
        dataset = Ml1mDataset(data_dict)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            generator=torch.Generator(device=self.device),
        )


def load_implicit_data(device="cpu"):
    data = load_ml1m_data()
    implicit_data = ImplicitData(data, device=device)

    return implicit_data


if __name__ == "__main__":
    split_data_loo()
