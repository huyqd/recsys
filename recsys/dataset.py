import numpy as np
import pandas as pd
import scipy.sparse as scs

from recsys.utils import col, path


def _ratings_time_rank():
    """Read in ratings file and rank items according to timestamp group by users"""
    ratings = pd.read_csv(
        path.ml1m_ratings,
        sep="::",
        header=None,
        names=[col.user_id, col.movie_id, col.rating, col.timestamp],
        dtype={
            col.user_id: np.int32,
            col.movie_id: np.int32,
            col.rating: np.float32,
            col.timestamp: np.int64,
        },
        engine="python",
        encoding="ISO-8859-1",
    )

    ratings[col.user_code] = ratings[col.user_id].astype("category").cat.codes
    ratings[col.movie_code] = ratings[col.movie_id].astype("category").cat.codes

    ratings = ratings.assign(
        rank=ratings.groupby(col.user_code)[col.timestamp]
        .rank(method="first", ascending=False)
        .sub(1)
    )

    return ratings


def split_data_loo(n_test_codes=100):
    """Split data into train and test sets, with loo (leave one out, i.e. latest one) logic"""
    ratings = _ratings_time_rank()
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
    test_loo = test_loo.assign(negative_code=negative_codes.tolist())

    # Save dataframe to disk
    train_loo.to_parquet(path.ml1m_train_loo, index=False)
    test_loo.to_parquet(path.ml1m_test_loo, index=False)

    # Save npz to disk
    inputs = scs.csr_matrix(
        (train_loo[col.rating], (train_loo[col.user_code], train_loo[col.movie_code]))
    )
    inputs[inputs.nonzero()] = 1
    labels = test_loo[[col.movie_code]].to_numpy()
    test_codes = np.hstack([labels, negative_codes])
    test_labels = np.zeros(shape=test_codes.shape, dtype=float)
    test_labels[:, 0] = 1
    npz = {
        "inputs": user_movie_matrix,
        "labels": labels,
        "test_codes": test_codes,
        "test_labels": test_labels,
        "negative_samples": negative_samples,
    }
    np.savez(path.ml1m_implicit_npz, data=npz)


def load_implicit_data():
    data = np.load(path.ml1m_implicit_npz, allow_pickle=True)["data"].item()

    return data


if __name__ == "__main__":
    split_data_loo()
