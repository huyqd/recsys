from pathlib import Path
import numpy as np


class col:
    user_id = "user_id"
    user_code = "user_code"
    movie_id = "movie_id"
    movie_code = "movie_code"
    rating = "rating"
    timestamp = "timestamp"
    title = "title"
    genre = "genre"
    gender = "gender"
    age = "age"
    occupation = "occupation"
    zipcode = "zipcode"
    negative = "negative_code"


class path:
    base = Path(__file__).parent.parent
    data = base / "data"

    # ml1m
    ml1m = data / "ml-1m"
    ml1m_ratings = ml1m / "ratings.dat"
    ml1m_users = ml1m / "users.dat"
    ml1m_movies = ml1m / "movies.dat"
    ml1m_split = ml1m / "split"
    ml1m_train_loo = ml1m_split / "train_loo.parquet"
    ml1m_test_loo = ml1m_split / "test_loo.parquet"
    ml1m_npz = ml1m_split / "ml1m.npz"


def topk(scores, array=None, subset=None, k=10):
    if subset is not None:
        scores = np.take_along_axis(
            scores,
            subset,
            axis=1,
        )
        array = subset

    sorted_scores = np.argsort(scores, axis=1)[:, ::-1][:, :k]

    if array is not None:
        sorted_scores = np.take_along_axis(
            array,
            sorted_scores,
            axis=1,
        )

    return sorted_scores
