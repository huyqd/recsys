from pathlib import Path


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
    negative = "negative"


class path:
    base = Path(__file__).parent.parent
    data = base / "data"

    # ml1m
    ml1m = data / "ml-1m"
    ml1m_ratings = ml1m / "ratings.dat"
    ml1m_split = ml1m / "split"
    ml1m_train_loo = ml1m_split / "train_loo.parquet"
    ml1m_test_loo = ml1m_split / "test_loo.parquet"
    ml1m_binary_npz = ml1m_split / "ml1m_binary.npz"
