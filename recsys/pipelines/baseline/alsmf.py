import implicit
import numpy as np

from recsys.metrics import ndcg_score, hr_score
from recsys.dataset import load_implicit_data


def train_alsmf(data, k=10):
    inputs, labels, test_codes = data["inputs"], data["labels"], data["test_codes"]

    model = implicit.als.AlternatingLeastSquares(
        factors=128, iterations=50, use_gpu=True, calculate_training_loss=True
    )

    model.fit(inputs)
    user_factors, movie_factors = model.user_factors, model.item_factors
    user_factors = user_factors.to_numpy()
    movie_factors = movie_factors.to_numpy()
    scores = user_factors.dot(movie_factors.T)

    # all items result
    all_preds = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    print(
        f"all ndcg: {ndcg_score(labels, all_preds):.4f}, all hr: {hr_score(labels, all_preds):.4f}"
    )

    # subset result
    test_scores = np.take_along_axis(scores, test_codes, axis=1)
    preds = np.take_along_axis(
        test_codes,
        np.argsort(test_scores, axis=1)[:, ::-1],
        axis=1,
    )[:, :k]

    print(f"ndcg: {ndcg_score(labels, preds):.4f}, hr: {hr_score(labels, preds):.4f}")


def run_alsmf():
    data = load_implicit_data()
    train_alsmf(data)


if __name__ == "__main__":
    run_alsmf()
