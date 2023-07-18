import numpy as np
import scipy.sparse as scs

from recsys.metrics import ndcg_score, hr_score
from recsys.dataset import load_data


def train_svd(inputs, labels, test_codes, k=10):
    user_factors, _, movie_factors = scs.linalg.svds(
        inputs.astype(float),
        128,
    )
    movie_factors = movie_factors
    scores = user_factors.dot(movie_factors)

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


def svd_pipeline():
    inputs, labels, test_codes = load_data()
    train_svd(inputs, labels, test_codes)


if __name__ == "__main__":
    svd_pipeline()
