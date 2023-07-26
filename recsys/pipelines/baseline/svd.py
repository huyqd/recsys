import scipy.sparse as scs

from recsys.dataset import load_ml1m_data
from recsys.metrics import compute_metrics
from recsys.utils import topk


def train_svd(data, k=10):
    inputs, labels, test_codes = data["inputs"], data["labels"], data["test_codes"]
    user_factors, _, movie_factors = scs.linalg.svds(
        inputs.astype(float),
        128,
    )
    movie_factors = movie_factors
    scores = user_factors.dot(movie_factors)

    print("All item predictions metrics")
    all_preds = topk(scores)
    _ = compute_metrics(labels, all_preds)

    print("Subset of item predictions metrics")
    sub_preds = topk(scores, subset=test_codes)
    _ = compute_metrics(labels, sub_preds)


def run_svd():
    data = load_ml1m_data()
    train_svd(data)


if __name__ == "__main__":
    run_svd()
