import implicit

from recsys.dataset import load_ml1m_data
from recsys.metrics import compute_metrics
from recsys.utils import topk


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

    print("All item predictions metrics")
    all_preds = topk(scores)
    _ = compute_metrics(labels, all_preds)

    print("Subset of item predictions metrics")
    sub_preds = topk(scores, subset=test_codes)
    _ = compute_metrics(labels, sub_preds)


def run_alsmf():
    data = load_ml1m_data()
    train_alsmf(data)


if __name__ == "__main__":
    run_alsmf()
