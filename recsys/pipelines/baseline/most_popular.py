import numpy as np

from recsys.dataset import load_implicit_data
from recsys.metrics import compute_metrics
from recsys.utils import topk


def train_popular(data, k=10):
    inputs, labels, test_codes = data["inputs"], data["labels"], data["test_codes"]
    # Most popular retrieval
    scores = np.asarray(inputs.sum(axis=0)).repeat(inputs.shape[0], axis=0)
    scores[inputs.nonzero()] = -1

    print("Predict all metrics")
    all_preds = topk(scores)
    _ = compute_metrics(labels, all_preds)

    print("Predict subset metrics")
    sub_preds = topk(scores, subset=test_codes)
    _ = compute_metrics(labels, sub_preds)


def run_popular():
    data = load_implicit_data()
    train_popular(data)


if __name__ == "__main__":
    run_popular()
