import numpy as np

from recsys.dataset import load_implicit_data
from recsys.metrics import ndcg_score, hr_score


def train_popular(data, k=10):
    (
        inputs,
        y_true,
        test_codes,
        negative_samples,
    ) = (
        data["inputs"],
        data["labels"],
        data["test_codes"],
        data["negative_samples"],
    )
    # Most popular retrieval
    retrieval = np.asarray(inputs.sum(axis=0)).repeat(inputs.shape[0], axis=0)
    retrieval[inputs.nonzero()] = -1
    retrieval = np.argsort(retrieval, axis=1)[:, ::-1]

    y_pred = retrieval[:, :k]
    print(
        f"ndcg@{k}: {ndcg_score(y_true, y_pred):.4f}, hr@{k}: {hr_score(y_true, y_pred):.4f}"
    )


def run_popular():
    data = load_implicit_data()
    train_popular(data)


if __name__ == "__main__":
    run_popular()
