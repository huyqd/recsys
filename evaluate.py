from dataset import ML1mDataset
from metrics import get_eval_metrics
from models import MODELS

if __name__ == '__main__':
    k = 10
    embedding_dim = 20
    model_name = "AlsMF"

    ds = ML1mDataset()

    model = MODELS[model_name]
    model = model(embedding_dim)
    model.fit(ds)
    scores = model(ds)
    ncdg, apak, hr = get_eval_metrics(scores, ds, k)
