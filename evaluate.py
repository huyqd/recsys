from dataset import ML1mDataset
from models import MODELS
from metrics import get_eval_metrics

if __name__ == '__main__':
    k = 10
    model_name = "AlsMF"

    ds = ML1mDataset()

    model = MODELS[model_name]
    model = model(20)
    model.fit(ds)
    scores = model.predict(ds)
    ncdg, apak, hr = get_eval_metrics(scores, ds, k)
