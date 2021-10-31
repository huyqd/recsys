from dataset import ML1mDataset
from models.baseline import Popularity
from metrics import get_eval_metrics

if __name__ == '__main__':
    k = 10
    ds = ML1mDataset()
    pop_model = Popularity()
    pop_scores = pop_model.predict(ds)
    ncdg, apak, hr = get_eval_metrics(pop_scores, ds, k)
