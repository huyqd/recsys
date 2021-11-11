import pytorch_lightning as pl

from dataset import ML1mDataset
from metrics import get_eval_metrics
from models import MODELS
from recsys.utils import LightningMF

if __name__ == '__main__':
    k = 10
    embedding_dim = 20
    model_name = "TorchMF"

    ds = ML1mDataset()
    n_users, n_items = ds.train_ds.n_users, ds.train_ds.n_items

    if model_name in ("Popularity", "AlsMF"):
        model = MODELS[model_name](embedding_dim)
        model.fit(ds)
        scores = model(ds)
        labels = ds.test_ds.test_pos[:, [1]]
        ncdg, apak, hr = get_eval_metrics(scores, labels, k)
    else:
        model = MODELS[model_name](n_users, n_items, embedding_dim)
        recommender = LightningMF(model)
        trainer = pl.Trainer(
            max_epochs=200,
            logger=False,
            check_val_every_n_epoch=1,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
        )

        trainer.fit(recommender, train_dataloaders=ds.train_dl, val_dataloaders=ds.test_dl)
