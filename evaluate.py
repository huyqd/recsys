import pytorch_lightning as pl

from dataset import ML1mDataset
from metrics import get_eval_metrics
from models import *
from utils import Engine

if __name__ == '__main__':
    k = 10
    embedding_dim = 32
    n_negative_samples = 4
    model = VanillaMF

    ds = ML1mDataset(n_workers=0)
    n_users, n_items = ds.train_ds.n_users, ds.train_ds.n_items

    if model in (Popularity, AlsMF):
        model = model(embedding_dim)
        model.fit(ds)
        scores = model(ds)
        labels = ds.test_ds.test_pos[:, [1]]
        ndcg, apak, hr = get_eval_metrics(scores, labels, k)
        metrics = {
            'ndcg': ndcg,
            'apak': apak,
            'hr': hr,
        }
        print(metrics)
    else:
        model = model(n_users, n_items, embedding_dim)
        recommender = Engine(model, n_negative_samples)
        # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
            max_epochs=10,
            logger=False,
            check_val_every_n_epoch=1,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            # callbacks=[lr_monitor],
        )

        trainer.fit(recommender, train_dataloaders=ds.train_dl, val_dataloaders=ds.test_dl)
