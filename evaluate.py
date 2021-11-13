import argparse

import pytorch_lightning as pl

from dataset import ML1mDataset
from metrics import get_eval_metrics
from models import *
from utils import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="k")
    parser.add_argument("--embedding-dim", type=int, default=32, help="embedding-dim")
    parser.add_argument("--n-negative-samples", type=int, default=4,
                        help="number of negative examples for neg sampling")
    parser.add_argument("--n-workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    model = VanillaMF

    ds = ML1mDataset(n_workers=args.n_workers)
    n_users, n_items = ds.train_ds.n_users, ds.train_ds.n_items

    if model in (Popularity, AlsMF):
        model = model(args.embedding_dim)
        model.fit(ds)
        scores = model(ds)
        labels = ds.test_ds.test_pos[:, [1]]
        ndcg, apak, hr = get_eval_metrics(scores, labels, args.k)
        metrics = {
            'ndcg': ndcg,
            'apak': apak,
            'hr': hr,
        }
        print(metrics)
    else:
        pl.seed_everything(args.seed)
        model = model(n_users, n_items, args.embedding_dim)
        recommender = Engine(model, args.n_negative_samples, k=args.k)
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
